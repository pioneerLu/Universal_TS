########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import pandas as pd
from .utils import compress_parameter_names
import matplotlib.pyplot as plt
from .timer.models.Timer import Model as TimerModel
CHUNK_LEN = 16

class Timer_Configs:
    def __init__(self):
        self.ckpt_path = ''  # 如果有预训练模型路径，可以在这里指定
        self.patch_len = 16
        self.d_model = 512
        self.d_ff = 2048
        self.e_layers = 6
        self.n_heads = 8
        self.dropout = 0.1
        self.output_attention = False
        self.factor = 3
        self.activation = 'gelu'

timer_configs = Timer_Configs()

# #  Timer test # # 
# model = TimerModel(timer_configs).cuda()
# B, L, M = 12, 96, 7  # 批量大小、序列长度、特征数量
# x_enc = torch.randn(B, L, M).cuda()
# output = model(x_enc).cuda()
# print(output.shape)
# print(output)  # 输出形状应为 [B, T, D]

class WaveNetEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=2, dilation_base=2):
        super(WaveNetEmbedding, self).__init__()
        self.layers = nn.ModuleList()
        current_dilation = 1
        for _ in range(num_layers):
            current_padding = (kernel_size - 1) * current_dilation
            # 1D 因果卷积层
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 因果卷积通常使用 kernel_size=2
                    dilation=current_dilation,
                    padding=current_padding  # 保持输出长度与输入长度相同
                )
            )
            in_channels = out_channels
            current_dilation *= dilation_base

        for name, layer in enumerate(self.layers):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch_size, in_channels, sequence_length)
        for layer in self.layers:
            x = layer(x)
            x = x[:, :, :x.size(2) - (layer.dilation[0])]
            x = self.activation(x)
        # (batch_size, out_channels, sequence_length)
        return x
    
# class SharedCausalCNN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers, kernel_size=2, dilation_base=2):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         current_dilation = 1
        
#         for _ in range(num_layers):
#             current_padding = (kernel_size - 1) * current_dilation
#             layer = nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 dilation=current_dilation,
#                 padding=current_padding
#             )
#             self.layers.append(layer)
            
#             in_channels = out_channels
#             current_dilation *= dilation_base

#         self.activation = nn.ReLU()

    # def forward(self, x):
    #     # x: [B, N, T] 
    #     B, N, T = x.size()
        
    #     x = x.view(B * N, 1, T)
        
    #     for layer in self.layers:
    #         x = layer(x)
    #         x = x[:, :, :x.size(2) - (layer.dilation[0])]
    #         x = self.activation(x)

    #     x = x.view(B, N, -1)
    #     return x

class CausalMovingAverage(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(1, 1, kernel_size=window_size, stride=1, padding=window_size-1, bias=False)
        nn.init.constant_(self.conv.weight, 1/window_size)  # 固定权重
        self.conv.weight.requires_grad_(False)  

    def forward(self, x):
        # [B, T, 1]
        x = x.transpose(1, 2)  # [B, 1, T]
        x = self.conv(x)  # out [B, 1, T + window_size-1]
        x = x[:, :, :-self.window_size+1]  
        return x.transpose(1, 2)  # [B, T, 1]
    
class CLSModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 后面在一个 batch 内会复制扩展
        self.cls_token = nn.Parameter(torch.randn(1,1,1,hidden_dim,))

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, T+1, D]
        """
        B = x.size(0)
        num_vars = x.size(2)
        # 在 batch 维度上复制
        cls_token = self.cls_token.expand(B, -1, num_vars,-1)  # [B, 1,num_vars, D]
        return torch.cat([cls_token, x], dim=1)       # [B, T+1,num_vars, D]


class MultiVariableAttention(nn.Module):
    def __init__(self, num_vars, embedding_dim, num_heads=4):
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.var_interaction = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        
        B, T,num_vars, D = x.size()
        
        # [B*T, num_vars,D]
        x = x.view(B * T,num_vars,D)
        
        attn_output, _ = self.attention(x, x, x)
        
        # 重塑回 [B, num_vars, T, D]
        attn_output = attn_output.view(B,T, num_vars, D)
        
        # x = attn_output.mean(dim=1)
        x = attn_output  # [B, T, num_vars, D]
        x = self.var_interaction(x)  # [B, T,num_vars, D]
        return x


class UniversalRWKVTimeSeries(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = WaveNetEmbedding(1, args.n_embd, args.n_emb_layer, dilation_base=2)
        self.cls_module = CLSModule(args.n_embd)
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, 1, bias=False)
        self.num_vars = args.num_vars
        self.att = MultiVariableAttention(num_vars=self.num_vars, embedding_dim=args.n_embd,num_heads=4)
        self.sma_window = getattr(args, 'sma_window', 3) # default 3
        self.loss_type = getattr(args, 'loss_type', 'mse')
        self.start_var_idx = args.start_var_idx
        self.select_indices = args.select_indices
        self.pre_len = args.forecast_len
        self.timer = TimerModel(timer_configs)
        if self.sma_window > 1:
            self.smooth = CausalMovingAverage(self.sma_window)
        
        self.best_val_loss = float('inf')
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


    def configure_optimizers(self):
        zero_weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad] 

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        optim_groups = []
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [{"params": weight_decay_group, "weight_decay": self.args.weight_decay}]
                rank_zero_info(f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}")
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    # def pad_left(self, x, num_tokens_to_pad):
    #     # pad left with eos token embedding
    #     if num_tokens_to_pad != 0:
    #         # left padding by add eos token at the beginning
    #         pad_emb = torch.zeros(
    #             x.size(0), num_tokens_to_pad, x.size(2),
    #             device=x.device, dtype=x.dtype
    #             )
    #         x = torch.cat((pad_emb, x), dim=1)
    #     return x

    def pad_left(self, x, num_tokens_to_pad):
        # x: [B, T,num_vars, D]
        if num_tokens_to_pad != 0:
            pad_emb = torch.zeros(
                x.size(0),# Batch size
                num_tokens_to_pad,  # Padding tokens
                x.size(2),  # Number of variables 
                x.size(3),  # Embedding dimension
                device=x.device, 
                dtype=x.dtype
            )
            x = torch.cat((pad_emb, x), dim=1)
        return x
    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:,:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        # print(x.shape)
        # x: [B, T, num_vars]
        B, T, C = x.size()                         ## INPUT
        x = self.timer(x)
        # print(x.shape)   #torch.Size([128, 96, 10]) (B, T , num_vars)

        # look_back_len = T - self.pre_len
        # x = x[:,:look_back_len,:].contiguous()  # [B, look_back_ken, num_vars]
        # print(x.shape, look_back_len, T) 
       

        return x
    def training_step(self, batch, batch_idx):
        '''
        batch: dict with keys "input_ids", "labels" and "input_text"
        '''
        seq_x = batch["seq_x"]
        if self.args.precision == "bf16":
            seq_x = seq_x.bfloat16() 

        
        predicts = self(seq_x[..., :-self.pre_len, :].contiguous())
        targets = seq_x[..., self.pre_len:, self.select_indices].contiguous()
        # shift_predicts = predicts[..., :-self.pre_len, :].contiguous()
        # shift_targets = targets[..., self.pre_len:, self.select_indices].contiguous()


        if self.loss_type == 'mse':
            loss = F.mse_loss(predicts, targets)

        elif self.loss_type == 'mae':
            loss = F.l1_loss(predicts, targets)
        return loss

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
    
    def validation_step(self, batch, batch_idx):
        seq_x = batch["seq_x"]
        if self.args.precision == "bf16":
            seq_x = seq_x.bfloat16()
    
        
        predicts = self(seq_x[..., :-self.pre_len, :].contiguous())
        targets = seq_x[..., self.pre_len:, self.select_indices].contiguous()
        
        # shift_predicts = predicts[..., :-self.pre_len, :].contiguous()
        # shift_targets = targets[..., self.pre_len:, self.select_indices].contiguous()

        if self.loss_type == 'mse':
            loss = F.mse_loss(predicts,targets)

        elif self.loss_type == 'mae':
            loss = F.l1_loss(predicts, targets)
        
        

        return {
            "loss": loss,
            "predicts": predicts.detach(),
            "targets": targets.detach()
        }

    def validation_epoch_end(self, outputs):
        # 多GPU数据聚合，如果不开多卡可以注释掉这里
        all_predicts = self.all_gather(torch.cat([out["predicts"] for out in outputs]))
        all_targets = self.all_gather(torch.cat([out["targets"] for out in outputs]))
        
        predicts_numpy = all_predicts.cpu().float().numpy()
        targets_numpy = all_targets.cpu().float().numpy()
        
        save_dir = os.path.join(self.args.proj_dir, "data_records")
        os.makedirs(save_dir, exist_ok=True)
        self._save_to_excel(predicts_numpy, targets_numpy, save_dir)
        
        val_loss = F.mse_loss(all_predicts, all_targets).item()
        self.log("val_loss_mse", val_loss, sync_dist=True)
        

        if val_loss < self.best_val_loss and self.current_epoch >= 1:
            self._save_best_model(val_loss)
            self._plot_predictions(outputs, val_loss)

    # def _save_to_excel(self, preds, targets, save_dir):
    #     df = pd.DataFrame({
    #         'predict': preds.flatten(),
    #         'target': targets.flatten()
    #     })
    #     save_path = os.path.join(save_dir, f"epoch_{self.current_epoch}.xlsx")
    #     df.to_excel(save_path, index=False)

    def _save_to_excel(self, preds, targets, save_dir):
        self._fallback_save(preds, targets, save_dir)


    def _fallback_save(self, preds, targets, save_dir):
        df = pd.DataFrame({
            'predict': preds.flatten(),
            'target': targets.flatten()
        }).iloc[:10000, :]  #太长了会超限
        print(df.shape)
        
        truncate_path = os.path.join(save_dir, f"epoch_{self.current_epoch}_TRUNCATED.csv")
        df.to_csv(truncate_path, index=False)
        
    def _save_best_model(self, val_loss):
        ckpt_dir = os.path.join(self.args.proj_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        old_model = os.path.join(ckpt_dir, f"best-{self.best_val_loss:.3f}.pth")
        if os.path.exists(old_model):
            os.remove(old_model)

        new_model = os.path.join(ckpt_dir, f"best-{val_loss:.3f}.pth")
        torch.save(self.state_dict(), new_model)
        self.best_val_loss = val_loss

    def _plot_predictions(self, outputs, val_loss):
        plot_dir = os.path.join(self.args.proj_dir, "visualization", f"best_plots_{val_loss:.3f}")
        os.makedirs(plot_dir, exist_ok=True)
        
        time_scales = [
            ('day', 1),
            ('week', 7),
            ('month', 30)
        ]
        
        for scale_name, window in time_scales:
            scale_pred = []
            scale_target = []
            
            for i in range(0, len(outputs), window):
                batch_pred = torch.cat([out["predicts"] for out in outputs[i:i+window]])
                batch_target = torch.cat([out["targets"] for out in outputs[i:i+window]])
                scale_pred.append(batch_pred.to(torch.float).cpu().numpy())
                scale_target.append(batch_target.to(torch.float).cpu().numpy())
            
            self._plot_scale(scale_pred, scale_target, plot_dir, scale_name)

    def _plot_scale(self, preds, targets, save_dir, prefix):
        plt.figure(figsize=(15, 6))
        for i, (p, t) in enumerate(zip(preds, targets)):
            plt.clf()
            plt.plot(t.flatten()[:672], label='Target', alpha=0.7)
            plt.plot(p.flatten()[:672], label='Prediction', linestyle='--')
            plt.title(f"{prefix.capitalize()} {i+1} Prediction vs Target")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{prefix}_{i}.png"), dpi=150)
            plt.close()



