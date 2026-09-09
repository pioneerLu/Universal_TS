########################################################################################################
# LoRA (Low-Rank Adaptation) Layers for RWKV Time Series Model
########################################################################################################

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA适配的线性层
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 获取原始层的输入输出维度
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 创建LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 冻结原始层参数
        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 原始前向传播
        original_output = self.original_layer(x)

        # LoRA适配
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """将LoRA权重合并到原始权重中（用于推理加速）"""
        with torch.no_grad():
            merged_weight = self.original_layer.weight + \
                           (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.copy_(merged_weight)

            # 清空LoRA参数
            self.lora_A.zero_()
            self.lora_B.zero_()

class LoRAWKVModel(nn.Module):
    """
    包装RWKV模型以支持LoRA微调
    """
    def __init__(self, model, target_modules=['receptance', 'key', 'value', 'output'], rank=8, alpha=16):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha

        self._replace_modules()

    def _replace_modules(self):
        """递归替换目标模块为LoRA版本"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, nn.Linear):
                # 检查是否是目标模块
                is_target = any(target in name for target in self.target_modules)
                if is_target:
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]

                    parent = self.model
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)

                    # 替换为LoRA层
                    lora_layer = LoRALinear(module, self.rank, self.alpha)
                    setattr(parent, attr_name, lora_layer)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def merge_lora_weights(self):
        """将所有LoRA权重合并到原始权重中（用于推理加速）"""
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

    def get_lora_parameters(self):
        """获取所有LoRA参数"""
        lora_params = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
        return lora_params

    def get_trainable_parameters(self):
        """获取所有可训练参数（包括LoRA和其他可能需要训练的参数）"""
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
