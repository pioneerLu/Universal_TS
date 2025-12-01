########################################################################################################
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
def parse_indices(value):

    if isinstance(value, int):
        return [value]
    
    if isinstance(value, str):
        try:
            # "[1,2,3]" / "1,2,3"
            return [int(x.strip()) for x in value.strip('[]').split(',')]
        except ValueError:
            # 默认1
            return [1]
    try:
        return [int(x) for x in value]
    except (ValueError, TypeError):
        return [1]
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str, help="path of rwkv model")  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--run_name", default='demo_run', type=str, 
                        help="run name for wandb. force to consider what is the purpose of this run")
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

    parser.add_argument("--n_emb_layer", default=4, type=int)
    parser.add_argument("--sma_window", default=3, type=int)
    parser.add_argument("--validate_only", default=0, type=int)
    parser.add_argument("--num_vars", default=1, type=int)
    parser.add_argument("--start_var_idx", default=0, type=int)
    parser.add_argument("--forecast_len",default=1, type=int)
    parser.add_argument("--feature_used", default=[0], type=parse_indices)
    parser.add_argument("--select_indices", default=[0], type=parse_indices)
    parser.add_argument("--loss_type",default="mse",type=str)
    parser.add_argument('--horizon',default=100,type=int)
    parser.add_argument("--do_normalize",default=False,type=bool)
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"], help="Device to run inference on")
    
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    

    assert set(args.select_indices).issubset(set(args.feature_used)),'select_indices should be a subset of feature_used'
    args.num_vars = len(args.feature_used)
    args.select_indices_positions = [args.feature_used.index(idx) for idx in args.select_indices]

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    # 根据设备设置 MODE
    if args.device == 'cpu':
        os.environ["Mode"] = 'inference_cpu'
    else:
        os.environ["Mode"] = 'cuda'
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    #args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["json"]

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################
    from src.trainer import train_callback
    # import pdb
    # pdb.set_trace()
    from src.model import UniversalRWKVTimeSeries

    if args.load_model:
        ckpt_path = os.path.join(args.proj_dir, args.load_model)
        # model = UniversalRWKVTimeSeries.load_from_checkpoint(torch.load(ckpt_path, map_location="cpu"))
        from io import BytesIO

        with open(args.load_model, 'rb') as f:
            buffer = BytesIO(f.read())
        model = UniversalRWKVTimeSeries(args)
        checkpoint = torch.load(buffer, map_location='cpu')
        
        if 'pytorch-lightning_version' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:  # if old version of pytorch-lightning
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[7:]  # remove prefix 'encoder.' or 'decoder.' for DDP/DDP
                if k.startswith('_forward_module.'):
                    k = k[16:]
                new_state_dict[k] = v
            # import pdb; pdb.set_trace()
            
            filtered_dict = {k: v for k, v in new_state_dict.items() if "smooth.conv.weight" not in k}

            model.load_state_dict(filtered_dict,strict=False)

        model.eval()
        # Use bfloat16 on CUDA, float32 on CPU (CPU may not support bfloat16)
        if args.device == 'cuda' and torch.cuda.is_available():
            model = model.to(dtype=torch.bfloat16)
        else:
            # CPU inference - use float32
            model = model.to(dtype=torch.float32)
        

        rank_zero_info(f"Loaded pretrained RWKV from {ckpt_path}")

    else:
        raise ValueError("Please provide a checkpoint path with --load_model")
    


    def autoregressive_predict(model, initial_sequence, horizon, window_size, device, feature_count):
        """
        Generate autoregressive predictions using the model.
        
        Args:
            model: The RWKV time series model
            initial_sequence: Starting sequence data (numpy array) with shape [T, F] where T is time steps and F is features
            horizon: Number of steps to predict
            window_size: Context window size
            device: Device to run inference on
            feature_count: Number of features
            
        Returns:
            Array of predictions with shape [T, F] where F corresponds to select_indices features
        """
        model.to(device)
        if len(initial_sequence.shape) != 2:
            print(f"Expected 2D input with shape [T, F], got {initial_sequence.shape}. Reshaping...")
            if len(initial_sequence.shape) == 3 and initial_sequence.shape[0] == 1:
                # Remove batch dimension if present
                initial_sequence = initial_sequence[0]
            else:
                # Try to reshape or transpose as needed
                initial_sequence = initial_sequence.reshape(-1, feature_count)
        
        print(f"Input sequence shape: {initial_sequence.shape}")
        
        # Extract relevant portion that fits in context window
        if initial_sequence.shape[0] > window_size:
            context = initial_sequence[-window_size:, :]
            print(f"Using last {window_size} time steps from input sequence")
        else:
            # Pad if needed
            pad_size = window_size - initial_sequence.shape[0]
            context = np.pad(initial_sequence, ((pad_size, 0), (0, 0)), mode='edge')
            print(f"Padded input sequence with {pad_size} time steps to reach window size {window_size}")
        
        # Add batch dimension for model input [1, T, F]
        input_seq = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
        # Use bfloat16 only on CUDA, float32 on CPU
        # device can be either a string or torch.device object
        device_obj = torch.device(device) if isinstance(device, str) else device
        if model.args.precision == "bf16" and device_obj.type == 'cuda':
            input_seq = input_seq.bfloat16()
        
        print(f"Starting autoregressive generation for {horizon} steps")
        
        # Store all predictions
        all_preds = []
        
        with torch.no_grad():
            for step in range(horizon):
                # Forward pass to get prediction
                # import pdb; pdb.set_trace()
                output = model(input_seq)
                
                # Get the last prediction (shape [1, 1, num_selected_features])
                next_pred = output[:, -1:, :]
                # Save prediction without batch dimension for return
                all_preds.append(next_pred.cpu().float().numpy()[0])
                
                # Create tensor with all features for next time step
                full_features = torch.zeros(
                    (1, 1, model.num_vars), 
                    dtype=input_seq.dtype, 
                    device=input_seq.device
                )
                
                # Place the predictions at the selected indices
                for i, idx in enumerate(model.select_indices):
                    full_features[:, :, idx] = next_pred[:, :, i]
                
                # Shift window - remove first element, append prediction
                input_seq = torch.cat([input_seq[:, 1:, :], full_features], dim=1)
                
                if step % 10 == 0 or step == horizon - 1:
                    print(f"Generated step {step+1}/{horizon}")
        
        # Stack all predictions along time dimension [horizon, num_selected_features]
        predictions = np.stack(all_preds, axis=0)
        
        # Remove any extra dimensions to get [T, F]
        predictions = predictions.squeeze()
        
        # If only predicting one feature and one step, ensure we have correct shape
        if predictions.ndim == 0:
            predictions = predictions.reshape(1, 1)
        elif predictions.ndim == 1 and horizon > 1:
            # Single feature case with multiple time steps
            predictions = predictions.reshape(-1, 1)
        
        print(f"Completed autoregressive generation with output shape {predictions.shape}")
        
        return predictions


    def plot_predictions(original_data, predictions, args):
        """
        Plot and save the predictions for each variable in separate figures.
        
        Args:
            original_data: Original time series data with shape [T, F]
            predictions: Predicted values with shape [horizon, num_selected_features]
            args: Command line arguments
        """
        os.makedirs(args.proj_dir, exist_ok=True)
        
        # Get indices of selected features
        select_indices = parse_indices(args.select_indices)
        feature_names = [f"Feature_{i}" for i in select_indices]
        
        context_size = min(96, original_data.shape[0])
        
        # Ensure predictions has correct shape for plotting
        if len(predictions.shape) == 1:
            # If only one feature is predicted, reshape to [T, 1]
            predictions = predictions.reshape(-1, 1)
        
        print(f"Plotting {len(select_indices)} features with historical context of {context_size} time steps")
        print(f"Prediction shape for plotting: {predictions.shape}")
        
        # Create a separate figure for each feature
        for i, feature_idx in enumerate(select_indices):
            plt.figure(figsize=(15, 8))
            
            # Plot original data (context) - 使用取200和原始数据长度中较小的值
            historical_data = original_data[-context_size:, feature_idx]
            historical_x = range(original_data.shape[0] - context_size, original_data.shape[0])
            
            plt.plot(
                historical_x,
                historical_data,
                label=f"Historical {feature_names[i]}", 
                color='blue',
                linewidth=2
            )
            
            # Plot predictions - extracting the right column if multiple features
            pred_values = predictions[:, i] if predictions.shape[1] > 1 else predictions[:, 0]
            pred_x = range(original_data.shape[0], original_data.shape[0] + len(pred_values))
            
            plt.plot(
                pred_x,
                pred_values,
                label=f"Predicted {feature_names[i]}", 
                color='red',
                linestyle='--',
                linewidth=2
            )
            
            # Add vertical line at prediction start point (分割线)
            plt.axvline(x=original_data.shape[0], color='black', linestyle='-', alpha=0.8, linewidth=2)
            
            # Add text annotation for the split line
            plt.text(original_data.shape[0] + len(pred_values) * 0.02, 
                    plt.ylim()[1] * 0.95, 
                    'Prediction Start', 
                    rotation=90, 
                    verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f"{feature_names[i]} - Autoregressive Forecast")
            
            # Add x and y labels
            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            
            # Save individual plot
            save_path = os.path.join(args.proj_dir, f"forecast_plot_feature_{feature_idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"Saved prediction plot for feature {feature_idx} to {save_path}")
        
        # Save raw predictions
        np_save_path = os.path.join(args.proj_dir, "predictions.npy")
        np.save(np_save_path, predictions)
        
        # Also save as CSV for easier analysis
        import pandas as pd
        pred_df = pd.DataFrame(predictions, columns=[f"Feature_{i}" for i in range(predictions.shape[1])])
        csv_path = os.path.join(args.proj_dir, "predictions.csv")
        pred_df.to_csv(csv_path, index=False)
        
        print(f"Saved predictions to {np_save_path} and {csv_path}")

    def load_data(args):
        try:
            # Load all columns from the original data file
            data = np.load(args.data_file)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim == 2:
                pass  # Already 2D
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
            print(f"Loaded numpy data with shape {data.shape}")
            return data
        except:
            import pandas as pd
            data = pd.read_csv(args.data_file).values
            print(f"Loaded CSV data with shape {data.shape}")
            return data
    
    def normalize(data,args):
        """
        Normalize each channel independently.
        Note: data should contain all original features, not just feature_used.
        """
        means = np.mean(data[-args.ctx_len:], axis=0)
        stds = np.std(data[-args.ctx_len:], axis=0)
        # Avoid division by zero
        stds[stds == 0] = 1e-5
        normalized = (data - means) / stds
        return normalized, means, stds

    def denormalize(data, means, stds):
        """
        Denormalize each channel using its own mean and std.
        
        Args:
            data: Normalized data with shape [T, F]
            means: Array of means for each feature
            stds: Array of standard deviations for each feature
            
        Returns:
            denormalized_data: Original scale data
        """
        return data * stds + means
    
    # Load all original data
    input_data_full = load_data(args)
    
    # Select only the features specified in feature_used
    input_data_selected = input_data_full[:, args.feature_used]
    
    # Normalize the selected features
    input_data_norm, means_selected, stds_selected = normalize(input_data_selected, args)
    
    # Also compute means and stds for all original features (needed for denormalization)
    _, means_full, stds_full = normalize(input_data_full, args)

    feature_count = len(parse_indices(args.feature_used))
    predictions_norm = autoregressive_predict(
        model=model,
        initial_sequence=input_data_norm,
        horizon=args.horizon,
        window_size=args.ctx_len,
        device=args.device,
        feature_count=feature_count
    )

    # Use original feature indices to access means and stds from full data
    selected_means = means_full[args.select_indices]
    selected_stds = stds_full[args.select_indices]
    predictions = denormalize(predictions_norm, selected_means, selected_stds)

    # Pass full original data for plotting (needed to access features by original indices)
    plot_predictions(input_data_full, predictions, args)
