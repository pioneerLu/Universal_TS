########################################################################################################
# RWKV LoRA Fine-tuning Script - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

from src.cli import str2bool, parse_indices, apply_feature_args, configure_runtime

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    import pytorch_lightning as pl

    rank_zero_info("########## RWKV LoRA Fine-tuning ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str, help="path of rwkv model")  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--run_name", default='lora_finetune_run', type=str,
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
    parser.add_argument("--dataset_type",default='npy',type=str)
    parser.add_argument("--num_vars", default=1, type=int)
    parser.add_argument("--start_var_idx", default=0, type=int)
    parser.add_argument("--forecast_len",default=1, type=int)
    parser.add_argument("--feature_used", default=[0], type=parse_indices)
    parser.add_argument("--select_indices", default=[0], type=parse_indices)
    parser.add_argument("--loss_type",default="mse",type=str)
    parser.add_argument('--do_normalize', type=str2bool, default=False)
    parser.add_argument("--eps",default=1e-5,type=float)
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"], help="Device to run train/validate on")
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache directory for preprocessed data (only for npy dataset)")

    # LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default="receptance,key,value,output",
                       help="Comma-separated list of module names to apply LoRA")

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    apply_feature_args(args)

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    if args.strategy and "deepspeed" in str(args.strategy):
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    configure_runtime(args, mode="train")

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except Exception:
        deepspeed_version = None
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 LoRA Fine-tuning on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.dataset_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
# LoRA = rank {args.lora_rank}, alpha {args.lora_alpha}, target_modules: {args.lora_target_modules}
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

    from src.trainer import train_callback
    from src.lora_model import UniversalRWKVTimeSeriesLoRA
    from src.data_factory import build_datasets, build_dataloaders
    from src.checkpoint import load_weights

    model = UniversalRWKVTimeSeriesLoRA(args)
    if args.load_model:
        try:
            load_weights(model, args.load_model, strict=False)
        except Exception as e:
            rank_zero_info(f"Warning: Failed to load pretrained weights: {e}. Starting from random initialization.")

    train_data, val_data = build_datasets(args)
    data_loader, val_loader = build_dataloaders(train_data, val_data, args)

    print('IF VALIDATE ONLY: ',args.validate_only)
    if args.validate_only == 1:
        trainer = Trainer.from_argparse_args(
            args,
            max_epochs=0,
            callbacks=[train_callback(args)],
            check_val_every_n_epoch=1,
            enable_checkpointing=False
        )
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
            check_val_every_n_epoch=1
        )

    if args.strategy and "deepspeed" in str(args.strategy):
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        rank_zero_info('deepspeed config:', trainer.strategy.config)

    if args.validate_only==1:
        if args.load_model:
            rank_zero_info(f"Loaded LoRA model from {args.load_model}")
        else:
            raise ValueError("Please provide a checkpoint path with --load_model for validation")
        # 选择设备
        device = torch.device(args.device)
        model = model.to(device)
        model = model.to(dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, data_loader, val_loader)
        # 选择设备
        device = torch.device(args.device)
        model = model.to(device)
        model = model.to(dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        trainer.validate(model, val_loader)
    # trainer.fit(model, data_loader,val_loader)

    # trainer.validate(model, val_loader)
