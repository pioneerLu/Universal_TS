"""CLI helpers."""

import datetime
import os

import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise TypeError("Boolean value expected.")


def parse_indices(value):
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        try:
            return [int(x.strip()) for x in value.strip("[]").split(",") if x.strip() != ""]
        except ValueError:
            return [0]
    try:
        return [int(x) for x in value]
    except (ValueError, TypeError):
        return [0]


def apply_feature_args(args):
    if not set(args.select_indices).issubset(set(args.feature_used)):
        raise ValueError("select_indices must be a subset of feature_used")
    args.num_vars = len(args.feature_used)
    args.select_indices_positions = [args.feature_used.index(idx) for idx in args.select_indices]
    return args


def configure_runtime(args, mode="train"):
    """Set derived args and env vars."""
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = mode == "train"
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz

    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)

    use_cpu = getattr(args, "device", "cuda") == "cpu" or not torch.cuda.is_available()
    if use_cpu:
        os.environ["Mode"] = "inference_cpu"
    elif mode == "infer":
        os.environ["Mode"] = "cuda"
    else:
        os.environ["Mode"] = "train"

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

    os.makedirs(args.proj_dir, exist_ok=True)

    if getattr(args, "accumulate_grad_batches", None) in (None, 0):
        args.accumulate_grad_batches = 1

    if args.precision in (32, "32"):
        args.precision = "fp32"
    elif args.precision in (16, "16"):
        args.precision = "fp16"
    args.precision = str(args.precision)

    if args.precision not in ["fp32", "tf32", "fp16", "bf16"]:
        raise ValueError(f"Unsupported precision: {args.precision}")
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    os.environ["RWKV_JIT_ON"] = "0" if "deepspeed_stage_3" in str(args.strategy) else "1"

    if torch.cuda.is_available() and not use_cpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        allow_tf32 = args.precision != "fp32"
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    if "32" in str(args.precision):
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"
    return args


def load_timeseries_array(path):
    """Load npy/csv as [T, F]."""
    import numpy as np

    try:
        data = np.load(path, allow_pickle=True)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 3:
            data = data[0]
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError(f"Unsupported npy shape: {data.shape}")
        return data
    except (ValueError, OSError):
        import pandas as pd
        data = pd.read_csv(path).values
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
