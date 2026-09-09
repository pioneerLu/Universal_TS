"""Dataset factory."""

import os

SUPPORTED_TYPES = ("utsd", "npy", "csv", "test", "multi", "multi_npy")


def _normalize_flag(args, default=True):
    return bool(getattr(args, "do_normalize", default))


def build_datasets(args):
    dtype = (args.dataset_type or "").lower().strip()
    if dtype in ("ustd", "utsd"):
        from src.dataset import UTSDataset

        common = dict(
            dataset_path=args.data_file,
            epoch_steps=args.epoch_steps,
            micro_bsz=args.micro_bsz,
            input_len=args.ctx_len,
            output_len=0,
            scale=_normalize_flag(args, default=True),
        )
        train_data = UTSDataset(**common, flag="train")
        val_data = UTSDataset(**common, flag="val")
        return train_data, val_data

    if dtype == "csv":
        from src.TFB_data import DatasetForTransformer, read_data

        series = read_data(args.data_file, normalize=_normalize_flag(args, default=True))[0]
        train_data = DatasetForTransformer(
            dataset=series, history_len=100, prediction_len=1, label_len=100
        )
        val_data = DatasetForTransformer(
            dataset=series, history_len=100, prediction_len=1, label_len=100
        )
        return train_data, val_data

    if dtype == "npy":
        from src.universal_dataset import TimeSeriesDataset

        common = dict(
            dataset_path=args.data_file,
            input_len=args.ctx_len,
            output_len=0,
            split=0.8,
            norm=_normalize_flag(args, default=True),
            cache_dir=getattr(args, "cache_dir", None),
        )
        train_data = TimeSeriesDataset(**common, flag="train")
        val_data = TimeSeriesDataset(**common, flag="val")
        return train_data, val_data

    if dtype == "test":
        from src.universal_dataset import SingleVarDataset

        ratio = [0.7, 0.2, 0.1]
        zero_shot = 1 if args.validate_only == 1 else 0
        common = dict(
            data_path=args.data_file,
            size=(args.ctx_len, 1),
            normalize=_normalize_flag(args, default=True),
            target="OT",
            stride=1,
            split=ratio,
            zero_shot=zero_shot,
        )
        train_data = SingleVarDataset(**common, flag="train")
        val_data = SingleVarDataset(**common, flag="val")
        return train_data, val_data

    if dtype == "multi":
        from src.universal_dataset import MultiVariateTimeSeriesDataset

        ratio = [0.8, 0.2, 0]
        common = dict(
            data_path=args.data_file,
            size=(args.ctx_len, 1),
            features=args.feature_used,
            split=ratio,
            target=args.select_indices,
            normalize=_normalize_flag(args, default=True),
        )
        train_data = MultiVariateTimeSeriesDataset(**common, flag="train")
        val_data = MultiVariateTimeSeriesDataset(**common, flag="val")
        return train_data, val_data

    if dtype == "multi_npy":
        from src.universal_dataset import MultiVariateNPYTimeSeriesDataset

        ratio = [0.8, 0.2, 0]
        common = dict(
            data_path=args.data_file,
            size=(args.ctx_len, 1),
            features=args.feature_used,
            split=ratio,
            target=args.select_indices,
            normalize=_normalize_flag(args, default=True),
        )
        train_data = MultiVariateNPYTimeSeriesDataset(**common, flag="train")
        val_data = MultiVariateNPYTimeSeriesDataset(**common, flag="val")
        return train_data, val_data

    raise ValueError(
        f"Unknown dataset_type={args.dataset_type!r}. Choose from: {', '.join(SUPPORTED_TYPES)}"
    )


def build_dataloaders(train_data, val_data, args):
    import torch
    from torch.utils.data import DataLoader

    use_pin_memory = getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available()
    num_workers = 0 if getattr(args, "device", "cuda") == "cpu" else min(2, os.cpu_count() or 1)
    common = dict(
        pin_memory=use_pin_memory,
        batch_size=args.micro_bsz,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
