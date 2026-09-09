#!/usr/bin/env python3
"""CPU smoke tests."""

import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("RWKV_JIT_ON", "0")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
os.environ["Mode"] = "inference_cpu"


def _make_args(**kwargs):
    defaults = dict(
        n_layer=1,
        n_embd=64,
        n_emb_layer=2,
        dim_att=64,
        dim_ffn=128,
        num_vars=1,
        select_indices=[0],
        select_indices_positions=[0],
        start_var_idx=0,
        forecast_len=4,
        sma_window=1,
        dropout=0,
        do_normalize=False,
        eps=1e-5,
        loss_type="mse",
        precision="fp32",
        lr_init=1e-3,
        betas=(0.9, 0.99),
        adam_eps=1e-8,
        weight_decay=0,
        grad_cp=0,
        pre_ffn=0,
        head_size_a=64,
        head_size_divisor=8,
        ctx_len=32,
        strategy="auto",
        proj_dir="out/smoke",
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules="receptance,key,value,output",
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_timeseries_dataset_and_preprocess():
    from src.universal_dataset import TimeSeriesDataset
    from preprocess_data import preprocess_npy_dataset

    with tempfile.TemporaryDirectory() as td:
        data = np.sin(np.linspace(0, 20, 256, dtype=np.float32)).reshape(1, -1, 1)
        np.save(os.path.join(td, "uni.npy"), data)
        ds = TimeSeriesDataset(
            dataset_path=td, flag="train", split=0.8, input_len=32, output_len=0, norm=True
        )
        assert len(ds) > 0
        sample = ds[0]
        assert sample["seq_x"].shape[0] == 32

        cache_dir = os.path.join(td, "cache")
        preprocess_npy_dataset(
            data_file=td,
            cache_dir=cache_dir,
            input_len=32,
            output_len=0,
            split=0.8,
            norm=True,
            stride=1,
            batch_size=64,
            merge_interval=10,
            async_merge=False,
        )
        assert os.path.exists(os.path.join(cache_dir, "cache_info.json")) or os.path.exists(
            os.path.join(cache_dir, "chunks_info.json")
        )


def test_forward_and_train_step():
    from src.model import UniversalRWKVTimeSeries

    args = _make_args()
    model = UniversalRWKVTimeSeries(args)
    model.eval()
    x = torch.randn(2, 32, 1)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 2
    assert y.shape[2] == 1

    model.train()
    batch = {"seq_x": torch.randn(2, 32, 1)}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_multivar_dataset():
    from src.universal_dataset import MultiVariateNPYTimeSeriesDataset

    with tempfile.TemporaryDirectory() as td:
        arr = np.random.randn(400, 3).astype(np.float32)
        path = os.path.join(td, "m.npy")
        np.save(path, arr)
        ds = MultiVariateNPYTimeSeriesDataset(
            data_path=path,
            flag="train",
            size=(32, 1),
            features=[0, 1, 2],
            target=[2],
            normalize=True,
            split=[0.8, 0.2, 0],
        )
        item = ds[0]
        assert item["seq_x"].shape == (32, 3)
        assert item["seq_x"].dtype == np.float32


def test_npy_sort_and_no_norm():
    from src.universal_dataset import TimeSeriesDataset

    with tempfile.TemporaryDirectory() as td:
        a = np.ones((1, 128, 1), dtype=np.float32)
        b = np.full((1, 128, 1), 3.0, dtype=np.float32)
        np.save(os.path.join(td, "z_last.npy"), b)
        np.save(os.path.join(td, "a_first.npy"), a)
        ds = TimeSeriesDataset(
            dataset_path=td, flag="train", split=0.8, input_len=32, output_len=0, norm=False
        )
        sample = ds[0]
        assert sample["seq_x"].dtype == np.float32
        assert abs(float(sample["seq_x"].mean()) - 1.0) < 1e-5


def test_csv_header_and_inverse():
    import pandas as pd
    from src.universal_dataset import MultiVariateTimeSeriesDataset

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.csv")
        pd.DataFrame({"date": ["2020-01-01"] * 200, "a": np.arange(200), "b": np.arange(200) * 2}).to_csv(
            path, index=False
        )
        ds = MultiVariateTimeSeriesDataset(
            data_path=path,
            flag="train",
            size=(16, 1),
            features=[0, 1],
            target=[1],
            normalize=True,
            split=[0.8, 0.2, 0],
        )
        item = ds[0]
        assert item["seq_x"].shape[1] == 2
        recovered = ds.inverse_transform(item["seq_y"], variable_idx=1)
        assert recovered.shape[0] > 0


def test_build_dataloaders():
    from src.data_factory import build_dataloaders
    from src.universal_dataset import TimeSeriesDataset

    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, "uni.npy"), np.sin(np.linspace(0, 20, 256, dtype=np.float32)).reshape(1, -1, 1))
        ds = TimeSeriesDataset(dataset_path=td, flag="train", split=0.8, input_len=32, output_len=0, norm=False)
        args = _make_args(micro_bsz=4, device="cpu")
        train_loader, val_loader = build_dataloaders(ds, ds, args)
        from torch.utils.data import RandomSampler, SequentialSampler

        assert isinstance(train_loader.sampler, RandomSampler)
        assert train_loader.drop_last is True
        assert isinstance(val_loader.sampler, SequentialSampler)
        assert val_loader.drop_last is False
        batch = next(iter(train_loader))
        assert batch["seq_x"].shape[0] == 4


def test_lora_wrap():
    from src.lora_model import UniversalRWKVTimeSeriesLoRA

    args = _make_args()
    model = UniversalRWKVTimeSeriesLoRA(args)
    x = torch.randn(1, 32, 1)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 1


if __name__ == "__main__":
    test_timeseries_dataset_and_preprocess()
    print("ok dataset/preprocess")
    test_forward_and_train_step()
    print("ok forward/train_step")
    test_multivar_dataset()
    print("ok multi_npy")
    test_npy_sort_and_no_norm()
    print("ok npy sort/norm")
    test_csv_header_and_inverse()
    print("ok csv header")
    test_build_dataloaders()
    print("ok dataloaders")
    test_lora_wrap()
    print("ok lora")
    print("all smoke tests passed")
