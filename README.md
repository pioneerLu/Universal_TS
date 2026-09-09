# Universal-RWKV-TS

基于 RWKV-7 的时间序列预测框架，用于风电 / 光伏功率预测：先在大规模单变量序列上预训练，再在场站数据上做多变量微调或 LoRA 适配。

## 功能

| 入口 | 作用 |
|------|------|
| `train.py` | 全量预训练 / 微调 |
| `inference.py` | 自回归推理（CUDA / CPU） |
| `lora_train.py` / `lora_inference.py` | LoRA 微调与推理 |
| `preprocess_data.py` | 单变量 npy 窗口缓存（加速多卡训练） |
| `scripts/download_dataset.py` | 下载 UTSD 预训练数据 |

## 环境

建议 Python 3.10+，PyTorch 2.1+，`pytorch-lightning==1.9.5`。

```bash
pip install -r requirements.txt
# 或
conda env create -f environment.yml
```

数据和权重不进入 git。本地数据放在 `data/`，checkpoint 放在 `out/`。

## 数据

| 目录 | 用途 | `--dataset_type` |
|------|------|------------------|
| `data/pretrain/npy` | 小规模单变量 npy，预训练 / LoRA 默认 | `npy` |
| `data/pretrain/utsd` | UTSD HuggingFace 落盘 | `utsd` |
| `data/pretrain/utsd_npy/{1g,4g,12g}` | UTSD 转成 npy 后的预训练集 | `npy` |
| `data/station/wind` | 风电场站，默认 `f1-4.npy` `[T, 4]` | `multi_npy` |
| `data/station/pv` | 光伏场站，默认 `f1-9.npy` `[T, 8]` | `multi_npy` |
| `data/gift-eval/datasets` | GIFT-Eval 预训练 arrow（`single` / `Multivariate`） | — |
| `data/gift-eval/npy` | GIFT 转成的单变量 npy | `npy` |
| `data/benchmark` | ETT / ECL 等 csv | `test` / `multi` |
| `data/cache` | `preprocess_data.py` 窗口缓存 | — |
| `data/toy` | `examples/generate_toy_data.py` 生成 | — |
| `data/misc` | 未接入训练流程的旧数据 | — |

- `npy`：目录内多个 `.npy`，形状 `[N, T, 1]`，或单个文件。
- `multi_npy`：二维 `[T, C]`。
- `utsd`：`datasets.load_from_disk` 目录。

```bash
python examples/generate_toy_data.py --out data/toy
python scripts/download_dataset.py --output data/pretrain/utsd
# GIFT-Eval 官方评测：从仓库根目录 export GIFT_EVAL=data/gift-eval/datasets
```

## 训练

在项目根目录运行：

```bash
# 单变量预训练（示例）
bash scripts/train/uni_rwkvts_pretrain.sh

# 风电/光伏多变量微调
DATA_FILE=data/station/wind/f1-4.npy CKPT=out/pretrained/xxx.pth bash scripts/train/multi_from_uni.sh

# LoRA
CKPT=out/pretrained/xxx.pth bash scripts/train/lora_finetune.sh
```

常用参数：

- `--ctx_len` 历史窗口，`--forecast_len` 预测步长（常用 96）
- `--n_layer 6 --n_embd 512`
- `--dataset_type npy|multi_npy|utsd|csv|multi`
- `--feature_used` / `--select_indices` 输入通道与要预测的通道
- `--device cuda|cpu`，CPU 训练会跳过 CUDA kernel 编译

## 推理

```bash
python inference.py \
  --load_model out/pretrained/xxx.pth \
  --data_file data/station/wind/f1-4.npy \
  --ctx_len 816 --n_layer 6 --n_embd 512 \
  --feature_used 0,3 --select_indices 3 \
  --horizon 96 --device cpu \
  --proj_dir out/infer
```

## 预处理缓存

```bash
python preprocess_data.py \
  --data_file data/pretrain/npy \
  --cache_dir data/cache/pretrain \
  --input_len 816 --split 0.8 --norm True
```

训练时加 `--cache_dir data/cache/pretrain`。`input_len` 必须与 `--ctx_len` 一致。

## 仓库布局

```
train.py / inference.py / lora_*.py / preprocess_data.py
src/          模型、数据、LoRA、训练回调
cuda/         WKV7 CUDA / CPU C++ 扩展源码
scripts/      训练、推理、数据下载
examples/     实例数据生成
data/         本地数据集（不入库）
out/          训练输出与权重
```

`data/`、`out/`、`wandb/`、checkpoint、notebook 均被 `.gitignore` 排除。

## 许可

Apache-2.0
