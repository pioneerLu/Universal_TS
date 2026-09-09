# RWKV LoRA 微调指南

本项目支持使用LoRA (Low-Rank Adaptation) 技术对RWKV时间序列模型进行高效微调。LoRA版本的文件都带有`lora_`前缀，与原版本完全独立。

## 文件结构

### LoRA核心文件
- `src/lora_layers.py` - LoRA核心实现（LoRALinear和LoRAWKVModel类）
- `src/lora_model.py` - LoRA版本的模型类（UniversalRWKVTimeSeriesLoRA）

### LoRA脚本文件
- `lora_train.py` - LoRA微调训练脚本
- `lora_inference.py` - LoRA推理脚本
- `scripts/train/lora_finetune.sh` - LoRA微调训练专用脚本（可执行）
- `scripts/inference/lora_predict.sh` - LoRA推理专用脚本（可执行）

### 文档文件
- `README_LoRA.md` - 本文档

## LoRA 原理

LoRA通过在原始权重矩阵上添加低秩矩阵适配器来实现模型微调：
- **冻结原始权重**：预训练模型的权重保持不变
- **添加LoRA适配器**：在指定层添加低秩矩阵A和B
- **只训练适配器**：大幅减少可训练参数数量

## 使用方法

### 方法一：使用Python脚本直接调用

#### 1. LoRA 微调训练

```bash
python lora_train.py \
    --load_model "path/to/pretrained/model.pth" \
    --wandb "rwkvts_lora_finetune" \
    --proj_dir out/lora_finetune \
    --data_file /path/to/your/data \
    --ctx_len 816 \
    --epoch_steps 1000 \
    --epoch_count 10 \
    --micro_bsz 32 \
    --n_layer 6 \
    --n_embd 512 \
    --lr_init 5e-5 \
    --lr_final 1e-6 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target_modules "receptance,key,value,output"
```

#### 2. LoRA 推理

```bash
python lora_inference.py \
    --load_model "out/lora_finetune/checkpoints/best-lora-X.XXX.pth" \
    --proj_dir out/lora_inference \
    --data_file /path/to/your/data \
    --ctx_len 816 \
    --horizon 100 \
    --merge_lora True
```

### 方法二：使用专用Shell脚本

#### 1. LoRA 微调训练

```bash
# 修改脚本中的路径和参数后执行
bash scripts/train/lora_finetune.sh
```

#### 2. LoRA 推理

```bash
# 修改脚本中的路径和参数后执行
bash scripts/inference/lora_predict.sh
```

#### LoRA 参数说明

- `--lora_rank`: LoRA秩，通常设为8或16，越小参数越少
- `--lora_alpha`: LoRA缩放因子，通常设为rank的2倍
- `--lora_target_modules`: 应用LoRA的模块名，用逗号分隔
  - `receptance`: RWKV的门控层
  - `key`: 注意力层的key投影
  - `value`: 注意力层的value投影
  - `output`: 输出投影层

### 2. LoRA 推理

```bash
python lora_inference.py \
    --load_model "out/lora_finetune/checkpoints/best-lora-X.XXX.pth" \
    --proj_dir out/lora_inference \
    --data_file /path/to/your/data \
    --ctx_len 816 \
    --horizon 100 \
    --merge_lora True
```

#### 推理选项

- `--merge_lora True`: 将LoRA权重合并到原始权重中，加速推理
- `--merge_lora False`: 保持LoRA结构，内存占用稍大但更灵活

## 优势特点

### 1. 参数效率
- 原始RWKV模型可能有数亿参数
- LoRA微调通常只需要训练数百万参数（减少90%以上）
- 适合在消费级GPU上进行微调

### 2. 存储效率
- LoRA适配器文件通常只有几MB到几十MB
- 可以与多个不同的适配器共享同一个基础模型

### 3. 推理加速
- 支持权重合并，推理时无额外开销
- 合并后与原模型推理速度相同

### 4. 向后兼容
- LoRA版本与原版本完全独立
- 原有代码和模型文件不受影响

## 技术细节

### LoRA层实现

```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank

        # 创建LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 冻结原始层
        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 原始输出 + LoRA适配
        return self.original_layer(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### 权重合并

推理时可以选择合并LoRA权重：

```python
def merge_lora_weights(self):
    with torch.no_grad():
        merged_weight = self.original_layer.weight + \
                       (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.copy_(merged_weight)
        # 清空LoRA参数以节省内存
        self.lora_A.zero_()
        self.lora_B.zero_()
```

## 最佳实践

### 1. 选择合适的Rank
- 小数据集：rank=4或8
- 大数据集：rank=16或32
- 一般情况下rank=8是一个很好的起点

### 2. 学习率设置
- LoRA微调通常使用更小的学习率
- 建议比全量微调小10-100倍
- 使用warmup阶段效果更好

### 3. 目标模块选择
- 默认选择所有线性层：`"receptance,key,value,output"`
- 对于特定任务可以只微调部分层
- 注意力层(`key,value`)通常比输出层更重要

### 4. 训练策略
- 使用较小的batch_size（32-64）
- 训练epoch数通常比全量微调少
- 监控验证损失，选择最佳checkpoint

## 故障排除

### 1. 内存不足
- 减小`lora_rank`
- 减小`micro_bsz`
- 使用`--merge_lora True`进行推理

### 2. 训练不稳定
- 减小学习率`--lr_init`
- 增加warmup步数`--warmup_steps`
- 检查数据预处理是否正确

### 3. 推理结果异常
- 检查是否使用了正确的checkpoint
- 确认数据预处理与训练时一致
- 尝试不合并LoRA权重进行调试

## 示例应用

### 时间序列预测微调

```bash
# 1. 使用预训练模型进行LoRA微调
python lora_train.py \
    --load_model "pretrained_rwkv.pth" \
    --data_file "your_timeseries_data.npy" \
    --dataset_type "npy" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --epoch_count 20 \
    --lr_init 1e-4

# 2. 使用微调后的模型进行推理
python lora_inference.py \
    --load_model "best-lora-model.pth" \
    --data_file "test_data.npy" \
    --horizon 168 \
    --merge_lora True
```

这个LoRA实现提供了高效、灵活的模型微调方案，特别适合在资源受限的环境下对大型预训练模型进行任务特定的适配。
