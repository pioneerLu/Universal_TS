export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/../.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python inference.py --load_model "/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/out/test/checkpoints/best-0.191.pth/test.pth" \
    --wandb "rwkvts_test" --proj_dir /home/rwkv/RWKV-TS/Universal-RWKV-TS-main/test_infer \
    --data_file '/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/FV_WIND/Wind/f1-4.npy'\
    --data_type "json" --vocab_size 65536 \
    --ctx_len 816 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator cpu --devices 1 --precision bf16 --strategy 'ddp' \
    --enable_progress_bar True --sma_window 3 --validate_only 1  --num_vars 4 \
    --select_indices '3' --feature_used '0,3' --horizon 96 --grad_cp 0\
    --device cpu
