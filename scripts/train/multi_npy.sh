export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES='2'

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/../.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model "/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/out/scale_up_3_shuffle/checkpoints/best-0.172.pth" \
    --wandb "rwkvts_test" --proj_dir out/multi_npy \
    --data_file '/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/FV_WIND/FV/f1-9.npy' \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 720 --epoch_steps 200 --epoch_count 3 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 16 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_3 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type multi_npy --num_vars 3 \
    --select_indices '0,7' --feature_used '0,1,2,6,7'

