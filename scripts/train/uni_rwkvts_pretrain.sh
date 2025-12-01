export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES='2'

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/../.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
    


python train.py --load_model "/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/out/scale_up_3_shuffle/checkpoints/best-0.172.pth" \
    --wandb "rwkvts_test" --proj_dir out/test \
    --data_file /home/rwkv/RWKV-TS/Universal-RWKV-TS-main/FV_WIND/FV/npy_data\
    --data_type "json" --vocab_size 65536 \
    --ctx_len 112 --epoch_steps 5000 --epoch_count 3 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 64 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 2e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_3 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type npy --num_vars 1 \
    --forecast_len 16 --loss_type 'mse' --do_normalize False\



