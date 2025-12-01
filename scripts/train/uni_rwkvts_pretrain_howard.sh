export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/../.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
    

python train.py --load_model "" \
    --wandb "rwkvts_test" --proj_dir out/howard_test \
    --data_file /home/rwkv/RWKV-TS/Universal-RWKV-TS-main/single_npy \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 720 --epoch_steps 2000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 100 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 8e-5 --lr_final 1e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type npy --num_vars 1 \
    --forecast_len 96 --loss_type 'mse'



