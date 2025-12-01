export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/../.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
    

python train.py --load_model "/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/out/scale_up_3_shuffle/checkpoints/best-0.172.pth" \
    --wandb "rwkvts_test" --proj_dir out/scale_up_3_shuffle_test \
    --data_file /home/rwkv/RWKV-TS/Universal-RWKV-TS-main/test_dataset/ETTm1.csv \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 816 --epoch_steps 200 --epoch_count 5 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 96 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-5 --lr_final 5e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 1 --dataset_type test --num_vars 1 \
    --forecast_len 96 --loss_type 'mse' --do_normalize True\



