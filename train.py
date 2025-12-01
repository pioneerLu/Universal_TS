########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')
def parse_indices(value):

    if isinstance(value, int):
        return [value]
    
    if isinstance(value, str):
        try:
            # "[1,2,3]" / "1,2,3"
            return [int(x.strip()) for x in value.strip('[]').split(',')]
        except ValueError:
            # 默认1
            return [1]
    try:
        return [int(x) for x in value]
    except (ValueError, TypeError):
        return [1]
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str, help="path of rwkv model")  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--run_name", default='demo_run', type=str, 
                        help="run name for wandb. force to consider what is the purpose of this run")
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

    parser.add_argument("--n_emb_layer", default=4, type=int)
    parser.add_argument("--sma_window", default=3, type=int)
    parser.add_argument("--validate_only", default=0, type=int)
    parser.add_argument("--dataset_type",default='ustd',type=str)
    parser.add_argument("--num_vars", default=1, type=int)
    parser.add_argument("--start_var_idx", default=0, type=int)
    parser.add_argument("--forecast_len",default=1, type=int)
    parser.add_argument("--feature_used", default=[0], type=parse_indices)
    parser.add_argument("--select_indices", default=[0], type=parse_indices)
    parser.add_argument("--loss_type",default="mse",type=str)
    parser.add_argument('--do_normalize', type=str2bool, default=False)
    parser.add_argument("--eps",default=1e-5,type=float)
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"], help="Device to run train/validate on")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    

    assert set(args.select_indices).issubset(set(args.feature_used)),'select_indices should be a subset of feature_used'
    args.num_vars = len(args.feature_used)
    args.select_indices_positions = [args.feature_used.index(idx) for idx in args.select_indices]

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["Mode"] = 'train'
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    #args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["json"]

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################
    from src.trainer import train_callback
    # import pdb
    # pdb.set_trace()
    from src.model import UniversalRWKVTimeSeries
    #
    model = UniversalRWKVTimeSeries(args)
    if args.dataset_type.lower() == 'utsd':
        from src.dataset import UTSDataset
        train_data = UTSDataset(
            dataset_path=args.data_file, epoch_steps=args.epoch_steps, micro_bsz=args.micro_bsz,
            input_len=args.ctx_len, output_len=0, flag='train'
            )
        
        val_data = UTSDataset(
            dataset_path=args.data_file, epoch_steps=args.epoch_steps, micro_bsz=args.micro_bsz,
            input_len=args.ctx_len, output_len=0, flag='val'
            )

        
    elif args.dataset_type == 'csv':
        from src.TFB_data import DatasetForTransformer
        from src.TFB_data import read_data
        train_data= DatasetForTransformer(
            dataset=read_data(args.data_file,normalize=True)[0],history_len=100,prediction_len=1,label_len=100
        )

        val_data= DatasetForTransformer(
            dataset=read_data(args.data_file,normalize=True)[0],history_len=100,prediction_len=1,label_len=100
        )


    elif args.dataset_type == 'npy':
        from src.universal_dataset import TimeSeriesDataset
        train_data = TimeSeriesDataset(
            dataset_path=args.data_file, 
            input_len=args.ctx_len, output_len=0, flag='train',split=0.8,norm=True
            )
        
        val_data = TimeSeriesDataset(
            dataset_path=args.data_file,
            input_len=args.ctx_len, output_len=0, flag='val',split=0.8,norm=True
            )

    elif args.dataset_type == 'test':
        from src.universal_dataset import SingleVarDataset
        ratio = [0.7, 0.2, 0.1]
        zero_shot = 1 if args.validate_only == 1 else 0
        train_data = SingleVarDataset(
        data_path=args.data_file,
        flag='train',
        size=(args.ctx_len, 1),  
        normalize=True,
        target='OT',    
        stride=1,
        split=ratio,
        zero_shot=zero_shot
        )   

        val_data = SingleVarDataset(
        data_path=args.data_file,
        flag='val',
        size=(args.ctx_len, 1),  
        normalize=True,
        target='OT',    
        stride=1,
        split=ratio,
        zero_shot=zero_shot
        )

        test_data = SingleVarDataset(
        data_path=args.data_file,
        flag='test',
        size=(args.ctx_len, 1),  
        normalize=True,
        target='OT',    
        stride=1,
        split=ratio,
        zero_shot=zero_shot
        )
        
    elif args.dataset_type == 'multi':

        from src.universal_dataset import MultiVariateTimeSeriesDataset
        ratio = [0.8, 0.2, 0]
        train_data = MultiVariateTimeSeriesDataset(
                    data_path=args.data_file,
                    flag='train',
                    size=(args.ctx_len, 1),  
                    features=args.feature_used, 
                    split=ratio, 
                    target=args.select_indices,      
                    normalize=True
                )
        
        val_data = MultiVariateTimeSeriesDataset(
                    data_path=args.data_file,
                    flag='val',
                    size=(args.ctx_len, 1),  
                    features=args.feature_used,
                    split=ratio,  
                    target=args.select_indices,      
                    normalize=True
                )
    elif args.dataset_type == 'multi_npy':
        from src.universal_dataset import MultiVariateNPYTimeSeriesDataset
        ratio = [0.8 ,0.2, 0]
        train_data = MultiVariateNPYTimeSeriesDataset(
            data_path=args.data_file,
            flag='train',
            size=(args.ctx_len, 1),
            features=args.feature_used,
            split=ratio,
            target=args.select_indices,
            normalize=True
        )
        val_data = MultiVariateNPYTimeSeriesDataset(
            data_path=args.data_file,
            flag='val',
            size=(args.ctx_len, 1),
            features=args.feature_used,
            split=[0.8, 0.2, 0],
            target=args.select_indices,
            normalize=True
        )

    trainer = Trainer.from_argparse_args(args, callbacks=[train_callback(args)],check_val_every_n_epoch=1)

    print('IF VALIDATE ONLY: ',args.validate_only)
    if args.validate_only == 1:  
        trainer = Trainer.from_argparse_args(
            args,
            max_epochs=0,  
            callbacks=[train_callback(args)],
            check_val_every_n_epoch=1,
            enable_checkpointing=False  
        )
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
            check_val_every_n_epoch=1
        )

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        rank_zero_info('deepspeed config:', trainer.strategy.config)
    g = torch.Generator()
    g.manual_seed(42)   
    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=True,generator=g, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, 
                             persistent_workers=False, drop_last=True)

    val_loader = DataLoader(val_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, 
                            persistent_workers=False, drop_last=True)
    
    if args.validate_only==1:
        if args.load_model:
            ckpt_path = os.path.join(args.proj_dir, args.load_model)
            from io import BytesIO
            with open(args.load_model, 'rb') as f:
                buffer = BytesIO(f.read())
            model = UniversalRWKVTimeSeries(args)
            checkpoint = torch.load(buffer, map_location='cpu')
            if 'pytorch-lightning_version' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    elif k.startswith('_forward_module.'):
                        k = k[16:]
                    new_state_dict[k] = v
                filtered_dict = {k: v for k, v in new_state_dict.items() if "smooth.conv.weight" not in k}
                model.load_state_dict(filtered_dict,strict=False)
            model.eval()
        if args.load_model:
            rank_zero_info(f"Loaded pretrained RWKV from {ckpt_path}")
        else:
            raise ValueError("Please provide a checkpoint path with --load_model")
        # 选择设备
        device = torch.device(args.device)
        model = model.to(device)
        model = model.to(dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        trainer.validate(model, val_loader)
    else:
        if args.load_model:
            ckpt_path = os.path.join(args.proj_dir, args.load_model)
            from io import BytesIO
            with open(args.load_model, 'rb') as f:
                buffer = BytesIO(f.read())
            checkpoint = torch.load(buffer, map_location='cpu',weights_only=True)
            if 'pytorch-lightning_version' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    new_state_dict[k] = v
                filtered_dict = {k: v for k, v in new_state_dict.items() if "smooth.conv.weight" not in k}
                model.load_state_dict(filtered_dict,strict=False)
        trainer.fit(model, data_loader, val_loader)
        # 选择设备
        device = torch.device(args.device)
        model = model.to(device)
        model = model.to(dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        trainer.validate(model, val_loader)
    # trainer.fit(model, data_loader,val_loader)

    # trainer.validate(model, val_loader)


