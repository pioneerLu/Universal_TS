########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import shutil
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import pandas as pd
from .utils import compress_parameter_names
import matplotlib.pyplot as plt
def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
CHUNK_LEN = 16
########################################################################################################
# CUDA Kernel
########################################################################################################
MODE = os.environ.get("Mode", "cuda")
from torch.utils.cpp_extension import load

# 尝试加载 CUDA 扩展（仅在非 CPU 模式下）
CUDA_EXT_AVAILABLE = False
CPU_EXT_AVAILABLE = False

if MODE != 'inference_cpu':
    try:
        flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
        load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
        CUDA_EXT_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Failed to load CUDA extension, will use CPU fallback: {e}")
        CUDA_EXT_AVAILABLE = False

# 尝试加载 CPU C++ 扩展
try:
    import os
    # Get the cuda directory path
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    cuda_dir = os.path.join(project_root, 'cuda')
    
    cpu_sources = [
        os.path.join(cuda_dir, 'wkv7_cpu.cpp'),
        os.path.join(cuda_dir, 'wkv7_cpu_op.cpp')
    ]
    
    # Check if files exist
    if all(os.path.exists(f) for f in cpu_sources):
        load(name="wind_backstepping_cpu", sources=cpu_sources, is_python_module=False, verbose=True, 
             extra_cflags=['-O3', '-march=native', '-mtune=native', '-fopenmp'])
        CPU_EXT_AVAILABLE = True
        print("Successfully loaded CPU C++ extension")
    else:
        missing = [f for f in cpu_sources if not os.path.exists(f)]
        print(f"Warning: CPU C++ source files not found: {missing}, will use Python fallback")
        CPU_EXT_AVAILABLE = False
except Exception as e:
    print(f"Warning: Failed to load CPU C++ extension, will use Python fallback: {e}")
    import traceback
    traceback.print_exc()
    CPU_EXT_AVAILABLE = False

@torch.jit.script
def _wkv7_cpu_core(w_reshaped: torch.Tensor, q_reshaped: torch.Tensor, k_reshaped: torch.Tensor, 
                   v_reshaped: torch.Tensor, a_reshaped: torch.Tensor, b_reshaped: torch.Tensor,
                   C: int, T: int, CHUNK_LEN: int, device: torch.device) -> tuple:
    """
    JIT-compiled core computation for WKV7 CPU forward pass.
    This function is optimized for speed using JIT compilation.
    """
    B_H = w_reshaped.shape[0]
    y_out = torch.zeros(B_H, T, C, dtype=torch.float32, device=device)
    sa_out = torch.zeros(B_H, T, C, dtype=torch.float32, device=device)
    num_chunks = T // CHUNK_LEN
    s_out = torch.zeros(B_H, num_chunks, C, C, dtype=torch.float32, device=device)
    
    for bh_idx in range(B_H):
        state = torch.zeros(C, C, dtype=torch.float32, device=device)
        
        for t in range(T):
            a_t = a_reshaped[bh_idx, t, :]
            b_t = b_reshaped[bh_idx, t, :]
            w_t = w_reshaped[bh_idx, t, :]
            k_t = k_reshaped[bh_idx, t, :]
            q_t = q_reshaped[bh_idx, t, :]
            v_t = v_reshaped[bh_idx, t, :]
            
            sa_t = torch.mv(state, a_t)
            sa_out[bh_idx, t, :] = sa_t
            
            state = (state * w_t.unsqueeze(0) + 
                    sa_t.unsqueeze(1) * b_t.unsqueeze(0) + 
                    k_t.unsqueeze(0) * v_t.unsqueeze(1))
            
            y_out[bh_idx, t, :] = torch.mv(state, q_t)
            
            if (t + 1) % CHUNK_LEN == 0:
                chunk_idx = t // CHUNK_LEN
                s_out[bh_idx, chunk_idx, :, :] = state
    
    return y_out, sa_out, s_out

def wkv7_cpu_forward(w, q, k, v, z, b):
    """
    CPU fallback implementation of WKV7 forward pass (highly optimized).
    w, q, k, v, z, b: [B, T, H, C] tensors
    Returns: y [B, T, H, C], s [B, H, T//CHUNK_LEN, C, C], sa [B, T, H, C]
    
    Optimized to match CUDA kernel structure with minimal Python loops.
    Uses batch operations where possible.
    """
    B, T, H, C = w.shape
    device = w.device
    
    # Convert to float32 for computation (handle both bfloat16 and float32 inputs)
    w_f = w.float() if w.dtype != torch.float32 else w
    q_f = q.float() if q.dtype != torch.float32 else q
    k_f = k.float() if k.dtype != torch.float32 else k
    v_f = v.float() if v.dtype != torch.float32 else v
    z_f = z.float() if z.dtype != torch.float32 else z  # This is 'a' in the kernel
    b_f = b.float() if b.dtype != torch.float32 else b
    
    # w is actually -exp(-exp(w)), so we need to compute decay
    w_decay = torch.exp(-torch.exp(w_f))  # [B, T, H, C]
    
    y = torch.zeros_like(v_f)  # [B, T, H, C]
    s = torch.zeros(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=device)
    sa = torch.zeros(B, T, H, C, dtype=torch.float32, device=device)
    
    # Reshape for easier indexing: [B, T, H, C] -> [B*H, T, C]
    w_reshaped = w_decay.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)  # [B*H, T, C]
    q_reshaped = q_f.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)
    k_reshaped = k_f.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)
    v_reshaped = v_f.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)
    a_reshaped = z_f.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)
    b_reshaped = b_f.permute(0, 2, 1, 3).contiguous().view(B*H, T, C)
    
    # Use JIT-compiled core function for better performance
    try:
        y_reshaped, sa_reshaped, s_reshaped = _wkv7_cpu_core(
            w_reshaped, q_reshaped, k_reshaped, v_reshaped, a_reshaped, b_reshaped,
            C, T, CHUNK_LEN, device
        )
        
        # Reshape back: [B*H, T, C] -> [B, T, H, C]
        y = y_reshaped.view(B, H, T, C).permute(0, 2, 1, 3).contiguous()
        sa = sa_reshaped.view(B, H, T, C).permute(0, 2, 1, 3).contiguous()
        s = s_reshaped.view(B, H, T // CHUNK_LEN, C, C).contiguous()
    except Exception as e:
        # Fallback to non-JIT version if JIT fails
        print(f"Warning: JIT compilation failed, using fallback: {e}")
        # Process each (B*H) combination
        for bh_idx in range(B * H):
            bb = bh_idx // H
            hh = bh_idx % H
            
            state = torch.zeros(C, C, dtype=torch.float32, device=device)
            
            for t in range(T):
                a_t = a_reshaped[bh_idx, t, :]
                b_t = b_reshaped[bh_idx, t, :]
                w_t = w_reshaped[bh_idx, t, :]
                k_t = k_reshaped[bh_idx, t, :]
                q_t = q_reshaped[bh_idx, t, :]
                v_t = v_reshaped[bh_idx, t, :]
                
                sa_t = torch.mv(state, a_t)
                sa[bb, t, hh, :] = sa_t
                
                state = (state * w_t.unsqueeze(0) + 
                        sa_t.unsqueeze(1) * b_t.unsqueeze(0) + 
                        k_t.unsqueeze(0) * v_t.unsqueeze(1))
                
                y[bb, t, hh, :] = torch.mv(state, q_t)
                
                if (t + 1) % CHUNK_LEN == 0:
                    chunk_idx = t // CHUNK_LEN
                    s[bb, hh, chunk_idx, :, :] = state
    
    # Return y in the same dtype as input
    output_dtype = w.dtype
    if output_dtype == torch.bfloat16:
        return y.to(torch.bfloat16), s, sa
    else:
        return y, s, sa

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        
        # Check if we're on CPU
        if w.device.type == 'cpu':
            # Try to use C++ CPU extension first
            if CPU_EXT_AVAILABLE:
                try:
                    # Ensure contiguous and convert to float32 for C++ extension
                    w_f = w.contiguous().float() if w.dtype != torch.float32 else w.contiguous()
                    q_f = q.contiguous().float() if q.dtype != torch.float32 else q.contiguous()
                    k_f = k.contiguous().float() if k.dtype != torch.float32 else k.contiguous()
                    v_f = v.contiguous().float() if v.dtype != torch.float32 else v.contiguous()
                    z_f = z.contiguous().float() if z.dtype != torch.float32 else z.contiguous()
                    b_f = b.contiguous().float() if b.dtype != torch.float32 else b.contiguous()
                    
                    y = torch.empty_like(v_f)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                    
                    torch.ops.wind_backstepping_cpu.forward(w_f,q_f,k_f,v_f,z_f,b_f, y,s,sa)
                    
                    # Convert back to original dtype
                    if w.dtype != torch.float32:
                        y = y.to(w.dtype)
                    
                    ctx.save_for_backward(w, q, k, v, z, b, s, sa)
                    ctx.is_cpu = True
                    ctx.use_cpp = True
                    return y
                except Exception as e:
                    print(f"Warning: C++ CPU extension failed, using Python fallback: {e}")
                    # Fall through to Python implementation
            
            # Use Python CPU fallback
            y, s, sa = wkv7_cpu_forward(w, q, k, v, z, b)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            ctx.is_cpu = True
            ctx.use_cpp = False
            return y
        else:
            # Use CUDA extension
            if not CUDA_EXT_AVAILABLE:
                raise RuntimeError("CUDA extension not available but device is CUDA")
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            ctx.is_cpu = False
            return y
    
    @staticmethod
    def backward(ctx, dy):
        if hasattr(ctx, 'is_cpu') and ctx.is_cpu:
            # CPU backward pass
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            
            # Try to use C++ CPU extension if available
            if hasattr(ctx, 'use_cpp') and ctx.use_cpp and CPU_EXT_AVAILABLE:
                try:
                    dy_f = dy.contiguous().float() if dy.dtype != torch.float32 else dy.contiguous()
                    w_f = w.contiguous().float() if w.dtype != torch.float32 else w.contiguous()
                    q_f = q.contiguous().float() if q.dtype != torch.float32 else q.contiguous()
                    k_f = k.contiguous().float() if k.dtype != torch.float32 else k.contiguous()
                    v_f = v.contiguous().float() if v.dtype != torch.float32 else v.contiguous()
                    z_f = z.contiguous().float() if z.dtype != torch.float32 else z.contiguous()
                    b_f = b.contiguous().float() if b.dtype != torch.float32 else b.contiguous()
                    
                    dw = torch.empty_like(w_f)
                    dq = torch.empty_like(q_f)
                    dk = torch.empty_like(k_f)
                    dv = torch.empty_like(v_f)
                    dz = torch.empty_like(z_f)
                    db = torch.empty_like(b_f)
                    
                    torch.ops.wind_backstepping_cpu.backward(w_f,q_f,k_f,v_f,z_f,b_f,dy_f,s,sa, dw,dq,dk,dv,dz,db)
                    
                    # Convert back to original dtype
                    if w.dtype != torch.float32:
                        dw = dw.to(w.dtype)
                        dq = dq.to(q.dtype)
                        dk = dk.to(k.dtype)
                        dv = dv.to(v.dtype)
                        dz = dz.to(z.dtype)
                        db = db.to(b.dtype)
                    
                    return dw, dq, dk, dv, dz, db
                except Exception as e:
                    print(f"Warning: C++ CPU backward failed, using zeros: {e}")
            
            # Simplified implementation for inference-only use
            dw = torch.zeros_like(w)
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            dz = torch.zeros_like(z)
            db = torch.zeros_like(b)
            return dw, dq, dk, dv, dz, db
        else:
            # CUDA backward pass
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

def RWKV7_OP(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
    

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id != 0: # not needed for the first layer
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()


    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class WaveNetEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=2, dilation_base=2):
        super(WaveNetEmbedding, self).__init__()
        self.layers = nn.ModuleList()
        current_dilation = 1
        for _ in range(num_layers):
            current_padding = (kernel_size - 1) * current_dilation
            # 1D 因果卷积层
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 因果卷积通常使用 kernel_size=2
                    dilation=current_dilation,
                    padding=current_padding  # 保持输出长度与输入长度相同
                )
            )
            in_channels = out_channels
            current_dilation *= dilation_base

        for name, layer in enumerate(self.layers):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        self.activation = nn.ReLU()

    def forward(self, x):
        # (batch_size, in_channels, sequence_length)
        for layer in self.layers:
            x = layer(x)
            x = x[:, :, :x.size(2) - (layer.dilation[0])]
            x = self.activation(x)
        # (batch_size, out_channels, sequence_length)
        return x
    
# class SharedCausalCNN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers, kernel_size=2, dilation_base=2):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         current_dilation = 1
        
#         for _ in range(num_layers):
#             current_padding = (kernel_size - 1) * current_dilation
#             layer = nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 dilation=current_dilation,
#                 padding=current_padding
#             )
#             self.layers.append(layer)
            
#             in_channels = out_channels
#             current_dilation *= dilation_base

#         self.activation = nn.ReLU()

    # def forward(self, x):
    #     # x: [B, N, T] 
    #     B, N, T = x.size()
        
    #     x = x.view(B * N, 1, T)
        
    #     for layer in self.layers:
    #         x = layer(x)
    #         x = x[:, :, :x.size(2) - (layer.dilation[0])]
    #         x = self.activation(x)

    #     x = x.view(B, N, -1)
    #     return x

class CausalMovingAverage(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(1, 1, kernel_size=window_size, stride=1, padding=window_size-1, bias=False)
        nn.init.constant_(self.conv.weight, 1/window_size)  # 固定权重
        self.conv.weight.requires_grad_(False)  

    def forward(self, x):
        # [B, T, 1]
        x = x.transpose(1, 2)  # [B, 1, T]
        x = self.conv(x)  # out [B, 1, T + window_size-1]
        x = x[:, :, :-self.window_size+1]  
        return x.transpose(1, 2)  # [B, T, 1]
    
class CLSModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 后面在一个 batch 内会复制扩展
        self.cls_token = nn.Parameter(torch.randn(1,1,1,hidden_dim,))

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, T+1, D]
        """
        B = x.size(0)
        num_vars = x.size(2)
        # 在 batch 维度上复制
        cls_token = self.cls_token.expand(B, -1, num_vars,-1)  # [B, 1,num_vars, D]
        return torch.cat([cls_token, x], dim=1)       # [B, T+1,num_vars, D]


class MultiVariableAttention(nn.Module):
    def __init__(self, num_vars, embedding_dim, num_heads=4):
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.var_interaction = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        
        B, T,num_vars, D = x.size()
        
        # [B*T, num_vars,D]
        x = x.view(B * T,num_vars,D)
        
        attn_output, _ = self.attention(x, x, x)
        
        # 重塑回 [B, num_vars, T, D]
        attn_output = attn_output.view(B,T, num_vars, D)
        
        # x = attn_output.mean(dim=1)
        x = attn_output  # [B, T, num_vars, D]
        x = self.var_interaction(x)  # [B, T,num_vars, D]
        return x


class UniversalRWKVTimeSeries(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = WaveNetEmbedding(1, args.n_embd, args.n_emb_layer, dilation_base=2)
        self.cls_module = CLSModule(args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, 1, bias=False)
        self.num_vars = args.num_vars
        self.att = MultiVariableAttention(num_vars=self.num_vars, embedding_dim=args.n_embd,num_heads=4)
        self.sma_window = getattr(args, 'sma_window', 3) # default 3
        self.loss_type = getattr(args, 'loss_type', 'mse')
        self.start_var_idx = args.start_var_idx
        self.select_indices = getattr(args, 'select_indices_positions', args.select_indices)
        self.pre_len = args.forecast_len
        if self.sma_window > 1:
            self.smooth = CausalMovingAverage(self.sma_window)
        
        self.best_val_loss = float('inf')
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        
        self.do_normalize = getattr(args, 'do_normalize', False)
        self.eps = getattr(args, 'eps', 1e-8)


    def configure_optimizers(self):
        if not self.training:
            return None
        zero_weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad] 

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        optim_groups = []
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [{"params": weight_decay_group, "weight_decay": self.args.weight_decay}]
                rank_zero_info(f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}")
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        if not hasattr(self, 'trainer') or self.trainer is None:
            return False
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    # def pad_left(self, x, num_tokens_to_pad):
    #     # pad left with eos token embedding
    #     if num_tokens_to_pad != 0:
    #         # left padding by add eos token at the beginning
    #         pad_emb = torch.zeros(
    #             x.size(0), num_tokens_to_pad, x.size(2),
    #             device=x.device, dtype=x.dtype
    #             )
    #         x = torch.cat((pad_emb, x), dim=1)
    #     return x

    def pad_left(self, x, num_tokens_to_pad):
        # x: [B, T,num_vars, D]
        if num_tokens_to_pad != 0:
            pad_emb = torch.zeros(
                x.size(0),# Batch size
                num_tokens_to_pad,  # Padding tokens
                x.size(2),  # Number of variables 
                x.size(3),  # Embedding dimension
                device=x.device, 
                dtype=x.dtype
            )
            x = torch.cat((pad_emb, x), dim=1)
        return x
    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:,:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        args = self.args
        # x: [B, T, num_vars]
        B, T, C = x.size()
        
        # 归一化处理
        if self.do_normalize:
            # 计算每个变量的均值和标准差
            means = x.mean(dim=1, keepdim=True).detach()  # [B, 1, num_vars]
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()  # [B, 1, num_vars]
            x = (x - means) / stdev
        
        # reshape x to [B, num_vars, T]
        x = x.transpose(1, 2)

        # x = torch.stack([
        #         self.emb(x[:, i, :].unsqueeze(1)) 
        #         for i in range(args.num_vars)
        #     ], dim=1).transpose(1, 2)
        # import pdb;pdb.set_trace()
        x = self.emb(x.reshape(B*self.num_vars,1,T)).reshape(B,self.num_vars,-1,T).permute(0,3,1,2)
        # shape: (B,T,num_vars,D)
        x = self.cls_module(x)
        # shape: (B,T+1,num_vars,D)

        # x = self.emb(x).transpose(1, 2) # [B, D, T] -> [B, T, D]
        # t slice
        num_tokens_to_pad = (
            CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        )
        # print(num_tokens_to_pad)
        x = self.pad_left(x, num_tokens_to_pad)
        if args.dropout > 0:
            x = self.drop0(x)
        # (B,T,num_vars,D)
        x = self.att(x)
        #shape :[B, T,num_vars, D]
        # import pdb;pdb.set_trace()
        x = x.transpose(1,2).reshape(B*self.num_vars,T+1+num_tokens_to_pad,-1)
        #shape :[B*num_vars, T,D]

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)
        x = x.view(B, self.num_vars, T+num_tokens_to_pad+1,-1)
        # import pdb;pdb.set_trace()

        if self.select_indices is None:
            start_idx = getattr(self.args, 'start_var_idx', -1)  
            selected_vars = x[:, start_idx:, :, :]
        else:
            selected_vars = x[:, self.select_indices, :, :]

        # start_idx = self.args.start_var_idx  # 起始变量索引(默认OT在后)
        # selected_vars = x[:, start_idx:, :, :]

        B1, num_vars1, T1, D1 = selected_vars.size()
        selected_vars = selected_vars.reshape(B * num_vars1, T1, D1)

        selected_vars = self.ln_out(selected_vars)

        outputs = self.head(selected_vars)
        
        # x = self.ln_out(x)
        # x = self.head(x)

        if hasattr(self, 'smooth') and self.sma_window > 1:
            outputs = self.smooth(outputs)
            
        outputs = outputs.view(B,num_vars1, T1,-1)
        
        # return self.unpad(outputs, num_tokens_to_pad).squeeze(-1).transpose(1,2)#(B,T,C)# if use cls, return this

        outputs= self.unpad(outputs, num_tokens_to_pad).squeeze(-1).transpose(1,2)[:,1:,:]
        
        if self.do_normalize:
            if self.select_indices is None:
                start_idx = getattr(self.args, 'start_var_idx', -1)
                means = means[:, :, start_idx:]
                stdev = stdev[:, :, start_idx:]
            else:
                means = means[:, :, self.select_indices]
                stdev = stdev[:, :, self.select_indices]
            outputs = outputs * stdev + means
            
        return outputs
    def training_step(self, batch, batch_idx):
        '''
        batch: dict with keys "input_ids", "labels" and "input_text"
        '''
        seq_x = batch["seq_x"]
        if self.args.precision == "bf16":
            seq_x = seq_x.bfloat16() 
        predicts = self(seq_x)
        targets = seq_x
        shift_predicts = predicts[..., :-self.pre_len, :].contiguous()
        shift_targets = targets[..., self.pre_len:, self.select_indices].contiguous()
        if self.loss_type == 'mse':
            loss = F.mse_loss(shift_predicts, shift_targets)

        elif self.loss_type == 'mae':
            loss = F.l1_loss(shift_predicts, shift_targets)
        return loss

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
    
    def validation_step(self, batch, batch_idx):
        seq_x = batch["seq_x"]
        if self.args.precision == "bf16":
            seq_x = seq_x.bfloat16()
        predicts = self(seq_x)
        targets = seq_x
        
        shift_predicts = predicts[..., :-self.pre_len, :].contiguous()
        shift_targets = targets[..., self.pre_len:, self.select_indices].contiguous()
        if self.loss_type == 'mse':
            loss = F.mse_loss(shift_predicts, shift_targets)

        elif self.loss_type == 'mae':
            loss = F.l1_loss(shift_predicts, shift_targets)
        
        

        return {
            "loss": loss,
            "predicts": shift_predicts.detach(),
            "targets": shift_targets.detach()
        }

    def validation_epoch_end(self, outputs):
        # 多GPU数据聚合，如果不开多卡可以注释掉这里
        all_predicts = self.all_gather(torch.cat([out["predicts"] for out in outputs]))
        all_targets = self.all_gather(torch.cat([out["targets"] for out in outputs]))
        
        predicts_numpy = all_predicts.cpu().float().numpy()
        targets_numpy = all_targets.cpu().float().numpy()
        
        save_dir = os.path.join(self.args.proj_dir, "data_records")
        os.makedirs(save_dir, exist_ok=True)
        
        # self._save_to_excel(predicts_numpy, targets_numpy, save_dir)
        
        val_loss = F.mse_loss(all_predicts, all_targets).item()
        self.log("val_loss_mse", val_loss, sync_dist=True)
        

        if val_loss < self.best_val_loss and self.current_epoch >= 1:
            self._save_best_model(val_loss)
            self._plot_predictions(outputs, val_loss)

    def _save_to_excel(self, preds, targets, save_dir):
        self._fallback_save(preds, targets, save_dir)


    def _fallback_save(self, preds, targets, save_dir):
        df = pd.DataFrame({
            'predict': preds.flatten(),
            'target': targets.flatten()
        }).iloc[:10000, :]  #太长了会超限
        print(df.shape)
        
        truncate_path = os.path.join(save_dir, f"epoch_{self.current_epoch}_TRUNCATED.csv")
        df.to_csv(truncate_path, index=False)
        
    def _save_best_model(self, val_loss):
        old_model_path = os.path.join(self.args.proj_dir, "checkpoints", f"best-{self.best_val_loss:.3f}.pth")
        
        if os.path.exists(old_model_path):
            try:
                if os.path.isfile(old_model_path):
                    os.remove(old_model_path)
                else:
                    shutil.rmtree(old_model_path)
            except Exception as e:
                print(f"删除 {old_model_path}时 : {e}")
                
        new_model_path = os.path.join(self.args.proj_dir, "checkpoints", f"best-{val_loss:.3f}.pth")
        # try:
        #     torch.save(self.state_dict(), new_model_path)
        #     self.best_val_loss = val_loss
        # except (RuntimeError, OSError, PermissionError) as e:
           
        #     os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        #     try:
        #         torch.save(self.state_dict(), new_model_path)
        #         self.best_val_loss = val_loss
        #     except Exception as e:
        #         print(f"Failed to save model after creating directory: {e}")
        #         raise
        try:
            self.trainer.save_checkpoint(new_model_path, weights_only=True)
            self.best_val_loss = val_loss
            rank_zero_info(f"Saved new best model to {new_model_path} with val_loss: {val_loss}")
        except Exception as e:
            rank_zero_info(f"Failed to save best model: {e}")

    def _plot_predictions(self, outputs, val_loss):
        plot_dir = os.path.join(self.args.proj_dir, "visualization", f"best_plots_{val_loss:.3f}")
    
        try:
            os.makedirs(plot_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Failed to create directory {plot_dir}: {e}")
            raise
        
        time_scales = [
            ('day', 1),
            ('week', 7),
            ('month', 30)
        ]
        
        for scale_name, window in time_scales:
            scale_pred = []
            scale_target = []
            
            for i in range(0, len(outputs), window):
                batch_pred = torch.cat([out["predicts"] for out in outputs[i:i+window]])
                batch_target = torch.cat([out["targets"] for out in outputs[i:i+window]])
                scale_pred.append(batch_pred.to(torch.float).cpu().numpy())
                scale_target.append(batch_target.to(torch.float).cpu().numpy())
            
            self._plot_scale(scale_pred, scale_target, plot_dir, scale_name)

    def _plot_scale(self, preds, targets, save_dir, prefix):
        plt.figure(figsize=(15, 6))
        for i, (p, t) in enumerate(zip(preds, targets)):
            plt.clf()
            plt.plot(t.flatten()[:672], label='Target', alpha=0.7)
            plt.plot(p.flatten()[:672], label='Prediction', linestyle='--')
            plt.title(f"{prefix.capitalize()} {i+1} Prediction vs Target")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{prefix}_{i}.png"), dpi=150)
            plt.close()

    def generate(self, seq, length):
        """
        Args:
            seq:  [B, T, num_vars]
            length: timesteps needed to predict
        Returns:
            predictions: [B, length, num_selected_vars]
        """
        
        B, T, _ = seq.shape
        predictions = []

        current_seq = seq.clone()
    
        if self.args.precision == "bf16":
            current_seq = current_seq.bfloat16()
        
        for step in range(length):
            with torch.no_grad():
                output = self(current_seq)
                
            last_step_pred = output[:, -1:, :]  # [B, 1, num_selected_vars]
            predictions.append(last_step_pred)
            
            if self.select_indices is None:
                # 默认使用从start_var_idx开始的所有变量
                start_idx = getattr(self.args, 'start_var_idx', -1)
                num_selected_vars = self.num_vars - start_idx if start_idx >= 0 else self.num_vars
                
                # 创建新的一步，填充预测值
                new_step = torch.zeros(B, 1, self.num_vars, device=current_seq.device, dtype=current_seq.dtype)
                new_step[:, :, start_idx:] = last_step_pred
                
                if start_idx > 0:
                    new_step[:, :, :start_idx] = current_seq[:, -1:, :start_idx]
            else:
                # 如果有明确的选择索引，则将预测值放入对应位置
                new_step = torch.zeros(B, 1, self.num_vars, device=current_seq.device, dtype=current_seq.dtype)
                for i, idx in enumerate(self.select_indices):
                    new_step[:, :, idx] = last_step_pred[:, :, i:i+1]
                
                # 复制非预测变量的最后一个值
                for i in range(self.num_vars):
                    if i not in self.select_indices:
                        new_step[:, :, i] = current_seq[:, -1:, i]
            
            current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)
        
        predictions = torch.cat(predictions, dim=1)  # [B, length, num_selected_vars]
        
        return predictions
