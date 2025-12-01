#include <torch/extension.h>
#include <cmath>
#include <cstring>

// CPU forward implementation matching CUDA kernel logic exactly
// Parameter mapping: z -> a_ in CUDA kernel, a -> b_ in CUDA kernel
// Note: 'b' parameter renamed to 'b_param' in usage to avoid conflict with PyTorch headers
void cpu_forward(int B, int T, int H, int C, int CHUNK_LEN,
                 const float* w, const float* q, const float* k, const float* v, 
                 const float* z, const float* b_param,
                 float* y, float* s, float* sa) {
    // z is a_ in CUDA kernel, b_param is b_ in CUDA kernel
    const float* a_vals = z;  // a_ in CUDA kernel (this is -kk in Python)
    const float* b_vals = b_param;  // b_ in CUDA kernel (this is kk*a in Python)
    
    // Process each batch, head, and position i (matching CUDA kernel: bb, hh, i)
    for (int bb = 0; bb < B; bb++) {
        for (int hh = 0; hh < H; hh++) {
            for (int i = 0; i < C; i++) {
                // Each (bb, hh, i) maintains its own state[C] vector
                float state[C];
                std::memset(state, 0, C * sizeof(float));
                
                for (int t = 0; t < T; t++) {
                    // Calculate index: bb*T*H*C + t*H*C + hh*C + i (matching CUDA kernel)
                    int ind = bb * T * H * C + t * H * C + hh * C + i;
                    
                    // Get current time step values for position i
                    // Note: w, q, k, a, b are shared across all i for this (bb, hh, t)
                    // v is specific to position i
                    float v_val = v[ind];
                    
                    // Pre-compute w_decay for all j positions (shared across i)
                    float w_decay[C];
                    float a_local[C], b_local[C], k_vals[C], q_vals[C];
                    for (int j = 0; j < C; j++) {
                        int shared_ind = bb * T * H * C + t * H * C + hh * C + j;
                        w_decay[j] = std::exp(-std::exp(w[shared_ind]));  // exp(-exp(w))
                        a_local[j] = a_vals[shared_ind];  // a_ in CUDA kernel (z in Python)
                        b_local[j] = b_vals[shared_ind];  // b_ in CUDA kernel (a in Python)
                        k_vals[j] = k[shared_ind];
                        q_vals[j] = q[shared_ind];
                    }
                    
                    // Compute sa = sum(a[j] * state[j]) for this position i
                    float sa_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        sa_val += a_local[j] * state[j];
                    }
                    sa[ind] = sa_val;
                    
                    // Update state and compute output
                    float y_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        // Update state: state[j] = state[j] * w[j] + sa * b[j] + k[j] * v
                        state[j] = state[j] * w_decay[j] + sa_val * b_local[j] + k_vals[j] * v_val;
                        
                        // Accumulate output: y = sum(state[j] * q[j])
                        y_val += state[j] * q_vals[j];
                    }
                    y[ind] = y_val;
                    
                    // Save state every CHUNK_LEN steps
                    // Matching CUDA kernel indexing: base = (bb*H+hh)*(T/CHUNK_LEN)*C*C + chunk_idx*C*C + i
                    // Then s[base + j*C] = state[j]
                    if ((t + 1) % CHUNK_LEN == 0) {
                        int chunk_idx = t / CHUNK_LEN;
                        int base = (bb * H + hh) * (T / CHUNK_LEN) * C * C + chunk_idx * C * C + i;
                        for (int j = 0; j < C; j++) {
                            s[base + j * C] = state[j];
                        }
                    }
                }
            }
        }
    }
}

// CPU backward implementation (simplified for inference-only use)
// Parameter mapping: z -> a_ in CUDA kernel, b_param -> b_ in CUDA kernel
void cpu_backward(int B, int T, int H, int C, int CHUNK_LEN,
                  const float* w, const float* q, const float* k, const float* v,
                  const float* z, const float* b_param, const float* dy,
                  const float* s, const float* sa,
                  float* dw, float* dq, float* dk, float* dv, float* dz, float* db) {
    // For inference-only, backward can be zeros
    // This is a placeholder - full implementation would require reverse pass
    std::memset(dw, 0, B * T * H * C * sizeof(float));
    std::memset(dq, 0, B * T * H * C * sizeof(float));
    std::memset(dk, 0, B * T * H * C * sizeof(float));
    std::memset(dv, 0, B * T * H * C * sizeof(float));
    std::memset(dz, 0, B * T * H * C * sizeof(float));
    std::memset(db, 0, B * T * H * C * sizeof(float));
}

