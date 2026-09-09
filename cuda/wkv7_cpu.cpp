#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD
#endif

// Helper function for horizontal sum of AVX register
#ifdef USE_SIMD
inline float horizontal_sum_avx(__m256 x) {
    __m128 low = _mm256_extractf128_ps(x, 0);
    __m128 high = _mm256_extractf128_ps(x, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(low);
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

// CPU forward implementation with SIMD and OpenMP optimizations
// Parameter mapping: z -> a_ in CUDA kernel, a -> b_ in CUDA kernel
void cpu_forward(int B, int T, int H, int C, int CHUNK_LEN,
                 const float* w, const float* q, const float* k, const float* v, 
                 const float* z, const float* b_param,
                 float* y, float* s, float* sa) {
    // z is a_ in CUDA kernel, b_param is b_ in CUDA kernel
    const float* a_vals = z;
    const float* b_vals = b_param;
    
    // Pre-compute w_decay for all positions (avoid repeated exp calculations)
    float* w_decay = new float[B * T * H * C];
    #pragma omp parallel for
    for (int i = 0; i < B * T * H * C; i++) {
        w_decay[i] = std::exp(-std::exp(w[i]));
    }
    
    // Parallelize over batch and head dimensions
    #pragma omp parallel for collapse(2)
    for (int bb = 0; bb < B; bb++) {
        for (int hh = 0; hh < H; hh++) {
            // Process each position i
            for (int i = 0; i < C; i++) {
                float state[C];
                std::memset(state, 0, C * sizeof(float));
                
                for (int t = 0; t < T; t++) {
                    int ind = bb * T * H * C + t * H * C + hh * C + i;
                    float v_val = v[ind];
                    
                    // Get base index for this (bb, hh, t)
                    int base_ind = bb * T * H * C + t * H * C + hh * C;
                    
                    // Compute sa = sum(a[j] * state[j]) using SIMD if available
                    float sa_val = 0.0f;
                    #ifdef USE_SIMD
                    if (C >= 8) {
                        __m256 sa_vec = _mm256_setzero_ps();
                        int j = 0;
                        for (; j <= C - 8; j += 8) {
                            __m256 a_vec = _mm256_loadu_ps(&a_vals[base_ind + j]);
                            __m256 state_vec = _mm256_loadu_ps(&state[j]);
                            sa_vec = _mm256_fmadd_ps(a_vec, state_vec, sa_vec);
                        }
                        sa_val = horizontal_sum_avx(sa_vec);
                        // Handle remaining elements
                        for (; j < C; j++) {
                            sa_val += a_vals[base_ind + j] * state[j];
                        }
                    } else {
                        for (int j = 0; j < C; j++) {
                            sa_val += a_vals[base_ind + j] * state[j];
                        }
                    }
                    #else
                    for (int j = 0; j < C; j++) {
                        sa_val += a_vals[base_ind + j] * state[j];
                    }
                    #endif
                    sa[ind] = sa_val;
                    
                    // Update state and compute output using SIMD
                    float y_val = 0.0f;
                    #ifdef USE_SIMD
                    if (C >= 8) {
                        __m256 sa_broadcast = _mm256_set1_ps(sa_val);
                        __m256 v_broadcast = _mm256_set1_ps(v_val);
                        int j = 0;
                        for (; j <= C - 8; j += 8) {
                            __m256 state_vec = _mm256_loadu_ps(&state[j]);
                            __m256 w_vec = _mm256_loadu_ps(&w_decay[base_ind + j]);
                            __m256 b_vec = _mm256_loadu_ps(&b_vals[base_ind + j]);
                            __m256 k_vec = _mm256_loadu_ps(&k[base_ind + j]);
                            __m256 q_vec = _mm256_loadu_ps(&q[base_ind + j]);
                            
                            // state[j] = state[j] * w[j] + sa * b[j] + k[j] * v
                            __m256 kv = _mm256_mul_ps(k_vec, v_broadcast);
                            __m256 sab = _mm256_mul_ps(sa_broadcast, b_vec);
                            __m256 sw = _mm256_mul_ps(state_vec, w_vec);
                            state_vec = _mm256_add_ps(_mm256_add_ps(sw, sab), kv);
                            _mm256_storeu_ps(&state[j], state_vec);
                            
                            // y += state[j] * q[j]
                            __m256 y_vec = _mm256_mul_ps(state_vec, q_vec);
                            y_val += horizontal_sum_avx(y_vec);
                        }
                        // Handle remaining elements
                        for (; j < C; j++) {
                            state[j] = state[j] * w_decay[base_ind + j] + 
                                      sa_val * b_vals[base_ind + j] + 
                                      k[base_ind + j] * v_val;
                            y_val += state[j] * q[base_ind + j];
                        }
                    } else {
                        for (int j = 0; j < C; j++) {
                            state[j] = state[j] * w_decay[base_ind + j] + 
                                      sa_val * b_vals[base_ind + j] + 
                                      k[base_ind + j] * v_val;
                            y_val += state[j] * q[base_ind + j];
                        }
                    }
                    #else
                    for (int j = 0; j < C; j++) {
                        state[j] = state[j] * w_decay[base_ind + j] + 
                                  sa_val * b_vals[base_ind + j] + 
                                  k[base_ind + j] * v_val;
                        y_val += state[j] * q[base_ind + j];
                    }
                    #endif
                    y[ind] = y_val;
                    
                    // Save state every CHUNK_LEN steps
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
    
    delete[] w_decay;
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
