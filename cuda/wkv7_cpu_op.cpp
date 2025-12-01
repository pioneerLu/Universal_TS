#include <torch/extension.h>

// Forward declarations
// Parameter mapping: z -> a_ in CUDA kernel, a -> b_ in CUDA kernel
void cpu_forward(int B, int T, int H, int C, int CHUNK_LEN,
                 const float* w, const float* q, const float* k, const float* v,
                 const float* z, const float* b_param,
                 float* y, float* s, float* sa);

void cpu_backward(int B, int T, int H, int C, int CHUNK_LEN,
                  const float* w, const float* q, const float* k, const float* v,
                  const float* z, const float* b_param, const float* dy,
                  const float* s, const float* sa,
                  float* dw, float* dq, float* dk, float* dv, float* dz, float* db);

// Convert bfloat16 to float if needed
template<typename T>
float to_float(const T& val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        // For bfloat16, convert to float
        return static_cast<float>(val);
    }
}

// Forward function - handles both float32 and bfloat16
// Note: z corresponds to 'a' in CUDA kernel, a corresponds to 'b' in CUDA kernel
void forward_cpu(torch::Tensor w, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                 torch::Tensor z, torch::Tensor a,
                 torch::Tensor y, torch::Tensor s, torch::Tensor sa) {
    int B = w.sizes()[0];
    int T = w.sizes()[1];
    int H = w.sizes()[2];
    int C = w.sizes()[3];
    int CHUNK_LEN = 16;  // Should match CHUNK_LEN from Python
    
    // Convert inputs to float32 if needed
    auto w_f = w.dtype() == torch::kFloat32 ? w : w.to(torch::kFloat32);
    auto q_f = q.dtype() == torch::kFloat32 ? q : q.to(torch::kFloat32);
    auto k_f = k.dtype() == torch::kFloat32 ? k : k.to(torch::kFloat32);
    auto v_f = v.dtype() == torch::kFloat32 ? v : v.to(torch::kFloat32);
    auto z_f = z.dtype() == torch::kFloat32 ? z : z.to(torch::kFloat32);
    auto a_f = a.dtype() == torch::kFloat32 ? a : a.to(torch::kFloat32);
    
    // Parameter mapping: 
    // Python z -> CUDA kernel a_ (this is -kk)
    // Python a -> CUDA kernel b_ (this is kk*a)
    cpu_forward(B, T, H, C, CHUNK_LEN,
                w_f.data_ptr<float>(), q_f.data_ptr<float>(), k_f.data_ptr<float>(), v_f.data_ptr<float>(),
                z_f.data_ptr<float>(), a_f.data_ptr<float>(),  // z is 'a_' in kernel, a is 'b_' in kernel
                y.data_ptr<float>(), s.data_ptr<float>(), sa.data_ptr<float>());
    
    // Convert output back to original dtype if needed
    if (w.dtype() != torch::kFloat32) {
        y = y.to(w.dtype());
    }
}

// Backward function
// Parameter mapping: z -> a_ in CUDA kernel, a -> b_ in CUDA kernel
void backward_cpu(torch::Tensor w, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  torch::Tensor z, torch::Tensor a, torch::Tensor dy,
                  torch::Tensor s, torch::Tensor sa,
                  torch::Tensor dw, torch::Tensor dq, torch::Tensor dk,
                  torch::Tensor dv, torch::Tensor dz, torch::Tensor da) {
    int B = w.sizes()[0];
    int T = w.sizes()[1];
    int H = w.sizes()[2];
    int C = w.sizes()[3];
    int CHUNK_LEN = 16;
    
    // Convert inputs to float32 if needed
    auto w_f = w.dtype() == torch::kFloat32 ? w : w.to(torch::kFloat32);
    auto q_f = q.dtype() == torch::kFloat32 ? q : q.to(torch::kFloat32);
    auto k_f = k.dtype() == torch::kFloat32 ? k : k.to(torch::kFloat32);
    auto v_f = v.dtype() == torch::kFloat32 ? v : v.to(torch::kFloat32);
    auto z_f = z.dtype() == torch::kFloat32 ? z : z.to(torch::kFloat32);
    auto a_f = a.dtype() == torch::kFloat32 ? a : a.to(torch::kFloat32);
    auto dy_f = dy.dtype() == torch::kFloat32 ? dy : dy.to(torch::kFloat32);
    
    // Create db tensor for backward (corresponds to gradient of 'a' parameter)
    auto db_f = torch::zeros_like(a_f);
    
    cpu_backward(B, T, H, C, CHUNK_LEN,
                 w_f.data_ptr<float>(), q_f.data_ptr<float>(), k_f.data_ptr<float>(), v_f.data_ptr<float>(),
                 z_f.data_ptr<float>(), a_f.data_ptr<float>(), dy_f.data_ptr<float>(),
                 s.data_ptr<float>(), sa.data_ptr<float>(),
                 dw.data_ptr<float>(), dq.data_ptr<float>(), dk.data_ptr<float>(),
                 dv.data_ptr<float>(), dz.data_ptr<float>(), db_f.data_ptr<float>());
    
    // Copy db to da (da is the gradient of 'a' parameter in Python)
    da.copy_(db_f);
    
    // Convert outputs back to original dtype if needed
    if (w.dtype() != torch::kFloat32) {
        dw = dw.to(w.dtype());
        dq = dq.to(q.dtype());
        dk = dk.to(k.dtype());
        dv = dv.to(v.dtype());
        dz = dz.to(z.dtype());
        da = da.to(a.dtype());
    }
}

TORCH_LIBRARY(wind_backstepping_cpu, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_cpu, CPU, m) {
    m.impl("forward", &forward_cpu);
    m.impl("backward", &backward_cpu);
}

