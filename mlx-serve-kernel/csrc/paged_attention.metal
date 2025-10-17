// Inspired by https://github.com/codelion/openevolve/blob/main/examples/mlx_metal_kernel_opt/best_program.py
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T, int HEAD_DIM, int N_GQA, int VEC_SIZE = 8, int SLIDING_WINDOW = -1>
[[kernel]] void paged_prefill_group_query_attention(
    const device T* query, // NHD layout
    const device int* position_ids,
    const device int* seq_ids,
    device T* output, // NHD layout
    const device T* k_cache,
    const device T* v_cache,
    const device int* kv_indices,
    constant const float &scale_val,
    constant const size_t &num_heads,
    constant const size_t &num_kv_heads,
    constant const size_t &kv_indices_stride,
    constant const size_t input_output_stride[2],
    constant const size_t kv_cache_stride[2], // NHD layout
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {

    uint batch_offset = pos.x;
    uint kv_head_idx = pos.y;
    uint sequence_offset = position_ids[batch_offset];
    uint seq_id = seq_ids[batch_offset];
    
    const uint VEC_PER_HEAD = HEAD_DIM / VEC_SIZE;
    using Vector = vec<T, VEC_PER_HEAD>;
    using Vector_FP32 = vec<float, VEC_PER_HEAD>;

    uint input_output_index = batch_offset * input_output_stride[0] + N_GQA * input_output_stride[2];
    Vector *output_ptr = (Vector*)(output + input_output_index);
    const Vector* query_ptr = (Vector*)(query + input_output_index);

    Vector query_vec_v[N_GQA][VEC_PER_HEAD];
    Vector_FP32 O_buffer[N_GQA][VEC_PER_HEAD];
    for (uint i = 0; i < N_GQA; i++) {
        for (uint d_vec = 0; d_vec < VEC_PER_HEAD; d_vec++) {
            query_vec_v[d_vec] = ((device vec<T, 8>*) (query_ptr))[d_vec];
        }
    }

    float max_scores[N_GQA];
    float log_sum_exps[N_GQA];

    for (uint i = 0; i < N_GQA; i++) {
        max_scores[i] = float(-INFINITY);
        log_sum_exps[i] = 0.0;
    }

    uint start_pos = 0;
    uint end_pos = sequence_offset;
    if constexpr (SLIDING_WINDOW != -1) {
        start_pos = max(start_pos, end_pos - SLIDING_WINDOW + 1);
    }

    Vector_FP32 v_buffer[VEC_PER_HEAD];

    for (uint key_pos = start_pos; key_pos <= end_pos; key_pos++) {
        float scores[N_GQA];
        for (int i = 0; i < N_GQA; i++) {
            scores[i] = T(0.0);
        }

        // Compute k_ptr index
        int k_index = kv_indices[seq_id * kv_indices_stride + key_pos];
        T* k_ptr = k_cache + k_index * kv_cache_stride[0] + kv_head_idx * kv_cache_stride[1];
        T* v_ptr = v_cache + k_index * kv_cache_stride[0] + kv_head_idx * kv_cache_stride[1];

        // Compute Q @ K^T for this key position using vectorized dot product
        for (uint d_vec = 0; d_vec < VEC_PER_HEAD; d_vec++) {
            Vector k_vec = ((device Vector*) (k_ptr))[d_vec];
            for (int i = 0; i < N_GQA; i++) {
                scores[i] += static_cast<float>(dot(query_vec_v[i][d_vec], k_vec));
            }
            Vector v_vec = ((device Vector*) (v_ptr))[d_vec];
            v_buffer[d_vec] = Vector_FP32(v_vec);
        }
        for (int i = 0; i < N_GQA; i++) {
            const float m_old = max_scores[i];
            const float m_new = max(max_scores[i], scores[i] * scale_val);
            const float exp_diff = exp(m_old - m_new);
            log_sum_exps[i] = log_sum_exps[i] * exp_diff + exp_diff;

            max_scores[i] = m_new;
            for (int d_vec = 0; d_vec < VEC_PER_HEAD; d_vec++) {
                O_buffer[i][d_vec] = O_buffer[i][d_vec] * exp_diff + exp_diff * v_buffer[d_vec];
            }
        }
    }

    for (int i = 0; i < N_GQA; i++) {
        for (int j = 0; j < VEC_PER_HEAD; j++) {
            Vector_FP32 tmp = O_buffer[i][j] / log_sum_exps[i];
            output_ptr[i * VEC_PER_HEAD + j] = static_cast<Vector>(tmp);
        }
    }
}