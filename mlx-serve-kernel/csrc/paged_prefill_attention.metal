#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

namespace {

constant short kFragRows = 16;
constant short kFragCols = 16;
constant short kElemsPerFrag = (kFragRows * kFragCols) / 32;
constant short kElemRows = 2;
constant short kElemCols = 4;
constant short kElemRowsJump = 8;

template <typename T>
using FragVec = metal::vec<T, 8>;

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct BaseFrag {
  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtr, typename StrX, typename StrY>
  METAL_FUNC static void load(
      thread FragVec<T>& dst,
      SrcPtr src,
      StrX str_x,
      StrY str_y,
      short off_x = 0,
      short off_y = 0) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    for (short i = 0; i < kElemRows; ++i) {
      const short r = off_x + i * kElemRowsJump;
      const short c = off_y;
      for (short j = 0; j < kElemCols; ++j) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[r * str_x + (c + j) * str_y]);
      }
    }
  }

  template <typename T, typename SrcPtr, typename StrX, typename StrY>
  METAL_FUNC static void load_rows(
      thread FragVec<T>& dst,
      SrcPtr src,
      StrX str_x,
      StrY str_y,
      short lim_x,
      short off_x = 0,
      short off_y = 0) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    const short lx = lim_x - sc.y;
    for (short i = 0; i < kElemRows; ++i) {
      const short r = off_x + i * kElemRowsJump;
      const short c = off_y;
      for (short j = 0; j < kElemCols; ++j) {
        dst[i * kElemCols + j] =
            (r < lx) ? static_cast<T>(src[r * str_x + (c + j) * str_y]) : T(0);
      }
    }
  }

  template <typename SrcT, typename DstT, typename StrX, typename StrY>
  METAL_FUNC static void store_rows(
      const thread FragVec<SrcT>& src,
      device DstT* dst,
      StrX str_x,
      StrY str_y,
      short lim_x,
      short off_x = 0,
      short off_y = 0) {
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    const short lx = lim_x - sc.y;
    for (short i = 0; i < kElemRows; ++i) {
      const short r = off_x + i * kElemRowsJump;
      const short c = off_y;
      if (r < lx) {
        for (short j = 0; j < kElemCols; ++j) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<DstT>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static void row_reduce(
      const thread FragVec<T>& inp_vals,
      thread T* reduced_vals) {
    for (short i = 0; i < kElemRows; ++i) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static void row_bin_op(
      thread FragVec<T>& inp_vals,
      thread T* row_vals) {
    for (short i = 0; i < kElemRows; ++i) {
      for (short j = 0; j < kElemCols; ++j) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <typename CType, typename AType, typename BType>
  METAL_FUNC static void mma_abtn(
      thread FragVec<CType>& cn0,
      thread FragVec<CType>& cn1,
      const thread FragVec<AType>& a,
      const thread FragVec<BType>& bn0,
      const thread FragVec<BType>& bn1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        false,
        true,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();
    for (short i = 0; i < kElemsPerFrag; ++i) {
      ct_a[i] = a[i];
      ct_b[i] = bn0[i];
      ct_b[kElemsPerFrag + i] = bn1[i];
      ct_c[i] = cn0[i];
      ct_c[kElemsPerFrag + i] = cn1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    for (short i = 0; i < kElemsPerFrag; ++i) {
      cn0[i] = ct_c[i];
      cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <typename CType, typename AType, typename BType>
  METAL_FUNC static void mma_abnn(
      thread FragVec<CType>& cn0,
      thread FragVec<CType>& cn1,
      const thread FragVec<AType>& a,
      const thread FragVec<BType>& bn0,
      const thread FragVec<BType>& bn1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        false,
        false,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();
    for (short i = 0; i < kElemsPerFrag; ++i) {
      ct_a[i] = a[i];
      ct_b[i] = bn0[i];
      ct_b[kElemsPerFrag + i] = bn1[i];
      ct_c[i] = cn0[i];
      ct_c[kElemsPerFrag + i] = cn1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    for (short i = 0; i < kElemsPerFrag; ++i) {
      cn0[i] = ct_c[i];
      cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <typename T>
METAL_FUNC void load_paged_k_frag(
    thread FragVec<T>& dst,
    const device T* cache,
    const device int32_t* kv_indices,
    int kv_start,
    int kv_block_start,
    int valid_n,
    int cur_kv_head,
    int head_dim,
    int num_kv_heads,
    int dk_base,
    int frag_row_base) {
  const short2 sc = BaseFrag::get_coord();
  for (short i = 0; i < kElemRows; ++i) {
    const short row = frag_row_base + i * kElemRowsJump + sc.y;
    for (short j = 0; j < kElemCols; ++j) {
      const short col = dk_base + sc.x + j;
      const short idx = i * kElemCols + j;
      if (row < valid_n) {
        const int page_idx = kv_indices[kv_start + kv_block_start + row];
        const int offset =
            page_idx * (num_kv_heads * head_dim) + cur_kv_head * head_dim +
            col;
        dst[idx] = static_cast<T>(cache[offset]);
      } else {
        dst[idx] = T(0);
      }
    }
  }
}

template <typename T>
METAL_FUNC void load_paged_v_frag(
    thread FragVec<T>& dst,
    const device T* cache,
    const device int32_t* kv_indices,
    int kv_start,
    int kv_block_start,
    int valid_n,
    int cur_kv_head,
    int head_dim,
    int num_kv_heads,
    int dv_base,
    int frag_col_base) {
  const short2 sc = BaseFrag::get_coord();
  for (short i = 0; i < kElemRows; ++i) {
    const short row = i * kElemRowsJump + sc.y;
    for (short j = 0; j < kElemCols; ++j) {
      const short col = frag_col_base + sc.x + j;
      const short idx = i * kElemCols + j;
      if (row < valid_n) {
        const int page_idx = kv_indices[kv_start + kv_block_start + row];
        const int offset =
            page_idx * (num_kv_heads * head_dim) + cur_kv_head * head_dim +
            dv_base + col;
        dst[idx] = static_cast<T>(cache[offset]);
      } else {
        dst[idx] = T(0);
      }
    }
  }
}

template <typename T>
METAL_FUNC void apply_length_mask(
    thread FragVec<T>& frag,
    int valid_m,
    int valid_n) {
  constexpr T neg_inf = -HUGE_VALF;
  const short2 sc = BaseFrag::get_coord();
  for (short i = 0; i < kElemRows; ++i) {
    for (short j = 0; j < kElemCols; ++j) {
      const short row = i * kElemRowsJump + sc.y;
      const short col = sc.x + j;
      const short idx = i * kElemCols + j;
      if (!((row < valid_m) && (col < valid_n))) {
        frag[idx] = neg_inf;
      }
    }
  }
}

template <typename T>
METAL_FUNC void apply_causal_mask(
    thread FragVec<T>& frag,
    int q_block_start,
    int kv_block_start,
    int cur_prefix_len) {
  constexpr T neg_inf = -HUGE_VALF;
  const short2 sc = BaseFrag::get_coord();
  for (short i = 0; i < kElemRows; ++i) {
    for (short j = 0; j < kElemCols; ++j) {
      const short row = i * kElemRowsJump + sc.y;
      const short col = sc.x + j;
      const short idx = i * kElemCols + j;
      const int kv_pos = kv_block_start + col;
      if (kv_pos >= cur_prefix_len) {
        const int k_extend_offset = kv_pos - cur_prefix_len;
        const int q_offset = q_block_start + row;
        if (q_offset < k_extend_offset) {
          frag[idx] = neg_inf;
        }
      }
    }
  }
}

template <typename T>
METAL_FUNC void scale_frag(thread FragVec<T>& frag, T scale) {
  for (short i = 0; i < kElemsPerFrag; ++i) {
    frag[i] *= scale;
  }
}

template <typename T>
METAL_FUNC void exp_sub_frag(
    thread FragVec<T>& frag,
    thread metal::vec<T, kElemRows>& row_max) {
  for (short i = 0; i < kElemRows; ++i) {
    for (short j = 0; j < kElemCols; ++j) {
      const short idx = i * kElemCols + j;
      frag[idx] = fast::exp2(frag[idx] - row_max[i]);
    }
  }
}

template <typename T>
METAL_FUNC void clear_frag(thread FragVec<T>& frag) {
  frag = FragVec<T>(T(0));
}

} // namespace

template <
    typename T,
    short DK,
    short DV,
    short BLOCK_M,
    short BLOCK_N>
[[kernel]] void paged_prefill_attention(
    const device T* Q                    [[buffer(0)]],
    device T* O                          [[buffer(1)]],
    const device T* K_cache              [[buffer(2)]],
    const device T* V_cache              [[buffer(3)]],
    const device int32_t* qo_indptr      [[buffer(4)]],
    const device int32_t* kv_indptr      [[buffer(5)]],
    const device int32_t* kv_indices     [[buffer(6)]],
    const device int32_t* prefix_lens    [[buffer(7)]],
    constant float& sm_scale             [[buffer(8)]],
    constant int& num_q_heads            [[buffer(9)]],
    constant int& num_kv_heads           [[buffer(10)]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]]) {
  static_assert(BLOCK_M == 64, "paged_prefill_attention expects BLOCK_M == 64");
  static_assert(BLOCK_N == 32, "paged_prefill_attention expects BLOCK_N == 32");
  static_assert(DK % 16 == 0, "paged_prefill_attention expects DK multiple of 16");
  static_assert(DV % 16 == 0, "paged_prefill_attention expects DV multiple of 16");
  static_assert((DV / 16) % 2 == 0, "paged_prefill_attention expects DV/16 even");

  using Frag = BaseFrag;
  using FragT = FragVec<T>;
  using FragF = FragVec<float>;

  constexpr short TDK = DK / 16;
  constexpr short TDV = DV / 16;
  constexpr short kRowsPT = kElemRows;

  const int cur_seq = tgpig.x;
  const int cur_head = tgpig.y;
  const int cur_block_m = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  const int cur_kv_head = cur_head / kv_group_num;

  const int q_start = qo_indptr[cur_seq];
  const int cur_seq_q_len = qo_indptr[cur_seq + 1] - q_start;
  const int kv_start = kv_indptr[cur_seq];
  const int cur_seq_kv_len = kv_indptr[cur_seq + 1] - kv_start;
  const int cur_prefix_len = prefix_lens[cur_seq];

  constexpr int ROWS_PER_SIMDGROUP = 16;
  const int q_block_start = cur_block_m * BLOCK_M;
  const int q_simd_start = q_block_start + int(sgitg) * ROWS_PER_SIMDGROUP;
  if (q_simd_start >= cur_seq_q_len) {
    return;
  }
  const int valid_m = min(ROWS_PER_SIMDGROUP, cur_seq_q_len - q_simd_start);

  const int q_stride_head = DK;
  const int q_stride_token = num_q_heads * DK;
  const int o_stride_head = DV;
  const int o_stride_token = num_q_heads * DV;
  const float sm_scale_log2e = sm_scale * M_LOG2E_F;
  const int causal_full_limit = cur_prefix_len + q_simd_start;
  const int kv_compute_limit =
      min(cur_seq_kv_len, cur_prefix_len + q_simd_start + valid_m);

  FragF o_frags[TDV];
  for (short i = 0; i < TDV; ++i) {
    clear_frag(o_frags[i]);
  }

  metal::vec<float, kRowsPT> max_score(-HUGE_VALF);
  metal::vec<float, kRowsPT> sum_score(0.0f);

  const device T* q_ptr =
      Q + (q_start + q_simd_start) * q_stride_token + cur_head * q_stride_head;

  for (int kv_block_start = 0; kv_block_start < kv_compute_limit; kv_block_start += BLOCK_N) {
    const int valid_n = min((int)BLOCK_N, kv_compute_limit - kv_block_start);
    const bool needs_causal_mask =
        (kv_block_start + BLOCK_N) > causal_full_limit;

    FragF s0;
    FragF s1;
    clear_frag(s0);
    clear_frag(s1);

    for (short dk_tile = 0; dk_tile < TDK; ++dk_tile) {
      const int dk_base = dk_tile * 16;

      FragT q_frag;
      if (valid_m < BLOCK_M) {
        Frag::load_rows(q_frag, q_ptr + dk_base, q_stride_token, 1, valid_m);
      } else {
        Frag::load(q_frag, q_ptr + dk_base, q_stride_token, 1);
      }

      FragT k_frag0;
      FragT k_frag1;
      load_paged_k_frag(
          k_frag0,
          K_cache,
          kv_indices,
          kv_start,
          kv_block_start,
          valid_n,
          cur_kv_head,
          DK,
          num_kv_heads,
          dk_base,
          0);
      load_paged_k_frag(
          k_frag1,
          K_cache,
          kv_indices,
          kv_start,
          kv_block_start,
          valid_n,
          cur_kv_head,
          DK,
          num_kv_heads,
          dk_base,
          16);

      Frag::mma_abtn(s0, s1, q_frag, k_frag0, k_frag1);
    }

    scale_frag(s0, sm_scale_log2e);
    scale_frag(s1, sm_scale_log2e);
    apply_length_mask(s0, valid_m, min(valid_n, 16));
    apply_length_mask(s1, valid_m, max(0, valid_n - 16));
    if (needs_causal_mask) {
      apply_causal_mask(s0, q_simd_start, kv_block_start, cur_prefix_len);
      apply_causal_mask(s1, q_simd_start, kv_block_start + 16, cur_prefix_len);
    }

    metal::vec<float, kRowsPT> new_max = max_score;
    Frag::row_reduce<MaxOp>(s0, reinterpret_cast<thread float*>(&new_max));
    Frag::row_reduce<MaxOp>(s1, reinterpret_cast<thread float*>(&new_max));

    exp_sub_frag(s0, new_max);
    exp_sub_frag(s1, new_max);

    metal::vec<float, kRowsPT> factor;
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] *= factor[i];
    }

    Frag::row_reduce<SumOp>(s0, reinterpret_cast<thread float*>(&sum_score));
    Frag::row_reduce<SumOp>(s1, reinterpret_cast<thread float*>(&sum_score));

    for (short dv_tile = 0; dv_tile < TDV; dv_tile += 2) {
      const int dv_base = dv_tile * 16;

      for (short row = 0; row < kRowsPT; ++row) {
        for (short col = 0; col < kElemCols; ++col) {
          o_frags[dv_tile][row * kElemCols + col] *= factor[row];
          o_frags[dv_tile + 1][row * kElemCols + col] *= factor[row];
        }
      }

      FragT v_frag00;
      FragT v_frag01;
      FragT v_frag10;
      FragT v_frag11;
      load_paged_v_frag(
          v_frag00,
          V_cache,
          kv_indices,
          kv_start,
          kv_block_start,
          min(valid_n, 16),
          cur_kv_head,
          DV,
          num_kv_heads,
          dv_base,
          0);
      load_paged_v_frag(
          v_frag01,
          V_cache,
          kv_indices,
          kv_start,
          kv_block_start,
          min(valid_n, 16),
          cur_kv_head,
          DV,
          num_kv_heads,
          dv_base,
          16);
      load_paged_v_frag(
          v_frag10,
          V_cache,
          kv_indices,
          kv_start + 16,
          kv_block_start,
          max(0, valid_n - 16),
          cur_kv_head,
          DV,
          num_kv_heads,
          dv_base,
          0);
      load_paged_v_frag(
          v_frag11,
          V_cache,
          kv_indices,
          kv_start + 16,
          kv_block_start,
          max(0, valid_n - 16),
          cur_kv_head,
          DV,
          num_kv_heads,
          dv_base,
          16);

      Frag::mma_abnn(o_frags[dv_tile], o_frags[dv_tile + 1], s0, v_frag00, v_frag01);
      Frag::mma_abnn(o_frags[dv_tile], o_frags[dv_tile + 1], s1, v_frag10, v_frag11);
    }
  }

  metal::vec<float, kRowsPT> inv_sum;
  for (short i = 0; i < kRowsPT; ++i) {
    inv_sum[i] = (sum_score[i] > 0.0f) ? (1.0f / sum_score[i]) : 0.0f;
  }

  for (short dv_tile = 0; dv_tile < TDV; ++dv_tile) {
    Frag::row_bin_op<MulOp>(
        o_frags[dv_tile], reinterpret_cast<thread float*>(&inv_sum));
    device T* o_ptr =
        O + (q_start + q_simd_start) * o_stride_token + cur_head * o_stride_head +
        dv_tile * 16;
    Frag::store_rows(o_frags[dv_tile], o_ptr, o_stride_token, 1, valid_m);
  }
}

#define instantiate_prefill(type_name, type, dk, dv) \
  instantiate_kernel("paged_prefill_attention_" #type_name "_dk" #dk "_dv" #dv, \
    paged_prefill_attention, type, dk, dv, 64, 32)

instantiate_prefill(bfloat16, bfloat16_t, 128, 128)
instantiate_prefill(bfloat16, bfloat16_t, 256, 256)
