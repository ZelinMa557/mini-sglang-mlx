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
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), CType>();
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
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), CType>();
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
METAL_FUNC void clear_frag(thread FragVec<T>& frag) {
  frag = FragVec<T>(T(0));
}

template <typename T>
METAL_FUNC void scale_frag(thread FragVec<T>& frag, T scale) {
  for (short i = 0; i < kElemsPerFrag; ++i) {
    frag[i] *= scale;
  }
}

template <typename T>
METAL_FUNC void exp2_sub_frag(
    thread FragVec<T>& frag,
    thread metal::vec<T, kElemRows>& row_max) {
  for (short i = 0; i < kElemRows; ++i) {
    for (short j = 0; j < kElemCols; ++j) {
      frag[i * kElemCols + j] = (row_max[i] == -HUGE_VALF)
        ? T(0)
        : fast::exp2(frag[i * kElemCols + j] - row_max[i]);
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
      if (!((row < valid_m) && (col < valid_n))) {
        frag[i * kElemCols + j] = neg_inf;
      }
    }
  }
}

template <typename T>
METAL_FUNC void load_paged_k_frag(
    thread FragVec<T>& dst,
    const device T* cache,
    const device int32_t* kv_indices,
    int kv_start,
    int block_start,
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
      const short idx = i * kElemCols + j;
      if (row < valid_n) {
        const int page_idx = kv_indices[kv_start + block_start + row];
        const int offset =
            page_idx * (num_kv_heads * head_dim) + cur_kv_head * head_dim +
            dk_base + sc.x + j;
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
    int block_start,
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
      const short idx = i * kElemCols + j;
      if (row < valid_n) {
        const int page_idx = kv_indices[kv_start + block_start + row];
        const int offset =
            page_idx * (num_kv_heads * head_dim) + cur_kv_head * head_dim +
            dv_base + frag_col_base + sc.x + j;
        dst[idx] = static_cast<T>(cache[offset]);
      } else {
        dst[idx] = T(0);
      }
    }
  }
}

} // namespace

// ============================================================================
// Stage 1: Grouped Paged Decode Attention
// ============================================================================

template <typename T, short DK, short DV, short BLOCK_H>
[[kernel]] void paged_decode_attention_stage1(
    const device T* Q                 [[buffer(0)]],
    const device T* K_cache           [[buffer(1)]],
    const device T* V_cache           [[buffer(2)]],
    const device int32_t* kv_indptr   [[buffer(3)]],
    const device int32_t* kv_indices  [[buffer(4)]],
    const device int32_t* num_kv_splits_buf [[buffer(5)]],
    device float* Att_Out             [[buffer(6)]],
    device float* Att_Lse             [[buffer(7)]],
    constant float& sm_scale          [[buffer(8)]],
    constant int& num_q_heads         [[buffer(9)]],
    constant int& num_kv_heads        [[buffer(10)]],
    constant int& max_kv_splits       [[buffer(11)]],
    ushort tiisg [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]]) {
  static_assert(BLOCK_H == 16, "paged_decode_attention_stage1 expects BLOCK_H == 16");
  static_assert(DK % 16 == 0, "paged_decode_attention_stage1 expects DK multiple of 16");
  static_assert(DV % 16 == 0, "paged_decode_attention_stage1 expects DV multiple of 16");
  static_assert((DV / 16) % 2 == 0, "paged_decode_attention_stage1 expects DV/16 even");

  constexpr int BLOCK_N = 32;
  constexpr short TDK = DK / 16;
  constexpr short TDV = DV / 16;
  constexpr short kRowsPT = kElemRows;
  using FragT = FragVec<T>;
  using FragF = FragVec<float>;

  const int cur_batch = tgpig.x;
  const int cur_kv_head = tgpig.y;
  const int split_kv_id = tgpig.z;

  const int kv_group_num = num_q_heads / num_kv_heads;
  const int actual_heads = kv_group_num;
  if (actual_heads > BLOCK_H) {
    return;
  }
  const int cur_q_head_start = cur_kv_head * kv_group_num;

  const int kv_start = kv_indptr[cur_batch];
  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_start;
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];
  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split =
      ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) /
      MIN_BLOCK_KV * MIN_BLOCK_KV;
  const int split_kv_start = kv_len_per_split * split_kv_id;
  const int split_kv_end = min(split_kv_start + kv_len_per_split, cur_batch_seq_len);
  if (split_kv_start >= split_kv_end) return;

  const float sm_scale_log2e = sm_scale * M_LOG2E_F;
  const int q_stride_batch = num_q_heads * DK;
  const int out_stride_batch = num_q_heads * max_kv_splits * DV;
  const int out_stride_head = max_kv_splits * DV;
  const int lse_stride_batch = num_q_heads * max_kv_splits;
  const int lse_stride_head = max_kv_splits;
  const int out_head_base = cur_q_head_start * out_stride_head;
  const int lse_head_base = cur_q_head_start * lse_stride_head;

  const device T* q_ptr =
      Q + cur_batch * q_stride_batch + cur_q_head_start * DK;

  FragF o_frags[TDV];
  for (short i = 0; i < TDV; ++i) {
    clear_frag(o_frags[i]);
  }

  metal::vec<float, kRowsPT> max_score(-HUGE_VALF);
  metal::vec<float, kRowsPT> sum_score(0.0f);

  for (int block_start = split_kv_start; block_start < split_kv_end; block_start += BLOCK_N) {
    const int valid_n = min(BLOCK_N, split_kv_end - block_start);

    FragF s0;
    FragF s1;
    clear_frag(s0);
    clear_frag(s1);

    for (short dk_tile = 0; dk_tile < TDK; ++dk_tile) {
      const int dk_base = dk_tile * 16;

      FragT q_frag;
      BaseFrag::load_rows(q_frag, q_ptr + dk_base, DK, 1, actual_heads);

      FragT k_frag0;
      FragT k_frag1;
      load_paged_k_frag(
          k_frag0, K_cache, kv_indices, kv_start, block_start, valid_n,
          cur_kv_head, DK, num_kv_heads, dk_base, 0);
      load_paged_k_frag(
          k_frag1, K_cache, kv_indices, kv_start, block_start, valid_n,
          cur_kv_head, DK, num_kv_heads, dk_base, 16);

      BaseFrag::mma_abtn(s0, s1, q_frag, k_frag0, k_frag1);
    }

    scale_frag(s0, sm_scale_log2e);
    scale_frag(s1, sm_scale_log2e);
    apply_length_mask(s0, actual_heads, min(valid_n, 16));
    apply_length_mask(s1, actual_heads, max(0, valid_n - 16));

    metal::vec<float, kRowsPT> new_max = max_score;
    BaseFrag::row_reduce<MaxOp>(s0, reinterpret_cast<thread float*>(&new_max));
    BaseFrag::row_reduce<MaxOp>(s1, reinterpret_cast<thread float*>(&new_max));

    bool row_active[kRowsPT];
    for (short i = 0; i < kRowsPT; ++i) {
      row_active[i] = isfinite(new_max[i]);
      if (!row_active[i]) {
        new_max[i] = 0.0f;
      }
    }

    exp2_sub_frag(s0, new_max);
    exp2_sub_frag(s1, new_max);

    metal::vec<float, kRowsPT> factor;
    for (short i = 0; i < kRowsPT; ++i) {
      if (row_active[i]) {
        factor[i] = (new_max[i] == -HUGE_VALF)
          ? 1.0f
          : fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
        sum_score[i] *= factor[i];
      } else {
        factor[i] = 0.0f;
        max_score[i] = 0.0f;
        sum_score[i] = 0.0f;
      }
    }

    BaseFrag::row_reduce<SumOp>(s0, reinterpret_cast<thread float*>(&sum_score));
    BaseFrag::row_reduce<SumOp>(s1, reinterpret_cast<thread float*>(&sum_score));

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
          v_frag00, V_cache, kv_indices, kv_start, block_start, min(valid_n, 16),
          cur_kv_head, DV, num_kv_heads, dv_base, 0);
      load_paged_v_frag(
          v_frag01, V_cache, kv_indices, kv_start, block_start, min(valid_n, 16),
          cur_kv_head, DV, num_kv_heads, dv_base, 16);
      load_paged_v_frag(
          v_frag10, V_cache, kv_indices, kv_start + 16, block_start, max(0, valid_n - 16),
          cur_kv_head, DV, num_kv_heads, dv_base, 0);
      load_paged_v_frag(
          v_frag11, V_cache, kv_indices, kv_start + 16, block_start, max(0, valid_n - 16),
          cur_kv_head, DV, num_kv_heads, dv_base, 16);

      BaseFrag::mma_abnn(o_frags[dv_tile], o_frags[dv_tile + 1], s0, v_frag00, v_frag01);
      BaseFrag::mma_abnn(o_frags[dv_tile], o_frags[dv_tile + 1], s1, v_frag10, v_frag11);
    }
  }

  metal::vec<float, kRowsPT> inv_sum;
  for (short i = 0; i < kRowsPT; ++i) {
    inv_sum[i] = (sum_score[i] > 0.0f) ? (1.0f / sum_score[i]) : 0.0f;
  }

  for (short dv_tile = 0; dv_tile < TDV; ++dv_tile) {
    BaseFrag::row_bin_op<MulOp>(o_frags[dv_tile], reinterpret_cast<thread float*>(&inv_sum));
    device float* out_ptr =
        Att_Out + cur_batch * out_stride_batch + out_head_base +
        split_kv_id * DV + dv_tile * 16;
    BaseFrag::store_rows(o_frags[dv_tile], out_ptr, out_stride_head, 1, actual_heads);
  }

  const short2 sc = BaseFrag::get_coord();
  if (sc.x == 0) {
    for (short i = 0; i < kElemRows; ++i) {
      const short row = i * kElemRowsJump + sc.y;
      if (row < actual_heads) {
        const int lse_offset =
            cur_batch * lse_stride_batch + lse_head_base +
            row * lse_stride_head + split_kv_id;
        Att_Lse[lse_offset] = max_score[i] * M_LN2_F + log(sum_score[i]);
      }
    }
  }
}

// ============================================================================
// Stage 2: Cross-Split Reduction
// ============================================================================

template <typename T, short DV>
[[kernel]] void paged_decode_attention_stage2(
    const device float* Att_Out       [[buffer(0)]],
    const device float* Att_Lse       [[buffer(1)]],
    device T* O                       [[buffer(2)]],
    const device int32_t* kv_indptr   [[buffer(3)]],
    const device int32_t* num_kv_splits_buf [[buffer(4)]],
    constant int& num_q_heads         [[buffer(5)]],
    constant int& max_kv_splits       [[buffer(6)]],
    uint2 tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]) {

  (void)sgitg;

  const int cur_batch = tgpig.x;
  const int cur_head = tgpig.y;

  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - kv_indptr[cur_batch];
  if (cur_batch_seq_len <= 0) return;

  const int kv_splits = num_kv_splits_buf[cur_batch];
  constexpr int MIN_BLOCK_KV = 32;
  const int kv_len_per_split =
      ((cur_batch_seq_len + kv_splits - 1) / kv_splits + MIN_BLOCK_KV - 1) /
      MIN_BLOCK_KV * MIN_BLOCK_KV;

  const int out_stride_batch = num_q_heads * max_kv_splits * DV;
  const int out_stride_head = max_kv_splits * DV;
  const int lse_stride_batch = num_q_heads * max_kv_splits;
  const int lse_stride_head = max_kv_splits;

  constexpr int NW = 32;
  float e_max = -HUGE_VALF;
  float e_sum = 0.0f;

  constexpr int ELEMS_PER_THREAD = (DV + NW - 1) / NW;
  float acc[ELEMS_PER_THREAD];
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    acc[i] = 0.0f;
  }

  for (int split = 0; split < kv_splits; split++) {
    int split_start = kv_len_per_split * split;
    int split_end = min(split_start + kv_len_per_split, cur_batch_seq_len);
    if (split_start >= split_end) break;

    int lse_offset = cur_batch * lse_stride_batch + cur_head * lse_stride_head + split;
    float lse_val = Att_Lse[lse_offset];

    float new_max = max(e_max, lse_val);
    float old_scale = fast::exp2((e_max - new_max) * M_LOG2E_F);
    float new_scale = fast::exp2((lse_val - new_max) * M_LOG2E_F);

    int out_offset = cur_batch * out_stride_batch + cur_head * out_stride_head + split * DV;
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      int d = tiisg + i * NW;
      if (d < DV) {
        acc[i] = acc[i] * old_scale + new_scale * Att_Out[out_offset + d];
      }
    }

    e_sum = e_sum * old_scale + new_scale;
    e_max = new_max;
  }

  int o_offset = cur_batch * num_q_heads * DV + cur_head * DV;
  float inv_sum = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    int d = tiisg + i * NW;
    if (d < DV) {
      O[o_offset + d] = static_cast<T>(acc[i] * inv_sum);
    }
  }
}

#define instantiate_stage1(type_name, type, dk, dv, block_h) \
  instantiate_kernel("paged_decode_attention_stage1_" #type_name "_dk" #dk "_dv" #dv "_bh" #block_h, \
    paged_decode_attention_stage1, type, dk, dv, block_h)

#define instantiate_stage2(type_name, type, dv) \
  instantiate_kernel("paged_decode_attention_stage2_" #type_name "_dv" #dv, \
    paged_decode_attention_stage2, type, dv)

instantiate_stage1(float16, half, 128, 128, 16)
instantiate_stage1(float16, half, 256, 256, 16)
instantiate_stage2(float16, half, 128)
instantiate_stage2(float16, half, 256)

instantiate_stage1(bfloat16, bfloat16_t, 128, 128, 16)
instantiate_stage1(bfloat16, bfloat16_t, 256, 256, 16)
instantiate_stage2(bfloat16, bfloat16_t, 128)
instantiate_stage2(bfloat16, bfloat16_t, 256)
