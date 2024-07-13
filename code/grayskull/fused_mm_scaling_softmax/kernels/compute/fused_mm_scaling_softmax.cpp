/*
 * The fused compute kernel running on the three middle RISC-V cores
 * I added attention score scaling and softmax to the matrix multiplication kernel from Tenstorrent:
 * https://github.com/tenstorrent/tt-metal/blob/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm.cpp
 * The beginning of my implementation is indicated below the matrix multiplication part.
 */

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "compute_kernel_api/reduce.h"

namespace NAMESPACE {
void MAIN {
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);               // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8);           // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9);           // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    uint32_t n_cores_row = get_compile_time_arg_val(11);
    uint32_t n_rows_per_core = get_compile_time_arg_val(12);

    // Circular buffers
    constexpr uint32_t in0_cb = tt::CB::c_in0;
    constexpr uint32_t in1_cb = tt::CB::c_in1;
    constexpr uint32_t scaler_cb = tt::CB::c_in2;
    constexpr uint32_t other_cores_cb = tt::CB::c_in3;
    constexpr uint32_t attention_scaler_cb = tt::CB::c_in4;
    constexpr uint32_t mm_interm_cb = tt::CB::c_intermed0;
    constexpr uint32_t softmax_interm_cb = tt::CB::c_intermed1;
    constexpr uint32_t max_cb = tt::CB::dataflow0;
    constexpr uint32_t sum_cb = tt::CB::dataflow1;
    constexpr uint32_t out_cb = tt::CB::c_out0;

    // Matrix Multiplication (described in detail in section V of the paper)
    {
        DeviceZoneScopedN("Matrix Multiplication");
        mm_init();
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);
            cb_wait_front(in0_cb, in0_block_num_tiles);
            cb_wait_front(in1_cb, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst(tt::DstMode::Half);

                    if (enable_reload) {
                        copy_tile_to_dst_init_short();
                        cb_wait_front(mm_interm_cb, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(mm_interm_cb, i, i);
                        }
                        cb_pop_front(mm_interm_cb, out_subblock_num_tiles);
                        mm_init_short();
                    }

                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(in0_cb, in1_cb, in0_index, in1_index, dst_index, false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    cb_reserve_back(mm_interm_cb, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, mm_interm_cb);
                    }
                    cb_push_back(mm_interm_cb, out_subblock_num_tiles);

                    release_dst(tt::DstMode::Half);
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill)
                enable_reload = true;

            cb_pop_front(in0_cb, in0_block_num_tiles);
            cb_pop_front(in1_cb, in1_block_num_tiles);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // My implementation starts here
    ///////////////////////////////////////////////////////////////////////////////

    {                                                  // Separate scopes for profiling
        DeviceZoneScopedN("Attention score scaling");  // For profiling with Tracy
        cb_wait_front(attention_scaler_cb, 1);
        mul_tiles_bcast_scalar_init_short();
        // For each tile
        for (uint32_t i = 0; i < in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles; i++) {
            acquire_dst(tt::DstMode::Half);
            // Perform attention score scaling
            mul_tiles_bcast<BroadcastType::SCALAR>(mm_interm_cb, attention_scaler_cb, 0, 0, 0);
            cb_pop_front(mm_interm_cb, 1);
            // Move from register to intermediate buffer
            cb_reserve_back(mm_interm_cb, 1);
            pack_tile(0, mm_interm_cb);
            cb_push_back(mm_interm_cb, 1);
            release_dst(tt::DstMode::Half);
        }
    }
    {
        DeviceZoneScopedN("Softmax");
        uint32_t mm_interm_tile_idx = 0;
        {
            DeviceZoneScopedN("Reduce max");
            cb_wait_front(scaler_cb, 1);
            reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(PoolType::MAX, ReduceDim::REDUCE_ROW);
            // For each subblock
            for (uint32_t sbh = 0; sbh < in0_num_subblocks; sbh++) {
                acquire_dst(tt::DstMode::Half);
                // Find local maxima in rows of subblock
                for (uint32_t sbw = 0; sbw < in1_num_subblocks; sbw++) {
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                                mm_interm_cb, scaler_cb, mm_interm_tile_idx, 0, h);
                            mm_interm_tile_idx++;
                        }
                    }
                }
                // Move them from registers into buffer to share them with cores in the same row
                cb_reserve_back(max_cb, out_subblock_h);
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    pack_tile(h, max_cb);
                }
                cb_push_back(max_cb, out_subblock_h);
                release_dst(tt::DstMode::Half);
            }
            reduce_revert_delta();
        }
        {
            DeviceZoneScopedN("Reduce max other cores");
            reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(PoolType::MAX, ReduceDim::REDUCE_ROW);
            // For each group of 8 rows (steps of 8 to utilize registers efficiently)
            for (uint32_t i = 0; i < n_rows_per_core; i += 8) {
                acquire_dst(tt::DstMode::Half);
                // For each row
                for (uint32_t j = 0; j < 8; j++) {
                    // Read local maxima from other Tensix cores in the same row to find global maxima
                    for (uint32_t w = 0; w < n_cores_row; w++) {
                        cb_wait_front(other_cores_cb, 1);
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(other_cores_cb, scaler_cb, 0, 0, j);
                        cb_pop_front(other_cores_cb, 1);
                    }
                }
                // Move tiles from register to intermediate buffer
                cb_reserve_back(softmax_interm_cb, 8);
                for (uint32_t j = 0; j < 8; j++) {
                    pack_tile(j, softmax_interm_cb);
                }
                cb_push_back(softmax_interm_cb, 8);
                release_dst(tt::DstMode::Half);
            }
            reduce_revert_delta();
        }
        {
            DeviceZoneScopedN("Subtract max");
            sub_bcast_cols_init_short();
            int h_start = 0;
            // For each subblock
            for (uint32_t sbh = 0; sbh < in0_num_subblocks; sbh++) {
                for (uint32_t sbw = 0; sbw < in1_num_subblocks; sbw++) {
                    acquire_dst(tt::DstMode::Half);
                    int dst_index = 0;
                    // Subtract global maxima from attention scores
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            sub_tiles_bcast<BroadcastType::COL>(
                                mm_interm_cb, softmax_interm_cb, dst_index, h_start + h, dst_index);
                            dst_index++;
                        }
                    }
                    cb_pop_front(mm_interm_cb, out_subblock_num_tiles);
                    // Move from registers to intermediate buffer
                    cb_reserve_back(mm_interm_cb, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, mm_interm_cb);
                    }
                    cb_push_back(mm_interm_cb, out_subblock_num_tiles);
                    release_dst(tt::DstMode::Half);
                }
                h_start += out_subblock_h;
            }
            cb_pop_front(softmax_interm_cb, n_rows_per_core);
        }
        {
            DeviceZoneScopedN("Exponentiate");
            copy_tile_to_dst_init_short();
            exp_tile_init<false>();
            // For each group of 8 rows (steps of 8 to utilize registers efficiently)
            for (uint32_t i = 0; i < in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles; i += 8) {
                acquire_dst(tt::DstMode::Half);
                // Move from buffer to registers
                for (uint32_t j = 0; j < 8; j++) {
                    copy_tile(mm_interm_cb, j, j);
                }
                cb_pop_front(mm_interm_cb, 8);
                // Exponentiate and move to intermediate buffer
                cb_reserve_back(mm_interm_cb, 8);
                for (uint32_t j = 0; j < 8; j++) {
                    exp_tile<false>(j);
                    pack_tile(j, mm_interm_cb);
                }
                cb_push_back(mm_interm_cb, 8);
                release_dst(tt::DstMode::Half);
            }
        }
        // Computing the following steps (local sums, global sums, normalization) works like the previous steps (local
        // maxima, global maxima, subtracting)
        {
            DeviceZoneScopedN("Reduce sum");
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(PoolType::SUM, ReduceDim::REDUCE_ROW);
            mm_interm_tile_idx = 0;
            for (uint32_t sbh = 0; sbh < in0_num_subblocks; sbh++) {
                acquire_dst(tt::DstMode::Half);
                for (uint32_t sbw = 0; sbw < in1_num_subblocks; sbw++) {
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                                mm_interm_cb, scaler_cb, mm_interm_tile_idx, 0, h);
                            mm_interm_tile_idx++;
                        }
                    }
                }
                cb_reserve_back(sum_cb, out_subblock_h);
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    pack_tile(h, sum_cb);
                }
                cb_push_back(sum_cb, out_subblock_h);
                release_dst(tt::DstMode::Half);
            }
            reduce_revert_delta();
        }
        {
            DeviceZoneScopedN("Reduce sum other cores");
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(PoolType::SUM, ReduceDim::REDUCE_ROW);
            recip_tile_init();
            for (uint32_t i = 0; i < n_rows_per_core; i += 8) {
                acquire_dst(tt::DstMode::Half);
                for (uint32_t j = 0; j < 8; j++) {
                    for (uint32_t w = 0; w < n_cores_row; w++) {
                        cb_wait_front(other_cores_cb, 1);
                        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(other_cores_cb, scaler_cb, 0, 0, j);
                        cb_pop_front(other_cores_cb, 1);
                    }
                }
                cb_reserve_back(softmax_interm_cb, 8);
                for (uint32_t j = 0; j < 8; j++) {
                    recip_tile(j);
                    pack_tile(j, softmax_interm_cb);
                }
                cb_push_back(softmax_interm_cb, 8);
                release_dst(tt::DstMode::Half);
            }
            reduce_revert_delta();
        }
        {
            DeviceZoneScopedN("Normalize");
            mul_bcast_cols_init_short();
            int h_start = 0;
            for (uint32_t sbh = 0; sbh < in0_num_subblocks; sbh++) {
                for (uint32_t sbw = 0; sbw < in1_num_subblocks; sbw++) {
                    acquire_dst(tt::DstMode::Half);
                    int dst_index = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            mul_tiles_bcast<BroadcastType::COL>(
                                mm_interm_cb, softmax_interm_cb, dst_index, h_start + h, dst_index);
                            dst_index++;
                        }
                    }
                    cb_pop_front(mm_interm_cb, out_subblock_num_tiles);
                    // Move tiles from registers to output buffer
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        cb_reserve_back(out_cb, 1);
                        pack_tile(i, out_cb);
                        cb_push_back(out_cb, 1);
                    }
                    release_dst(tt::DstMode::Half);
                }
                h_start += out_subblock_h;
            }
        }
    }
}
}  // namespace NAMESPACE
