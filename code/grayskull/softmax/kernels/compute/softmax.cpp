/*
 * The compute kernel running on the three middle RISC-V cores.
 * It processes a subset of the rows.
 */

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    const uint32_t n_rows_tile_per_core = get_arg_val<uint32_t>(0);  // Number of rows of tiles per core
    const uint32_t n_cols_tiles = get_arg_val<uint32_t>(1);          // Number of tiles in a row
    const uint32_t n_tiles_block = get_arg_val<uint32_t>(2);         // 8

    // Circular buffers
    constexpr auto cb_input = tt::CB::c_in0;
    constexpr auto cb_bcast_scaler = tt::CB::c_in1;
    constexpr auto cb_row = tt::CB::c_intermed0;
    constexpr auto cb_reduce = tt::CB::c_intermed1;
    constexpr auto cb_output = tt::CB::c_out0;

    binary_op_init_common(cb_input, cb_bcast_scaler, cb_row);
    cb_wait_front(cb_bcast_scaler, 1);

    // For each row
    for (uint32_t i = 0; i < n_rows_tile_per_core; i++) {
        {                                     // Separate scopes for profiling
            DeviceZoneScopedN("Copy Input");  // For profiling with Tracy
            copy_tile_to_dst_init_short();
            // For each block of 8 tiles
            for (uint32_t j = 0; j < n_cols_tiles; j += n_tiles_block) {
                acquire_dst(tt::DstMode::Half);
                // Copy tiles from input buffer to registers
                cb_wait_front(cb_input, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    copy_tile(cb_input, k, k);
                }
                cb_pop_front(cb_input, n_tiles_block);
                // Move tiles from registers to intermediate buffer for intermediate results
                cb_reserve_back(cb_row, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    pack_tile(k, cb_row);
                }
                cb_push_back(cb_row, n_tiles_block);
                release_dst(tt::DstMode::Half);
            }
        }
        {
            DeviceZoneScopedN("Max");
            acquire_dst(tt::DstMode::Half);
            // Find all 32 maxima in row of tiles (a tile has 32 rows)
            reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(PoolType::MAX, ReduceDim::REDUCE_ROW);
            for (uint32_t j = 0; j < n_cols_tiles; j++) {
                cb_wait_front(cb_row, j + 1);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_row, cb_bcast_scaler, j, 0, 0);
            }
            reduce_revert_delta();
            // Move resulting maxima as single tile into circular buffer
            cb_reserve_back(cb_reduce, 1);
            pack_tile(0, cb_reduce);
            cb_push_back(cb_reduce, 1);
            release_dst(tt::DstMode::Half);
        }
        {
            DeviceZoneScopedN("Subtract Max");
            cb_wait_front(cb_reduce, 1);
            sub_bcast_cols_init_short();
            // For each block of 8 tiles
            for (uint32_t j = 0; j < n_cols_tiles; j += n_tiles_block) {
                acquire_dst(tt::DstMode::Half);
                // Subtract maxima and store in registers
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    sub_tiles_bcast<BroadcastType::COL>(cb_row, cb_reduce, k, 0, k);
                }
                cb_pop_front(cb_row, n_tiles_block);
                // Move tiles from registers to buffer for intermediate results
                cb_reserve_back(cb_row, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    pack_tile(k, cb_row);
                }
                cb_push_back(cb_row, n_tiles_block);
                release_dst(tt::DstMode::Half);
            }
            cb_pop_front(cb_reduce, 1);
        }
        {
            DeviceZoneScopedN("Exponentiate");
            copy_tile_to_dst_init_short();
            exp_tile_init<false>();
            // For each block of 8 tiles
            for (uint32_t j = 0; j < n_cols_tiles; j += n_tiles_block) {
                acquire_dst(tt::DstMode::Half);
                // Move tiles of intermediate results to registers
                cb_wait_front(cb_row, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    copy_tile(cb_row, k, k);
                }
                cb_pop_front(cb_row, n_tiles_block);
                // Exponentiate tiles and move back to intermediate buffer
                cb_reserve_back(cb_row, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    exp_tile<false>(k);
                    pack_tile(k, cb_row);
                }
                cb_push_back(cb_row, n_tiles_block);
                release_dst(tt::DstMode::Half);
            }
        }
        {
            DeviceZoneScopedN("Sum");
            acquire_dst(tt::DstMode::Half);
            cb_reserve_back(cb_reduce, 1);
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(PoolType::SUM, ReduceDim::REDUCE_ROW);
            // Compute sum similar to finding maxima
            for (uint32_t j = 0; j < n_cols_tiles; j++) {
                cb_wait_front(cb_row, j + 1);
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_row, cb_bcast_scaler, j, 0, 0);
            }
            reduce_revert_delta();
            // Compute reciprocal for normalization
            recip_tile_init();
            recip_tile(0);
            pack_tile(0, cb_reduce);
            cb_push_back(cb_reduce, 1);
            release_dst(tt::DstMode::Half);
        }
        {
            DeviceZoneScopedN("Normalize");
            cb_wait_front(cb_reduce, 1);
            mul_bcast_cols_init_short();
            // For each block of 8 tiles
            for (uint32_t j = 0; j < n_cols_tiles; j += n_tiles_block) {
                acquire_dst(tt::DstMode::Half);
                // Multiply each exponential with the reciprocal for normalization
                // Immediately move tile from register to output buffer for writer kernel
                cb_reserve_back(cb_output, n_tiles_block);
                for (uint32_t k = 0; k < n_tiles_block; k++) {
                    mul_tiles_bcast<BroadcastType::COL>(cb_row, cb_reduce, j + k, 0, k);
                    pack_tile(k, cb_output);
                }
                cb_push_back(cb_output, n_tiles_block);
                release_dst(tt::DstMode::Half);
            }
            cb_pop_front(cb_reduce, 1);
            cb_pop_front(cb_row, n_cols_tiles);
        }
    }
}
}  // namespace NAMESPACE
