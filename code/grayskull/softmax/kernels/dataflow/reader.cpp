/*
 * The reader kernel running on the first RISC-V cores.
 * It reads tiles from the card's DRAM to the Tensix core's SRAM.
 */

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t input_dram_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles_block = get_arg_val<uint32_t>(1);          // 8
    const uint32_t n_rows_tiles_per_core = get_arg_val<uint32_t>(2);  // Number of rows of tiles per core
    const uint32_t n_cols_tiles = get_arg_val<uint32_t>(3);           // Number of tiles in a row
    const uint32_t start_tile_idx = get_arg_val<uint32_t>(4);  // First tile (different for different Tensix cores)

    // Circular buffers
    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_bcast_scaler = tt::CB::c_in1;

    const uint32_t tile_size = get_tile_size(cb_input);  // 2 * 32 * 32
    const InterleavedAddrGenFast<true> input_dram_buffer_address_interleaved = {
        .bank_base_address = input_dram_buffer_address,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_input)};

    // The hex constant equals 1.0 => No scaling inside reduce operations
    generate_reduce_scaler(cb_bcast_scaler, 0x3f803f80);

    uint32_t tile_idx = start_tile_idx;
    // For each row
    for (uint32_t i = 0; i < n_rows_tiles_per_core; i++) {
        // Read row in blocks of 8 tiles and push them to the circular buffer for the compute kernel
        for (uint32_t j = 0; j < n_cols_tiles; j += n_tiles_block) {
            cb_reserve_back(cb_input, n_tiles_block);
            uint32_t l1_write_address = get_write_ptr(cb_input);
            for (uint32_t k = 0; k < n_tiles_block; k++) {
                noc_async_read_tile(tile_idx, input_dram_buffer_address_interleaved, l1_write_address);
                tile_idx++;
                l1_write_address += tile_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input, n_tiles_block);
        }
    }
}
