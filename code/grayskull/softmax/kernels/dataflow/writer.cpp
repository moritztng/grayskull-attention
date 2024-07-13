/*
 * The writer kernel running on the last RISC-V cores.
 * It writes tiles from the Tensix core's SRAM to the card's DRAM.
 */

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t output_dram_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles_per_core = get_arg_val<uint32_t>(1);  // Number of rows of tiles per core
    const uint32_t n_tiles_block = get_arg_val<uint32_t>(2);     // 8
    const uint32_t start_tile_idx = get_arg_val<uint32_t>(3);    // First tile (different for different Tensix cores)

    constexpr uint32_t cb_output = tt::CB::c_out0;        // circular buffer
    const uint32_t tile_size = get_tile_size(cb_output);  // 2 * 32 * 32
    const InterleavedAddrGenFast<true> output_dram_buffer_address_interleaved = {
        .bank_base_address = output_dram_buffer_address,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_output)};

    uint32_t tile_idx = start_tile_idx;
    // For each block of 8 tiles
    for (uint32_t i = 0; i < n_tiles_per_core; i += n_tiles_block) {
        // Write tiles from output buffer in SRAM to the card's DRAM
        cb_wait_front(cb_output, n_tiles_block);
        uint32_t l1_read_address = get_read_ptr(cb_output);
        for (uint32_t j = 0; j < n_tiles_block; j++) {
            noc_async_write_tile(tile_idx, output_dram_buffer_address_interleaved, l1_read_address);
            tile_idx++;
            l1_read_address += tile_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, n_tiles_block);
    }
}
