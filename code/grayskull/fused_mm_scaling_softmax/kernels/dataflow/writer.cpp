/*
 * Kernel writing outputs from Tensix core's SRAM to card's DRAM. It runs on the last RISC-V core.
 * From Tenstorrent:
 * https://github.com/tenstorrent/tt-metal/blob/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp
 */

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    constexpr uint32_t out0_cb = tt::CB::c_out0;

    // single-tile
    const uint32_t tile_size = get_tile_size(out0_cb);
    const DataFormat data_format = get_dataformat(out0_cb);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = out_tensor_addr, .page_size = tile_size, .data_format = data_format};

    uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
    for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
            for (uint32_t h = 0; h < out_subblock_h; h++) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    cb_wait_front(out0_cb, 1);
                    uint32_t l1_read_addr = get_read_ptr(out0_cb);
                    noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(out0_cb, 1);
                    out_tensor_tile_id += out_tensor_stride_w;
                }
                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
            }
            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
    }
}
