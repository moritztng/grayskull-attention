/*
 * Kernel reading tiles of both input matrices from the card's DRAM and broadcasting them to other cores in its row.
 * It runs on the first RISC-V core.
 * I added attention score scaling and softmax to the matrix multiplication kernel from Tenstorrent:
 * https://github.com/tenstorrent/tt-metal/blob/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp
 * The beginning of my implementation is indicated below the matrix multiplication part.
 */

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_addr = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_addr = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_addr = get_arg_val<uint32_t>(34);

    uint32_t attention_scaler = get_arg_val<uint32_t>(35);

    constexpr uint32_t in0_cb = tt::CB::c_in0;
    constexpr uint32_t in1_cb = tt::CB::c_in1;
    constexpr uint32_t scaler_cb = tt::CB::c_in2;
    constexpr uint32_t other_cores_cb = tt::CB::c_in3;
    constexpr uint32_t attention_scaler_cb = tt::CB::c_in4;
    constexpr uint32_t max_cb = tt::CB::dataflow0;
    constexpr uint32_t sum_cb = tt::CB::dataflow1;

    const uint32_t tile_size = get_tile_size(in0_cb);
    const DataFormat data_format = get_dataformat(in0_cb);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in1_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_sender_semaphore_addr);

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = tile_size, .data_format = data_format};

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = tile_size, .data_format = data_format};

    generate_reduce_scaler(scaler_cb, 0x3f803f80);

    cb_reserve_back(attention_scaler_cb, 1);
    (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(attention_scaler_cb)))[0] = attention_scaler;
    cb_push_back(attention_scaler_cb, 1);

    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(in0_cb, in0_block_num_tiles);
        l1_write_addr_in0 = get_write_ptr(in0_cb);
        uint32_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
        uint32_t in0_block_size_bytes = 0;               // can be optimized later, pass it to kernel
        {
            // Copy in0 block into CB, as the default kernel
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += tile_size;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                    in0_block_size_bytes += tile_size;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();
        }

        // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its value
        // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
        noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
        noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t in0_multicast_data_addr = get_noc_multicast_addr(
            in0_mcast_dest_noc_end_x,
            in0_mcast_dest_noc_end_y,
            in0_mcast_dest_noc_start_x,
            in0_mcast_dest_noc_start_y,
            in0_start_address);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(
            in0_start_address, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_dests);

        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same
        // cmd_buf Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

        // We should also multicast the flag to destinations
        uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
            in0_mcast_dest_noc_end_x,
            in0_mcast_dest_noc_end_y,
            in0_mcast_dest_noc_start_x,
            in0_mcast_dest_noc_start_y,
            in0_mcast_receiver_semaphore_addr);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_semaphore_set_multicast(
            in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_dests);

        cb_push_back(in0_cb, in0_block_num_tiles);

        // Operand 1
        cb_reserve_back(in1_cb, in1_block_num_tiles);
        l1_write_addr_in1 = get_write_ptr(in1_cb);

        uint32_t in1_start_address = l1_write_addr_in1;  // copy start address of block, to be used for mcasting
        uint32_t in1_block_size_bytes = 0;               // can be optimized later, pass it to kernel

        // Copy in1 block into CB, as the default kernel
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for (uint32_t w = 0; w < in1_block_w; w++) {
                noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                l1_write_addr_in1 += tile_size;
                in1_tensor_tile_id += in1_tensor_stride_w;
                in1_block_size_bytes += tile_size;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

        // Barrier! make sure the reads are done
        noc_async_read_barrier();

        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value
        // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
        noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
        noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t in1_multicast_data_addr = get_noc_multicast_addr(
            in1_mcast_dest_noc_end_x,
            in1_mcast_dest_noc_end_y,
            in1_mcast_dest_noc_start_x,
            in1_mcast_dest_noc_start_y,
            in1_start_address);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(
            in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_dests);

        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same
        // cmd_buf Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

        // We should also multicast the flag to destinations
        uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
            in1_mcast_dest_noc_end_x,
            in1_mcast_dest_noc_end_y,
            in1_mcast_dest_noc_start_x,
            in1_mcast_dest_noc_start_y,
            in1_mcast_receiver_semaphore_addr);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_semaphore_set_multicast(
            in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_dests);

        cb_push_back(in1_cb, in1_block_num_tiles);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // My implementation starts here
    ///////////////////////////////////////////////////////////////////////////////

    const uint32_t n_cores_row = in0_mcast_num_dests + 1;
    const uint32_t current_core_y_pos = in0_mcast_dest_noc_start_y;

    // Once the local maxima of this core are computed, signal all other cores in the same row
    cb_wait_front(max_cb, in0_block_h);
    for (uint32_t i = 1; i < n_cores_row + 1; i++) {
        noc_semaphore_inc(get_noc_addr(i, current_core_y_pos, in0_mcast_sender_semaphore_addr), 1);
    }

    // Once a signal from all cores in the same row was received, read local maxima from all other cores
    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, n_cores_row);
    noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);
    uint32_t max_cb_l1_read_addr = get_read_ptr(max_cb);
    for (uint32_t i = 0; i < in0_block_h; i++) {
        for (uint32_t j = 1; j < n_cores_row + 1; j++) {
            cb_reserve_back(other_cores_cb, 1);
            noc_async_read(
                get_noc_addr(j, current_core_y_pos, max_cb_l1_read_addr), get_write_ptr(other_cores_cb), tile_size);
            noc_async_read_barrier();
            cb_push_back(other_cores_cb, 1);
        }
        max_cb_l1_read_addr += tile_size;
    }

    // Once the local sums of this core are computed, signal all other cores in the same row
    cb_wait_front(sum_cb, in0_block_h);
    for (uint32_t i = 1; i < n_cores_row + 1; i++) {
        noc_semaphore_inc(get_noc_addr(i, current_core_y_pos, in0_mcast_sender_semaphore_addr), 1);
    }

    // Once a signal from all cores in the same row was received, read local sums from all other cores
    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, n_cores_row);
    noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);
    uint32_t sum_cb_l1_read_addr = get_read_ptr(sum_cb);
    for (uint32_t i = 0; i < in0_block_h; i++) {
        for (uint32_t j = 1; j < n_cores_row + 1; j++) {
            cb_reserve_back(other_cores_cb, 1);
            noc_async_read(
                get_noc_addr(j, current_core_y_pos, sum_cb_l1_read_addr), get_write_ptr(other_cores_cb), tile_size);
            noc_async_read_barrier();
            cb_push_back(other_cores_cb, 1);
        }
        sum_cb_l1_read_addr += tile_size;
    }
}
