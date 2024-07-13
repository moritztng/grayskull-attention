/*
 * Kernel receiving tiles of the input matrices from other cores.
 * It runs on the first RISC-V core.
 * I added attention score scaling and softmax to the matrix multiplication kernel from Tenstorrent:
 * https://github.com/tenstorrent/tt-metal/blob/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp
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

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    generate_reduce_scaler(scaler_cb, 0x3f803f80);

    cb_reserve_back(attention_scaler_cb, 1);
    (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(attention_scaler_cb)))[0] = attention_scaler;
    cb_push_back(attention_scaler_cb, 1);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Operand 0
        cb_reserve_back(in0_cb, in0_block_num_tiles);

        // Set in0 semaphore value to INVALID
        noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

        // Atomic increment source core counter
        uint64_t in0_mcast_sender_semaphore_noc_addr =
            get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);
        noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

        cb_push_back(in0_cb, in0_block_num_tiles);

        // Operand 1
        cb_reserve_back(in1_cb, in1_block_num_tiles);

        // Set in1 semaphore value to INVALID
        noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

        uint64_t in1_mcast_sender_semaphore_noc_addr =
            get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
        noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

        // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

        cb_push_back(in1_cb, in1_block_num_tiles);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // My implementation starts here
    ///////////////////////////////////////////////////////////////////////////////

    const uint32_t tile_size = get_tile_size(0);
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

    // Once the local sum of this core are computed, signal all other cores in the same row
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
