/*
 * The host program running on the CPU.
 * It dispatchtes kernels to Grayskull, tests for correctness, and measures the runtime.
 * This is the implementation for the fused matrix multiplication, scaling of attention scores, and Softmax operations.
 * It builds on top of the matrix multiplication implementation from Tenstorrent:
 *   - Host pogram:
 *     https://github.com/tenstorrent/tt-metal/blob/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp
 *   - Kernels:
 *     https://github.com/tenstorrent/tt-metal/tree/e85e46876d0818ab787b1290884be743fbf2366e/tt_metal/programming_examples/matmul_common/kernels
 */

#include <cfloat>
#include <chrono>

#include "tt_metal/common/tilize_untilize.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"

using uint32_t = std::uint32_t;
using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t M = 4096;  // Sequence length
constexpr uint32_t N = 4096;  // Sequence length
constexpr uint32_t K = 128;   // Query and key dimension (d_k)

// Number of tiles instead of scalars
constexpr uint32_t Mt = M / TILE_HEIGHT;
constexpr uint32_t Kt = K / TILE_WIDTH;
constexpr uint32_t Nt = N / TILE_WIDTH;

constexpr tt::DataFormat CB_DATA_FORMAT = tt::DataFormat::Float16_b;
const uint32_t TILE_SIZE = detail::TileSize(CB_DATA_FORMAT);  // 32 x 32 tiles of Bfloat16 => 1024 * 2 bytes
const uint32_t DRAM_BUFFER_IN0_SIZE = TILE_SIZE * Mt * Kt;
const uint32_t DRAM_BUFFER_IN1_SIZE = TILE_SIZE * Nt * Kt;
const uint32_t DRAM_BUFFER_OUT_SIZE = TILE_SIZE * Mt * Nt;

// Each Tensix core processes blocks of tiles from both input matrices
// Change constants for different experiments
constexpr uint32_t IN0_BLOCK_W = 2;  // Width of blocks from first input matrix and height of blocks from second one
constexpr uint32_t PER_CORE_M = 16;  // Number of rows of tiles per core
constexpr uint32_t PER_CORE_N = 16;  // Number of columns of tiles per core
constexpr uint32_t OUT_SUBBLOCK_H =
    4;  // Because registers in Tensix cores are limited, even more partitioning: Matrix => Blocks => Subblocks
constexpr uint32_t OUT_SUBBLOCK_W = 2;
constexpr uint32_t NUM_BOCKS_Y = Mt / PER_CORE_M;
constexpr uint32_t NUM_BOCKS_X = Nt / PER_CORE_N;

constexpr uint32_t PROFILING_ITERATIONS = 10;  // For measuring runtime

// CPU implementation to test against
void golden_mm_scaling_softmax(
    vector<bfloat16>& a, vector<bfloat16>& b, vector<bfloat16>& out, uint32_t M, uint32_t N, uint32_t K) {
    vector<float> row(N, 0);
    uint32_t out_idx = 0;
    // Calculate scaling factor
    float recip_sqrt_k = 1 / sqrt(K);
    // For each row
    for (int i = 0; i < M; i++) {
        float max = -FLT_MAX;
        // For each column
        for (int j = 0; j < N; j++) {
            uint32_t a_idx = i * K;
            uint32_t b_idx = j;
            float x = 0;
            // Dot product
            for (int k = 0; k < K; k++) {
                x += a[a_idx].to_float() * b[b_idx].to_float();
                a_idx++;
                b_idx += N;
            }
            // Attention score scaling
            x *= recip_sqrt_k;
            // Find maximum
            if (x > max) {
                max = x;
            }
            // Cache
            row.at(j) = x;
        }
        // Subtract max, exponentiate, sum and cache
        float sum = 0;
        for (int j = 0; j < N; j++) {
            float x = expf(row.at(j) - max);
            sum += x;
            row.at(j) = x;
        }
        // Normalize
        for (int j = 0; j < N; j++) {
            out.at(out_idx) = bfloat16(row.at(j) / sum);
            out_idx++;
        }
    }
}

int main(int argc, char** argv) {
    // Represents kernels, run- and compile-time arguments, placement on Tensix cores
    Program program = CreateProgram();
    // Represents Grayskull card
    Device* device = CreateDevice(0);
    // For dispatching commands to the card
    CommandQueue& command_queue = device->command_queue();

    // 9 x 12 core grid for computation and storage
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t max_cores_x = compute_with_storage_grid_size.x;
    uint32_t max_cores_y = compute_with_storage_grid_size.y;
    // Determines the core grid that is actually used
    CoreCoord core_range = bmm_op_utils::get_core_range(NUM_BOCKS_Y, NUM_BOCKS_X, max_cores_y, max_cores_x);
    size_t start_core_x = 0;
    size_t start_core_y = 0;
    uint32_t num_cores_x = core_range.x;
    uint32_t num_cores_y = core_range.y;
    log_info(tt::LogVerif, "Num cores x: {}, Num cores y: {}", num_cores_x, num_cores_y);

    CoreRange all_cores({start_core_x, start_core_y}, {start_core_x + num_cores_x - 1, start_core_y + num_cores_y - 1});
    CoreRange left_column({start_core_x, start_core_y}, {start_core_x, start_core_y + num_cores_y - 1});
    CoreRange all_except_left_column(
        {start_core_x + 1, start_core_y}, {start_core_x + num_cores_x - 1, start_core_y + num_cores_y - 1});
    // Four different classes of cores:
    // The top, left core is in the first column and first row => has to read tiles of both input matrices from card's
    // DRAM and broadcast to other cores in its row
    CoreRange in0_sender_in1_sender({start_core_x, start_core_y}, {start_core_x, start_core_y});
    // Cores in first column have to read tiles of the first input matrix from the card's DRAM and broadcast
    CoreRange in0_sender_in1_receiver({start_core_x, start_core_y + 1}, {start_core_x, start_core_y + num_cores_y - 1});
    // Cores in first row have to read tiles of the second input matrix from the card's DRAM and broadcast
    CoreRange in0_receiver_in1_sender({start_core_x + 1, start_core_y}, {start_core_x + num_cores_x - 1, start_core_y});
    // Rest of the cores only have to receive tiles from the reading core in their column/row
    CoreRange in0_receiver_in1_receiver(
        {start_core_x + 1, start_core_y + 1}, {start_core_x + num_cores_x - 1, start_core_y + num_cores_y - 1});

    // Allocate circular buffers in the SRAM of Tensix cores
    // For blocks of first input matrix
    uint32_t in0_cb = CB::c_in0;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(PER_CORE_M * IN0_BLOCK_W * 2 * TILE_SIZE, {{in0_cb, CB_DATA_FORMAT}})
            .set_page_size(in0_cb, TILE_SIZE);  // Double buffer for efficiency => multiply with two
    tt_metal::CreateCircularBuffer(program, all_cores, cb_in0_config);
    // For blocks of second input matrix
    uint32_t in1_cb = CB::c_in1;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(PER_CORE_N * IN0_BLOCK_W * 2 * TILE_SIZE, {{in1_cb, CB_DATA_FORMAT}})
            .set_page_size(in1_cb, TILE_SIZE);  // Double buffer for efficiency => multiply with two
    tt_metal::CreateCircularBuffer(program, all_cores, cb_in1_config);
    // For reduction scaler (implementation detail, not too important)
    uint32_t scaler_cb = CB::c_in2;
    auto cb_scaler_config =
        CircularBufferConfig(TILE_SIZE, {{scaler_cb, CB_DATA_FORMAT}}).set_page_size(scaler_cb, TILE_SIZE);
    CreateCircularBuffer(program, all_cores, cb_scaler_config);
    // For maxima and sums from other cores
    uint32_t other_cores_cb = CB::c_in3;
    CircularBufferConfig cb_other_cores_config =
        CircularBufferConfig(2 * TILE_SIZE, {{other_cores_cb, CB_DATA_FORMAT}})
            .set_page_size(other_cores_cb, TILE_SIZE);  // Double buffer for efficiency => multiply with two
    tt_metal::CreateCircularBuffer(program, all_cores, cb_other_cores_config);
    // For attention score scaling factor
    uint32_t attention_scaler_cb = CB::c_in4;
    auto cb_attention_scaler_config = CircularBufferConfig(TILE_SIZE, {{attention_scaler_cb, CB_DATA_FORMAT}})
                                          .set_page_size(attention_scaler_cb, TILE_SIZE);
    CreateCircularBuffer(program, all_cores, cb_attention_scaler_config);
    // For intermediate results of matrix multiplication
    uint32_t mm_interm_cb = CB::c_intermed0;
    CircularBufferConfig cb_interm0_config =
        CircularBufferConfig(PER_CORE_M * PER_CORE_N * TILE_SIZE, {{mm_interm_cb, CB_DATA_FORMAT}})
            .set_page_size(mm_interm_cb, TILE_SIZE);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_interm0_config);
    // For intermediate results of softmax
    uint32_t softmax_interm_cb = CB::c_intermed1;
    CircularBufferConfig cb_interm1_config =
        CircularBufferConfig(2 * PER_CORE_M * TILE_SIZE, {{softmax_interm_cb, CB_DATA_FORMAT}})
            .set_page_size(softmax_interm_cb, TILE_SIZE);  // Double buffer for efficiency => multiply with two
    tt_metal::CreateCircularBuffer(program, all_cores, cb_interm1_config);
    // For sharing local maxima with other cores
    uint32_t max_cb = CB::dataflow0;
    CircularBufferConfig cb_dataflow0_config =
        CircularBufferConfig(PER_CORE_M * TILE_SIZE, {{max_cb, CB_DATA_FORMAT}}).set_page_size(max_cb, TILE_SIZE);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_dataflow0_config);
    // For sharing local sums with other cores
    uint32_t sum_cb = CB::dataflow1;
    CircularBufferConfig cb_dataflow1_config =
        CircularBufferConfig(PER_CORE_M * TILE_SIZE, {{sum_cb, CB_DATA_FORMAT}}).set_page_size(sum_cb, TILE_SIZE);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_dataflow1_config);
    // For final outputs
    uint32_t out_cb = CB::c_out0;
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * TILE_SIZE, {{out_cb, CB_DATA_FORMAT}}).set_page_size(out_cb, TILE_SIZE);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // Add reader kernels to the program
    // There are four different reader kernels for the four different classes of cores described before
    auto reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/"
        "reader_in0_sender_in1_sender.cpp",
        in0_sender_in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});
    auto reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/"
        "reader_in0_sender_in1_receiver.cpp",
        in0_sender_in1_receiver,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});
    auto reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/"
        "reader_in0_receiver_in1_sender.cpp",
        in0_receiver_in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});
    auto reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/"
        "reader_in0_receiver_in1_receiver.cpp",
        in0_receiver_in1_receiver,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Add writer kernel to program (same for all cores)
    // Cores in the first column use different NoC connection
    auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        all_except_left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_1_default,
        });

    // Add compute kernel with compile-time arguments to program (same for all cores)
    constexpr uint32_t NUM_BLOCKS = Kt / IN0_BLOCK_W;
    constexpr uint32_t IN0_NUM_SUBBLOCKS = PER_CORE_M / OUT_SUBBLOCK_H;
    constexpr uint32_t IN0_BLOCK_NUM_TILES = OUT_SUBBLOCK_H * IN0_BLOCK_W * IN0_NUM_SUBBLOCKS;
    constexpr uint32_t IN0_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * IN0_BLOCK_W;
    constexpr uint32_t IN1_NUM_SUBBLOCKS = PER_CORE_N / OUT_SUBBLOCK_W;
    constexpr uint32_t IN1_BLOCK_NUM_TILES = OUT_SUBBLOCK_W * IN0_BLOCK_W * IN1_NUM_SUBBLOCKS;
    constexpr uint32_t OUT_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * OUT_SUBBLOCK_W;
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "kernels/compute/fused_mm_scaling_softmax.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .compile_args = {
                IN0_BLOCK_W,
                IN0_NUM_SUBBLOCKS,
                IN0_BLOCK_NUM_TILES,
                IN0_SUBBLOCK_NUM_TILES,
                IN1_NUM_SUBBLOCKS,
                IN1_BLOCK_NUM_TILES,
                PER_CORE_N,
                NUM_BLOCKS,
                OUT_SUBBLOCK_H,
                OUT_SUBBLOCK_W,
                OUT_SUBBLOCK_NUM_TILES,
                num_cores_x,
                PER_CORE_M}});

    // Allocat buffers for input and output matrices in the card's DRAM
    tt_metal::InterleavedBufferConfig dram_config_in0{
        .device = device,
        .size = DRAM_BUFFER_IN0_SIZE,
        .page_size = TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig dram_config_in1{
        .device = device,
        .size = DRAM_BUFFER_IN1_SIZE,
        .page_size = TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = DRAM_BUFFER_OUT_SIZE,
        .page_size = TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto in0_dram_buffer = CreateBuffer(dram_config_in0);
    auto in1_dram_buffer = CreateBuffer(dram_config_in1);
    auto out_dram_buffer = CreateBuffer(dram_config_out);

    // Create semaphores for dataflow between the Tensix cores
    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Add runtime-arguments for kernels to program
    // They vary strongly among Tensix cores, because:
    //   - Each core has to write the output to a different address in the card's DRAM
    //   - Different reading cores read from different addresses of the input matrices
    //   - Different cores have to receive from and send to different cores
    for (int core_idx_y = 0; core_idx_y < num_cores_y; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_x; core_idx_x++) {
            CoreCoord core = {start_core_x + core_idx_x, start_core_y + core_idx_y};

            CoreCoord left_core = {start_core_x, (size_t)core.y};
            CoreCoord left_core_plus_one = {start_core_x + 1, (size_t)core.y};
            CoreCoord right_core = {start_core_x + num_cores_x - 1, (size_t)core.y};
            CoreCoord top_core = {(size_t)core.x, start_core_y};
            CoreCoord top_core_plus_one = {(size_t)core.x, start_core_y + 1};
            CoreCoord bottom_core = {(size_t)core.x, start_core_y + num_cores_y - 1};

            auto left_core_physical = device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

            std::vector<uint32_t> mm_reader_args = {
                (uint32_t)in0_dram_buffer->address(),    // in0_buffer_addr
                (uint32_t)Kt * PER_CORE_M * core_idx_y,  // in0_buffer_start_tile_id
                (uint32_t)1,                             // in0_buffer_stride_w
                (uint32_t)Kt,                            // in0_buffer_stride_h
                (uint32_t)IN0_BLOCK_W,                   // in0_buffer_next_block_stride

                (uint32_t)IN0_BLOCK_W,               // in0_block_w
                (uint32_t)PER_CORE_M,                // in0_block_h
                (uint32_t)IN0_BLOCK_W * PER_CORE_M,  // in0_block_num_tiles

                (uint32_t)in1_dram_buffer->address(),  // in1_buffer_addr
                (uint32_t)PER_CORE_N * core_idx_x,     // in1_buffer_start_tile_id
                (uint32_t)1,                           // in1_buffer_stride_w
                (uint32_t)Nt,                          // in1_buffer_stride_h
                (uint32_t)IN0_BLOCK_W * Nt,            // in1_buffer_next_block_stride

                (uint32_t)PER_CORE_N,                // in1_block_w
                (uint32_t)IN0_BLOCK_W,               // in1_block_h
                (uint32_t)PER_CORE_N * IN0_BLOCK_W,  // IN1_BLOCK_NUM_TILES

                (uint32_t)Kt / IN0_BLOCK_W,  // num_blocks

                (uint32_t)right_core_physical.x,          // in0_mcast_dest_noc_start_x
                (uint32_t)right_core_physical.y,          // in0_mcast_dest_noc_start_y
                (uint32_t)left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
                (uint32_t)left_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
                (uint32_t)(num_cores_x - 1),              // in0_mcast_num_dests
                (uint32_t)left_core_physical.x,           // in0_mcast_sender_noc_x
                (uint32_t)left_core_physical.y,           // in0_mcast_sender_noc_y
                (uint32_t)in0_mcast_sender_semaphore,
                (uint32_t)in0_mcast_receiver_semaphore,

                (uint32_t)bottom_core_physical.x,        // in0_mcast_dest_noc_start_x
                (uint32_t)bottom_core_physical.y,        // in0_mcast_dest_noc_start_y
                (uint32_t)top_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
                (uint32_t)top_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
                (uint32_t)(num_cores_y - 1),             // in0_mcast_num_dests
                (uint32_t)top_core_physical.x,           // in0_mcast_sender_noc_x
                (uint32_t)top_core_physical.y,           // in0_mcast_sender_noc_y
                (uint32_t)in1_mcast_sender_semaphore,
                (uint32_t)in1_mcast_receiver_semaphore,

                (uint32_t)bfloat16((float)(1 / sqrt(K))).to_uint16()  // attention score scaler
            };

            std::vector<uint32_t> writer_args = {
                (uint32_t)out_dram_buffer->address(),                              // out_buffer_addr
                (uint32_t)core_idx_x * PER_CORE_N + core_idx_y * PER_CORE_M * Nt,  // out_buffer_start_tile_id
                (uint32_t)1,                                                       // out_buffer_stride_w
                (uint32_t)Nt,                                                      // out_buffer_stride_h
                (uint32_t)OUT_SUBBLOCK_W,                                          // out_buffer_next_subblock_stride_w
                (uint32_t)OUT_SUBBLOCK_H * Nt,                                     // out_buffer_next_subblock_stride_h
                (uint32_t)OUT_SUBBLOCK_W,                                          // out_subblock_w
                (uint32_t)OUT_SUBBLOCK_H,                                          // out_subblock_h
                (uint32_t)(OUT_SUBBLOCK_W * OUT_SUBBLOCK_H),                       // out_subblocks_w * out_subblocks_h
                (uint32_t)(PER_CORE_N / OUT_SUBBLOCK_W),                           // out_num_subblocks_w
                (uint32_t)(PER_CORE_M / OUT_SUBBLOCK_H),                           // out_num_subblocks_h
            };

            if (core_idx_x == 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(
                    program, reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args);         // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);  // RISCV_1_default
            } else if (core_idx_x == 0 and core_idx_y != 0) {
                tt_metal::SetRuntimeArgs(
                    program, reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args);       // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);  // RISCV_1_default
            } else if (core_idx_x != 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(
                    program, reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args);       // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);  // RISCV_0_default
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_kernel_in0_receiver_in1_receiver_id,
                    core,
                    mm_reader_args);                                                                // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);  // RISCV_0_default
            }
        }
    }

    // Compute the expected results for testing
    std::vector<bfloat16> in0 = create_random_vector_of_bfloat16_native(DRAM_BUFFER_IN0_SIZE, 1, 123, -0.4);
    std::vector<bfloat16> in1 = create_random_vector_of_bfloat16_native(DRAM_BUFFER_IN1_SIZE, 1, 12522, -0.3);
    vector<bfloat16> golden_out(M * N, 0);
    golden_mm_scaling_softmax(in0, in1, golden_out, M, N, K);

    // Elements of each tile should be stored consecutively
    tilize(in0, M, K);
    tilize(in1, K, N);

    // Write inputs from host's DRAM to card's DRAM (non-blocking call)
    EnqueueWriteBuffer(command_queue, in0_dram_buffer, in0.data(), false);
    EnqueueWriteBuffer(command_queue, in1_dram_buffer, in1.data(), false);

    vector<bfloat16> output(DRAM_BUFFER_OUT_SIZE / sizeof(bfloat16));
    // First execution takes longer => Warmup for profiling
    // Start the kernels without waiting for completion
    EnqueueProgram(command_queue, program, false);
    // Wait for completion
    Finish(command_queue);

    // Multiple iterations for profiling
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < PROFILING_ITERATIONS; i++) {
        EnqueueProgram(command_queue, program, false);
        Finish(command_queue);
    }
    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;
    log_info(tt::LogVerif, "Program average time: {} Seconds", duration.count() / PROFILING_ITERATIONS);

    // Also profile inside the kernels with Tracy (https://github.com/wolfpld/tracy)
    detail::DumpDeviceProfileResults(device);

    // Read outputs from card's DRAM to host's DRAM (blocking call)
    EnqueueReadBuffer(command_queue, out_dram_buffer, output.data(), true);
    untilize(output, M, N);

    // Test with expected results
    float pcc = check_bfloat16_vector_pcc(golden_out, output);
    log_info(tt::LogVerif, "Metalium vs Golden PCC: {}", pcc);
    TT_FATAL(pcc > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pcc);

    CloseDevice(device);
    return 0;
}
