/*
 * The host program running on the CPU.
 * It dispatchtes kernels to Grayskull, tests for correctness and measures the runtime.
 * This is the dedicated single- and multi-core Softmax implementation.
 */

#include <cfloat>
#include <chrono>

#include "tt_metal/common/tilize_untilize.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;
using namespace tt;

// Change constants for different experiments
constexpr bool MULTI_CORE = true;  // Single-core vs. multi-core flag
constexpr uint32_t N_ROWS = 8192;  // Matrix dimensions
constexpr uint32_t N_COLS = 8192;
constexpr uint32_t N_ELEMENTS = N_ROWS * N_COLS;
constexpr uint32_t N_ROWS_TILES = N_ROWS / constants::TILE_HEIGHT;  // Number of rows of tiles (instead of scalars)
constexpr uint32_t N_COLS_TILES = N_COLS / constants::TILE_WIDTH;
constexpr uint32_t N_TILES = N_ROWS_TILES * N_COLS_TILES;
constexpr uint32_t N_TILES_BLOCK = 8;          // Each row is processed in blocks of 8 tiles for efficiency
constexpr uint32_t PROFILING_ITERATIONS = 10;  // For measuring of runtime
constexpr tt::DataFormat CB_DATA_FORMAT = tt::DataFormat::Float16_b;
const uint32_t TILE_SIZE = detail::TileSize(CB_DATA_FORMAT);  // 32 x 32 tiles of Bfloat16 => 1024 * 2 bytes

// Testing for correctness with pearson correlation coefficient between results and expected values
// From Tenstorrent:
// "https://github.com/tenstorrent/tt-metal/blob/a6f2ac0b132b90d934e4dd7ba5e56e9352f3b499/tt_metal/programming_examples/matmul_common/bmm_op.hpp#L74
inline float check_bfloat16_vector_pcc(const vector<bfloat16>& vec_a, const vector<bfloat16>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;
    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += vec_a[i].to_float();
        y_mean += vec_b[i].to_float();
    }
    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;
    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = vec_a[i].to_float() - x_mean;
        float y_diff = vec_b[i].to_float() - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }
    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (sqrt(x_stddev) * sqrt(y_stddev));
    return correlation_coefficient_;
}

// CPU implementation to test against
void golden_softmax(vector<bfloat16>& input, vector<bfloat16>& output, uint32_t M, uint32_t N) {
    size_t input_idx_start = 0, output_idx = 0;
    vector<float> row(N, 0);
    // For each row
    for (int i = 0; i < M; i++) {
        // Find max
        float max = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            float x = input.at(input_idx_start + j).to_float();
            if (x > max) {
                max = x;
            }
        }
        // Subtract max, exponentiate, sum and cache
        float sum = 0;
        for (int j = 0; j < N; j++) {
            float x = expf(input.at(input_idx_start + j).to_float() - max);
            sum += x;
            row.at(j) = x;
        }
        // Normalize
        for (int j = 0; j < N; j++) {
            output.at(output_idx) = bfloat16(row.at(j) / sum);
            output_idx++;
        }
        input_idx_start += N;
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
    auto grid = device->compute_with_storage_grid_size();
    // If single-core, limit to one core
    uint32_t max_cores_x = 1;
    uint32_t max_cores_y = 1;
    if (MULTI_CORE) {
        max_cores_x = grid.x;
        max_cores_y = grid.y;
    }

    // Determine the minimum number of rows per core and distribute the rest across the core grid (see paper)
    // Some cores have to process one additional row
    uint32_t max_cores = max_cores_x * max_cores_y;
    uint32_t min_rows_tiles_per_core = N_ROWS_TILES / max_cores;
    uint32_t remainder_rows_tiles = N_ROWS_TILES % max_cores;
    size_t rectangle_width = min(N_ROWS_TILES, max_cores_x);
    size_t rectangle_height = min(N_ROWS_TILES / max_cores_x, max_cores_y);
    CoreRangeSet cores({{{0, 0}, {rectangle_width - 1, rectangle_height - 1}}});
    size_t rest = N_ROWS_TILES % max_cores_x;
    if (N_ROWS_TILES > max_cores_x && N_ROWS_TILES < max_cores && rest > 0) {
        cores = cores.merge({{{0, rectangle_height}, {rest - 1, rectangle_height}}});
    }

    // Add reader, compute and writer kernels to the program
    // Place them on the cores determined before
    // Specify the RISC-V cores and the NoC connection for reader and writer kernels
    auto reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto softmax_kernel_id = CreateKernel(
        program,
        "kernels/compute/softmax.cpp",
        cores,
        tt_metal::ComputeConfig{});

    // Allocate circular buffers in the SRAM of Tensix cores
    // For input blocks
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(2 * N_TILES_BLOCK * TILE_SIZE, {{CB::c_in0, CB_DATA_FORMAT}})
            .set_page_size(CB::c_in0, TILE_SIZE));  // Double buffer for efficiency => multiply with two
    // For reduction scaler (implementation detail, not too important)
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(TILE_SIZE, {{CB::c_in1, CB_DATA_FORMAT}}).set_page_size(CB::c_in1, TILE_SIZE));
    // For caching a row of intermediate results
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(N_COLS_TILES * TILE_SIZE, {{CB::c_intermed0, CB_DATA_FORMAT}})
            .set_page_size(CB::c_intermed0, TILE_SIZE));
    // For maxima and sums
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(TILE_SIZE, {{CB::c_intermed1, CB_DATA_FORMAT}}).set_page_size(CB::c_intermed1, TILE_SIZE));
    // For output blocks
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(2 * N_TILES_BLOCK * TILE_SIZE, {{CB::c_out0, CB_DATA_FORMAT}})
            .set_page_size(CB::c_out0, TILE_SIZE));  // Double buffer for efficiency

    // Allocate buffers in DRAM of the card
    InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = N_TILES * TILE_SIZE,
        .page_size = TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = N_TILES * TILE_SIZE,
        .page_size = TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto input_dram_buffer = CreateBuffer(input_dram_config);
    auto output_dram_buffer = CreateBuffer(output_dram_config);
    uint32_t output_dram_buffer_address = output_dram_buffer->address();
    uint32_t input_dram_buffer_address = input_dram_buffer->address();

    // Pass run-time arguments to the kernels
    // Some arguments vary among the Tensix cores, because they read/write different rows from/to DRAM
    uint32_t start_tile_idx = 0;
    for (uint32_t i = 0; i < cores.num_cores(); i++) {
        CoreCoord core = {i % max_cores_x, i / max_cores_x};
        uint32_t n_rows_tiles_per_core = min_rows_tiles_per_core + (i < remainder_rows_tiles ? 1 : 0);
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input_dram_buffer_address, N_TILES_BLOCK, n_rows_tiles_per_core, N_COLS_TILES, start_tile_idx});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output_dram_buffer_address, n_rows_tiles_per_core * N_COLS_TILES, N_TILES_BLOCK, start_tile_idx});
        SetRuntimeArgs(program, softmax_kernel_id, core, {n_rows_tiles_per_core, N_COLS_TILES, N_TILES_BLOCK});
        start_tile_idx += n_rows_tiles_per_core * N_COLS_TILES;
    }

    // Compute the expected results for testing
    vector<bfloat16> input = create_random_vector_of_bfloat16_native(N_ELEMENTS * 2, 1, 123, -0.4);
    vector<bfloat16> golden_output(N_ELEMENTS, 0);
    golden_softmax(input, golden_output, N_ROWS, N_COLS);

    // Elements of each tile should be stored consecutively
    tilize(input, N_ROWS, N_COLS);

    // Write inputs from host's DRAM to card's DRAM (non-blocking call)
    EnqueueWriteBuffer(command_queue, input_dram_buffer, input.data(), false);

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
    vector<bfloat16> output(N_ELEMENTS);
    EnqueueReadBuffer(command_queue, output_dram_buffer, output.data(), true);
    untilize(output, N_ROWS, N_COLS);

    // Test with expected results
    float pcc = check_bfloat16_vector_pcc(golden_output, output);
    log_info(tt::LogVerif, "Metalium vs Golden PCC: {}", pcc);
    TT_FATAL(pcc > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pcc);

    CloseDevice(device);
    return 0;
}
