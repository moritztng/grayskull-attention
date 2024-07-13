/*
 * CPU implementation for the Softmax operation.
 * The first implementation caches the exponentials and the second implementation recomputes them.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Change constants for different experiments
const int RECOMPUTE = 0;    // Flag for selecting the implementation
const int MIN_ROWS = 1024;  // Define the min and max dimensions for profiling
const int MAX_ROWS = 8192;
const int MIN_COLS = 1024;
const int MAX_COLS = 8192;
const float GROWTH_RATE_ROWS = 2;  // Define the growth rate of the dimensions for profiling
const float GROWTH_RATE_COLS = 2;
const int PROFILING_ITERATIONS = 100;  // For averaging profiling results
const int MAX_ELS = MAX_ROWS * MAX_COLS;

// Implementation caching the exponentials
void softmax(const float* input, float* output, int n_rows, int n_cols) {
    int row_start_idx = 0;
    // For each row
    for (int i = 0; i < n_rows; i++) {
        // Find maximum
        float max = input[row_start_idx];
        for (int j = 1; j < n_cols; j++) {
            int k = row_start_idx + j;
            if (input[k] > max) {
                max = input[k];
            }
        }
        // Subtract the maximum, exponentiate, cache exponentials, and compute sum
        float sum = 0;
        for (int j = 0; j < n_cols; j++) {
            int k = row_start_idx + j;
            output[k] = exp(input[k] - max);
            sum += output[k];
        }
        // Normalize
        for (int j = 0; j < n_cols; j++) {
            int k = row_start_idx + j;
            output[k] /= sum;
        }
    }
    row_start_idx += n_cols;
}

// Implementation recomputing the exponentials
void softmax_recompute(const float* input, float* output, int n_rows, int n_cols) {
    int row_start_idx = 0;
    // For each row
    for (int i = 0; i < n_rows; i++) {
        // Find maximum
        float max = input[row_start_idx];
        for (int j = 1; j < n_cols; j++) {
            int k = row_start_idx + j;
            if (input[k] > max) {
                max = input[k];
            }
        }
        // Subtract the maximum, exponentiate, and compute sum (no caching)
        float sum = 0;
        for (int j = 0; j < n_cols; j++) {
            int k = row_start_idx + j;
            sum += exp(input[k] - max);
        }
        // Recompute exponentials and normalize
        for (int j = 0; j < n_cols; j++) {
            int k = row_start_idx + j;
            output[k] = exp(input[k] - max) / sum;
        }
        row_start_idx += n_cols;
    }
}

int main() {
    // Allocate DRAM for input and output matrices
    // Two allocations to prevent distorting profiling results by resetting input values (See paper)
    float* input = (float*)malloc(MAX_ELS * sizeof(float));
    float* output = (float*)malloc(MAX_ELS * sizeof(float));
    if (output == NULL || input == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }

    printf(
        "\nRead it in this format: ((rows, cols), runtime, runtime per element)\nBoth runtimes averaged and in "
        "seconds.\n");
    // First iteration as warmup for profiling
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            printf("\n--------Warmup--------\n");
        } else {
            printf("\n--------Profiling--------\n");
        }
        // Increase dimensions from min to max by growth rate
        for (int m = MIN_ROWS, n = MIN_COLS; m <= MAX_ROWS && n <= MAX_COLS;
             m *= GROWTH_RATE_ROWS, n *= GROWTH_RATE_COLS) {
            // Average runtime over multiple iterations
            clock_t start = clock();
            for (int j = 0; j < PROFILING_ITERATIONS; j++) {
                if (RECOMPUTE) {
                    softmax_recompute(input, output, m, n);
                } else {
                    softmax(input, output, m, n);
                }
            }
            double duration = (double)(clock() - start) / PROFILING_ITERATIONS / CLOCKS_PER_SEC;
            printf("((%d, %d), %e, %e),", m, n, duration, duration / (m * n));
            fflush(stdout);
        }
        printf("\n");
    }
    free(input);
    free(output);
    return 0;
}
