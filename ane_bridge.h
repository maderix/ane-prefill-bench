// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle
typedef struct ANEKernelHandle ANEKernelHandle;

// Initialize ANE runtime (load private framework, resolve classes)
// Returns 0 on success, -1 on failure
int ane_bridge_init(void);

// Compile a MIL program with weight blobs into an ANE kernel
// mil_text: UTF-8 MIL program text
// mil_len: length of MIL text
// weight_data: raw weight blob (can be NULL)
// weight_len: length of weight blob
// n_inputs: number of input tensors
// input_sizes: array of byte sizes for each input
// n_outputs: number of output tensors
// output_sizes: array of byte sizes for each output
// Returns kernel handle or NULL on failure
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
// weight_names: array of weight file paths (e.g. "@model_path/weights/wq.bin")
// weight_datas: array of weight data pointers
// weight_lens: array of weight data lengths
// n_weights: number of weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
// Returns true on success
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to kernel input tensor
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Read data from kernel output tensor
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);

// Reset compile count
void ane_bridge_reset_compile_count(void);

// Build a weight blob in ANE format (128-byte header + fp16 data)
// src: float32 weights [rows x cols]
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);

// Build a transposed weight blob in ANE format
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);

// Free a blob allocated by ane_bridge_build_weight_blob*
void ane_bridge_free_blob(void *ptr);

// Enable dynamic weights via weightsBuffer IOSurface
// Allocates a weight IOSurface and rebuilds the request with weightsBuffer.
// weight_data: raw fp16 data (no blob header), weight_bytes long.
// After calling this, ane_bridge_eval() uses the dynamic weights.
// Returns true on success.
bool ane_bridge_set_dynamic_weights(ANEKernelHandle *kernel,
                                     const void *weight_data, size_t weight_bytes);

// Update dynamic weight data (fast — just copies to existing IOSurface)
// Must call ane_bridge_set_dynamic_weights() first to allocate.
void ane_bridge_update_dynamic_weights(ANEKernelHandle *kernel,
                                        const void *weight_data, size_t weight_bytes);

// Fused multi-head attention on ANE
// Q, K, V: [H*S, D] row-major float arrays (as i64 pointers)
// O: output [H*S, D] row-major float array (as i64 pointer)
// H: number of heads, S: sequence length, D: head dimension
void ane_runtime_mha(int64_t ptrQ, int64_t ptrK, int64_t ptrV, int64_t ptrO,
                     int64_t H, int64_t S, int64_t D);

// Plan-aware runtime entry points. Generic NPU planning passes lower target
// metadata into these calls; ANE is the first concrete implementation.
void ane_runtime_matmul_plan(int64_t ptrA, int64_t ptrB, int64_t ptrC,
                             int64_t M, int64_t N, int64_t K,
                             int64_t seq_tile, int64_t proj_oc_tile,
                             int64_t ffn_k_tile, int64_t head_tile,
                             int64_t local_memory_bytes);

void ane_runtime_mha_plan(int64_t ptrQ, int64_t ptrK, int64_t ptrV,
                          int64_t ptrO, int64_t H, int64_t S, int64_t D,
                          int64_t seq_tile, int64_t head_tile,
                          int64_t local_memory_bytes);

void ane_runtime_deltanet_recurrence_plan(
    int64_t ptrQ, int64_t ptrK, int64_t ptrV, int64_t ptrBeta, int64_t ptrG,
    int64_t ptrState, int64_t ptrO, int64_t state_offset_bytes,
    int64_t H, int64_t D, int64_t C,
    int64_t recurrence_token_tile, int64_t recurrence_head_tile,
    int64_t local_memory_bytes);

void ane_runtime_q4k_prefill_matmul_plan(
    int64_t ptrQWeights, int64_t ptrScales, int64_t ptrMins,
    int64_t ptrInput, int64_t ptrOutput, int64_t OC, int64_t S, int64_t K,
    int64_t seq_tile, int64_t proj_oc_tile, int64_t local_memory_bytes);

void ane_runtime_q5k_prefill_matmul_plan(
    int64_t ptrQVals, int64_t ptrScales, int64_t ptrMins,
    int64_t ptrInput, int64_t ptrOutput, int64_t OC, int64_t S, int64_t K,
    int64_t seq_tile, int64_t proj_oc_tile, int64_t local_memory_bytes);

void ane_runtime_q6k_prefill_matmul_plan(
    int64_t ptrQVals, int64_t ptrScales, int64_t ptrInput, int64_t ptrOutput,
    int64_t OC, int64_t S, int64_t K,
    int64_t seq_tile, int64_t proj_oc_tile, int64_t local_memory_bytes);

// Direct IOSurface access for zero-copy Metal/ANE interop
// Returns the IOSurfaceRef for a kernel's input or output buffer.
// The caller must IOSurfaceLock/Unlock around access.
void *ane_bridge_get_input_surface(ANEKernelHandle *kernel, int idx);
void *ane_bridge_get_output_surface(ANEKernelHandle *kernel, int idx);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
