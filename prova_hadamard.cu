#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main() {
    
    const int nIndexBits = 2;  // 2 qubits
    const int nSvSize    = (1 << nIndexBits); // 4 states: |00>, |01>, |10>, |11>
    const int nTargets   = 2;
    const int adjoint    = 0;

    int targets[]  = {0,1};

    // Define the initial state vector in host memory
    cuDoubleComplex h_sv[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    }; // |00⟩

    cuDoubleComplex matrix_hadamard[] = {
        {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {-1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {-1.0 / 2.0, 0.0},  {1.0 / 2.0, 0.0}
    };

    cuDoubleComplex matrix_hadamard_2x2[] = {
        {1.0 / sqrt(2), 0.0},   {1.0 / sqrt(2), 0.0},
        {1.0 / sqrt(2), 0.0},  {-1.0 / sqrt(2), 0.0}
    };

    cuDoubleComplex* d_sv;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event using CUDA event
    cudaEventRecord(start);

    cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex));

    cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------

    

    // custatevec handle initialization
    custatevecHandle_t handle;
    custatevecCreate(&handle);

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    //--------------------------------------------------------------------------
    // Apply Hadamard gates to the initial state |00>
    custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix_hadamard, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, nullptr,
        nullptr, 0, CUSTATEVEC_COMPUTE_64F,
        extraWorkspace, extraWorkspaceSizeInBytes);

    //--------------------------------------------------------------------------

    // Apply Hadamard gates again to the output to measure
    custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix_hadamard, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, nullptr,
        nullptr, 0, CUSTATEVEC_COMPUTE_64F,
        extraWorkspace, extraWorkspaceSizeInBytes);
    //--------------------------------------------------------------------------

    // destroy handle
    custatevecDestroy(handle);

    //--------------------------------------------------------------------------

    // Copy result back to host memory
    cuDoubleComplex h_sv_result[nSvSize];
    cudaMemcpy(h_sv_result, d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    cudaFree(d_sv);

    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total Time elapsed: %f ms\n", milliseconds);

    // Check if the result is correct after applying Hadamard
    bool correct = true;
    for (int i = 0; i < nSvSize; i++) {
        if ((h_sv_result[i].x != h_sv[i].x) ||
            (h_sv_result[i].y != h_sv[i].y)) {
            correct = false;
            break;
        }
    }

    if (correct)
        printf("Hadamard gates test PASSED\n");
    else
        printf("Hadamard gates test FAILED: wrong result\n");

    return EXIT_SUCCESS;
}