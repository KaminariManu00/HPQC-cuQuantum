#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define ORACLE_TYPE_BALANCED 0
#define ORACLE_TYPE_CONSTANT 1

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <oracle_type>\n", argv[0]);
        printf("0: Balanced Oracle\n1: Constant Oracle\n");
        return EXIT_FAILURE;
    }

    int oracle_type = atoi(argv[1]);
    if (oracle_type == ORACLE_TYPE_BALANCED) {
        printf("Oracle type: Balanced\n");
    } else {
        printf("Oracle type: Constant\n");
    }

    const int nIndexBits = 2;  // 2 qubits
    const int nSvSize    = (1 << nIndexBits); // 4 states: |00>, |01>, |10>, |11>
    const int nTargets   = 2;
    const int adjoint    = 0;

    int targets[]  = {0,1};

    // Define the initial state vector in host memory
    cuDoubleComplex h_sv[] = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    }; // |01⟩
    // Define the initial state vector in host memory
    cuDoubleComplex h_sv_result_balanced[] = {
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
    }; // |10⟩
    // Define the initial state vector in host memory
    cuDoubleComplex h_sv_result_constant[] = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    }; // |01⟩

    // Oracle matrices for balanced and constant cases
    cuDoubleComplex matrix_balanced[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
    };
    cuDoubleComplex matrix_constant[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
    };

    cuDoubleComplex matrix_hadamard[] = {
        {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {-1.0 / 2.0, 0.0},
        {1.0 / 2.0, 0.0},   {-1.0 / 2.0, 0.0},  {-1.0 / 2.0, 0.0},  {1.0 / 2.0, 0.0}
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

    // Determine the Oracle matrix based on the input
    cuDoubleComplex* matrix;
    if (oracle_type == ORACLE_TYPE_BALANCED) {
        matrix = matrix_balanced;
    } else {
        matrix = matrix_constant;
    }
    //--------------------------------------------------------------------------

    // Apply Hadamard gates to the initial state |01>
    printf("\nInitial State:\n");
    for (int i = 0; i < nSvSize; i++) {
        printf("(%f, %f)\t", h_sv[i].x, h_sv[i].y);
    }
    printf("\n");
    custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix_hadamard, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, nullptr,
        nullptr, 0, CUSTATEVEC_COMPUTE_64F,
        extraWorkspace, extraWorkspaceSizeInBytes);

    //--------------------------------------------------------------------------

    // Apply Oracle (Gate 1)
    printf("\nState after Hadamard gates:\n");
    cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < nSvSize; i++) {
        printf("(%f, %f)\t", h_sv[i].x, h_sv[i].y);
    }
    printf("\n");

    custatevecApplyMatrix(
       handle, d_sv, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, nullptr,
       nullptr, 0, CUSTATEVEC_COMPUTE_64F,
       extraWorkspace, extraWorkspaceSizeInBytes);

    //--------------------------------------------------------------------------

    // Apply Hadamard gates again to the output to measure
    /*printf("\nState after Oracle application:\n");
    cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < nSvSize; i++) {
        printf("(%f, %f)\t", h_sv[i].x, h_sv[i].y);
    }
    printf("\n");*/

    custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix_hadamard, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, nullptr,
        nullptr, 0, CUSTATEVEC_COMPUTE_64F,
        extraWorkspace, extraWorkspaceSizeInBytes);

    //--------------------------------------------------------------------------

    // destroy handle
    custatevecDestroy(handle);

    //--------------------------------------------------------------------------

    cuDoubleComplex* h_sv_result;
    if (oracle_type == ORACLE_TYPE_BALANCED) {
        h_sv_result = h_sv_result_balanced;
    } else {
        h_sv_result = h_sv_result_constant;
    }

    cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    cudaFree(d_sv);

    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total Time elapsed: %f ms\n", milliseconds);

    bool correct = true;
    for (int i = 0; i < nSvSize; i++) {
        if ((h_sv[i].x != h_sv_result[i].x) ||
            (h_sv[i].y != h_sv_result[i].y)) {
            correct = false;
            break;
        }
    }

    printf("\nState vector:\n");
    for (int i = 0; i < nSvSize; i++) {
        printf("(%f, %f)\t", h_sv[i].x, h_sv[i].y);
    }
    printf("\n");

    if (correct)
        printf("Deutsch-Jozsa algorithm PASSED\n");
    else
        printf("Deutsch-Jozsa algorithm FAILED: wrong result\n");

    return EXIT_SUCCESS;
}
