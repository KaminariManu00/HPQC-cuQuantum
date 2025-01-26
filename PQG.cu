#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cmath>              // cos and sin

// Helper function for checking CUDA errors
#define CHECK_CUDA(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA Error: %s:%d, '%s'\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// Helper function for checking cuStateVec errors
#define CHECK_CUSTATEVEC(call)                                       \
    do {                                                             \
        custatevecStatus_t status = call;                            \
        if (status != CUSTATEVEC_STATUS_SUCCESS) {                   \
            fprintf(stderr, "cuStateVec Error: %s:%d\n",             \
                    __FILE__, __LINE__);                             \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)
float* get_quantum_measures(double* angles, int num_angles, int depth);

#define MATRIX_SIZE 4

// Function to create RX matrix
cuDoubleComplex* create_rx_matrix(double angle);

// Function to create RY matrix
cuDoubleComplex* create_ry_matrix(double angle);

// Function to create RZ matrix
cuDoubleComplex* create_rz_matrix(double angle);

int main(int argc, char* argv[]) {
    if (argc < 38) {
        printf("Usage: %s <depth> <angle1> <angle2> ... <angle36>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int depth = atoi(argv[1]);

    // Store the 9 parameters from command line into a vector
    double angles[36];
    for (int i = 1; i < argc; ++i) {
        angles[i - 1] = atof(argv[i]);  // Convert command-line argument to double
    }

    float* output = get_quantum_measures(angles, 36, depth);  // Call the function to get measurements

    // Output the array of 8 values
    for (int i = 0; i < 8; ++i) {
        printf("Probability[%d]: %f\n", i, output[i]);
    }

    return 0;
}


float* get_quantum_measures(double* angles, int num_angles, int depth) {
    if (num_angles < 36) {
        fprintf(stderr, "Error: At least 36 angles are required.\n");
        exit(EXIT_FAILURE);
    }

    const int nIndexBits = 3;                 // 3 qubits
    const int nSvSize    = (1 << nIndexBits); // 8 states 
    const int nTargets   = 1;                 //Always 1 target at a time
    const int nControls  = 1;                 //Always 1 control at a time
    const int adjoint    = 0;

    int32_t target0 = 0;
    int32_t target1 = 1;
    int32_t target2 = 2;

    int32_t control0 = 0;
    int32_t control1 = 1;

    // Define the initial state vector in host memory
    cuDoubleComplex h_sv[nSvSize] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    }; // |000⟩

    //Hadamard matrix
    cuDoubleComplex matrix_hadamard_2x2[] = {
        {1.0 / sqrt(2), 0.0}, {1.0 / sqrt(2), 0.0},
        {1.0 / sqrt(2), 0.0}, {-1.0 / sqrt(2), 0.0}
    };

    // Create matrices for each qubit
    cuDoubleComplex* rx_matrices[nIndexBits];
    cuDoubleComplex* ry_matrices[nIndexBits];
    cuDoubleComplex* rz_matrices[nIndexBits];

    //CNOT matrix
    cuDoubleComplex matrix_cnot[] = {{0.0, 0.0}, {1.0, 0.0},
                                     {1.0, 0.0}, {0.0, 0.0}};

    cuDoubleComplex* d_sv;
    CHECK_CUDA(cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));

    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    //--------------------------------------------------------------------------

    // Apply Hadamard gates to the initial state
    for (int qubit = 0; qubit < 3; ++qubit) {
        int* target = (qubit == 0) ? &target0 : (qubit == 1) ? &target1 : &target2;
        custatevecApplyMatrix(
            handle, d_sv, CUDA_C_64F, nIndexBits, matrix_hadamard_2x2, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, target, nTargets, nullptr,
            nullptr, 0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes
        );
    }

    //--------------------------------------------------------------------------

    // Loop to apply each set of gates depth times
    for (int k = 0; k <= depth; k++) {
            
        // Create matrices for each gate
        for (int j = 0; j < nIndexBits; ++j) {
            int baseIndex = k * 9;  // Base index in the angles array for each depth level
            rx_matrices[j] = create_rx_matrix(angles[baseIndex + j]);       // rx_matrices[j] corresponds to angles[k * 9 + j]
            ry_matrices[j] = create_ry_matrix(angles[baseIndex + nIndexBits + j]);  // ry_matrices[j] corresponds to angles[k * 9 + 3 + j]
            rz_matrices[j] = create_rz_matrix(angles[baseIndex + 2 * nIndexBits + j]);  // rz_matrices[j] corresponds to angles[k * 9 + 6 + j]
        }
        
        // Apply RX gates
        for (int j = 0; j < nIndexBits; ++j) {
            int32_t* target = (j == 0) ? &target0 : (j == 1) ? &target1 : &target2;
            custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, rx_matrices[j], CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, target, nTargets, nullptr,
                nullptr, 0, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes
            );
        }

        // Apply RY gates
        for (int j = 0; j < nIndexBits; ++j) {
            int32_t* target = (j == 0) ? &target0 : (j == 1) ? &target1 : &target2;
            custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, ry_matrices[j], CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, target, nTargets, nullptr,
                nullptr, 0, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes
            );
        }

        // Apply RZ gates
        for (int j = 0; j < nIndexBits; ++j) {
            int32_t* target = (j == 0) ? &target0 : (j == 1) ? &target1 : &target2;
            custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, rz_matrices[j], CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, target, nTargets, nullptr,
                nullptr, 0, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes
            );
        }

        if(k != depth){
            // Apply the CNOT gates
            custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, matrix_cnot, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target1, nTargets, &control0,
                nullptr, nControls, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes
            );
            custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, matrix_cnot, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target2, nTargets, &control1,
                nullptr, nControls, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes
            ); 
        } 
    }
    //--------------------------------------------------------------------------

    // Copy result back to host memory
    cuDoubleComplex h_sv_result[nSvSize];
    CHECK_CUDA(cudaMemcpy(h_sv_result, d_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    static float output[nSvSize];

    // Measure probabilities
    for (int i = 0; i < nSvSize; i++) {
        output[i] = h_sv_result[i].x * h_sv_result[i].x  + h_sv_result[i].y * h_sv_result[i].y;
    }
    // Clean up
    CHECK_CUSTATEVEC(custatevecDestroy(handle));
    CHECK_CUDA(cudaFree(d_sv));

    return output;
}

// Function to create RX matrix
cuDoubleComplex* create_rx_matrix(double angle) {
    cuDoubleComplex* matrix = (cuDoubleComplex*)malloc(MATRIX_SIZE * sizeof(cuDoubleComplex));
    matrix[0] = (cuDoubleComplex){ cos(0.5 * angle), 0.0 };
    matrix[1] = (cuDoubleComplex){ 0.0, -sin(0.5 * angle) };
    matrix[2] = (cuDoubleComplex){ 0.0, -sin(0.5 * angle) };
    matrix[3] = (cuDoubleComplex){ cos(0.5 * angle), 0.0 };
    return matrix;
}

// Function to create RY matrix
cuDoubleComplex* create_ry_matrix(double angle) {
    cuDoubleComplex* matrix = (cuDoubleComplex*)malloc(MATRIX_SIZE * sizeof(cuDoubleComplex));
    matrix[0] = (cuDoubleComplex){ cos(0.5 * angle), 0.0 };
    matrix[1] = (cuDoubleComplex){ -sin(0.5 * angle), 0.0 };
    matrix[2] = (cuDoubleComplex){ sin(0.5 * angle), 0.0 };
    matrix[3] = (cuDoubleComplex){ cos(0.5 * angle), 0.0 };
    return matrix;
}

// Function to create RZ matrix
cuDoubleComplex* create_rz_matrix(double angle) {
    cuDoubleComplex* matrix = (cuDoubleComplex*)malloc(MATRIX_SIZE * sizeof(cuDoubleComplex));
    matrix[0] = (cuDoubleComplex){ cos(0.5 * angle), -sin(0.5 * angle) };
    matrix[1] = (cuDoubleComplex){ 0.0, 0.0 };
    matrix[2] = (cuDoubleComplex){ 0.0, 0.0 };
    matrix[3] = (cuDoubleComplex){ cos(0.5 * angle), sin(0.5 * angle) };
    return matrix;
}
