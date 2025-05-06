code = ""
       "
#include <iostream>
#include <cstdlib>

#define N 3 // Matrix size N x N
#define TILE_WIDTH 16

    using namespace std;

// CUDA Kernel
__global__ void MatrixMulKernel(int *A, int *B, int *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < width && col < width)
    {
        int sum = 0;
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];

        C[row * width + col] = sum;
    }
}

// Host function
void printMatrix(int *M, int width)
{
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
            cout << M[i * width + j] << "\t";
        cout << endl;
    }
}

int main()
{
    int size = N * N * sizeof(int);

    // Allocate host memory
    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    cout << "Matrix A:" << endl;
    printMatrix(h_A, N);
    cout << "Matrix B:" << endl;
    printMatrix(h_B, N);
    cout << "Matrix C (Result):" << endl;
    printMatrix(h_C, N);

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
""
    "

    with open("matrix_mul.cu", "w") as f : f.write(code) !nvcc
    - arch = sm_75 matrix_mul.cu - o matrix_mul
                                       !./
                                       matrix_mul