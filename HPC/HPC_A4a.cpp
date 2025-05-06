code = """
#include <iostream>
#include <vector>
using namespace std;

// CUDA kernel for vector addition
__global__ void vecAdd(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        C[tid] = A[tid] + B[tid];
}

// Sequential vector addition for reference
void addition(vector<int>& vecA, vector<int>& vecB, vector<int>& vecC, const int& size) {
    for (int i = 0; i < size; i++)
        vecC[i] = vecA[i] + vecB[i];
}

// Print vector
void Print(const vector<int>& vc) {
    for (auto x : vc)
        cout << x << "\t";
    cout << endl;
}

int main() {
    const int size = 7;
    vector<int> A = {2, 3, 4, 5, 6, 7, 8};
    vector<int> B = {4, 5, 6, 12, 34, 54, 2};
    vector<int> C(size);  // Result vector

    cout << "Vec A: \t\t"; Print(A);
    cout << "Vec B: \t\t"; Print(B);
    cout << endl;

    // Sequential addition
    addition(A, B, C, size);
    cout << "Seq Vec C: \t"; Print(C);

    // Allocate memory on GPU
    int totalBytes = size * sizeof(int);
    int *X, *Y, *Z;
    cudaMalloc(&X, totalBytes);
    cudaMalloc(&Y, totalBytes);
    cudaMalloc(&Z, totalBytes);

    // Copy data to device
    cudaMemcpy(X, A.data(), totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B.data(), totalBytes, cudaMemcpyHostToDevice);

    // Launch kernel with proper configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, size);
    cudaDeviceSynchronize();  // Wait for kernel to finish

    // Copy result back to host
    cudaMemcpy(C.data(), Z, totalBytes, cudaMemcpyDeviceToHost);

    // Print parallel result
    cout << "Par Vec C: \t"; Print(C);

    // Free GPU memory
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}
"""

# Save and compile
with open("test.cu", "w") as f:
    f.write(code)

!nvcc test.cu -o test_cuda -arch=sm_70
!./test_cuda


