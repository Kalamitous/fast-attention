#include <fstream>
#include <iomanip>
#include <iostream>

#include <math_constants.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

const int N = 4096;
const int M = 4096;
const int d = 1024;

// transposes src into dst
__global__ void matrix_transpose(
    const float* src,
    float* dst,
    int N, int M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= M) return;

    dst[col * N + row] = src[row * M + col];
}

// C = A * B
__global__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int N, int M, int d
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= M) return;

    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += A[row * d + i] * B[i * M + col];
    }
    C[row * M + col] = sum;
}

// divides array by value in place
__global__ void array_divide(
    float* array,
    float value,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] /= value;
    }
}

// computes softmax of matrix in place
// ref: https://maharshi.bearblog.dev/optimizing-softmax-cuda/
__global__ void matrix_softmax(
    float* matrix,
    int N, int M
) {
    extern __shared__ float smem[];

    // ensure memory coalescing
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    int warp_size = 32;

    // perform online softmax which fuses the step of finding the local max
    // and the step of computing the local denominator
    float local_max = -CUDART_INF_F;
    float local_sum_exp = 0.0f;
    for (int col = tid; col < M; col += blockDim.x) {
        float cur = matrix[row * M + col];
        if (cur > local_max) {
            // correct local denominator up until this point
            local_sum_exp *= expf(local_max - cur);
            local_max = cur;
        }
        local_sum_exp += expf(cur - local_max);
    }
    __syncthreads();

    // warp-level reduction via warp shuffling
    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    // now the first thread in each warp has its local max
    if (blockDim.x > warp_size) {
        // block-level reduction on the first thread of each warp
        if (tid % warp_size == 0){
            smem[tid / warp_size] = val;
        }
        __syncthreads();
        // now the first warp holds all local maxes
        if (tid < warp_size) {
            // warp-level reduction on the first warp
            // there are CEIL_DIV(blockDim.x, warp_size) local maxes
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -CUDART_INF_F;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            // now the first thread holds the global max
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    float global_max = smem[0];
    __syncthreads();

    // repeat the reduction process to obtain the global denominator
    // correct the local denominator with the global max
    val = local_sum_exp * expf(local_max - global_max);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0){
            smem[tid / warp_size] = val;
        }
        __syncthreads();
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    float global_sum_exp = smem[0];
    __syncthreads();

    // normalize
    for (int col = tid; col < M; col += blockDim.x) {
        int i = row * M + col;
        matrix[i] = expf(matrix[i] - global_max) / global_sum_exp;
    }
}

// computes attention from Q, K, V into O
void attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int M, int d
) {
    float *K_T, *scores;
    cudaMalloc(&K_T, d * M * sizeof(float));
    cudaMalloc(&scores, N * M * sizeof(float));

    dim3 blockDim1(32, 32);
    dim3 gridDim1(CEIL_DIV(M, blockDim1.x), CEIL_DIV(d, blockDim1.y));
    matrix_transpose<<<gridDim1, blockDim1>>>(K, K_T, M, d);
    cudaDeviceSynchronize();

    dim3 blockDim2(32, 32);
    dim3 gridDim2(CEIL_DIV(N, blockDim2.x), CEIL_DIV(M, blockDim2.y));
    matrix_multiply<<<gridDim2, blockDim2>>>(Q, K_T, scores, N, M, d);
    cudaDeviceSynchronize();

    dim3 blockDim3(1024);
    dim3 gridDim3(CEIL_DIV(N * M, blockDim3.x));
    array_divide<<<gridDim3, blockDim3>>>(scores, sqrtf((float)d), N * M);
    cudaDeviceSynchronize();

    dim3 blockDim4(1024);
    dim3 gridDim4(N);
    int shared_bytes = blockDim4.x * sizeof(float);
    matrix_softmax<<<gridDim4, blockDim4, shared_bytes>>>(scores, N, M);
    cudaDeviceSynchronize();

    dim3 blockDim5(32, 32);
    dim3 gridDim5(CEIL_DIV(N, blockDim5.x), CEIL_DIV(d, blockDim5.y));
    matrix_multiply<<<gridDim5, blockDim5>>>(scores, V, O, N, d, M);
    cudaDeviceSynchronize();

    cudaFree(K_T);
    cudaFree(scores);
}

int main(int argc, char* argv[]) {
    float *Q, *K, *V, *O;
    cudaMallocManaged(&Q, N * d * sizeof(float));
    cudaMallocManaged(&K, M * d * sizeof(float));
    cudaMallocManaged(&V, M * d * sizeof(float));
    cudaMallocManaged(&O, N * d * sizeof(float));

    // initialize pseudo-random test matrices
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < d; ++col) {
            Q[row * d + col] = ((row * 7 + col * 13) % 31) * 0.1f - 1.5f;
        }
    }
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < d; ++col) {
            K[row * d + col] = ((row * 11 + col * 17) % 29) * 0.1f - 1.3f;
            V[row * d + col] = ((row * 5 + col * 19) % 37) * 0.1f - 1.1f;
        }
    }

    // prefetch matrices to gpu
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;
    cudaMemPrefetchAsync(Q, N * d * sizeof(float), loc, 0);
    cudaMemPrefetchAsync(K, M * d * sizeof(float), loc, 0);
    cudaMemPrefetchAsync(V, M * d * sizeof(float), loc, 0);
    cudaMemPrefetchAsync(O, N * d * sizeof(float), loc, 0);

    attention(Q, K, V, O, N, M, d);
    
    // write output to file
    if (argc > 1) {
        std::ofstream file(argv[1]);
        if (!file) {
            std::cerr << "Error opening file: " << argv[1] << std::endl;
        }
        for (int i = 0; i < N * d; i++) {
            file << std::fixed << std::setprecision(6) << O[i];
            if (i < N * d - 1) {
                file << " ";
            }
        }
        file.close();
        std::cout << "Output written to file: " << argv[1] << std::endl;
    }

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
}