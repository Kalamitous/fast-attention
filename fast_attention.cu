#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <math_constants.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

const int WARP_SIZE = 32;

const int N = 4096;
const int M = 4096;
const int d = 1024;

// transpose parameters
const int TILE_DIM = 64;
const int BLOCK_ROWS = TILE_DIM / 8;

// matrix multiply parameters
const int BN = 128;
const int BM = 128;
const int Bd = 16;
const int TN = 8;
const int TM = 8;

// transposes src into dst
// ref: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template <const int TILE_DIM, const int BLOCK_ROWS>
__global__ void matrix_transpose(
    const float* src,
    float* dst,
    int N, int M
) {
    static_assert(TILE_DIM % BLOCK_ROWS == 0, "TILE_DIM must be divisible by BLOCK_ROWS");

    // ensure memory coalescing by mapping columns to the horizontal axis
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // pad width to avoid bank conflicts when
    // accessing the same column in different rows
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    // copy tile into smem
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((row + i) < N && col < M) {
            tile[thread_row + i][thread_col] = src[(row + i) * M + col];
        }
    }
    __syncthreads();

    // transpose global indices
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    col = blockIdx.y * TILE_DIM + threadIdx.x;

    // transpose from smem
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((row + i) < M && col < N) {
            dst[(row + i) * N + col] = tile[thread_col][thread_row + i];
        }
    }
}

// C = A * B
// ref: https://siboehm.com/articles/22/CUDA-MMM
template <const int BN, const int BM, const int Bd, const int TN, const int TM>
__global__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int N, int M, int d
) {
    // calculate indices
    int outer_row = blockIdx.y;
    int outer_col = blockIdx.x;

    int num_threads = (BN * BM) / (TN * TM);
    assert(num_threads == blockDim.x);
    
    int inner_row_a = threadIdx.x / Bd;
    int inner_col_a = threadIdx.x % Bd;
    int stride_a = num_threads / Bd;

    int inner_row_b = threadIdx.x / BM;
    int inner_col_b = threadIdx.x % BM;
    int stride_b = num_threads / BM;

    int thread_row = threadIdx.x / (BM / TM);
    int thread_col = threadIdx.x % (BM / TM);

    // allocate space for a block tile in smem 
    __shared__ float As[BN * Bd];
    __shared__ float Bs[Bd * BM];

    // allocate space for a thread tile in registers
    float Ar[TN] = {0.0f};
    float Br[TM] = {0.0f};
    float Cr[TN * TM] = {0.0f};

    // point to the first block tile
    A += outer_row * BN * d;
    B += outer_col * BM;
    C += outer_row * BN * M + outer_col * BM;

    // iterate through block tiles
    for (int block = 0; block < d; block += Bd) {
        // copy block tile to smem
        for (int i = 0; i < BN; i += stride_a) {
            As[(inner_row_a + i) * Bd + inner_col_a] = A[(inner_row_a + i) * d + inner_col_a];
        }
        for (int i = 0; i < Bd; i += stride_b) {
            Bs[(inner_row_b + i) * BM + inner_col_b] = B[(inner_row_b + i) * M + inner_col_b];
        }
        __syncthreads();

        // advance to next block tile
        A += Bd;
        B += Bd * M;

        for (int dot_idx = 0; dot_idx < Bd; ++dot_idx) {
            // copy thread tile to registers
            for (int row = 0; row < TN; ++row) {
                Ar[row] = As[(thread_row * TN + row) * Bd + dot_idx];
            }
            for (int col = 0; col < TM; ++col) {
                Br[col] = Bs[dot_idx * BM + thread_col * TM + col];
            }
            // compute dot products
            for (int row = 0; row < TN; ++row) {
                for (int col = 0; col < TM; ++col) {
                    Cr[row * TM + col] += Ar[row] * Br[col];
                }
            }
        }
        __syncthreads();
    }

    // write per-thread results to gmem
    for (int row = 0; row < TN; ++row) {
        for (int col = 0; col < TM; ++col) {
            C[(thread_row * TN + row) * M + thread_col * TM + col] = Cr[row * TM + col];
        }
    }
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    // now the first thread in each warp has its local max
    if (blockDim.x > WARP_SIZE) {
        // block-level reduction on the first thread of each warp
        if (tid % WARP_SIZE == 0){
            smem[tid / WARP_SIZE] = val;
        }
        __syncthreads();
        // now the first warp holds all local maxes
        if (tid < WARP_SIZE) {
            // warp-level reduction on the first warp
            // there are CEIL_DIV(blockDim.x, WARP_SIZE) local maxes
            val = (tid < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tid] : -CUDART_INF_F;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (blockDim.x > WARP_SIZE) {
        if (tid % WARP_SIZE == 0){
            smem[tid / WARP_SIZE] = val;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            val = (tid < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tid] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
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

    dim3 blockDim1(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim1(CEIL_DIV(d, TILE_DIM), CEIL_DIV(M, TILE_DIM));
    matrix_transpose<TILE_DIM, BLOCK_ROWS><<<gridDim1, blockDim1>>>(K, K_T, M, d);
    cudaDeviceSynchronize();

    int num_threads = (BN * BM) / (TN * TM);
    dim3 blockDim2(num_threads);
    dim3 gridDim2(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    matrix_multiply<BN, BM, Bd, TN, TM><<<gridDim2, blockDim2>>>(Q, K_T, scores, N, M, d);
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

    dim3 blockDim5(num_threads);
    dim3 gridDim5(CEIL_DIV(d, BM), CEIL_DIV(N, BN));
    matrix_multiply<BN, BM, Bd, TN, TM><<<gridDim5, blockDim5>>>(scores, V, O, N, d, M);
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