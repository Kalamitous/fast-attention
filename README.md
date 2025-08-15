# Fast Attention

This repository contains a CUDA implementation of the attention mechanism (forward pass), primarily composed of kernels for transpose, matrix multiply, and softmax.

`naive_attention.cu` is a straightforward 2D parallelization approach.\
`fast_attention.cu` improves upon this through several optimizations detailed below.

## Benchmarks

Profiling was done with Nsight Compute on an NVIDIA GeForce RTX 4080 SUPER.\
The dimensions used for `Q`, `K`, and `V` are 4096 x 1024.

| Version | Time |
| - | - |
| naive_attention.cu | 219.09 ms |
| fast_attention.cu | 3.71 ms |

This is a ~59x speedup!

## Optimizations
- **matrix_transpose**: Divided the input matrix into tiles, each loaded into shared memory. Block indices are maintained to ensure coalesced global memory reads/writes, and the shared memory tile is padded to avoid bank conflicts. This reduces global memory latency and improves throughput.
- **matrix_multiply**: Applied similar block tiling and shared memory usage, mapping columns to the horizontal thread axis to ensure memory coalescing. Arithmetic intensity is further increased with thread tiling and register blocking. 
- **matrix_softmax**: Implemented online softmax to fuse the loops of finding the max and computing the denominator. Each warp performs reductions via shuffle instructions to find the local values, followed by a final block-level and warp-level reduction to obtain the global values. This increases parallelism by distributing each row across a warp instead of a single thread. Memory coalescing is ensured because threads in a warp access consecutive elements in memory.

## Running the Program
1. Compile with NVCC:
```bash
nvcc fast_attention.cu -o fast_attention
```
2. Run with an output file:
```bash
./fast_attention out
```
3. Test correctness:
```bash
python test_correctness.py out
```

## Next Steps
- Improve occupancy
- Fuse kernels
- Write a script to autotune kernel parameters

## References
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/)
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)