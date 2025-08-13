# Fast Attention

This repository contains a CUDA implementation of the attention mechanism (forward pass), primarily composed of kernels for transpose, matrix multiply, and softmax.

# Benchmarks

Profiling was done with Nsight Compute on an NVIDIA GeForce RTX 4080 SUPER. The dimensions used for `Q`, `K`, and `V` are 4096 x 1024.

`naive_attention.cu`: 219.09 ms
`fast_attention.cu`: 209.88 ms

# Optimizations
- `matrix_softmax`: Implemented online softmax to fuse the loops of finding the max and computing the denominator. Each warp performs reductions via shuffle instructions to find the local values, followed by a final block-level and warp-level reduction to obtain the global values. This increases parallelism by distributing each row across a warp instead of a single thread. Memory coalescing is ensured because threads in a warp access consecutive elements in memory.

## Running the Program
1. Compile with NVCC:
```bash
nvcc naive_attention.cu -o naive_attention
```
2. Run the program with an output file
```bash
./naive_attention out
```
3. Test correctness
```bash
python test_correctness.py out
```