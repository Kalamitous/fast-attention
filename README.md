# Fast Attention

This repository contains a CUDA implementation of the feed-forward attention mechanism, primarily composed of kernels for transpose, matrix multiply, and softmax.

# Benchmarks

Profiling was done with Nsight Compute on an NVIDIA GeForce RTX 4080 SUPER. The dimensions used for `Q`, `K`, and `V` are 4096 x 1024.

`naive_attention.cu`: 219.09 ms

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