import torch
import torch.nn.functional as F
import sys

TOLERANCE = 1e-5

N = 4096
M = 4096
d = 1024

if len(sys.argv) < 2:
    print("CUDA output file not specified")
    sys.exit(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# get pytorch output
rows_N = torch.arange(N, device=device).unsqueeze(1).expand(N, d)
cols_d = torch.arange(d, device=device).unsqueeze(0).expand(N, d)

rows_M = torch.arange(M, device=device).unsqueeze(1).expand(M, d)
cols_d_M = torch.arange(d, device=device).unsqueeze(0).expand(M, d)

Q = ((rows_N * 7 + cols_d * 13) % 31) * 0.1 - 1.5
K = ((rows_M * 11 + cols_d_M * 17) % 29) * 0.1 - 1.3
V = ((rows_M * 5 + cols_d_M * 19) % 37) * 0.1 - 1.1

pytorch_output = F.scaled_dot_product_attention(Q, K, V)

# get cuda output
with open(sys.argv[1], 'r') as f:
    rows = f.readlines()
    cuda_output = torch.tensor(
        [list(map(float, row.split())) for row in rows],
        dtype=torch.float32,
        device=device
    ).view(N, d)

if torch.allclose(pytorch_output, cuda_output, rtol=TOLERANCE, atol=TOLERANCE):
    print('Outputs match within tolerance')
else:
    print('Outputs do not match within tolerance')