import time

import torch

N = 8192
SECONDS = 15


def bench(dtype: torch.dtype) -> float:
    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < SECONDS:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        iters += 10
    dt = time.perf_counter() - t0
    return 2 * N**3 * iters / dt / 1e12


print(torch.cuda.get_device_name(0))
print(f"fp16 matmul {N}x{N}: {bench(torch.float16):.1f} TFLOPS ({SECONDS}s)")
print(f"bf16 matmul {N}x{N}: {bench(torch.bfloat16):.1f} TFLOPS ({SECONDS}s)")
