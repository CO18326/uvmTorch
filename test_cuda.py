import torch
import time
import nvtx

# ----------------------------
# Configuration
# ----------------------------
ITERATIONS = 20
N = 1 << 24        # ~64MB float32
CPU_SLEEP_MS = 20

assert torch.cuda.is_available(), "CUDA not available"

device = torch.device("cuda")

# ----------------------------
# Allocate memory
# ----------------------------
# Pinned host memory → enables async memcpy
h_tensor = torch.empty(N, dtype=torch.float32, pin_memory=True)
h_tensor.fill_(1.0)

# Device tensor
d_tensor = torch.empty(N, device=device)

# CUDA stream
stream = torch.cuda.Stream()

# ----------------------------
# CPU compute (busy wait)
# ----------------------------
def cpu_compute(ms):
    nvtx.range_push("CPU Compute")
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        pass
    nvtx.range_pop()

# ----------------------------
# GPU compute workload
# ----------------------------
def gpu_compute(t):
    nvtx.range_push("GPU Compute")
    for _ in range(10):
        t = t * 1.000001 + 0.000001
    nvtx.range_pop()
    return t

# ----------------------------
# Main loop
# ----------------------------
for i in range(ITERATIONS):
    nvtx.range_push(f"Iteration {i}")

    # CPU work (overlaps GPU)
    cpu_compute(CPU_SLEEP_MS)

    with torch.cuda.stream(stream):
        # HtoD memcpy
        nvtx.range_push("HtoD memcpy")
        d_tensor.copy_(h_tensor, non_blocking=True)
        nvtx.range_pop()

        # GPU compute
        d_tensor = gpu_compute(d_tensor)

        # DtoH memcpy
        nvtx.range_push("DtoH memcpy")
        h_tensor.copy_(d_tensor, non_blocking=True)
        nvtx.range_pop()

    nvtx.range_pop()

# Ensure all work is done
torch.cuda.synchronize()

print("Done.")
