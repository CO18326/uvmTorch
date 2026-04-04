import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._python_dispatch import TorchDispatchMode

# -----------------------------
# Dispatcher logger
# -----------------------------

class AccessLogger(TorchDispatchMode):

    def __init__(self, log_list):
        self.log = log_list

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        tensors = []

        def collect(x):
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, (list, tuple)):
                for i in x:
                    collect(i)
            elif isinstance(x, dict):
                for i in x.values():
                    collect(i)

        collect(args)
        collect(kwargs)

        entry = [func.__name__]

        for t in tensors:
            entry.append((t.data_ptr(), tuple(t.shape)))

        self.log.append(entry)

        return func(*args, **kwargs)


# -----------------------------
# Simple model
# -----------------------------

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)


device = "cuda"

model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

batch = 32

static_input = torch.randn(batch, 1024, device=device)
static_target = torch.randint(0, 10, (batch,), device=device)

# -----------------------------
# STEP 0 (normal execution)
# -----------------------------

log_step0 = []

optimizer.zero_grad(set_to_none=True)

with AccessLogger(log_step0):

    out = model(static_input)
    loss = criterion(out, static_target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

print("Logged ops step0:", len(log_step0))


# -----------------------------
# Warmup
# -----------------------------

for _ in range(3):

    optimizer.zero_grad(set_to_none=True)

    out = model(static_input)
    loss = criterion(out, static_target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()


# -----------------------------
# CUDA graph capture
# -----------------------------

g = torch.cuda.CUDAGraph()

optimizer.zero_grad(set_to_none=True)

with torch.cuda.graph(g):

    out = model(static_input)
    loss = criterion(out, static_target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()


# -----------------------------
# Replay
# -----------------------------

g.replay()

torch.cuda.synchronize()


# -----------------------------
# STEP 1 (normal execution again)
# -----------------------------

log_step1 = []

optimizer.zero_grad(set_to_none=True)

with AccessLogger(log_step1):

    out = model(static_input)
    loss = criterion(out, static_target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

print("Logged ops step1:", len(log_step1))


# -----------------------------
# Compare patterns
# -----------------------------

same = True

for a, b in zip(log_step0, log_step1):

    if a != b:
        same = False
        print("Mismatch:", a, b)
        break

print("Access pattern identical:", same)
