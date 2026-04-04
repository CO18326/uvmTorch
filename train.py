import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._python_dispatch import TorchDispatchMode

# -----------------------------
# Dispatcher hook
# -----------------------------

class TensorAccessLogger(TorchDispatchMode):

    def __init__(self):
        self.op_id = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        tensors = []

        def extract(x):
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, (list, tuple)):
                for i in x:
                    extract(i)
            elif isinstance(x, dict):
                for i in x.values():
                    extract(i)

        extract(args)
        extract(kwargs)

        info = []
        for t in tensors:
            info.append(
                f"addr=0x{t.data_ptr():x},shape={tuple(t.shape)},device={t.device}"
            )

        print(
            f"OP {self.op_id} | {func.__name__} | " +
            " | ".join(info)
        )

        self.op_id += 1

        return func(*args, **kwargs)


# -----------------------------
# Simple model
# -----------------------------

class SimpleModel(nn.Module):

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


# -----------------------------
# Setup
# -----------------------------

device = "cuda"

model = SimpleModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

batch = 32

static_input = torch.randn(batch, 1024, device=device)
static_target = torch.randint(0, 10, (batch,), device=device)

# -----------------------------
# Warmup (important for CUDA graphs)
# -----------------------------

for _ in range(3):

    optimizer.zero_grad(set_to_none=True)

    with TensorAccessLogger():
        out = model(static_input)
        loss = criterion(out, static_target)
        loss.backward()
        optimizer.step()

torch.cuda.synchronize()

# -----------------------------
# CUDA Graph Capture
# -----------------------------

g = torch.cuda.CUDAGraph()

optimizer.zero_grad(set_to_none=True)

with torch.cuda.graph(g):

    with TensorAccessLogger():
        out = model(static_input)
        loss = criterion(out, static_target)
        loss.backward()
        optimizer.step()

torch.cuda.synchronize()

print("\n==== GRAPH REPLAY ====\n")

# -----------------------------
# Replay loop
# -----------------------------

for i in range(2):

    new_input = torch.randn_like(static_input)
    new_target = torch.randint_like(static_target, 10)

    static_input.copy_(new_input)
    static_target.copy_(new_target)

    g.replay()

torch.cuda.synchronize()
