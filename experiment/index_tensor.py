import torch
import time
import csv
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from torch.utils.cpp_extension import load
import os

# Adjust if CUDA installed elsewhere
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

cupti_include = f"{CUDA_HOME}/extras/CUPTI/include"
cupti_lib = f"{CUDA_HOME}/extras/CUPTI/lib64"

cupti_ext = load(
    name="cupti_timestamp",
    sources=["cupti_timestamp.cpp"],
    extra_cflags=[
    "-I/usr/local/cuda/include",
    "-I/usr/local/cuda/extras/CUPTI/include",
    ],
    extra_ldflags=[
    "-L/usr/local/cuda/lib64",
    "-L/usr/local/cuda/extras/CUPTI/lib64",
    "-lcupti"
    ],
    verbose=True)


import torch
import csv
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

class DataInspectorMode(TorchDispatchMode):
    def __init__(self, base_name="tensor_log_3000_4_optimiser_offload"):
        self.base_name = base_name
        self.current_step = None
        self.csv_file = None
    
    def get_time_ns(self):
        return time.time_ns()

    def set_step(self, step):
        if step != self.current_step:
            self.current_step = step
            self.csv_file = f"{self.base_name}.csv"

            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_ns",
                    "kernel_name",
                    "tensor_id",
                    "tensor_address",
                    "tensor_shape",
                    "tensor_size_bytes"
                ])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        timestamp = self.get_time_ns()
        kernel_name = str(func)

        rows = []

        def inspect(x):
            if isinstance(x, torch.Tensor) and x.is_cuda:
                size_bytes = x.numel() * x.element_size()

                rows.append([
                    timestamp,
                    kernel_name,
                    id(x),
                    hex(x.data_ptr()),
                    tuple(x.shape),
                    size_bytes
                ])
            return x

        tree_map(inspect, args)
        if kwargs:
            tree_map(inspect, kwargs)

        if rows and self.csv_file:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        return func(*args, **(kwargs or {}))
