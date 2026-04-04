#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode
from common import (train_without_cuda_graph, setup_model_and_data,
                    create_model, create_profiler, save_and_print_profile)

import weakref
import time
import ctypes
import queue
import threading
from cpu_adm import DeepSpeedCPUAdam as ds_opt
tensor_queue=queue.Queue()

my_lib = ctypes.CDLL("./prefetch_async.so")
my_lib.prefetch_memory.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int]
my_lib.prefetch_memory.restype = ctypes.c_int
my_lib.pin_memory_hint.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int]
my_lib.pin_memory_hint.restype = ctypes.c_int
#my_lib.prefetch_memory_batch.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
#my_lib.prefetch_memory_batch.restype = ctypes.c_int
my_lib.cuda_malloc.argtypes=[ctypes.c_ulong]
my_lib.cuda_malloc.restype=ctypes.c_int
try:
    my_lib.print_first_byte.restype = ctypes.c_int
except Exception:
    pass



def prefetch_tensor_if_large(tensor, stream_idx=1, threshold_bytes=2 * 1024 * 1024):
    if tensor is None or not torch.is_tensor(tensor):
        return
    size_in_bytes = tensor.nelement() * tensor.element_size()
    _prefetch_tensor_bytes(tensor.data_ptr(), size_in_bytes, stream_idx)


def _prefetch_tensor_bytes(data_ptr, size_in_bytes, stream_idx):
    if size_in_bytes <= 0:
        return
    my_lib.prefetch_memory(data_ptr, size_in_bytes, 1)


def prefetch_worker():
    global tensor_queue
    stream = torch.cuda.Stream()

    while not tensor_queue.empty():

        t = tensor_queue.get()

        if t is None:
            break

        with torch.cuda.stream(stream):

            prefetch_tensor_if_large(t())
        
        tensor_queue.task_done()


class TensorAddressLogger(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):

        global tensor_queue
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

        for t in tensors:
            tensor_queue.put(weakref.ref(t))

        return func(*args, **kwargs)




def prepare_cuda_graph(model, loss_fn, optimizer, static_input, static_target):
    """Warmup and capture CUDA graph (not profiled)."""

    print("  Performing warmup iterations...")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for i in range(3):
            with torch.cuda.amp.autocast(True):
                optimizer.zero_grad(set_to_none=True)

                y_pred = model(static_input,labels=static_target)
                loss = y_pred.loss

                loss.backward()
                optimizer.step()

    torch.cuda.current_stream().wait_stream(s)

    print("  Capturing CUDA graph...")

    g = torch.cuda.CUDAGraph()

    optimizer.zero_grad(set_to_none=True)

    with TensorAddressLogger():
        with torch.cuda.graph(g):
            with torch.cuda.amp.autocast(True):
                static_y_pred = model(static_input,labels=static_target)
                static_loss = static_y_pred.loss

                static_loss.backward()
                optimizer.step()

    return g, static_loss


def train_with_cuda_graph(graph,
                          data_loader,
                          static_input,
                          static_target,
                          static_loss,
                          profiler=None):

    print("  Training with graph replay...")

    for i, data in enumerate(data_loader):

        #with record_function("## copy_input_data ##"):
        static_input.copy_(data["input_ids"])
        static_target.copy_(data["labels"])

            #with record_function("## graph.replay ##"):
        threading.Thread(target=prefetch_worker, daemon=True).start()        
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        end = time.perf_counter()

        print("Duration:", end - start, "seconds")

        if profiler is not None:
            profiler.step()
        if i==4:
            break
        print(i)

    print(f"  Completed  iterations.")
    print()


def main():

    print("CUDA Graph Whole Network Capture Example")
    print("=" * 70)

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. This example requires a CUDA-capable GPU."
        )
        return

    device = torch.device('cuda')

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print()

    trace_dir = "traces"

    data_loader = setup_model_and_data(device)

    static_input = torch.empty((4,2000), device='cuda', dtype=torch.int32)
    static_target = torch.empty((4, 2000), device='cuda', dtype=torch.long)

    # =====================================================
    # WITHOUT CUDA GRAPH
    # =====================================================

    print("=" * 70)
    print("SCENARIO 1: Training WITHOUT CUDA Graph")
    print("=" * 70)

    '''model_no_graph = create_model(None, device)
    loss_fn_no_graph = torch.nn.CrossEntropyLoss()
    optimizer_no_graph = torch.optim.SGD(model_no_graph.parameters(), lr=0.1)'''

    #with create_profiler() as prof_no_graph:

    '''train_without_cuda_graph(model_no_graph,
                                 loss_fn_no_graph,
                                 optimizer_no_graph,
                                 data_loader)'''

    # =====================================================
    # WITH CUDA GRAPH
    # =====================================================

    print("=" * 70)
    print("SCENARIO 2: Training WITH CUDA Graph")
    print("=" * 70)

    model_with_graph = create_model(None, device)
    loss_fn_with_graph = torch.nn.CrossEntropyLoss()
    optimizer_with_graph = ds_opt(model_with_graph.parameters(), lr=1e-5)

    print("Preparing CUDA graph (warmup + capture)...")

    graph, static_loss = prepare_cuda_graph(model_with_graph,
                                            loss_fn_with_graph,
                                            optimizer_with_graph, static_input,
                                            static_target)

    print("CUDA graph ready.")
    print()
    train_with_cuda_graph(graph,
                              data_loader,
                              static_input,
                              static_target,
                              static_loss)

    trace_file_with_graph = trace_dir + "/" + "trace_with_manual_capture.json"

    print("=" * 70)


if __name__ == "__main__":
    torch._C._cuda_beginUvmAllocate()
    main()
    torch._C._cuda_endUvmAllocate()


