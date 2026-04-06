import torch
from torch.utils._python_dispatch import TorchDispatchMode

class LoggingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. Logic before kernel launch
        print(f"Intercepting: {func.__module__}.{func.__name__}")
        
        # You can inspect or modify args here
        # Example: if func == torch.ops.aten.addmm.default:
        #    print("Injecting logic specifically for matrix multiplication")

        # 2. Proceed to actual execution (or return custom result)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

# Usage
x = torch.randn(2, 2, device='cuda')
y = torch.randn(2, 2, device='cuda')
mat = torch.randn(2, 2, device='cuda')

with LoggingMode():
    # This will trigger intercepts for 'aten::addmm'
    result = torch.addmm(mat, x, y)
