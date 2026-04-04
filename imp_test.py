import torch

x = torch.tensor([3.0], requires_grad=True)

y = x**3
loss = y.sum()

print("Before change:", x.data_ptr())
clone=torch.tensor([100.0])
print("clone ",clone.data_ptr())
with torch.no_grad():
    x.data=clone

print("After change:", x.data_ptr())
print("Grad fn:", y.grad_fn)
print("Next functions:", y.grad_fn.next_functions)
x.detach()
clone.detach()
loss.backward()

print("Gradient:", x.grad)
