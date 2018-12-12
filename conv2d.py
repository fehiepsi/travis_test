import time
import torch


torch.manual_seed(0)
x = torch.randn(1000, 1, 28, 28)
conv = torch.nn.Conv2d(1, 10, kernel_size=5)
start = time.time()
o = conv(x)
print(time.time() - start)
print(o.sum().item())

