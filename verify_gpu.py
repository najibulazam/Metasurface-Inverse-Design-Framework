# verify_gpu.py
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Supported archs:", torch.cuda.get_arch_list())
    x = torch.randn(4000, 4000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Matmul on GPU successful.")