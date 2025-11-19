import torch
print("cuda available:", torch.cuda.is_available())
# Depends on NVIDIA drivers and CUDA toolkit installation
# If false, your either lacking the hardware or PyTorch is installed wrong
