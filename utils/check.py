"""
Small script for ensuring that PyTorch is detecting the GPU correctly."
This is functionallity not necessary, it's just here to double check in case training is taking unexpectedly long.
"""

import torch
print("cuda available:", torch.cuda.is_available())
# Depends on NVIDIA drivers and CUDA toolkit installation
# If false, your either lacking the hardware or PyTorch is installed wrong
