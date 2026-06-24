"""
Installs all required packages for BioInfo Finals project.
Run this once: python scripts/install_deps.py

Install order matters:
  1. PyTorch with CUDA 12.4 (RTX 4060 compatible)
  2. torch-geometric and scatter/sparse (must match PyTorch version)
  3. remaining packages
"""
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PYGEOM_TORCH = "2.6.0"
PYGEOM_CUDA = "cu124"

def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"[FAIL] Exit code {result.returncode}")
        sys.exit(result.returncode)
    return result

# 1. PyTorch
run([sys.executable, "-m", "pip", "install",
     "torch==2.5.0", "torchvision==0.20.0",
     "--index-url", TORCH_INDEX])

# 2. torch-geometric (version that matches torch 2.5 + cu124)
run([sys.executable, "-m", "pip", "install",
     "torch-geometric",
     "--index-url", "https://data.pyg.org/whl/torch-2.5.0+cu124.html"])

# 3. torch-scatter and torch-sparse (needed by HGTConv)
for pkg in ["torch-scatter", "torch-sparse"]:
    run([sys.executable, "-m", "pip", "install", pkg,
         "--index-url", f"https://data.pyg.org/whl/torch-2.5.0+cu124.html"])

# 4. Other packages
run([sys.executable, "-m", "pip", "install",
     "scikit-learn", "pandas", "numpy", "pyyaml",
     "matplotlib", "seaborn", "tqdm", "jsonlines",
     "jupyter", "ipykernel"])

# 5. Verify
print("\n=== Verification ===")
import importlib
for pkg in ["torch", "torch_geometric", "sklearn", "pandas", "numpy"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "?")
        print(f"  {pkg}: {v}")
    except ImportError as e:
        print(f"  {pkg}: MISSING ({e})")

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
