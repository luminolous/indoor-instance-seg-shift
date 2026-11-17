# indoor-instance-seg-shift

## Setup (Windows + CUDA)

Tested target: Windows 11, NVIDIA RTX 4050 Laptop GPU (6 GB), Python 3.12, [uv](https://docs.astral.sh/uv/).

### 1. Check the NVIDIA driver

```powershell
nvidia-smi
```

The header must show a driver that supports CUDA 12.6 or newer (driver >= 560). The PyTorch
wheels bundle their own CUDA runtime, so a separate CUDA Toolkit install is **not** needed.

### 2. Create the virtual environment

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Install PyTorch (CUDA build) first

PyPI only ships CPU wheels of PyTorch for Windows, so install torch from the PyTorch index
**before** the other dependencies:

```powershell
uv pip install "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu126
```

`cu126` works with any driver >= 560. With a newer driver you may use `cu130` instead.

### 4. Install the project and the remaining dependencies

```powershell
uv pip install -e ".[dev]"
```

Optional Weights & Biases logging: `uv pip install -e ".[dev,wandb]"`.

Without `uv`, replace `uv pip` with `python -m pip` in the commands above.

### 5. Verify

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: the version ends with `+cu126`, followed by `True` and the GPU name. If it prints
`False` or a `+cpu` version, a later install replaced torch with the CPU wheel. Re-run step 3.

### Notes

- `pycocotools` ships prebuilt Windows wheels; no Visual C++ build tools are required.
- detectron2 is intentionally not used (difficult to install on native Windows). Mask R-CNN
  comes from `torchvision`, Mask2Former from Hugging Face `transformers`.
- Runs that do not fit in 6 GB VRAM fall back to Google Colab.
