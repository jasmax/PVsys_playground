## Requirements

- Windows or Linux with an NVIDIA GPU. The development computer has an RTX 3070 Ti with 8 GB.
- Python 3.10 or later in a conda environment (this project uses `pv_gpu_env`).


## Installation

1. **Make the environment.** In the project folder, run these commands:
   ```
   set SAM2_BUILD_CUDA=1
   conda env create -f environment.yml
   conda activate pv_gpu_env
   pip install -r requirements.txt
   ```
   - **PyTorch:** the CUDA 13.0 build, from pip, not from conda. The conda `pytorch` package did not load cuDNN.
   ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   ``` 
   - **SAM 2 and Depth Pro:** the copies in `ssam2/` and `ml-depth-pro/`, installed in editable mode.
   ```
   git clone https://github.com/apple/ml-depth-pro
   cd ml-depth-pro
   pip install -e .
   ```
   ```
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   ```
