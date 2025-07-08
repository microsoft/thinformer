# ImageNet Classification with T2T-ViT model

This example folder recreates the T2T-ViT ImageNet classification experiments of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) (Section 4.2).

These experiments were carried out using Python 3.12.9, PyTorch 2.8.0.dev20250407+cu128, and an Ubuntu 22.04.5 LTS server with an AMD EPYC 7V13 64-Core Processor, 220 GB RAM, and a single NVIDIA A100 GPU (80 GB memory, CUDA 12.8, driver version 570.124.04).

The settings and implementations for all methods other than Thinformer were provided by the authors of KDEformer (Zandieh et al., 2023), and our experiment code builds on their open-source repository https://github.com/majid-daliri/kdeformer.

## Prerequisites

1. Download the ILSVRC2012 validation dataset from https://www.image-net.org/download.php. You will need to login and submit the terms of access. The total size is roughly 6.3 GB.

2. Extract validation data using [extract_ILSVRC.sh](extract_ILSVRC.sh).

3. Download the pretrained T2T-ViT model from the [T2T-ViT repo](https://github.com/yitu-opensource/T2T-ViT/releases). We use the ``82.6_T2T_ViTt_24`` model which can be downloaded by running the following command:
```sh
mkdir -p checkpoints
wget -P checkpoints https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
```

## Dependencies

### Prepare conda environment with dependencies, including Scatterbrain

To install Scatterbrain with GPU support, we found it important to ensure that the GPU driver CUDA version (reported by `nvidia-smi`),
the compiler CUDA version (reported by `nvcc --version`), and PyTorch CUDA version (`python -c "import torch; print(torch.version.cuda)"`) all matched.

```bash
# Create environment with nvcc, cudart, and cuda-toolkit for scatterbrain
yes | conda create -n scatter python=3.12 cuda-nvcc cuda-cudart cuda-toolkit pip -c nvidia
# Activate environment
conda activate scatter
# Install torch version that matches local cuda version (12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Install scatterbrain
git clone https://github.com/idiap/fast-transformers.git
# update this line (https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/setup.py#L81) for Nvidia Ampere GPUs
sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_80"]/' fast-transformers/setup.py
pip install fast-transformers/
# Install Python dependencies of T2T-ViT
pip install "timm==0.3.4" pyyaml
# Install Python dependencies of run_imagenet
pip install einops lightning lightning-bolts
# Install other needed Python packages used by experiment
pip install numpy matplotlib pandas tabulate
# Replace outdated helpers file in installed timm package
cp helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
# Install thinformer
pip install git+https://github.com/microsoft/thinformer.git
```

> \[!TIP\]
> On Nvidia Hopper GPUs (e.g., H100), `sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_80"]/' fast-transformers/setup.py` should be replaced by `sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_90"]/' fast-transformers/setup.py`.

You can find the export of this environment in [environment-scatter.yml](environment-scatter.yml).

### Prepare conda environment with dependencies, excluding Scatterbrain

```bash
# Create conda environment
conda create -n thinformer python=3.12
conda activate thinformer
# Install Python dependencies of T2T-ViT
pip install "timm==0.3.4" pyyaml
# Install Python dependencies of the imagenet.py moddule
pip install einops lightning lightning-bolts
# Install other needed Python packages
pip install numpy matplotlib pandas tabulate
# Replace outdated helpers file in installed timm package
cp helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
# Install thinformer
pip install git+https://github.com/microsoft/thinformer.git
```

You can find the export of this environment in [environment.yml](environment.yml).

## Results

To obtain the accuracy numbers in Table 2 for all methods, please run:

```bash
./slurm/accuracy.slurm /PATH/TO/IMAGENET OUTPUT_PATH
```

To obtain the runtime numbers in Table 2 for all methods, please run:

```bash
./slurm/runtime.slurm /PATH/TO/IMAGENET OUTPUT_PATH
```
