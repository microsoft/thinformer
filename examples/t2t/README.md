# ImageNet Classification with T2T-ViT model

This example folder recreates the T2T-ViT ImageNet classification experiments of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) (Section 4.2).

These experiments were carried out using Python 3.12.5, PyTorch 2.6.0, and an Ubuntu 20.04 server with 8 CPU cores (Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz), 100 GB RAM, and a single Nvidia A6000 GPU (48 GB memory, CUDA 12.1, driver version 530.30.02).

The settings and implementations for all methods other than Thinformer were provided by the authors of KDEformer (Zandieh et al., 2023), and our experiment code builds on their open-source repository https://github.com/majid-daliri/kdeformer.

## Prerequisites

1. Download the ILSVRC2012 validation dataset from https://www.image-net.org/download.php. You will need to login and submit the terms of access. The total size is roughly 6.3 GB.

2. Extract validation data using (extract_ILSVRC.sh).

3. Download the pretrained T2T-ViT model from the [T2T-ViT repo](https://github.com/yitu-opensource/T2T-ViT/releases). We choose ``82.6_T2T_ViTt_24`` model which can be downloaded by running the following command:
```sh
mkdir -p checkpoints
wget -P checkpoints https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
```

## Dependencies

The following dependences are needed to run the accuracy and runtime scripts:

```bash
# Create conda environment ()
conda create -n thinformer python=3.12
conda activate thinformer
# Install RUST without user interaction for T2T-ViT
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Install Python dependencies of T2T-ViT
pip install "timm==0.3.4" pyyaml
# Install Python dependencies of the imagenet.py moddule
pip install einops lightning lightning-bolts
# Install other needed Python packages
pip install numpy matplotlib pandas tabulate
# Replace outdated helpers file in installed timm package
cp helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
```

> \[!TIP\]
> A copy of our `environment.yml` file is included in this subdirectory to promote reproducilibity of our experiments.

- [ ] @AshettyV: I'm curious if the code runs without installing rust. Can you try that first? If it works, then we can just leave that step out.
- [ ] @AshettyV: We definitely need `timm`, but I'm not sure if the pyyaml dependency is actually needed since it doesn't show up when I do a global Ctrl + F. Can you also leave this out when setting up for the first time?

**Baseline-specific dependencies**: Some baselines require additional setup. See below:

<details>
<summary>ScatterBrain</summary>

1. Ensure that the GPU driver CUDA version, compiler (nvcc) CUDA version, and pytorch CUDA version all match!
- To check GPU driver: `nvidia-smi`
- To check nvcc: `nvcc --version`
- To check pytorch: `conda list | grep pytorch-cuda`

2. Install the `fast-transformers` package from source.
```bash
git clone https://github.com/albertgong1/fast-transformers.git
pip install fast-transformers/
```
</details>

## Results

To obtain the accuracy numbers in Table 2 for all methods, please run:

```bash
./slurm/accuracy.slurm /PATH/TO/IMAGENET OUTPUT_PATH
```

To obtain the runtime numbers in Table 2 for all methods, please run:

```bash
./slurm/runtime.slurm /PATH/TO/IMAGENET OUTPUT_PATH
```
