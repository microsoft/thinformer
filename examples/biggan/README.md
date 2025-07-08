# Image Generation with BigGAN Model

This example folder recreates the BigGAN image generation experiment of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) (Section 4.2).

These experiments were carried out using Python 3.12.9, PyTorch 2.6.0, and an Ubuntu 22.04.5 LTS server with an Intel(R) Xeon(R) Gold 5218 CPU Processor, 100 GB RAM, and a single NVIDIA A6000 GPU (48 GB memory, CUDA 12.1, driver version 530.30.02). See [environment.yml](./environment.yml) for our reference conda environment.

The settings and implementations for all methods other than Thinformer were provided by the authors of KDEformer (Zandieh et al., 2023), and our experiment code builds on their open-source repository https://github.com/majid-daliri/kdeformer.

## Prerequisites

We provide the Imagenet Validation statistics at [imagenet_val_inception_moments.npz](./imagenet_val_inception_moments.npz), so there is no need to download Imagenet for the BigGAN experiment. Similarly, model checkpoints will be automatically downloaded.

## Dependencies

Please follow the [t2t dependency instructions](../t2t/README.md#dependencies) to prepare a conda environment with needed dependencies.

## Results

Please follow the steps below to recreate the BigGAN experiment:

1. Compute FID and IS scores:

```bash
./slurm/generate.slurm
```

> \[!TIP\]
> The FID and IS scores are outputed to the console and to `fid_score_results.txt`.

2. Compute and format runtimes:
```bash
./slurm/runtime.slurm OUTPUT_PATH
python plot_times.py -op OUTPUT_PATH
```
