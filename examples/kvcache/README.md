# Accelerating KV Cache Compression

This example folder recreates the accelerating KV cache compression experiments of [Express Language Modeling](https://arxiv.org/pdf/2606.10944) (Section 4).

These experiments were carried out using Python 3.13.9, PyTorch 2.10.0, and an Ubuntu 24.04.4 LTS server with an Intel Xeon Silver 4316 processor (2 x 20 cores), 1 TB RAM, and a single NVIDIA RTX A6000 GPU (48 GB memory, CUDA 12.8, driver version 570.124.06).

The settings and implementations for all methods other than Thinformer were provided by the authors of Expected Attention (Devoto et al., 2025), and our experiment code builds on their open-source repository https://github.com/NVIDIA/kvpress.

## Dependencies

```bash
conda create -n kvcache python=3.13.9
conda activate kvcache
pip install uv
# install a CUDA 12.8 build of PyTorch 2.10
uv pip install --torch-backend=cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
# install kvpress (editable) with its dev group and eval extra.
cd kvpress
uv pip install --group dev -e ".[eval]"
uv pip install transformers==5.1.0
cd ..
# install thinformer
uv pip install git+https://github.com/microsoft/thinformer.git
```

## Results

1. Launch the SLURM jobs to record timings and accuracy.

```bash
sbatch -p PARTITION slurm/longbench-e.slurm OUTPUT_DIR hotpotqa_e
sbatch -p PARTITION slurm/longbench-e.slurm OUTPUT_DIR trec_e
sbatch -p PARTITION slurm/longbench-e.slurm OUTPUT_DIR triviaqa_e
```

2. Generate Figure 2:

```bash
python format_accuracy_vs_time_3panel.py --model meta-llama/Meta-Llama-3.1-8B-Instruct -od OUTPUT_DIR
```
