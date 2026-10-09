# Memory- and compute-constrained long-form decoding

This example folder recreates the memory- and compute-constrained long-form decoding experiments of [Express Language Modeling](https://arxiv.org/pdf/2606.10944) (Section 4).

These experiments were carried out using Python 3.13.9, PyTorch 2.10.0, and an Ubuntu 24.04.4 LTS server with an Intel Xeon Silver 4316 processor (2 x 20 cores), 1 TB RAM, and a single NVIDIA RTX A6000 GPU (48 GB memory, CUDA 12.8, driver version 570.124.06).

The settings and implementations for all methods other than Thinformer were provided by the authors of Expected Attention (Devoto et al., 2025), and our experiment code builds on their open-source repository https://github.com/NVIDIA/kvpress.

## Dependencies

```bash
conda create -n decoding3 python=3.13.9
conda activate decoding3
pip install uv
# Install a CUDA 12.8 build of PyTorch 2.10
uv pip install --torch-backend=cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
# Install kvpress (editable) with its dev group and eval extra.
cd kvpress
uv pip install --group dev -e ".[eval]"
uv pip install transformers==4.57.1
cd ..
# Overlay the patched files onto stock transformers 4.57.1
# (transformers==4.57.1 is already installed by the command above).
cp modeling_llama.py "$(python -c 'import transformers.models.llama.modeling_llama as m; print(m.__file__)')"
cp masking_utils.py  "$(python -c 'import transformers.masking_utils as m; print(m.__file__)')"
# Install thinformer
uv pip install git+https://github.com/microsoft/thinformer.git
```

## Results

The launch scripts below submit SLURM job arrays. Cluster-specific settings are read from environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `SLURM_PARTITION` | Partition to submit to | cluster default |
| `SLURM_GRES` | GPU request | `gpu:nvidia_rtx_a6000:1` |
| `SLURM_EXCLUDE` | Comma-separated nodes to exclude | none |

1. Launch jobs to obtain accuracy-vs.-cache size results:

```bash
./launch-math500.sh OUTPUT_DIR
```

2. Generate Figure 4a (accuracy vs. cache size):

```bash
python format_accuracy_math500.py -od OUTPUT_DIR
```

3. Launch jobs to record timings and accuracy on all 500 problems:

```bash
./launch-math500-time.sh OUTPUT_DIR
```

4. Generate Figure 4b (accuracy vs. decoding time):

```bash
python format_accuracy_vs_time_math500_scatter.py -od OUTPUT_DIR
```
