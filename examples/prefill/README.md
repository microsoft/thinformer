# Accelerating Prefill

This example folder recreates the accelerating prefill experiments of [Express Language Modeling](https://arxiv.org/pdf/2606.10944) (Section 4).

These experiments were carried out using Python 3.12.11, PyTorch 2.8.0, and an Ubuntu 24.04.4 LTS server with an Intel Xeon Silver 4316 processor (2 x 20 cores), 1 TB RAM, and a single NVIDIA RTX A6000 GPU (48 GB memory, CUDA 12.8, driver version 570.124.06).

The settings and implementations for all methods other than Thinformer were provided by the authors of KDEformer (Zandieh et al., 2023), and our experiment code builds on their open-source repository https://github.com/insuhan/hyper-attn.

## Dependencies

```bash
conda create -n prefill python=3.12
conda activate prefill
# install a CUDA 12.8 build of PyTorch 2.8 and a matching prebuilt FlashAttention wheel (cu12 / torch2.8 / cxx11abiTRUE / cp312)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install transformers==4.56.1 einops
# NOTE: `datasets<4.0.0` is needed since LongBench uses a script data loader.
pip install tqdm datasets==3.6.0 sentencepiece
# install formatting dependencies
pip install matplotlib scipy python-slugify tabulate
# install thinformer
pip install git+https://github.com/microsoft/thinformer.git
```

### Alternative Python 3.13 / transformers 5.17.0 setup (used for the Chameleon error bounds)

The prebuilt FlashAttention wheels are the binding constraint: the latest release with a full wheel matrix (v2.8.3.post1) ships cu12 wheels for torch 2.4-2.8 and Python 3.9-3.13 only, so torch 2.8.0 is the newest version that works with CUDA 12.8 and Python 3.13 without compiling FlashAttention from source. 

```bash
conda create -n prefill313 python=3.13
conda activate prefill313
# install a CUDA 12.8 build of PyTorch 2.8 and a matching prebuilt FlashAttention wheel (cu12 / torch2.8 / cxx11abiTRUE / cp313)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
pip install transformers==5.17.0 accelerate einops
# NOTE: `datasets<4.0.0` is needed since LongBench uses a script data loader.
pip install tqdm datasets==3.6.0 sentencepiece
# install formatting dependencies
pip install matplotlib scipy python-slugify tabulate
# install thinformer
pip install -e ../..
```

NOTE: `datasets==3.6.0` predates `huggingface-hub` 1.x (required by `transformers` 5.x). pip resolves the pair, but runtime compatibility is untested upstream.

## Results

### Reproducing Figure 1a: Masked prefill

1. Launch job array for timings:

```bash
sbatch --partition=PARTITION slurm/benchmark_single_attention_array.slurm OUTPUT_DIR 0
```

2. Plot speedup over Torch SDPA (FlashAttention 2) for context lengths 32k-512k when timing out-of-network:

```bash
python format_single_attention.py -op OUTPUT_DIR -bs 1 -hs 32 -d 128 -s 1 --speedup --causal 1
```

### Reproducing Figure 1b: Masked prefill ($n=32768$)

1. Launch job array for quality:

```bash
sbatch --partition=PARTITION slurm/benchmark_patch_llm_array.slurm OUTPUT_DIR
```

2. Launch job array for timings (`1` = in-network, `32` = sink/window size N_FIRST=N_LAST=32):

```bash
sbatch --partition=PARTITION slurm/benchmark_single_attention_array.slurm OUTPUT_DIR 1 32
```

3. Plot perplexity and speed-up over exact attention vs. number of layers replaced:

```bash
python plot_combined.py -op OUTPUT_DIR
```

4. Plot the quality/cost trade-off (speed-up and runtime vs. relative and inverse
   perplexity, one figure each). Requires both job arrays above, since it joins the
   two result tables on (method, layers replaced, `h`):

```bash
python plot_tradeoff.py -op OUTPUT_DIR
```

### Reproducing Figure 1c: Attention approximation error

1. Generate attention outputs for exact and Thinformer Express attention:
   
```bash
python benchmark_single_attention.py -m exact -bs 1 -hs 32 -d 128 -s 1 -op out-test --force \
    --num_runs 1 --num_warmup 0 --seq_len 524288 --input_source glm_embed --save_outputs
for h in 7 8 9 10; do
python benchmark_single_attention.py -m thinformer_triton --log2_target_cache_size $h -bs 1 -hs 32 -d 128 -s 1 \
    -op out-test --force --num_runs 1 --num_warmup 0 --seq_len 524288 --input_source glm_embed --save_outputs
done
```

2. Plot attention approximation error:

```bash
python compute_maxnorm_truncate.py --out maxnorm_trunc_sweep_glm_embed.pdf --plot_stride 1 \
    --data_dir Out_data_sweep_glm_embed
# restrict to a subset of h, and thin the plotted points to keep the PDF small
python compute_maxnorm_truncate.py --data_dir Out_data_sweep_glm_embed --h 8 9 10 --plot_stride 1024
```

### Reproducing Figure 1d: Unmasked prefill

1. Launch job array:

```bash
sbatch --partition=PARTITION slurm/benchmark_single_attention_no_causal_array.slurm OUTPUT_DIR
```

4. Plot in-network runtime speedup over Torch SDPA (FlashAttention 2) for context lengths 1k-32k:

```bash
python format_single_attention.py -op OUTPUT_DIR -bs 1 -hs 32 -d 128 -s 1 --in_network --speedup --causal 0
```

### Reproducing Chameleon error bound factors

1. Record the query, key, and value token embeddings from Chameleon:

```bash
# first sequence with >= 32768 Chameleon tokens, layer 0 (use a list like 0,5,31 or 'all' for more layers)
python benchmark_patch_llm.py --model_name facebook/chameleon-7b --attn_method flash \
    --seq_len 32768 -od out-cham --record_qkv_layer 0 --max_seqs 1
# 65536 tokens: chunk the position-wise MLPs to fit a 24 GB GPU
python benchmark_patch_llm.py --model_name facebook/chameleon-7b --attn_method flash \
    --seq_len 65536 -od out-cham --record_qkv_layer 0 --max_seqs 1 --mlp_chunk 8192
```

2. Compute the rho_n and conservative upper bound error factors reported in Sec. 3.2:
   
```bash
python plot_bounds.py -od out-cham --pattern "states__model=facebook-chameleon-7b_layer=0_*bid=0.pt" \
    --out cham_l0/bounds.pdf --center_keys --center_values
```


