# BigGAN Experiment

Setup instructions:

```bash
pip install pytorch-fid
```

## Results

Please follow the steps below to recreate the BigGAN experiment: 

1. Generate images:

```bash
python compress_bigGAN_script.py --num_classes 1000 --data_per_class 5 --beta 0.5 --attention METHOD --path OUTPUT_PATH/generations/
```

2. Compute FID scores:

> \[!NOTE\]
> This following script expects a file containing the ImageNet statistics at `OUTPUT_PATH/scores/imagenet_statistics.npz`. If this file doesn't exist, it will be created.

```bash
python compute_fid.py -dp PATH/TO/IMAGENET/val -op OUTPUT_PATH --method METHOD
```

3. Compute Inception scores:

```bash
python demo_inception_scores.py  # TODO (Albert)
```

4. Compute runtimes:
```bash
python runtime.py --attention=METHOD
```
Optional: `--num_runs=RUNS` to get +- std on timings.