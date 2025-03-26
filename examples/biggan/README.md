# BigGAN Experiment

Setup instructions:

```bash
pip install pytorch-fid
```

## Results

Please follow the steps below to recreate the BigGAN experiment: 

1. Compute ImageNet statistics:

```bash
python compute_fid.py -dp DATASET_PATH
```

2. Generate images:

```bash
python compress_bigGAN_script.py --num_classes 1000 --data_per_class 5 --beta 0.5 --attention METHOD
```

3. Compute FID scores:

```bash
python compute_fid.py -dp PATH/TO/IMAGENET/val
```

4. Compute Inception scores:

```bash
python demo_inception_scores.py  # TODO (Albert)
```