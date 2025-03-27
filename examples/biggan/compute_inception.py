"""Script to compute Inception Score of BigGAN generations.

Adapted from code snippet provided by authors of KDEformer (see `demo_inception_scores.py`)

Example usage:
```bash
python compute_inception.py -dp OUTPUT_PATH -op OUTPUT_PATH --attention METHOD
```
"""

import torchvision.transforms as transforms
# import utils  # NOTE (Albert): code not provided
from torch.utils.data import DataLoader
import torchvision
import inception_utils
from compute_fid import CenterCropLongEdge
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import pandas as pd
from util_experiments import get_base_parser

parser = get_base_parser()
args = parser.parse_args()
dataset_path = args.dataset_path
attention_method = args.attention
output_path = args.output_path

# loader = utils.get_data_loaders('I128', data_root="/home/ih244/workspace/data")
# imgs, labels = next(iter(loader[0]))
image_size = 128
batch_size = 64
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
shuffle = False
transform = transforms.Compose([
    CenterCropLongEdge(),
    transforms.Resize(image_size),
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std)
])
# NOTE: original code
# data = torchvision.datasets.ImageNet("/home/ih244/workspace/data/imagenet/", transform=transform, split='val')
# NOTE (Albert): use ImageFolder to load imagenet dataset
generations_path = os.path.join(dataset_path, 'generations', 'biggan_deep_512', attention_method, "KV_0.5_0.5")
print(f"Loading generations from {generations_path}...")
data = torchvision.datasets.ImageFolder(generations_path, transform=transform)
loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
imgs, labels = next(iter(loader))

net = inception_utils.load_inception_net()
net = net.to('cuda')

# transcribed the function inception_utils.accumulate_inception_activations
pool, logits = [], []
print(f"loader len: {len(loader)}")
with torch.no_grad():
    for i, (images, labels) in tqdm(enumerate(loader)):
        images, lables = images.to('cuda'), labels.to('cuda')
        pool_val, logits_val = net(images.float())
        pool += [pool_val]
        logits += [F.softmax(logits_val, 1)]
pool, logits = torch.cat(pool, 0), torch.cat(logits, 0)

IS_mean, IS_std = inception_utils.calculate_inception_score(logits.cpu().numpy(), num_splits=10)
print(f"IS_mean: {IS_mean}, IS_std: {IS_std}")

# NOTE (Albert): save IS_mean and IS_std to a csv file
df = pd.DataFrame(data={
    'method': [attention_method],
    'IS_mean': [IS_mean], 
    'IS_std': [IS_std],
    'score': ['inception']
})
save_path = os.path.join(output_path, 'scores', f'inception-{attention_method}.csv')
print(f"Saving Inception Scores to {save_path}...")
df.to_csv(save_path, index=False)
