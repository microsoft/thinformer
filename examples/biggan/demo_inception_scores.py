import torchvision.transforms as transforms
import utils
from torch.utils.data import DataLoader
import torchvision
import inception_utils

# loader = utils.get_data_loaders('I128', data_root="/home/ih244/workspace/data")
# imgs, labels = next(iter(loader[0]))
image_size = 128
batch_size = 64
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
shuffle = False
transform = transforms.Compose([
    utils.CenterCropLongEdge(),
    transforms.Resize(image_size),
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std)
])
data = torchvision.datasets.ImageNet("/home/ih244/workspace/data/imagenet/", transform=transform, split='val')
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