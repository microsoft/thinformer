import os
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as TF
from torchvision.models.inception import inception_v3
from tqdm import tqdm
from scipy.stats import entropy

    
# From https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
def inception_score(imgs, cuda=True, batch_size=128, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)





from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.image.fid import _compute_fid
from torchmetrics.utilities.data import dim_zero_cat


def get_inception_features(imgs, dim=2048, batch_size=128):
    assert imgs.dtype == torch.uint8
    model = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(dim)]).to('cuda')
    model.eval()
    with torch.no_grad():
        features = []
        num_batches = len(imgs) // batch_size + 1
        for idx in tqdm(range(num_batches)):
            batch_idx = list(range(idx * batch_size, min(len(imgs), (idx+1) * batch_size)))
            if len(batch_idx) == 0:
                continue

            # Generate an image
            output = model(imgs[batch_idx].to('cuda')).to('cpu')
            features.append(output)
        return torch.cat(features)


def compute_fid(real_features, fake_features):
    real_features = dim_zero_cat(real_features)
    fake_features = dim_zero_cat(fake_features)
    # computation is extremely sensitive so it needs to happen in double precision
    orig_dtype = real_features.dtype
    real_features = real_features.double()
    fake_features = fake_features.double()

    # calculate mean and covariance
    n = real_features.shape[0]
    m = fake_features.shape[0]
    mean1 = real_features.mean(dim=0)
    mean2 = fake_features.mean(dim=0)
    diff1 = real_features - mean1
    diff2 = fake_features - mean2
    cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
    cov2 = 1.0 / (m - 1) * diff2.t().mm(diff2)

    # compute fid
    return _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)


def compute_inception(features, splits=10):
    features = dim_zero_cat(features)
    # random permute the features
    idx = torch.randperm(features.shape[0])
    features = features[idx]

    # calculate probs and logits
    prob = features.softmax(dim=1)
    log_prob = features.log_softmax(dim=1)

    # split into groups
    prob = prob.chunk(splits, dim=0)
    log_prob = log_prob.chunk(splits, dim=0)

    # calculate score per split
    mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
    kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
    kl = torch.stack(kl_)

    # return mean and std
    return kl.mean(), kl.std()


def test2():
    from PIL import Image
    import torchvision.transforms as TF

    path = "/home/ih244/workspace/data/imagenet/val/"
    files = []
    for root, dirs, files_ in os.walk(path, topdown=True):
        if root == path:
            continue
        cnt = 0
        for name in files_:
            files.append(os.path.join(root, name))
            cnt += 1
    files = sorted(files)
    print(len(files))
    transform = TF.Compose([
        TF.Resize(256),
        TF.CenterCrop(224),
        TF.ToTensor(),
    ])

    import sys
    sys.path.append("/home/ih244/workspace/kde/BigGAN-PyTorch")
    import inception_utils
    net = inception_utils.load_inception_net(parallel=True)
    import pdb; pdb.set_trace();

    from tqdm import tqdm
    # aa = transform(Image.open(files[0]).convert('RGB'))
    # imgs = []
    # cnt = 0
    # for fi in tqdm(files):
    #     imgs.append((transform(Image.open(fi).convert('RGB')) * 255).type(torch.uint8).unsqueeze(0))
    #     cnt += 1
    #     # if cnt > 10:
    #         # break
    # imgs = torch.cat(imgs, 0)
    real_imgs = torch.load("./valid_imgs_uint8.pth")
    real_features = get_inception_features(real_imgs)
    print(real_features)
    import pdb; pdb.set_trace();
    
    fake_imgs = torch.load("tmp.pth")

    quit(-1)
    fake_features = get_inception_features(fake_imgs)

    print(compute_fid(real_features, fake_features))
    print(compute_inception(real_features))
    print(compute_inception(fake_features))

    import pdb; pdb.set_trace();


    # from torchmetrics.image.fid import FrechetInceptionDistance
    # fid = FrechetInceptionDistance(feature=2048)
    from torchmetrics.image.inception import InceptionScore
    fid = InceptionScore(feature=2048)
    # features = fid.inception(imgs)
    fid.inception = fid.inception.to('cuda')
    batch_size = 128

    features = []
    num_batches = len(imgs) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(imgs), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue

        # Generate an image
        output = fid.inception(imgs[batch_idx].to('cuda')).to('cpu')
        features.append(output)
    features = torch.cat(features)
    del imgs

    # fid.real_features.append(features)
    fid.features.append(features)
    print(fid.features[0].shape)
    print(fid.compute())

    import pdb; pdb.set_trace();

    fid2 = InceptionScore(feature=2048)
    fid2.update(fake_imgs)
    fid2.compute()
    
    features2 = []
    num_batches = len(fake_imgs) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(fake_imgs), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue

        # Generate an image
        output = fid.inception(fake_imgs[batch_idx].to('cuda')).to('cpu')
        features2.append(output)
    
    features2 = torch.cat(features2)

    # del fake_imgs

    # fid.fake_features.append(features2)
    fid.features = [features2]
    print(fid.compute())
    # fid.update(imgs, real=True)
    # fid.update(fake_imgs, real=False)
    # print(fid.compute())
    print()

    import pdb; pdb.set_trace();

    print("computing FID score.")
    output_all = (fake_imgs / 255.).float() # range in [0, 1]
    # torch.save(output_all, f"./{attention}_{output_all.shape[0]}.pth")
    # import pdb; pdb.set_trace();

    from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to('cuda')
    inception_model.eval()
    pred = get_activations(output_all, inception_model, batch_size=batch_size, device='cuda')
    m1 = np.mean(pred, axis=0)
    s1 = np.cov(pred, rowvar=False)

    # pre-computed statistics of ImageNet validation set.
    res = torch.load("/home/ih244/workspace/kde/pytorch-pretrained-BigGAN/imagenet_val_stats.pth")
    m2 = res['m']
    s2 = res['s']

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID  : {fid_value}")


    
    import pdb; pdb.set_trace();


def get_logits(imgs, batch_size=128, net=None):
    import sys
    sys.path.append("/home/ih244/workspace/kde/BigGAN-PyTorch")
    import inception_utils

    if net is None:
        net = inception_utils.load_inception_net()
        net = net.to('cuda')

    pool, logits = [], []
    with torch.no_grad():
        num_batches = len(imgs) // batch_size + 1
        for idx in tqdm(range(num_batches)):
            batch_idx = list(range(idx * batch_size, min(len(imgs), (idx+1) * batch_size)))
            if len(batch_idx) == 0:
                continue

            # Generate an image
            pool_val, logits_val = net(imgs[batch_idx].to('cuda'))
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]
    pool, logits = torch.cat(pool, 0), torch.cat(logits, 0)
    return pool , logits



def test3(): # This is based on BigGAN official code (https://github.com/ajbrock/BigGAN-PyTorch)
    import sys
    sys.path.append("/home/ih244/workspace/kde/BigGAN-PyTorch")
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


    import pdb; pdb.set_trace();
    




if __name__ == "__main__":
    test3()