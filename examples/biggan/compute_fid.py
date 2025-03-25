import torch
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3



def get_activations(dataloader, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataloader  : PyTorch dataloader object containing images
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """

    # Get layer of inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    num_samples = len(dataloader.dataset)
    activations = np.zeros((num_samples, dims), dtype=np.float32)

    start_idx = 0

    for batch in tqdm(dataloader, desc="Extracting Features"):
        batch, labels = batch
        batch = batch.to(device)

        with torch.no_grad():
            pred = inception_model(batch)[0]
            batch_size = pred.size(0)
            
            activations[start_idx:start_idx + batch_size] = pred.squeeze().cpu().numpy()
            start_idx += batch_size

    return activations


def calculate_fid_score(generations_root, data_statistics):
    """
    Image folder of generated images and data statistics of real images
    """
    data_statistics = np.load(data_statistics)

    # transform = transforms.Compose([
    #     transforms.Resize((299, 299)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    generated_dataset = torchvision.datasets.ImageFolder(root=generations_root, transform=transform)
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False)
    mu, sigma = calculate_statistics(generated_loader)

    fid = calculate_frechet_distance(mu, sigma, data_statistics['mu'], data_statistics['sigma'])

    return fid, mu, sigma



def calculate_statistics(dataloader, dims=2048, device='cpu'):

    activations = get_activations(dataloader, dims=dims, device=device)
    mu = np.mean(activations, axis=0).squeeze()
    sigma = np.cov(activations.squeeze(), rowvar=False)
    return mu, sigma




def calculate_statistics_from_folder(data_root, transforms = transforms.ToTensor(), 
                                     batch_size=64, dims=2048, num_workers =1, device='cpu'):
    """
    Calculates statistics of inception activations for images stored in an image folder of the following structure:
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png
    """

    dataset = torchvision.datasets.ImageFolder(data_root, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return calculate_statistics(dataloader, dims=dims, device=device)





def calculate_imagenet_statistics(data_root = None, image_size = 128, batch_size=64, device = 'cpu', num_workers=0):
    """
    Calculates statistics of inception activations for ImageNet images from the validation dataset.
    """
    
    image_net = load_imagenet(data_root, image_size, batch_size=batch_size)
    dataloader = DataLoader(image_net, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mu, sigma = calculate_statistics(dataloader, device=device)
    np.savez_compressed('./scores/imagenet_statistics.npz', mu=mu, sigma=sigma)

    return mu, sigma





def load_imagenet(data_root = None, image_size = 128, batch_size=64, shuffle=False):

    if data_root is None:
        data_root = './data/imagenet'

    IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp', 'JPEG']

    norm_mean = [0.5,0.5,0.5]
    norm_std = [0.5,0.5,0.5]

    transform = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(image_size),
        transforms.ToTensor(), 
        transforms.Normalize(norm_mean, norm_std)
    ]) 


    image_net = torchvision.datasets.ImageFolder(data_root, transform=transform)
    
    return image_net


class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Source: https://github.com/ajbrock/BigGAN-PyTorch.git utils.py
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', '-dp', type=str, default='data',
                        help="directory containing ImageNet val folder",)
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="PyTorch device: e.g., cuda or cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    print(f"Dataset path: {dataset_path}")
    device = args.device
    print(f"Device: {device}")

    print("Calculating ImageNet statistics...")
    # TODO: skip this step if the statistics file already exists
    mu, sigma = calculate_imagenet_statistics(data_root=dataset_path, device=device)

    # attention_method = 'ADD_ME'
    # fid, IS_mu, IS_sigma = calculate_fid_score('./data/generations/biggan_deep_512/'+attention_method, './data/scores/imagenet_statistics.npz')
    
    # print(fid)

    # with open(f'./examples/bigGAN/scores/fid_scores.txt', 'a') as f:
    #     #f.seek(0, 2)  # Move the cursor to the end of the file
    #     f.write('\n')
    #     f.write(f'Attention Method: {attention_method}\n')
    #     f.write(f'FID: {fid}\n')
    #     # f.write(f'IS_mu: {IS_mu}\n')
    #     # f.write(f'IS_sigma: {IS_sigma}\n')
    
