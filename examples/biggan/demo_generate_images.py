import argparse
import os
import torch
# import torchvision
import sys
import time
import numpy as np
from tqdm import tqdm
from biggan_models.model import BigGAN
from biggan_models.utils import (
    truncated_noise_sample, 
    one_hot_from_int, 
    save_as_images
)

from biggan_models.model_kdeformer import KDEformerBigGAN
from biggan_models.model_performer import PerformerBigGAN
from biggan_models.model_reformer import ReformerBigGAN
from biggan_models.model_sblocal import SBlocalBigGAN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str, default='biggan-deep-512')
    parser.add_argument("--num_classes",type=int, default=1000)
    parser.add_argument("--data_per_class",type=int, default=1)
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--attention",type=str, default='exact', choices=['exact', 'kdeformer', 'performer', 'reformer', 'sblocal'])
    parser.add_argument("--truncation",type=float, default=0.4)
    parser.add_argument("--no_store",action='store_true')
    parser.add_argument("--fid",action='store_true')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--postfix", type=str, default='')
    
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()

    for aa, bb in args.__dict__.items():
        print(f"{aa}: {bb}")

    # data = torchvision.datasets.ImageNet("/home/ih244/workspace/data/imagenet/", split='val')

    model_name = args.model_name
    num_classes = args.num_classes
    data_per_class = args.data_per_class
    batch_size = args.batch_size
    attention = args.attention
    truncation = args.truncation

    # Load pre-trained model tokenizer (vocabulary)
    if attention == 'exact':
        model = BigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer':
        model = KDEformerBigGAN.from_pretrained(model_name)
    elif attention == 'performer':
        model = PerformerBigGAN.from_pretrained(model_name)
    elif attention == 'reformer':
        model = ReformerBigGAN.from_pretrained(model_name)
    elif attention == 'sblocal':
        model = SBlocalBigGAN.from_pretrained(model_name)
    else:
        raise NotImplementedError("Invalid attention option")

    print(model.__class__)

    # Prepare a input
    labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
    class_vector = one_hot_from_int(labels, batch_size=len(labels))
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(labels), seed=args.seed)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    if torch.cuda.is_available():
        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        model = model.to('cuda')

    tic = time.time()
    model.eval()
    output_all = []
    num_batches = len(labels) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue
        # res_all.append(batch_idx)

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        output = model(n_vec, c_vec, truncation)
        output = output.to('cpu')

        output_all.append(output)

    time_generation = time.time() - tic

    output_all = torch.cat(output_all)
    print(f"output_all.shape: {output_all.shape}")
    print(f"generation time : {time_generation:.4f} sec")
    del model, noise_vector, class_vector

    if args.fid:
        print("computing FID & Inception scores ...")
        from demo_inception_score import get_inception_features, compute_fid, compute_inception, get_logits
        # sys.path.append("/home/ih244/workspace/kde/BigGAN-PyTorch")
        import inception_utils
        
        pool, logits = get_logits(output_all)
        import pdb; pdb.set_trace();
        is_mean_fake, is_std_fake = inception_utils.calculate_inception_score(logits.cpu().numpy(), num_splits=10)
        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), np.cov(pool.cpu().numpy(), rowvar=False)
        data_mu = np.load('imagenet_val_inception_moments.npz')['mu']
        data_sigma = np.load('imagenet_val_inception_moments.npz')['sigma']
        import pdb; pdb.set_trace();
        fid_value = inception_utils.numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
        # fake_features = get_inception_features((((output_all + 1.) / 2.) * 255).type(torch.uint8))
        # real_features = torch.load("./imagenet_val_inception_features.pth")

        # fid_value = compute_fid(real_features, fake_features)
        print(f"FID  : {fid_value}")

        # is_mean, is_std = compute_inception(fake_features)

        # is_mean_real, is_std_real = compute_inception(real_features)

        # output_all = (output_all + 1.) / 2. # range in [0, 1]
        # # torch.save(output_all, f"./{attention}_{output_all.shape[0]}.pth")
        # # import pdb; pdb.set_trace();

        # from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
        # from pytorch_fid.inception import InceptionV3
        # dims = 2048
        # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        # inception_model = InceptionV3([block_idx]).to('cuda')
        # inception_model.eval()
        # pred = get_activations(output_all, inception_model, batch_size=batch_size, device='cuda')
        # m1 = np.mean(pred, axis=0)
        # s1 = np.cov(pred, rowvar=False)

        # # pre-computed statistics of ImageNet validation set.
        # res = torch.load("/home/ih244/workspace/kde/pytorch-pretrained-BigGAN/imagenet_val_stats.pth")
        # m2 = res['m']
        # s2 = res['s']

        # fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        # print(f"FID  : {fid_value}")

        # torch.manual_seed(1)
        # from torchmetrics.image.inception import InceptionScore
        # inception = InceptionScore()
        # inception.update((output_all * 255).type(torch.uint8))
        # is_mean, is_std = inception.compute()
        print(f"Inception score : {is_mean_fake:.5f} (std : {is_std_fake:.5f})")
        res_str = f"model: {args.model_name}, attention: {attention:<10}, data_per_class: {data_per_class}, fid: {fid_value}, is_mean_fake: {is_mean_fake}, is_std_fake: {is_std_fake}\n"
        with open("./fid_score_results.txt", "a") as f:
            f.write(res_str)

#     print("float type is casting to uint8....", end='')
#     output_all = torch.clip((output_all + 1) / 2.0 * 256, 0, 255).type(torch.uint8)
#     print("done.")
#     # obj = output_all.detach().numpy()
#     # obj = obj.transpose((0, 2, 3, 1))
#     # obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

# #     img = []
# #     for i, out in enumerate(obj):
# #         out_array = np.asarray(np.uint8(out), dtype=np.uint8)
# #         img.append(PIL.Image.fromarray(out_array))
# #     return img

#     print("computing inception score .... ", end=''); tic = time.time()
#     from torchmetrics.image.inception import InceptionScore
#     inception = InceptionScore()
#     inception.update(output_all)
#     mean_, std_ = inception.compute()
#     print(f"done. ({time.time() - tic:.4f})")
#     import pdb; pdb.set_trace();


    if not args.no_store:
        output_path = f"./generations/{model_name.replace('-','_')}/{attention}{len(labels)}{args.postfix}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tic = time.time()
        print("saving images....")
        save_as_images(output_all, output_path + "/img")
        print(f"done. ({time.time() - tic:.4f} sec)")

        with open(f"{output_path}/time.txt", "a") as f:
            f.write(f"generation_time: {time_generation} sec")

    # /gpfs/gibbs/project/karbasi/ih244/conda_envs/insu/lib/python3.9/site-packages/pytorch_fid/fid_score.py


def fid_test():

    imgs = torch.randint(0, 255, (100, 3, 512, 512), dtype=torch.uint8)
    from torchmetrics.image.inception import InceptionScore
    aa = InceptionScore(feature=2048)
    xx = aa.inception(imgs)
    # # inception.update(imgs)
    # print(xx.shape)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # bb = FrechetInceptionDistance(feature=2048)
    # yy = bb.inception(imgs)
    # print(yy.shape)

    # from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])
    pred = get_activations((imgs / 255.).float(), inception_model)


    # from torchmetrics.image.fid import NoTrainInceptionV3
    # cc = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(2048)])
    # zz = cc(imgs)
    # is_mean, is_std = inception.compute()

    import pdb; pdb.set_trace();



if __name__ == "__main__":
    # fid_test()
    main()
    # src_path = "/home/ih244/workspace/data/imagenet/val/"
    # img_path_all = []
    # cnt = 0
    # for root, dirs, files in os.walk(src_path, topdown=True):
    #     for name in files:
    #         # print(os.path.join(root, name))
    #         img_path_all.append(os.path.join(root, name))
    #         cnt += 1

    # # transform = TF.Compose([
    # #     TF.Resize(256),
    # #     TF.CenterCrop(224),
    # #     TF.ToTensor(),
    # # ])
    # # dataset = ImagePathDataset(files, transforms=transform)
    # # # dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    # # dataloader = torch.utils.data.DataLoader(dataset,
    # #                                          batch_size=4,
    # #                                          shuffle=False,
    # #                                          drop_last=False,
    # #                                          num_workers=1)

    # # batch = next(iter(dataloader))
    # from pytorch_fid.fid_score import get_activations
    # from pytorch_fid.inception import InceptionV3
    # dims = 2048
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    # model = InceptionV3([block_idx]).to('cuda')
    # pred = get_activations(img_path_all, model, batch_size=128, device='cuda')
    # mu = np.mean(pred, axis=0)
    # sigma = np.cov(pred, rowvar=False)
    # # pred_2 = get_activations(batch, model, batch_size=4)
    # import pdb; pdb.set_trace();    
