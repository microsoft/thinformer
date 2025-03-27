import argparse
import os

import torch
import time
import numpy as np
from tqdm import tqdm

from biggan_models.model import BigGAN
from biggan_models.model_kdeformer import KDEformerBigGAN
from biggan_models.utils import one_hot_from_int, truncated_noise_sample, save_as_images_by_class

from util_experiments import get_model

def get_args():
    if False:
        parser = argparse.ArgumentParser()
        # Model Arguments
        parser.add_argument("--model_name",type=str, default='biggan-deep-512')
        parser.add_argument("--truncation",type=float, default=0.4)

        # Run Arguments
        parser.add_argument("--batch_size",type=int, default=32)
        parser.add_argument("--seed",type=int, default=123)

        # Data Arguments
        parser.add_argument("--num_classes",type=int, default=1000)
        parser.add_argument("--num_outputs",type=int, default=-1)
        parser.add_argument("--data_per_class",type=int, default=5)
        
        # Attention Arguments
        parser.add_argument("--attention",type=str, default='kdeformer', choices=['exact', 'kdeformer', 'performer', 'reformer', 'sblocal', 'thinformer', 'kdeformer-old'])
        parser.add_argument("--kernel", type = str, default = "KV")
        parser.add_argument("--alpha", type = float, default=0.5)
        parser.add_argument("--beta", type = float, default=0.5)

        parser.add_argument("--no_store", action='store_true')
        parser.add_argument("--path", type = str, default = "./data/generations/")
        parser.add_argument("--fid", type = bool, default=False)
    else:
        from util_experiments import get_base_parser
        parser = get_base_parser()
        parser.add_argument("--kernel", type = str, default = "KV")
        parser.add_argument("--alpha", type = float, default=0.5)
        parser.add_argument("--beta", type = float, default=0.5)
        parser.add_argument("--path", type = str, default = "./data/generations/")
        parser.add_argument("--fid", type = bool, default=False)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()

    for args_name, args_value in args.__dict__.items():
        print(f"{args_name}: {args_value}")

    model_name = args.model_name
    num_classes = args.num_classes
    data_per_class = args.data_per_class
    batch_size = args.batch_size
    attention = args.attention
    truncation = args.truncation

    # Load pre-trained model tokenizer (vocabulary)
    model = get_model(model_name, attention)

    print(model.__class__)

    # Prepare a input
    if args.data_per_class > 0:
        labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
    elif args.num_outputs > 0:
        labels = np.random.randint(num_classes, size=(args.num_outputs,))

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

    print(f"Setting torch random seed to {args.seed}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    model.eval()
    time_generations = []
    num_batches = len(labels) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        tic = time.time()
        output_all = []
        labels_all = []
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue
        # res_all.append(batch_idx)

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        output = model(n_vec, c_vec, truncation)
        output = output.to('cpu')

        #output_all.append(output)
        labels_all = labels_all + [labels[i] for i in batch_idx]
        #output_all = torch.cat(output_all)

        time_generation = time.time() - tic
        time_generations.append(time_generation)

        print(f"output_all.shape: {output.shape}")
        print(f"generation time : {time_generation:.4f} sec")

        if not args.no_store:  
            generations_dir = os.path.join(args.output_path, "generations")
            os.makedirs(generations_dir, exist_ok=True)
            output_path = os.path.join(generations_dir, f"{model_name.replace('-','_')}", attention, f"{args.kernel}_{args.alpha}_{args.beta}")
            os.makedirs(output_path, exist_ok=True)

            tic = time.time()
            print("saving images....")
            save_as_images_by_class(output, labels_all, output_path)
            print(f"done. ({time.time() - tic:.4f} sec)")


if __name__ == "__main__":
    main()