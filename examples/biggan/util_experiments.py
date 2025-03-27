"""Helper functions for the BigGAN attention experiments."""

import os
import argparse
import torch
from biggan_models.model import BigGAN
from biggan_models.model_performer import PerformerBigGAN
from biggan_models.model_reformer import ReformerBigGAN
from biggan_models.model_kdeformer import KDEformerBigGAN
from biggan_models.model_kdeformer_old import KDEformerBigGAN as KDEformerOldBigGAN
from biggan_models.model_thinformer import ThinformerBigGAN

CHECKPOINTPATH = "checkpoints"
DATASETPATH = "data"


def get_base_parser() -> argparse.ArgumentParser:
    """Get the base parser for the BigGAN attention experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        "-s",
        # default=1,
        default=123,
        type=int,
        help="random seed for both pytorch and thinformer",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="PyTorch device: e.g., cuda or cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--ckpt_path",
        "-cp",
        default=CHECKPOINTPATH,
        help="directory containing 82.6_T2T_ViTt_24.pth.tar",
    )
    parser.add_argument(
        "--dataset_path",
        "-dp",
        default=DATASETPATH,
        help="directory containing ImageNet val folder",
    )
    parser.add_argument(
        "--output_path", "-op", default="out", help="directory for storing output"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="if set, overwrite existing output file even when it exists",
    )

    parser.add_argument("--model_name",type=str, default='biggan-deep-512')
    parser.add_argument("--num_classes",type=int, default=1000)
    parser.add_argument("--num_outputs",type=int, default=-1)
    parser.add_argument("--data_per_class",type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--attention",type=str, default='exact', choices=['exact', 'kdeformer', 'performer', 'reformer', 'sblocal', 'thinformer', 'kdeformer-old'])
    parser.add_argument("--truncation",type=float, default=0.4)
    parser.add_argument("--no_store",action='store_true')    
    parser.add_argument("--g", "-g", type = int, default=None, 
                        help="KH-Compress oversampling factor" \
                            "If None, use the default value in the model config JSON file")
    return parser


#
# Model utils
#
def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    """Helper for loading the T2T-ViT checkpoint"""
    # fix from Insu >>>>
    state_dict = torch.load(checkpoint_path, map_location=device)

    # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
    import re

    if state_dict.keys() == {"state_dict_ema"}:
        state_dict = state_dict["state_dict_ema"]

    # Replace the names of some of the submodules
    def key_mapping(key: str) -> str:
        if key == "pos_embed":
            return "pos_embed.pe"
        elif key.startswith("tokens_to_token."):
            return re.sub("^tokens_to_token.", "patch_embed.", key)
        else:
            return key

    state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    # <<<< END

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # NOTE: Performer & Scatterbrain also have a projection_matrix term
    # that is not in the original T2T checkpoint
    # these projection matrices are initialized by the module
    # so we don't have to worry about them
    print(f"missing keys: {missing_keys}")
    assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"


# Load model using specified attention method
def get_model(
    model_name: str, attention: str, g: int | None = None
) -> torch.nn.Module:
    """Load the T2T-ViT model with the specified attention methods.
    Args:
        model_name: str, name of the model to load
        attention: str, attention method to use
        g: int, oversampling factor for KH-Compress
    Returns:
        model: torch.nn.Module, the loaded model
    """
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
    elif attention == 'thinformer':
        model = ThinformerBigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer-old':
        model = KDEformerOldBigGAN.from_pretrained(model_name)
    else:
        raise NotImplementedError("Invalid attention option")
    return model

#
# Timing utils
#
def get_modules(model: torch.nn.Module) -> dict:
    """Get the modules of the BigGAN model."""
    modules = {
        "generator": model.generator,
        "embed": model.embeddings,
        "attention": model.generator.layers[8],
        "attention-matrix": model.generator.layers[8].attn,
        "model": model
    }
    return modules
