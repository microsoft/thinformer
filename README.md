# Thinformer

`thinformer` is a Python package for approximating attention cheaply and accurately. 

It weds the **Thinformer** method for unmasked attention and **Thinformer Express** for causal attention with Triton implementations for I/O-aware efficiency.

See [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) and [Express Language Modeling](https://arxiv.org/abs/2606.10944) for a detailed description of the methods and their strong attention approximation guarantees.

```bib
@inproceedings{carrell2025lowrank,
  title={Low-Rank Thinning},
  author={Annabelle Michael Carrell and Albert Gong and Abhishek Shetty and Raaz Dwivedi and Lester Mackey},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=iAkg2nVmvN}
}
```

```bib
@inproceedings{gong2026express,
  title={Express Language Modeling},
  author={Gong, Albert and Carrell, Annabelle Michael and Dwivedi, Raaz and Mackey, Lester},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {39},
  year      = {2026},
}
```

## Getting Started
To install the `thinformer` package, use the following pip command:
```bash
pip install git+https://github.com/microsoft/thinformer.git
```

Then, simply use `ThinformerAttention` as a drop-in replacement for a standard attention layer:

```python
from thinformer.triton import ThinformerAttention

attention_layer = ThinformerAttention()

# In non-causally masked attention, we assume:
# - query has shape (B, T, H, E)
# - key has shape (B, S, H, E)
# - value has shape (B, S, H, D)
attn_output, attn_output_weights = attention_layer(query, key, value)

# In causally masked attention, we assume:
# - query has shape (B, T, H, E)
# - key has shape (B, T, H, E)
# - value has shape (B, T, H, D)
attn_output, attn_output_weights = attention_layer(query, key, value, causal=True)
```

For example usages, see our [T2T-ViT ImageNet classification experiments](./examples/t2t/README.md), [BigGAN image generation experiments](./examples/biggan/README.md), [prefill experiments](./examples/prefill/README.md), [KV cache compression experiments](./examples/kvcache/README.md), and [long-form decoding experiments](./examples/decoding/README.md).

This package has been tested with the following operating system, Python, and PyTorch combinations:
- Ubuntu 20.04, Python 3.12.9, Torch 2.4.0
- Ubuntu 20.04, Python 3.12.9, Torch 2.6.0
- Ubuntu 22.04.5, Python 3.12.9, Torch 2.8.0
- Ubuntu 24.04.4, Python 3.12.13, Torch 2.10.0
- Ubuntu 24.04.4, Python 3.13.9, Torch 2.10.0

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
