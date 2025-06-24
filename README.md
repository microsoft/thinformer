# Thinformer

**Thinformer** provides a fast, high-quality approximation to the scaled dot-product attention mechanism in Transformers.

For a detailed description of the **Thinformer** algorithm and its strong approximation guarantees, see [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063).

```bib
@inproceedings{carrell2025lowrank,
  title={Low-Rank Thinning},
  author={Annabelle Michael Carrell and Albert Gong and Abhishek Shetty and Raaz Dwivedi and Lester Mackey},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=iAkg2nVmvN}
}
```

## Getting Started
To install the `thinformer` package, use the following pip command:
```bash
pip install git+https://github.com/microsoft/thinformer.git
```

Then, simply use `ThinformerAttention` as a drop-in replacement for a standard attention layer:

```python
from thinformer import ThinformerAttention

attention_layer = ThinformerAttention()

# Assumes:
# - query has shape (B, T, H, E)
# - key has shape (B, S, H, E)
# - value has shape (B, S, H, D)

attn_output, attn_output_weights = attention_layer(query, key, value)
```

For an example usage, see our [T2T-ViT ImageNet classification experiments](./examples/t2t/README.md) and [BigGAN image generation experiments](./examples/biggan/README.md).

This package has been tested with the following operating system, Python, and PyTorch combinations:
- Ubuntu 20.04, Python 3.12.9, Torch 2.4.0
- Ubuntu 20.04, Python 3.12.9, Torch 2.6.0
- Ubuntu 22.04.5, Python 3.12.9, Torch 2.8.0

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
