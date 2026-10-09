# Thinformer and Thinformer Express

## Overview

Thinformer is a research codebase for approximating attention cheaply and accurately. It houses both the Thinformer method for approximating unmasked attention by thinning key-value pairs and Thinformer Express, which converts high-quality non-causal thinning approximations into efficiently updatable weighted caches for causal attention.

See [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) and [Express Language Modeling](https://arxiv.org/pdf/2606.10944) for a detailed description of the methods, theoretical guarantees, implementation details, and evaluation.

### WHAT CAN THINFORMER AND THINFORMER EXPRESS DO

Thinformer was developed to accelerate unmasked dot-product attention by using a key-value attention kernel and KH-COMPRESS to select a representative subset of key-value pairs before computing exact attention on that subset. Thinformer Express extends this approach to causal attention and was developed to reduce the runtime and memory requirements of long-context language-model inference, including prefill, KV-cache compression, and long-form decoding.

Detailed discussions of the original Thinformer method and Thinformer Express, including their development, theoretical guarantees, and evaluations, appear respectively in “Low-Rank Thinning” (arXiv:2502.12063) and “Express Language Modeling” (arXiv:2606.10944).

### INTENDED USES

Thinformer is best suited for research on efficient unmasked attention in transformer applications, while Thinformer Express is best suited for efficient causal attention, long-context prefill, KV-cache compression, and memory- or compute-constrained long-form decoding.

Thinformer and Thinformer Express are being shared with the research community to facilitate reproduction of the reported results and foster further research on thinning-based attention approximation.

Both methods are intended for machine-learning researchers and systems practitioners who can independently evaluate approximation quality, runtime, memory use, and downstream model behavior before acting on results.

### OUT-of-scope uses

Thinformer and Thinformer Express are not well suited for safety-critical deployment, unsupported models or hardware, or applications requiring exact attention outputs. Thinformer is not designed for causal masking without an extension such as Express, and Thinformer Express is not intended as a general replacement for non-causal attention methods in settings where streaming causal updates are unnecessary.

We do not recommend using Thinformer or Thinformer Express in commercial or real-world applications without further testing and development. Both are being released for research purposes.

Neither method was designed or evaluated for all possible downstream purposes. Developers should consider each method’s approximation assumptions, attention setting, and implementation constraints, and evaluate accuracy, safety, and fairness concerns specific to each intended downstream use.

Without further testing and development, Thinformer and Thinformer Express should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively affect an individual’s legal, financial, or life opportunities.

We do not recommend using Thinformer or Thinformer Express in high-risk decision making, including law enforcement, legal, finance, or healthcare contexts.

## How to get started

To begin using Thinformer or Thinformer Express, consult the project repository at https://github.com/microsoft/thinformer for installation requirements, supported configurations, example commands, and benchmark scripts. The original Thinformer experiments used a PyTorch implementation for unmasked attention, but the latest codebase also provides an I/O-aware Triton implementation of Thinformer and Thinformer Express. Verify compatible GPU, software, model, attention setting, and compression parameters before running experiments.

## Evaluation

Thinformer was evaluated on attention approximation quality, downstream image-classification and image-generation quality, and forward-pass runtime. Thinformer Express was evaluated on causal-attention approximation quality, long-context prefill speed and perplexity, KV-cache compression runtime and task accuracy, and long-form decoding accuracy, cache size, and runtime.

Detailed discussions of the evaluation methods and results can be found in “Low-Rank Thinning” (arXiv:2502.12063) for Thinformer and “Express Language Modeling” (arXiv:2606.10944) for Thinformer Express.

### EVALUATION METHODS

For Thinformer, we used maximum attention-output approximation error, ImageNet Top-1 accuracy, Frechet Inception Distance, Inception Score, and forward-pass runtime. For Thinformer Express, we used approximation error, wall-clock attention runtime and speedup, language-model perplexity, downstream task accuracy, KV-cache size, and computation time.

We compared Thinformer against exact attention, Performer, Reformer, ScatterBrain, and KDEformer using T2T-ViT image classification and BigGAN image generation. We compared Thinformer Express against exact attention and leading alternatives, including FlashAttention 2, HyperAttention, SnapKV, StreamingLLM, PyramidKV, and other cache-compression baselines, using long-context prefill workloads, LongBench-E, and MATH-500.

Thinformer evaluation used pretrained T2T-ViT on the ImageNet 2012 validation set and BigGAN image generation. Thinformer Express evaluation included long-context language models and configurations described in its paper, including ChatGLM2-6B-32K, Llama-3.1-8B-Instruct, and DeepSeek-R1-Distill-Llama-8B. Consult the associated model and dataset documentation and licenses before reuse.

Results may vary with a different model, dataset, attention setting, sequence length, kernel or compression configuration, hardware platform, PyTorch or Triton version, or compression budget.

The papers report theoretical, systems, and downstream task-performance evaluations; they do not describe a separate systematic Responsible AI or DSB evaluation.

### EVALUATION RESULTS

At a high level, Thinformer established stronger worst-case approximation guarantees than prior practical unmasked-attention approximations and performed strongly in vision benchmarks: on T2T-ViT it achieved 82.18% Top-1 ImageNet accuracy while running faster than the evaluated alternatives, and on BigGAN it achieved the best reported FID (30.54) and strong Inception Score (57.12) with substantially lower runtime than exact attention, KDEformer, Reformer, and ScatterBrain. Thinformer Express improved the best-known causal-attention approximation guarantees and achieved an 52× speedup over FlashAttention 2 at 512K-token prefill, reduced attention runtime with three KV-cache compression methods without sacrificing LongBench-E accuracy, retained 99.6% of exact-attention accuracy using only 64% of the cache size, and matched exact-attention accuracy with 58% of the computation time.

## LIMITATIONS

Thinformer and Thinformer Express were developed for research and experimental purposes. Further testing and validation are needed before considering commercial or real-world deployment.

Thinformer was empirically evaluated on selected computer-vision applications, while Thinformer Express was evaluated on selected English-language long-context understanding and mathematical reasoning workloads. Performance across other languages, tasks, modalities, and domains was not established and should be independently assessed.

Both methods approximate attention rather than generating content directly, but approximation error can alter downstream model outputs. Thinformer targets unmasked attention, while Thinformer Express targets causal attention through streaming weighted caches. Users are responsible for selecting the appropriate method, assessing output accuracy, and maintaining appropriate human oversight.

Thinformer and Thinformer Express inherit the biases, errors, omissions, and safety limitations of the base models and datasets used with them. Choose and evaluate base models appropriate to the intended use case.

The experiments use multiple pretrained models and configurations rather than a single required model. Consult both papers and each model’s documentation to understand applicable capabilities, limitations, and licensing terms.

Behavior depends on the base model, task data, attention setting, sequence length, kernel and compression configuration, and system environment; approximation effects may be amplified in settings unlike those evaluated.

There has not been a systematic effort to establish protection from security vulnerabilities in systems incorporating Thinformer or Thinformer Express. Deployments should apply appropriate software, model, data, and infrastructure security controls.

## BEST PRACTICES

Better performance can be achieved by first matching the method to the attention setting: use Thinformer for unmasked attention and Thinformer Express for causal or streaming attention. Select compression parameters appropriate to the target accuracy, memory, and runtime tradeoff; reproduce evaluated model and hardware configurations where possible; validate numerical and downstream quality against exact attention; and benchmark end-to-end workloads rather than attention kernels alone.

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications:

- [What is Azure AI Content Safety?] ([https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview))
- [Overview of Responsible AI practices for Azure OpenAI models] ([https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview))
- [Azure OpenAI Transparency Note]([https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note))
- [OpenAI’s Usage policies] ([https://openai.com/policies/usage-policies](https://openai.com/policies/usage-policies))
- [Azure OpenAI’s Code of Conduct] ([https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct))

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.

Users are reminded to be mindful of data privacy concerns and to review the privacy policies associated with any models, datasets, and storage solutions used with Thinformer or Thinformer Express.

It is the user’s responsibility to ensure that use of Thinformer or Thinformer Express complies with relevant data-protection regulations, model and dataset licenses, and organizational guidelines.

Developers should follow transparency best practices and inform end-users they are interacting with an AI system.

## LICENSE

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## TRADEMARKS

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## CONTACT

This research was conducted by members of the University of Cambridge, Cornell Tech, MIT, and Microsoft Research. We welcome feedback and collaboration. For questions about Thinformer, contact Annabelle Michael Carrell at ac2411@cam.ac.uk, Albert Gong at agong@cs.cornell.edu, Abhishek Shetty at ashetty1995@gmail.com, Raaz Dwivedi at dwivedi@cornell.edu, or Lester Mackey at lmackey@microsoft.com. For questions about Thinformer Express, contact Albert Gong, Annabelle Michael Carrell, Raaz Dwivedi, or Lester Mackey at the addresses above.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.
