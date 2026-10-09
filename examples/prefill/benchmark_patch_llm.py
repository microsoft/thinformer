"""Script for measuring the perplexity of a model with layers replaced.

Adapted from https://github.com/amirzandieh/HyperAttention/blob/main/benchmark_patch_llm.py

Example Usage:
```bash
python benchmark_single_attention.py -attn_method express -od /path/to/output/dir
```

"""

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
import json
import re
import pandas as pd
from slugify import slugify

from models.replace_llm_attention import patch_attention_layers
from util_experiments import seed_everything, get_base_parser

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_and_tokenizer(model_name):
    if model_name == "THUDM/chatglm2-6b-32k":
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
    elif model_name.startswith("facebook/chameleon"):
        # Chameleon is not registered with AutoModelForCausalLM; its
        # ChameleonForConditionalGeneration accepts text-only input_ids.
        from transformers import ChameleonForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ChameleonForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16)
    else:
        raise NotImplementedError("Currently we only support THUDM/chatglm2-6b-32k and facebook/chameleon-*")

    return model, tokenizer


class QKVRecorder:
    """Base class for forward hooks that record a layer's query/key/value states.

    The states are local variables inside the attention ``forward``, so a plain
    forward hook cannot read them. Subclasses receive the module's inputs via
    ``register_forward_hook(..., with_kwargs=True)`` and recompute the three
    tensors with the module's own projections/norms/RoPE, i.e. exactly the
    values that were passed to the attention kernel:

        query_states: [batch, num_heads,    seq_len, head_dim]
        key_states:   [batch, num_kv_heads, seq_len, head_dim]
        value_states: [batch, num_kv_heads, seq_len, head_dim]
        scaling:      softmax temperature used by the layer
    """

    def __init__(self):
        self.states = None

    def pop(self):
        states, self.states = self.states, None
        return states


class ChameleonQKVRecorder(QKVRecorder):
    """Forward hook recording query/key/value states of a ChameleonAttention layer.

    Mirrors ``ChameleonAttention.forward`` (transformers/models/chameleon/modeling_chameleon.py):
    q/k projections -> per-head ChameleonLayerNorm -> RoPE (applied jointly to q and k);
    v is the plain projection.
    """

    def __call__(self, module, args, kwargs, output):
        from transformers.models.chameleon.modeling_chameleon import apply_rotary_pos_emb

        hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        position_embeddings = kwargs.get("position_embeddings")
        bsz, q_len, _ = hidden_states.size()

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

        query_states = query_states.reshape(-1, module.num_heads, module.head_dim)
        query_states = module.q_norm(query_states)

        key_states = key_states.reshape(-1, module.num_key_value_heads, module.head_dim)
        key_states = module.k_norm(key_states)

        query_states = query_states.reshape(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        self.states = {
            "query_states": query_states.detach().cpu(),
            "key_states": key_states.detach().cpu(),
            "value_states": value_states.detach().cpu(),
            "scaling": float(module.scaling),  # head_dim ** -0.5
        }


def supports_chunked_head(model_name):
    return model_name.startswith("facebook/chameleon")


def token_losses_chunked(model, input_ids, loss_fct, chunk, model_name):
    """Per-token next-token losses without materializing the full [seq_len, vocab] logits.

    Equivalent to ``loss_fct(model(input_ids).logits[:, :-1].transpose(1, 2), input_ids[:, 1:])``
    but runs the backbone once and applies lm_head plus the model's own logit
    post-processing to ``chunk`` positions at a time. For Chameleon the
    image-token logits are masked to the dtype minimum, as in its forward.
    """
    if not model_name.startswith("facebook/chameleon"):
        raise NotImplementedError(f"chunked LM head is not supported for {model_name}")
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state  # [B, n, hidden]
    image_tokens = model.model.vocabulary_mapping.image_tokens

    n_pred = input_ids.shape[1] - 1
    losses = []
    for start in range(0, n_pred, chunk):
        stop = min(start + chunk, n_pred)
        logits = model.lm_head(hidden[:, start:stop])
        logits[:, :, image_tokens] = torch.finfo(logits.dtype).min
        losses.append(loss_fct(logits.transpose(1, 2), input_ids[:, start + 1:stop + 1]))
        del logits
    return torch.cat(losses, dim=1).float()


def get_decoder_layers(model, model_name):
    if model_name.startswith("facebook/chameleon"):
        return model.model.layers
    raise NotImplementedError(f"QKV recording is not supported for {model_name}")


def parse_layer_list(spec, num_layers):
    """'all' -> every layer; '0,5,31' -> those layers; '-1' / '' -> none."""
    if spec in ("", "-1", None):
        return []
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",")]


class ChunkedMLP(torch.nn.Module):
    """Run a position-wise MLP over sequence chunks so its intermediate activations
    ([seq_len, intermediate_size], ~1.4 GB per tensor for Chameleon-7B at 64k) never
    exist for the whole sequence at once. Mathematically exact (every token is
    processed identically); on GPU in bf16 results can differ at rounding level
    because cuBLAS may pick different kernels for different row counts."""

    def __init__(self, mlp, chunk):
        super().__init__()
        self.mlp = mlp
        self.chunk = chunk

    def forward(self, hidden_states, *args, **kwargs):
        out = [self.mlp(hidden_states[:, s:s + self.chunk], *args, **kwargs)
               for s in range(0, hidden_states.shape[1], self.chunk)]
        if isinstance(out[0], tuple):  # e.g. MoE blocks returning (hidden, router_logits)
            return tuple(torch.cat(parts, dim=1) if torch.is_tensor(parts[0]) and parts[0].dim() > 1 else parts[0]
                         for parts in zip(*out))
        return torch.cat(out, dim=1)


def wrap_mlp_chunks(model, model_name, mlp_chunk):
    for layer in get_decoder_layers(model, model_name):
        layer.mlp = ChunkedMLP(layer.mlp, mlp_chunk)
    print(f"Wrapped decoder MLPs to run in chunks of {mlp_chunk} positions")


def register_qkv_hook(model, model_name, layer_idx):
    """Register a QKV recorder on the self-attention module of decoder layer ``layer_idx`` (0-indexed)."""
    attn = get_decoder_layers(model, model_name)[layer_idx].self_attn
    if model_name.startswith("facebook/chameleon"):
        recorder = ChameleonQKVRecorder()
    else:
        raise NotImplementedError(f"QKV recording is not supported for {model_name}")

    handle = attn.register_forward_hook(recorder, with_kwargs=True)
    print(f"Registered QKV recorder on layer {layer_idx} ({type(attn).__name__}, layer_type={getattr(attn, 'layer_type', None)})")
    return recorder, handle


def get_arguments():
    parser = get_base_parser()
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--output_dir", "-od", type=str, default="data_save")
    # patch config
    parser.add_argument("--patch_config", type=str, default="last", choices=['last', 'first', 'even', 'odd'])
    parser.add_argument("--attn_method", type=str, default="hyper", choices=['flash', 'hyper-cuda', 'thinformer_triton'])
    parser.add_argument("--num_patch_layers", type=int, default=-1)
    # params of HyperAttention
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--lsh_num_projs", type=int, default=7)
    parser.add_argument("--min_seq_len", type=int, default=4096)
    # params of SubsamplingAttention
    parser.add_argument("--n", type=int, default=256)
    # dataset params
    parser.add_argument("--dataset_num", type=int, default=-1, choices=[-1,0, 8, 9, 19, 20], help="-1 for all datasets, 0 for narrativeqa, 8 for dureader, 9 for gov_report, 19 for passage_retrieval_en, 20 for passage_retrieval_zh")
    # memory
    parser.add_argument("--mlp_chunk", type=int, default=0, help="run each decoder layer's position-wise MLP in chunks of this many positions to cap activation memory; 0 disables (e.g. 8192 for Chameleon-7B at 64k on a 24 GB GPU)")
    parser.add_argument("--lm_head_chunk", type=int, default=2048, help="Chameleon: apply the LM head and the loss in chunks of this many positions so the full [seq_len, vocab] logits are never materialized (exact); 0 disables chunking")
    # QKV recording
    parser.add_argument("--record_qkv_layer", type=str, default="-1", help="0-indexed decoder layer(s) whose query/key/value states are saved to <output_dir>/states, one file per layer per sequence: a single index, a comma-separated list (0,5,31) or 'all'; -1 disables recording")
    parser.add_argument("--max_seqs", type=int, default=-1, help="only process the first N sequences (e.g. 1 when recording states); -1 processes all")
    parser.add_argument("--save_loss", action="store_true", help="save the per-token loss of every processed sequence to <output_dir>/loss/")

    return parser.parse_args()

def dummy_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, return_tensors=None, padding_side=None, **kwargs):
    return encoded_inputs


@torch.no_grad()
def main():
    args = get_arguments()
    if args.seq_len != 32768 and args.dataset_num != -1:
        raise ValueError("Dataset subselection only supported for seq_len 32768. Please set seq_len to 32768 or dataset_num to -1.")

    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    model, tokenizer = get_model_and_tokenizer(args.model_name)
    tokenizer.model_max_length = args.seq_len
    device = "cuda"
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    if args.seed:
        seed_everything(args.seed)
    # Load LongBench datasets
    dataset = 'longbench'

    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    filename_base = f'{args.output_dir}/data_save/{str(args.seq_len)}'
    if "chatglm" not in args.model_name:
        # The cached token ids are tokenizer-specific; keep the legacy chatglm path unchanged.
        filename_base += f'__model={slugify(args.model_name)}'
    filename = f'{filename_base}.pkl'

    if args.dataset_num != -1:
        dataset_names = [dataset_names[args.dataset_num]]
        filename = f'{filename_base}_{str(args.dataset_num)}.pkl'
    print(f"dataset_names: {dataset_names}")

    # Either load data from save of filter data
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            encoded_texts = pickle.load(file)
    else:
        data_subset_all = []
        for dataset in dataset_names:
            data_ = load_dataset('THUDM/LongBench', f"{dataset}", split='test')

            # chatglm2's custom ChatGLMTokenizer._pad() doesn't accept the
            # padding_side kwarg that newer transformers passes; override it.
            if "chatglm" in args.model_name:
                tokenizer._pad = dummy_pad

            data_subset = data_.filter(lambda x: len(tokenizer.encode(x['context'])) >= args.seq_len)
            if len(data_subset) > 0:
                data_subset_all.append(data_subset)
        data = concatenate_datasets(data_subset_all)

        encoded_texts = []
        pbar = tqdm(data)
        for i, data_i in enumerate(pbar):
            encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
            pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
            if len(encoded_text[0]) < args.seq_len:
                continue
            encoded_texts.append(encoded_text)
        print(f"# of data longer than {args.seq_len}: {len(encoded_texts)}")

        with open(filename, 'wb') as file:
            pickle.dump(encoded_texts, file, protocol=pickle.HIGHEST_PROTOCOL)

    if args.attn_method != 'flash':
        patch_attention_layers(model=model, **args.__dict__)

    if args.mlp_chunk > 0:
        wrap_mlp_chunks(model, args.model_name, args.mlp_chunk)

    model.to(device=device, dtype=dtype)
    model.eval()
    loss_fct = CrossEntropyLoss(reduction="none")

    qkv_recorders = {}
    for layer_idx in parse_layer_list(args.record_qkv_layer, len(get_decoder_layers(model, args.model_name))
                                      if args.record_qkv_layer not in ("-1", "") else 0):
        qkv_recorders[layer_idx], _ = register_qkv_hook(model, args.model_name, layer_idx)
    if qkv_recorders:
        states_dir = os.path.join(args.output_dir, "states")
        os.makedirs(states_dir, exist_ok=True)

    ppls = []
    n_seqs = len(encoded_texts) if args.max_seqs < 0 else min(args.max_seqs, len(encoded_texts))
    if args.save_loss:
        loss_dir = os.path.join(args.output_dir, "loss")
        os.makedirs(loss_dir, exist_ok=True)

    pbar = tqdm(range(n_seqs))
    for bid in pbar:
        encoded_batch = encoded_texts[bid:bid+1]
        if type(encoded_batch) == dict:
            attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
            encoded_batch = encoded_batch['input_ids']
        elif type(encoded_batch) == list:
            encoded_batch = encoded_batch[0]

        encoded_batch = encoded_batch.to(device)
        attn_mask = torch.ones_like(encoded_batch)

        chunk_lm_head = args.lm_head_chunk > 0 and supports_chunked_head(args.model_name)
        if chunk_lm_head:
            loss_ = token_losses_chunked(model, encoded_batch, loss_fct, args.lm_head_chunk, args.model_name)
        else:
            out_logits = model(encoded_batch).logits

        for layer_idx, qkv_recorder in qkv_recorders.items():
            states = qkv_recorder.pop()
            states_path = os.path.join(
                states_dir,
                f"states__model={slugify(args.model_name)}_layer={layer_idx}_method={args.attn_method}_seq_len={args.seq_len}-{args.dataset_num}_bid={bid}.pt",
            )
            torch.save(states, states_path)
            pbar.write(f"Saved {', '.join(f'{k}{tuple(v.shape)}' for k, v in states.items() if torch.is_tensor(v))} (scaling={states['scaling']}) to {states_path}")

        labels = encoded_batch
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        if not chunk_lm_head:
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
            del out_logits, shift_logits, shift_labels
        if args.save_loss:
            loss_path = os.path.join(
                loss_dir,
                f"loss__model={slugify(args.model_name)}_method={args.attn_method}_layers={args.num_patch_layers}_seq_len={args.seq_len}-{args.dataset_num}_bid={bid}.pt",
            )
            torch.save(loss_.cpu(), loss_path)

        perplexity_batch = torch.exp2(
            (loss_ * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls += perplexity_batch.tolist()

        pbar.set_description(f"[{bid:<4}/{n_seqs}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")

        del loss_, encoded_batch, attn_mask, shift_attention_mask_batch, perplexity_batch

    nan_cnt = sum(np.isnan(np.array(ppls)))
    ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])

    print(f"ppl: {ppl_mean}, nan_cnt: {nan_cnt}")
    res_str = f"model: {args.model_name}, dtype: {dtype}, seq_len: {args.seq_len}, num_patch_layers: {args.num_patch_layers}, n_data: {n_seqs}, ppl: {ppl_mean}, nan_cnt: {nan_cnt}, seed: {args.seed}\n"
    if args.dataset_num != -1:
        res_str = f"dataset: {dataset_names[0]}, " + res_str
    print(res_str)

    results = {
        "model_name": args.model_name,
        "dtype": str(dtype),
        "seq_len": args.seq_len,
        "num_patch_layers": args.num_patch_layers,
        "n_data": n_seqs,
        # "ppl": float(ppl_mean),
        "nan_cnt": nan_cnt,
        "seed": args.seed,
        "ppls": ppls,
        "h": args.log2_target_cache_size,
        # save args as a json string
        "args": json.dumps(vars(args)),
    }
    df = pd.DataFrame(results)

    accuracy_dir = os.path.join(args.output_dir, "acc")
    os.makedirs(accuracy_dir, exist_ok=True)
    accuracy_path = os.path.join(accuracy_dir, f"acc__method={args.attn_method}-h={args.log2_target_cache_size}__model={slugify(args.model_name)}_{dtype}_seq_len={args.seq_len}-{args.dataset_num}_layers={args.num_patch_layers}_s={args.seed}.csv")

    # Save results to CSV
    df.to_csv(accuracy_path, index=False)
    print(f"Results saved to {accuracy_path}")


if __name__ == "__main__":
    main()
