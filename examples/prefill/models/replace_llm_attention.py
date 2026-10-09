NUM_TOTAL_LAYERS = {
    "THUDM/chatglm2-6b-32k": 28,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
}


def patch_attention_layers(model, model_name, patch_config, num_patch_layers, **kwargs):
    num_total_layers = NUM_TOTAL_LAYERS[model_name]
    num_patch_layers = num_total_layers if num_patch_layers < 0 else num_patch_layers

    if patch_config == "last":
        patch_layer_indices = range(
            num_total_layers - 1, num_total_layers - num_patch_layers - 1, -1
        )

    elif patch_config == "first":
        patch_layer_indices = range(num_patch_layers)

    elif patch_config == "odd":
        patch_layer_indices = range(1, num_total_layers, 2)

    elif patch_config == "even":
        patch_layer_indices = range(0, num_total_layers, 2)

    elif patch_config == "odd_first":
        patch_layer_indices = range(1, 2 * num_patch_layers, 2)

    elif patch_config == "odd_last":
        patch_layer_indices = range(
            num_total_layers - 1, num_total_layers - num_patch_layers, -1
        )

    elif patch_config == "even_first":
        patch_layer_indices = range(0, num_total_layers, 2)[:num_patch_layers]

    elif patch_config == "even_last":
        patch_layer_indices = range(1, num_total_layers, 2)[-num_patch_layers:]

    elif patch_config == "second":
        patch_layer_indices = [1]

    else:
        raise NotImplementedError(f"Invalid patch_config option: {patch_config}")

    if model_name == "THUDM/chatglm2-6b-32k":
        from models.attention.modeling_chatglm_fast_attention import FastCoreAttention

        print(
            f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}"
        )
        for i in patch_layer_indices:
            model.transformer.encoder.layers[
                i
            ].self_attention.core_attention = FastCoreAttention(
                model.config, i, **kwargs
            )
    elif model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        from thinformer.modeling_llama import LlamaAttention

        print(
            f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}"
        )
        for i in patch_layer_indices:
            original_attn = model.model.layers[i].self_attn
            custom_attn = LlamaAttention(
                model.config,
                i,
                g_eps=kwargs.get("g_eps"),
                g_min=kwargs.get("g_min"),
            )
            custom_attn.load_state_dict(original_attn.state_dict())
            custom_attn = custom_attn.to(
                device=original_attn.q_proj.weight.device,
                dtype=original_attn.q_proj.weight.dtype
            )
            model.model.layers[i].self_attn = custom_attn
