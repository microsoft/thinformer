import math
from typing import List, Optional, Tuple, Union, Any, Dict
import torch


def indexing(key, sort_idx, block_size, value=None):
    indices = sort_idx.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    new_n = math.ceil(sort_idx.shape[-1] / block_size) * block_size
    if new_n < sort_idx.shape[-1]:
        import pdb; pdb.set_trace();
    out_key = torch.nn.functional.pad(key.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    out_value = None
    if value is not None:
        out_value = torch.nn.functional.pad(value.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    return out_key, out_value


def balanced_walk(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, sort_idx=None):
    if itrs == 0:
        return sort_idx
    b, h, n, d = key.shape

    if type(gamma_) != list:
        gamma_ = [gamma_] * itrs

    if type(block_size) != list:
        block_size = [block_size] * itrs

    for t in range(itrs):
        if sort_idx is not None:
            key_sorted, value_sorted = indexing(key, sort_idx, block_size[t], value)
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
            if value is not None:
                value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
        else:
            new_n = math.ceil(n / block_size[t]) * block_size[t]
            key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
            if value is not None:
                value_sorted = torch.nn.functional.pad(value, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)

        normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)
        # normal_keys = key_sorted
        kernel_ = torch.exp(temp_ * torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/math.sqrt(d) - beta_)
        # kernel_ = (torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/d**0.5).exp2()
        if value is not None:
            kernel_ *= (1e-8 + torch.einsum('...nd,...sd->...ns', value_sorted, value_sorted))

        signs = torch.zeros(kernel_.shape[:4], dtype=torch.int16, device=kernel_.device)
        signs[:, :, :, 0] = 1
        partial_quad_form = kernel_[:, :, :, 0, 0].detach().clone()
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)
        for i in range(1, kernel_.shape[3]):
            partial_inner_prod = (kernel_[:, :, :, i, :i] * signs[:, :, :, :i]).sum(dim=-1)
            # samp_prb = 0.5 - gamma_[t] * partial_inner_prod/torch.sqrt(partial_quad_form * kernel_[:, :, :, i, i])
            samp_prb = 0.5 - gamma_[t] * partial_inner_prod
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1
            partial_quad_form += (2 * signs[:, :, :, i] * partial_inner_prod + kernel_[:, :, :, i, i])

        signs = signs.view(b, h, -1)[:, :, :n]
        signs_argsort = torch.argsort(signs, dim=-1, stable=True)
        n = n//2
        if sort_idx is None:
            sort_idx = signs_argsort[:, :, :n]
        else:
            sort_idx = sort_idx.gather(2, signs_argsort[:, :, :n])
    return sort_idx


def balanced_walk2(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, sort_idx=None, query=None):
    b, h, n, d = key.shape
    if type(gamma_) != list:
        gamma_ = [gamma_] * itrs
    const_denom = 0.025 # change this to 0.00 to change the kernel back

    if type(block_size) != list:
        block_size = [block_size] * itrs
    weight_idx = None
    for t in range(itrs): #write range(1, itrs) to check everything still works
        if sort_idx is not None:
            key_sorted, value_sorted = indexing(key, sort_idx, block_size[t], value)
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
            if value is not None:
                weight_idx_padded = torch.nn.functional.pad(weight_idx, (0, math.ceil(n / block_size[t]) * block_size[t] - weight_idx.shape[-1]))
                value_sorted = value_sorted*weight_idx_padded.unsqueeze(-1)
                value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
        else:
            new_n = math.ceil(n / block_size[t]) * block_size[t]
            key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
            value_sorted = None
            if value is not None:
                value_sorted = torch.nn.functional.pad(value, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)

        normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)

        if query is not None:
            query_key_correlation = torch.softmax(torch.einsum('b h n d,b h s m d->b h s n m',query[:,::4,:,:],normal_keys),dim=-1).mean(-2,keepdim=True)
            kernel_ = query_key_correlation*query_key_correlation.transpose(-1,-2)
        else:
            kernel_ = torch.exp(temp_ * torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/math.sqrt(d) - beta_)
        if value is not None:
            kernel_ *= (1e-8 + torch.einsum('...nd,...sd->...ns', value_sorted, value_sorted)+const_denom)

        signs = torch.zeros(kernel_.shape[:4], dtype=torch.int16, device=kernel_.device)
        signs[:, :, :, 0] = 1
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)

        for i in range(1, kernel_.shape[3]): 
            partial_inner_prod = (kernel_[:, :, :, i, :] * signs).sum(dim=-1) 
            samp_prb = 0.5 - gamma_[t] * partial_inner_prod
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1

        signs = signs.view(b, h, -1)[:, :, :n]

        if signs.shape[-1]==0: # simply to deal with n==0
            sort_idx = signs[:, :, :0]
            weigth_idx = signs[:, :, :0]
            break
        cumsum_neg = (signs == -1).cumsum(dim=-1)
        cumsum_pos = (signs == 1).cumsum(dim=-1)

        c_neg = torch.argmax((cumsum_neg == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c_pos = torch.argmax((cumsum_pos == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c = torch.maximum(c_neg, c_pos)

        c = c.to(signs.device)

        weight = signs

        # Create an index tensor `[0, 1, ..., n-1]` for comparison
        indices = torch.arange(signs.shape[2], device=signs.device).view(1, 1, -1)
        # Set all values after `c[a, b]` to `1`
        mask_after_c = indices > c.unsqueeze(-1)  # True for all d > c[a, b]
        weight[mask_after_c] = torch.abs(weight[mask_after_c])  # Set those indices to `1`
        # Identify where `signs[a, b, c[a, b]] == 1`
        mask_flip_needed = (signs.gather(2, c.unsqueeze(-1)) == 1).squeeze(-1)
        # Create mask for all indices `<= c[a, b]`
        mask_before_c = indices <= c.unsqueeze(-1)
        weight[mask_before_c] *= 2
        # Apply flipping only when `signs[a, b, c] == 1`
        flip_mask = mask_before_c & mask_flip_needed.unsqueeze(-1)
        weight[flip_mask] *= -1  # Flip selected values

        

        weight_argsort = torch.argsort(-weight, dim=-1, stable=True)

        n = n//2
        if sort_idx is None:
            sort_idx = weight_argsort[:, :, :n]
            weight_idx = weight.gather(-1, weight_argsort[:, :, :n])
        else:
            sort_idx = sort_idx.gather(2, weight_argsort[:, :, :n])
            weigth_idx_1 = weight.gather(-1, weight_argsort[:, :, :n])
            weight_idx = weight_idx.gather(-1, weight_argsort[:, :, :n])
            weight_idx = weight_idx*weigth_idx_1

    return sort_idx, weight_idx

