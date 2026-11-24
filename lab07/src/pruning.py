import copy
import torch
from torch.nn.utils import prune
from torch import nn
from models import ConvNet
from engine import train_epochs, evaluate, count_nonzero_parameters, dense_size_bytes, sparse_format_size_bytes, measure_latency, save_checkpoint

PRUNABLE = (nn.Conv2d, nn.Linear)

def apply_magnitude_pruning(model, amount, mode="global"):
    targets = [(m, "weight") for m in model.modules() if isinstance(m, PRUNABLE)]
    if amount <= 0:
        return []
    if mode == "global":
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    elif mode == "layerwise":
        for module, name in targets:
            prune.l1_unstructured(module, name=name, amount=amount)
    else:
        raise ValueError(mode)
    masks = []
    for module, _ in targets:
        # Keep the parametrization active and enforce the mask after optimizer steps.
        masks.append((module.weight_orig, module.weight_mask))
    return masks

def remove_pruning_reparam(model):
    for m in model.modules():
        if isinstance(m, PRUNABLE) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")

def structured_compact_copy(model, rate):
    """Physically remove low-L2-norm output filters and corresponding input channels."""
    old = model
    device = next(old.parameters()).device
    convs = [old.conv1, old.conv2, old.conv3]
    keep_lists = []
    for conv in convs:
        norms = torch.linalg.vector_norm(conv.weight.detach(), ord=2, dim=(1,2,3))
        keep_n = max(8, int(round(conv.out_channels * (1.0-rate))))
        keep = torch.topk(norms, k=min(keep_n, conv.out_channels)).indices.sort().values
        keep_lists.append(keep)

    c1, c2, c3 = [len(x) for x in keep_lists]
    compact = ConvNet(channels=(c1,c2,c3),
                      hidden=old.classifier[1].out_features,
                      pool_size=old.adaptive.output_size[0]).to(device)
    with torch.no_grad():
        compact.conv1.weight.copy_(old.conv1.weight[keep_lists[0]])
        compact.conv1.bias.copy_(old.conv1.bias[keep_lists[0]])
        compact.conv2.weight.copy_(old.conv2.weight[keep_lists[1]][:, keep_lists[0]])
        compact.conv2.bias.copy_(old.conv2.bias[keep_lists[1]])
        compact.conv3.weight.copy_(old.conv3.weight[keep_lists[2]][:, keep_lists[1]])
        compact.conv3.bias.copy_(old.conv3.bias[keep_lists[2]])
        old_fc = old.classifier[1]
        new_fc = compact.classifier[1]
        p = old.adaptive.output_size[0]
        flat_idx = torch.cat([torch.arange(int(ch)*p*p, (int(ch)+1)*p*p, device=device)
                              for ch in keep_lists[2]])
        new_fc.weight.copy_(old_fc.weight[:, flat_idx])
        new_fc.bias.copy_(old_fc.bias)
        compact.classifier[4].weight.copy_(old.classifier[4].weight)
        compact.classifier[4].bias.copy_(old.classifier[4].bias)
    return compact
