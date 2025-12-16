import torch
from torch import nn
from torch.nn import functional as F

def contrastive_loss(orig_logits, gated_logits, margin=0.5):
    # 相似度差异
    sim_diff = (orig_logits * gated_logits).sum(dim=-1)  # 越高表示一致性越强
    loss = F.relu(margin - sim_diff.mean())
    return loss


def counterfactual_loss(orig_logits, masked_logits, alpha=0.7):
    # 反事实输出应与原始输出在关键实体上一致
    kl_loss = F.kl_div(
        masked_logits,
        orig_logits.detach(),
        reduction='batchmean'
    )
    return kl_loss