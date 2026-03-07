"""Distributed training utilities for contrastive learning.

The gather_features function is ported verbatim from
src/open_clip/loss.py to preserve exact numerical behaviour
in the multi-GPU contrastive loss computation.
"""
import torch

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
):
    """Gather features from all GPUs for contrastive loss computation.

    Ported verbatim from src/open_clip/loss.py to preserve numerical
    behaviour. When local_loss=False, the local rank's tensors are
    substituted back so that gradients flow to the local parameters.

    Args:
        image_features: Local image embeddings (B, D).
        text_features: Local text embeddings (B, D).
        local_loss: If True, only compute loss on local batch shard.
        gather_with_grad: If True, gather with gradient tracking.
        rank: Local rank.
        world_size: Total number of processes.
        use_horovod: Whether to use Horovod instead of torch.distributed.

    Returns:
        Tuple of (all_image_features, all_text_features) gathered across
        all ranks, shape (world_size * B, D).
    """
    assert has_distributed, (
        "torch.distributed did not import correctly, "
        "please use a PyTorch version with support."
    )
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
