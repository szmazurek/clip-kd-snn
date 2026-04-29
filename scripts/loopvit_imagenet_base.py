import hydra
import pydantic
import os
import time
import json
import math
import shutil
import yaml
import copy
import torch
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


from src.models.visual_encoders.loopvit import LoopViT, _compute_gate_regularizers
from src.models.visual_encoders.transformer_utils import get_mixup_cutmix
from torch.utils.dataloader import default_collate


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, DDP):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, DDP):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, DDP):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def store(self, module):
        if isinstance(module, DDP):
            module = module.module

        self.backup = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def restore(self, module):
        if isinstance(module, DDP):
            module = module.module

        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


@torch.no_grad()
def all_reduce_sum(x: torch.Tensor):
    if dist.is_initialized():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    return x


@torch.no_grad()
def all_reduce_mean(x: torch.Tensor):
    return all_reduce_sum(x) / dist.get_world_size() if dist.is_initialized() else x


@torch.no_grad()
def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    if targets.ndim == 2:
        targets = targets.argmax(dim=-1)
    return (pred == targets).sum()


@torch.no_grad()
def safe_mean_steps(steps: torch.Tensor) -> torch.Tensor:
    return steps.to(torch.float32).mean()


def build_scheduler(
    optimizer, epochs=300, warmup_epochs=30, warmup_start_factor=0.033, lr_min=0.0
):

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler


def model_initial_carry(model, batch: Dict[str, torch.Tensor]):
    if isinstance(model, DDP):
        return model.module.initial_carry(batch)
    return model.initial_carry(batch)


def model_forward(model, carry, batch: Dict[str, torch.Tensor]):
    return model(carry, batch)


def train_one_epoch(
    model,
    optimizer,
    scaler,
    scheduler,
    gate_entropy_weight,
    loop_penalty_weight,
    use_dynamic_exit,
    gate_threshold,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_norm: float = 1.0,
    ema_helper: Optional[EMAHelper] = None,
    label_smoothing: float = 0.1,
):
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    progbar = tqdm(loader)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    carry = None

    for it, (images, targets) in enumerate(progbar):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, metadata = model(
            images,
            dynamic_exit=use_dynamic_exit,
            gate_threshold=gate_threshold,
        )

        loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        gate_entropy_loss, loop_penalty = _compute_gate_regularizers(
            metadata, disable_exit_gate=False
        )
        if gate_entropy_weight > 0:
            loss = loss + gate_entropy_weight * gate_entropy_loss
        if loop_penalty_weight > 0:
            loss = loss + loop_penalty_weight * loop_penalty
        scaler.scale(loss).backward()

        with torch.no_grad():
            total_loss += loss.item()
            total_correct += top1_correct(logits, targets).item()
            total_count += targets.shape[0]

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema_helper is not None:
            ema_helper.update(model)

    loss_t = torch.tensor(total_loss / max(1, len(loader)), device=device)
    loss_m = all_reduce_mean(loss_t).item()
    corr_t = torch.tensor(total_correct, device=device)
    cnt_t = torch.tensor(total_count, device=device)

    corr_s = all_reduce_sum(corr_t).item()
    cnt_s = all_reduce_sum(cnt_t).item()
    acc = corr_s / max(1, cnt_s)

    return {"loss": loss_m, "accuracy": acc}


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    split: str,
):
    model.eval()
    progbar = tqdm(loader)

    total_correct = 0.0
    total_count = 0.0
    total_loss = 0.0

    for it, (images, targets) in enumerate(progbar):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, metadata = model(images, dynamic_exit=None, gate_threshold=None)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item()
        total_correct += top1_correct(logits, targets).item()
        total_count += targets.shape[0]

    loss_t = torch.tensor(total_loss / max(1, len(loader)), device=device)
    loss_m = all_reduce_mean(loss_t).item()
    corr_t = torch.tensor(total_correct, device=device)
    cnt_t = torch.tensor(total_count, device=device)
    corr_s = all_reduce_sum(corr_t).item()
    cnt_s = all_reduce_sum(cnt_t).item()
    acc = corr_s / max(1, cnt_s)

    return {"loss": loss_m, "accuracy": acc}


@hydra.main(config_path="configs", config_name="imagenet_vase_config")
def main(cfg: DictConfig):

    device = torch.device(cfg.device)

    learning_rate = cfg.training.learning_rate
    min_learning_rate = cfg.training.min_learning_rate
    weight_decay = cfg.training.weight_decay
    epochs = cfg.training.epochs
    warmup_epochs = cfg.training.warmup_epochs
    train_batch = cfg.training.train_batch
    test_batch = cfg.training.test_batch
    use_ema = cfg.training.use_ema
    ema_rate = cfg.training.ema_rate

    img_size = cfg.training.img_size
    patch_size = cfg.training.patch_size
    num_classes = cfg.training.num_classes
    in_chans = cfg.training.in_chans
    max_norm = cfg.training.max_norm
    max_loop_steps = cfg.training.max_loop_steps
    min_loop_steps = cfg.training.min_loop_steps
    gate_entropy_weight = cfg.training.gate_entropy_weight
    loop_penalty_weight = cfg.training.loop_penalty_weight
    mixup_alpha = cfg.training.mixup_alpha
    cutmix_alpha = cfg.training.cutmix_alpha
    label_smoothing = cfg.training.label_smoothing
    warmup_start_factor = cfg.training.warmup_start_factor

    embed_dim = cfg.model.embed_dim
    loop_core_depth = cfg.model.loop_core_depth
    num_heads = cfg.model.num_heads
    dropout = cfg.model.dropout
    mlp_ratio = cfg.model.mlp_ratio
    add_step_embeddings = cfg.model.add_step_embeddings
    use_exit_gate = cfg.model.use_exit_gate
    use_dynamic_exit = cfg.model.use_dynamic_exit
    gate_threshold = cfg.model.gate_threshold
    swiglu = cfg.model.swiglu

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=num_classes
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    dataset_train, dataset_val, dataset_test = get_dataset(cfg)

    train_loader = ...
    test_loader = ...
    val_loader = ...

    model = LoopViT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        in_chans=in_chans,
        loop_core_depth=loop_core_depth,
        max_loop_steps=max_loop_steps,
        min_loop_steps=min_loop_steps,
        add_step_embeddings=add_step_embeddings,
        use_exit_gate=use_exit_gate,
        gate_threshod=gate_threshold,
        swiglu=swiglu,
    ).to(device)

    def param_groups_weight_decay(model, weight_decay=0.05):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim == 1
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "step_embed" in name.lower()
                or "patch_pos_embed" in name
                or "cls" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(
        param_groups_weight_decay(model, weight_decay=weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler()
    scheduler = build_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        lr_min=min_learning_rate,
    )
    ema_helper = (
        EMAHelper(mu=ema_rate) if use_ema else None
    )  # EMA wegihts, this should be ONLY on rank 0
    if ema_helper is not None:
        ema_helper.register(model)

    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        tr = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            gate_entropy_weight=gate_entropy_weight,
            loop_penalty_weight=loop_penalty_weight,
            use_dynamic_exit=use_dynamic_exit,
            gate_threshold=gate_threshold,
            loader=train_loader,
            device=device,
            epoch=epoch,
            max_norm=max_norm,
            ema_helper=ema_helper,
            label_smoothing=label_smoothing,
        )
        scheduler.step()

        if val_loader:
            if ema_helper is not None:
                ema_helper.store(model)
                ema_helper.ema(model)

        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            split="val",
        )

        if ema_helper is not None:
            ema_helper.restore(model)

    if test_loader:
        if ema_helper is not None:
            ema_helper.store(model)
            ema_helper.ema(model)

        te = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            epoch=epochs,
            split="test",
        )

        if ema_helper is not None:
            ema_helper.restore(model)
