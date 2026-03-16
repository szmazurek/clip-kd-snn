"""ZeroShotEvalMixin: shared validation and test evaluation for CLIP modules.

Provides on_validation_epoch_start / validation_step / on_validation_epoch_end
and the parallel test hooks. Both CLIPModule and CLIPKDModule inherit from this
to avoid duplicating evaluation logic.

Requires the host module to expose:
    self.student    — open_clip CLIP model (CLIPWrapper)
    self.tokenizer  — text tokenizer callable
    self.device     — current device (standard Lightning attribute)
    self.trainer    — Lightning Trainer (standard Lightning attribute)
    self.log()      — Lightning logging method
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..evaluation.imagenet_eval import _accuracy
from ..evaluation.imagenet_zeroshot_data import (
    imagenet_classnames,
    imagenet_r_indices,
    openai_imagenet_template,
)
from ..evaluation.retrieval_eval import compute_retrieval_metrics, encode_dataset
from ..evaluation.zero_shot_classifier import build_zero_shot_classifier


class ZeroShotEvalMixin:
    """Mixin that adds zero-shot ImageNet + retrieval evaluation to a LightningModule.

    Plug in by listing it before L.LightningModule in the class bases:

        class CLIPModule(ZeroShotEvalMixin, L.LightningModule):
            ...

    Evaluation is triggered automatically by Lightning's validation / test loop.
    No changes to training_step or configure_optimizers are needed.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_val_dataset_names(self) -> list[str]:
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "val_datasets"):
            return []
        return list(dm.val_datasets.keys())

    def _build_zs_classifier(self) -> Optional[torch.Tensor]:
        """Build zero-shot classifier if any imagenet dataset is present."""
        names = self._get_val_dataset_names()
        if not any(n.startswith("imagenet") for n in names):
            return None
        self.student.eval()
        try:
            classifier = build_zero_shot_classifier(
                model=self.student,
                classnames=imagenet_classnames,
                templates=openai_imagenet_template,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        finally:
            self.student.train()
        return classifier

    def _eval_imagenet_batch(
        self,
        batch,
        dl_name: str,
        top1_acc: dict,
        top5_acc: dict,
        counts: dict,
        classifier: torch.Tensor,
    ) -> None:
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        with torch.no_grad():
            img_feats = F.normalize(self.student.encode_image(images), dim=-1)
            logits = 100.0 * img_feats @ classifier
        if dl_name == "imagenet_r":
            logits = logits[:, imagenet_r_indices]
        acc1, acc5 = _accuracy(logits, labels, topk=(1, 5))
        top1_acc[dl_name] += acc1
        top5_acc[dl_name] += acc5
        counts[dl_name] += images.size(0)


    def _log_imagenet_metrics(self, top1_acc: dict, top5_acc: dict, counts: dict, prefix: str) -> None:
        for name in top1_acc:
            n = counts[name]
            if n == 0:
                continue
            top1 = top1_acc[name] / n
            top5 = top5_acc[name] / n
            self.log(f"{prefix}/{name}/top1", top1, sync_dist=True, prog_bar=(name == "imagenet"))
            self.log(f"{prefix}/{name}/top5", top5, sync_dist=True, prog_bar=False)

    def _log_retrieval_metrics(self, prefix: str) -> None:
        """Run retrieval eval by re-iterating retrieval datasets from the datamodule."""
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "val_datasets"):
            return
        retrieval_names = [k for k in dm.val_datasets if k in ("mscoco", "flickr30k")]
        if not retrieval_names:
            return
        self.student.eval()
        try:
            for name in retrieval_names:
                loader = DataLoader(
                    dm.val_datasets[name],
                    batch_size=256,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
                img_feats, txt_feats = encode_dataset(self.student, loader, self.device)
                metrics = compute_retrieval_metrics(img_feats, txt_feats)
                for metric, val in metrics.items():
                    self.log(f"{prefix}/{name}/{metric}", val, sync_dist=True)
        finally:
            self.student.train()

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_top1: dict[str, float] = defaultdict(float)
        self._val_top5: dict[str, float] = defaultdict(float)
        self._val_n: dict[str, int] = defaultdict(int)
        self._zs_classifier = self._build_zs_classifier()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        names = self._get_val_dataset_names()
        if dataloader_idx >= len(names):
            return
        dl_name = names[dataloader_idx]
        if dl_name.startswith("imagenet") and self._zs_classifier is not None:
            self._eval_imagenet_batch(
                batch, dl_name,
                self._val_top1, self._val_top5, self._val_n,
                self._zs_classifier,
            )
        # Retrieval datasets are handled in on_validation_epoch_end

    def on_validation_epoch_end(self) -> None:
        self._log_imagenet_metrics(self._val_top1, self._val_top5, self._val_n, prefix="val")
        self._log_retrieval_metrics(prefix="val")

    # ------------------------------------------------------------------
    # Test hooks (identical logic, different metric prefix)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self._test_top1: dict[str, float] = defaultdict(float)
        self._test_top5: dict[str, float] = defaultdict(float)
        self._test_n: dict[str, int] = defaultdict(int)
        self._zs_classifier = self._build_zs_classifier()

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        names = self._get_val_dataset_names()
        if dataloader_idx >= len(names):
            return
        dl_name = names[dataloader_idx]
        if dl_name.startswith("imagenet") and self._zs_classifier is not None:
            self._eval_imagenet_batch(
                batch, dl_name,
                self._test_top1, self._test_top5, self._test_n,
                self._zs_classifier,
            )

    def on_test_epoch_end(self) -> None:
        self._log_imagenet_metrics(self._test_top1, self._test_top5, self._test_n, prefix="test")
        self._log_retrieval_metrics(prefix="test")
