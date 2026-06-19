"""Text encoder wrappers that expose per-layer hidden states for MACB.

Both wrappers satisfy the same protocol:

    encode_text_eos(token_ids)          → [B, embed_dim]  (EOS/pooled embedding)
    encode_text_all_layers(token_ids)   → list of K × [B, seq_len, hidden_dim]
    embed_dim: int
    hidden_dim: int
    n_layers: int   (== K above)

OpenCLIPTextWrapper  — hooks into open_clip's TextTransformer residual blocks.
LoopTextWrapper      — delegates to LoopText.encode_text_with_hidden_states().
HFCLIPTextWrapper    — hooks into a HuggingFace CLIPModel's text encoder
                       layers. Unlike OpenCLIPTextWrapper, HF's CLIPAttention
                       exposes q_proj/k_proj/v_proj/out_proj as separate
                       nn.Linear modules called via ordinary forward(), so
                       LoRA can target attention directly — matching HiVG's
                       actual released code, not just the MLP (open_clip's
                       limitation, since its fused nn.MultiheadAttention reads
                       weights as raw tensors, bypassing any submodule wrap).

Selected at build time by cfg.model.text_encoder: "open_clip" | "looptext" | "hf_clip".
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TextEncoderProtocol(Protocol):
    embed_dim: int
    hidden_dim: int
    n_layers: int

    def encode_text_eos(self, token_ids: Tensor) -> Tensor: ...
    def encode_text_all_layers(self, token_ids: Tensor) -> list[Tensor]: ...


# ---------------------------------------------------------------------------
# open_clip wrapper
# ---------------------------------------------------------------------------

class OpenCLIPTextWrapper(nn.Module):
    """Wraps an open_clip CLIP model's text encoder.

    Installs forward hooks on each resblock to capture hidden states.
    The underlying CLIP model is expected to be frozen; no parameters from
    this wrapper are trainable.

    Args:
        clip_model: An open_clip CLIP instance (has .encode_text() and
                    .transformer.resblocks).
        embed_dim: Output embedding dimension after text projection (e.g. 512).
    """

    def __init__(self, clip_model: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim

        # Locate the transformer resblocks. open_clip stores them at
        # clip.transformer.resblocks (TextTransformer path).
        resblocks = self._find_resblocks()
        self.n_layers = len(resblocks)
        self.hidden_dim = resblocks[0].attn.embed_dim if hasattr(resblocks[0], "attn") \
            else embed_dim

        # Stash for hook outputs; filled during forward
        self._layer_outputs: list[Tensor] = []

        self._hooks = []
        for block in resblocks:
            h = block.register_forward_hook(self._capture_hook)
            self._hooks.append(h)

    def _find_resblocks(self):
        # open_clip ViT text: model.transformer.resblocks
        if hasattr(self.clip, "transformer") and hasattr(self.clip.transformer, "resblocks"):
            return list(self.clip.transformer.resblocks)
        raise RuntimeError(
            "Cannot locate text transformer resblocks on the provided CLIP model. "
            "Expected clip.transformer.resblocks."
        )

    def _capture_hook(self, module, input, output):
        # output shape: [B, seq_len, hidden] — this open_clip version's
        # ResidualAttentionBlock is batch_first (see CLIP.encode_text's own
        # "[batch_size, n_ctx, d_model]" comment), no permute needed.
        self._layer_outputs.append(output)

    @torch.no_grad()
    def _run_text(self, token_ids: Tensor):
        self._layer_outputs = []
        # open_clip encode_text returns [B, embed_dim] (EOS projected)
        eos = self.clip.encode_text(token_ids)
        hidden_states = list(self._layer_outputs)
        self._layer_outputs = []
        return eos, hidden_states

    def encode_text_eos(self, token_ids: Tensor) -> Tensor:
        eos, _ = self._run_text(token_ids)
        return eos

    def encode_text_all_layers(self, token_ids: Tensor) -> list[Tensor]:
        _, hidden = self._run_text(token_ids)
        return hidden

    def forward_with_hidden(self, token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Single pass returning both EOS and all hidden states."""
        self._layer_outputs = []
        eos = self.clip.encode_text(token_ids)
        hidden = list(self._layer_outputs)
        self._layer_outputs = []
        return eos, hidden

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# LoopText wrapper
# ---------------------------------------------------------------------------

class LoopTextWrapper(nn.Module):
    """Wraps LoopText to expose per-block-execution hidden states for MACB.

    Args:
        looptext: A LoopText instance with encode_text_with_hidden_states().
        embed_dim: Output CLIP embedding dimension.
    """

    def __init__(self, looptext: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.looptext = looptext
        self.embed_dim = embed_dim
        # depth × steps block executions
        self.n_layers = looptext.max_loop_steps * len(looptext.blocks)
        self.hidden_dim = looptext.blocks[0].attn.qkv.in_features

    def encode_text_eos(self, token_ids: Tensor) -> Tensor:
        return self.looptext.encode_text(token_ids, normalize=False)

    def encode_text_all_layers(self, token_ids: Tensor) -> list[Tensor]:
        _, hidden = self.looptext.encode_text_with_hidden_states(token_ids)
        return hidden

    def forward_with_hidden(self, token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        eos, hidden = self.looptext.encode_text_with_hidden_states(token_ids)
        return eos, hidden


# ---------------------------------------------------------------------------
# HuggingFace CLIP wrapper
# ---------------------------------------------------------------------------

class HFCLIPTextWrapper(nn.Module):
    """Wraps a HuggingFace CLIPModel's text tower (text_model + text_projection).

    Args:
        text_model: A CLIPTextTransformer (`CLIPModel.text_model`).
        text_projection: The outer CLIPModel's text_projection Linear — HF
                          keeps this separate from CLIPTextTransformer, unlike
                          open_clip which folds it into clip.encode_text().
        embed_dim: CLIP projection dimension (e.g. 512).
    """

    def __init__(self, text_model: nn.Module, text_projection: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection
        self.embed_dim = embed_dim
        self.n_layers = len(text_model.encoder.layers)
        self.hidden_dim = text_model.config.hidden_size

        self._layer_outputs: list[Tensor] = []
        self._hooks = []
        for layer in text_model.encoder.layers:
            h = layer.register_forward_hook(self._capture_hook)
            self._hooks.append(h)

    def _capture_hook(self, module, input, output):
        # CLIPEncoderLayer.forward returns a plain [B, seq, hidden] tensor
        # (batch-first), no permute needed.
        self._layer_outputs.append(output)

    @torch.no_grad()
    def _run_text_no_grad(self, token_ids: Tensor):
        return self._run_text(token_ids)

    def _run_text(self, token_ids: Tensor):
        self._layer_outputs = []
        out = self.text_model(input_ids=token_ids)
        eos = self.text_projection(out.pooler_output)
        hidden_states = list(self._layer_outputs)
        self._layer_outputs = []
        return eos, hidden_states

    def encode_text_eos(self, token_ids: Tensor) -> Tensor:
        eos, _ = self._run_text_no_grad(token_ids)
        return eos

    def encode_text_all_layers(self, token_ids: Tensor) -> list[Tensor]:
        _, hidden = self._run_text_no_grad(token_ids)
        return hidden

    def forward_with_hidden(self, token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Single pass returning both EOS and all hidden states (gradients flow)."""
        return self._run_text(token_ids)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_text_wrapper(
    text_encoder_type: str,
    clip_model: nn.Module | None = None,
    looptext: nn.Module | None = None,
    hf_clip_model: nn.Module | None = None,
    embed_dim: int = 512,
) -> OpenCLIPTextWrapper | LoopTextWrapper | HFCLIPTextWrapper:
    """Instantiate the correct wrapper based on config string.

    Args:
        text_encoder_type: "open_clip", "looptext", or "hf_clip".
        clip_model: Required when text_encoder_type == "open_clip".
        looptext: Required when text_encoder_type == "looptext".
        hf_clip_model: Required when text_encoder_type == "hf_clip" — a
                        HuggingFace CLIPModel (uses .text_model/.text_projection).
        embed_dim: CLIP projection dimension.
    """
    if text_encoder_type == "open_clip":
        if clip_model is None:
            raise ValueError("clip_model required for open_clip text wrapper")
        return OpenCLIPTextWrapper(clip_model, embed_dim)
    elif text_encoder_type == "looptext":
        if looptext is None:
            raise ValueError("looptext required for looptext wrapper")
        return LoopTextWrapper(looptext, embed_dim)
    elif text_encoder_type == "hf_clip":
        if hf_clip_model is None:
            raise ValueError("hf_clip_model required for hf_clip text wrapper")
        return HFCLIPTextWrapper(hf_clip_model.text_model, hf_clip_model.text_projection, embed_dim)
    else:
        raise ValueError(f"Unknown text_encoder_type: {text_encoder_type!r}")
