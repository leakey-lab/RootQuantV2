"""DINOv3 ViT-L/16 backbone wrapper for RootQuantV2.

Loads Meta's DINOv3 ViT-L/16 (dinov3_vitl16), freezes all parameters, and
exposes `forward_intermediates` which taps selected transformer block outputs
without retaining all 24 intermediate activations in memory.

Key architectural facts about dinov3_vitl16 (confirmed by live inspection):
  - embed_dim=1024, depth=24 blocks, num_heads=16, patch_size=16
  - 4 register tokens (storage tokens in DINOv3 nomenclature)
  - RoPE positional encoding — no interpolation needed for arbitrary (H,W)
  - Token order in each block output: [CLS, reg_0..reg_3, patch_0..patch_N-1]
  - Block class: SelfAttentionBlock (from dinov3.layers)
    Direct children  : norm1 (LayerNorm), attn (SelfAttention), ls1 (LayerScale),
                        norm2 (LayerNorm), mlp (Mlp), ls2 (LayerScale)
    attn submodules  : qkv (LinearKMaskedBias), attn_drop (Dropout),
                        proj (Linear), proj_drop (Dropout)
    mlp submodules   : fc1 (Linear), act (GELU), fc2 (Linear), drop (Dropout)
  - NOTE: the attention QKV linear is named 'qkv' (NOT 'in_proj' or similar),
    and the MLP uses 'fc1'/'fc2' — timm-style naming confirmed.  No SwiGLU.

Weight loading (always from a local file; the weights are gated by Meta's
DINOv3 License, so nothing is downloaded automatically):
  1. env DINOV3_CHECKPOINT_PATH, if set
  2. the weights_path kwarg, if that file exists (checkpoints store the path
     they were trained with, which usually does not exist on another machine)
  3. RootQuantV2/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  4. otherwise FileNotFoundError saying where to put the file
The file is handed to torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16',
source='github', weights=<path>), which builds the architecture from the
facebookresearch/dinov3 code (fetched from GitHub on first use, then cached).

Pass pretrained=False to skip all weight loading (unit-test / arch-introspection
mode).  The architecture skeleton is still constructed via torch.hub with
pretrained=False so inspect_block_names() works without any checkpoint or
network access beyond the hub cache.

fp32 only — no autocast inside this module.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────────

_DINOV3_HUB_REPO = "facebookresearch/dinov3"
_DINOV3_HUB_MODEL = "dinov3_vitl16"
_DINOV3_WEIGHTS_FILE = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# Documented default location: RootQuantV2/dinov3/<file> (see dinov3/README.md).
_VENDORED_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dinov3", _DINOV3_WEIGHTS_FILE
)


def resolve_weights_path(weights_path: Optional[str] = None) -> str:
    """Return the local DINOv3 checkpoint file to load.

    Order: env ``DINOV3_CHECKPOINT_PATH`` (if set) → ``weights_path`` (if the
    file exists) → ``RootQuantV2/dinov3/<file>``. Raises FileNotFoundError with
    setup instructions when none of them is usable.
    """
    env_path = os.environ.get("DINOV3_CHECKPOINT_PATH")
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(
                f"DINOV3_CHECKPOINT_PATH={env_path!r} is not a file. Point it at "
                f"{_DINOV3_WEIGHTS_FILE}, or unset it to use {_VENDORED_WEIGHTS}."
            )
        return env_path
    if weights_path and os.path.isfile(weights_path):
        return weights_path
    if os.path.isfile(_VENDORED_WEIGHTS):
        return _VENDORED_WEIGHTS
    stale = (
        f"\n(The path stored in the config, {weights_path!r}, does not exist on this machine.)"
        if weights_path else ""
    )
    raise FileNotFoundError(
        "DINOv3 ViT-L/16 backbone weights not found. Download "
        f"{_DINOV3_WEIGHTS_FILE} (~1.2 GB) under the DINOv3 License from "
        "https://github.com/facebookresearch/dinov3, then either\n"
        f"  - put it at {_VENDORED_WEIGHTS}, or\n"
        f"  - set DINOV3_CHECKPOINT_PATH=/abs/path/to/{_DINOV3_WEIGHTS_FILE}\n"
        "See RootQuantV2/dinov3/README.md." + stale
    )


def _load_via_hub(
    hub_repo: str,
    hub_model: str,
    weights_path: str,
) -> nn.Module:
    """Load DINOv3 from torch.hub with weights from a local checkpoint file.

    The path is passed as the ``weights`` kwarg; the hub's ``dinov3_vitl16``
    parses the 8-character hash out of the ``-XXXXXXXX.pth`` file name, so the
    file must keep its original name.
    """
    kwargs: Dict = {"source": "github", "weights": weights_path, "pretrained": True}

    log.info(
        "Attempting torch.hub.load('%s', '%s', source='github', "
        "pretrained=%s, weights=%s)",
        hub_repo,
        hub_model,
        kwargs.get("pretrained"),
        weights_path,
    )
    model = torch.hub.load(hub_repo, hub_model, **kwargs)
    return model


def _load_via_hub_no_weights(
    hub_repo: str,
    hub_model: str,
) -> nn.Module:
    """Load architecture skeleton only via torch.hub (pretrained=False).

    Used in unit-test / inspection mode to construct the module without
    downloading weights.  Raises if hub is unavailable (no network / hub cache
    miss).
    """
    log.info(
        "Attempting torch.hub.load('%s', '%s', source='github', pretrained=False) "
        "— architecture skeleton only.",
        hub_repo,
        hub_model,
    )
    model = torch.hub.load(hub_repo, hub_model, source="github", pretrained=False)
    return model


# ────────────────────────────────────────────────────────────────────────────────
# Main class
# ────────────────────────────────────────────────────────────────────────────────


class DINOv3Backbone(nn.Module):
    """Thin frozen wrapper around Meta's DINOv3 ViT-L/16.

    Constructor arguments
    ---------------------
    weights_path : str or None
        Path to a local DINOv3 checkpoint file.  Used only when env
        ``DINOV3_CHECKPOINT_PATH`` is unset and the file exists; otherwise
        ``RootQuantV2/dinov3/<file>`` is tried (see ``resolve_weights_path``).
    hub_repo : str
        torch.hub repository string (default: 'facebookresearch/dinov3').
    hub_model : str
        torch.hub entry-point name (default: 'dinov3_vitl16').
    tap_layers : tuple of int
        Block indices whose outputs are captured by forward hooks for
        ``forward_intermediates``.  Default ``(11, 17, 21, 23)`` matches the
        ViTDet four-tap pattern biased toward later blocks.
    num_register_tokens : int
        Number of register (storage) tokens the model prepends after CLS.
        DINOv3 ViT-L/16 has 4.  Used for correct token-slice logic.
    embed_dim : int
        Feature dimension per token.  1024 for ViT-L.
    patch_size : int
        Spatial patch size in pixels.  16 for the entire DINOv3 family.
    pretrained : bool
        When False, constructs the module architecture without loading any
        weights.  Intended for unit tests and block-name introspection.
        The architecture is still instantiated via torch.hub (pretrained=False),
        so block names are authentic.

    Example usage (no weights — unit test / inspection mode)
    --------------------------------------------------------
    >>> m = DINOv3Backbone(pretrained=False)
    >>> m.inspect_block_names()
    >>> out = m.forward_intermediates(torch.zeros(1, 3, 640, 640))

    Environment variable
    --------------------
    ``DINOV3_CHECKPOINT_PATH`` — path to the DINOv3 checkpoint file.  Takes
    priority over the ``weights_path`` kwarg when set.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        hub_repo: str = _DINOV3_HUB_REPO,
        hub_model: str = _DINOV3_HUB_MODEL,
        tap_layers: Tuple[int, ...] = (11, 17, 21, 23),
        num_register_tokens: int = 4,
        embed_dim: int = 1024,
        patch_size: int = 16,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.tap_layers: Tuple[int, ...] = tap_layers
        self.num_register_tokens: int = num_register_tokens
        self.embed_dim: int = embed_dim
        self.patch_size: int = patch_size
        self._hub_repo = hub_repo
        self._hub_model = hub_model

        # ── Load the backbone ─────────────────────────────────────────────────
        if not pretrained:
            # Architecture-only mode: no checkpoint needed.
            self.model = self._load_skeleton(hub_repo, hub_model)
        else:
            self.model = self._load_pretrained(hub_repo, hub_model, weights_path)

        # ── Freeze all backbone parameters ───────────────────────────────────
        self._freeze()

        # ── Install forward hooks on the requested tap blocks ─────────────────
        # _hooks: list of removable hook handles so we can re-install if needed.
        # _taps:  dict[block_idx -> Tensor] populated during forward_intermediates.
        #         Cleared at the start of each call; only tap_layers are stored.
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._taps: Dict[int, torch.Tensor] = {}
        self._install_hooks(tap_layers)

    # ── Private: loading strategies ───────────────────────────────────────────

    def _load_skeleton(self, hub_repo: str, hub_model: str) -> nn.Module:
        """Load architecture shell (pretrained=False) via torch.hub."""
        try:
            return _load_via_hub_no_weights(hub_repo, hub_model)
        except Exception as exc:
            log.warning(
                "torch.hub skeleton load failed (%s).  "
                "Building the architecture needs the facebookresearch/dinov3 "
                "code, which torch.hub fetches from GitHub on first use and "
                "caches under ~/.cache/torch/hub.  Run once with network "
                "access, or copy that cache directory across.",
                exc,
            )
            raise

    def _load_pretrained(
        self,
        hub_repo: str,
        hub_model: str,
        weights_path: Optional[str],
    ) -> nn.Module:
        """Resolve the local weights file and build the model via torch.hub."""
        path = resolve_weights_path(weights_path)
        print(f"[DINOv3Backbone] backbone weights: {path}")
        try:
            return _load_via_hub(hub_repo, hub_model, path)
        except ImportError as exc:
            raise RuntimeError(
                f"Could not build DINOv3 ViT-L/16: {exc}. The facebookresearch/dinov3 "
                "torch.hub code needs this package; install the requirements "
                "(pip install -r requirements.txt)."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not build DINOv3 ViT-L/16 from {path}: {exc}\n"
                "  - Keep the original file name "
                f"({_DINOV3_WEIGHTS_FILE}); the torch.hub loader parses the "
                "8-character hash from it.\n"
                "  - The architecture code comes from the facebookresearch/dinov3 "
                "GitHub repo via torch.hub; the first run needs network access "
                "(later runs use ~/.cache/torch/hub)."
            ) from exc

    # ── Private: freeze ───────────────────────────────────────────────────────

    def _freeze(self) -> None:
        """Set requires_grad=False on all backbone params and put it in eval."""
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        log.info(
            "DINOv3 backbone frozen.  Total params: %d  (all requires_grad=False)",
            sum(p.numel() for p in self.model.parameters()),
        )

    # ── Private: hooks ────────────────────────────────────────────────────────

    def _install_hooks(self, layers: Iterable[int]) -> None:
        """Register a forward hook on each requested block index.

        The hook fires after the block's forward() completes (post-residual),
        storing the full sequence tensor [B, N_tokens, D] in self._taps[idx].
        Hooks are attached to the block module itself — not to sub-attn or
        sub-MLP — giving the correct post-residual tap for ViTDet-style fusion.

        Memory note: self._taps is cleared at the start of forward_intermediates,
        populated by hooks during the single forward pass, then read out and
        cleared at the end.  Only the ``tap_layers`` blocks are stored; the
        remaining 20 blocks' activations are freed by PyTorch's normal graph
        cleanup.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        blocks = self._get_blocks()
        if blocks is None:
            log.warning(
                "Could not locate model.blocks — forward hooks not installed.  "
                "forward_intermediates will not return patch features."
            )
            return

        for idx in layers:
            if idx >= len(blocks):
                raise ValueError(
                    f"tap_layers contains index {idx} but the model only has "
                    f"{len(blocks)} blocks (valid range 0..{len(blocks)-1})."
                )
            block = blocks[idx]

            # Closure captures idx by value via the factory function.
            def _make_hook(layer_idx: int):
                def _hook(module, input, output):  # noqa: ANN001
                    # output: post-residual Tensor [B, N_tokens, D].
                    # Detach to avoid holding the grad-graph in memory.
                    self._taps[layer_idx] = output.detach()

                return _hook

            handle = block.register_forward_hook(_make_hook(idx))
            self._hooks.append(handle)

        log.info("Forward hooks installed on blocks: %s", sorted(layers))

    def _get_blocks(self):
        """Return the list/ModuleList of transformer blocks, or None."""
        if hasattr(self.model, "blocks"):
            return self.model.blocks
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_intermediates(
        self,
        x: torch.Tensor,
        layers: Optional[Iterable[int]] = None,
    ) -> Dict[str, object]:
        """Run the backbone and return intermediate features at selected layers.

        A single forward pass is performed.  Forward hooks populate self._taps
        for all tap_layers during that pass.  The final-layer CLS and register
        tokens are extracted from the model's forward_features() API (which is
        the same computation as the forward() call, just with richer output).

        Parameters
        ----------
        x : Tensor[B, 3, H, W]
            Input images.  H and W must both be divisible by ``patch_size``
            (16).  DINOv3 uses RoPE, so arbitrary (H, W) are supported without
            positional-embedding interpolation.
        layers : iterable of int or None
            Block indices to return patch features for.  Must be a subset of
            ``tap_layers`` passed at construction (only those have hooks).
            If None, uses ``self.tap_layers``.

        Returns
        -------
        dict with keys:
            "cls"       : Tensor[B, embed_dim]
            "registers" : Tensor[B, num_register_tokens, embed_dim]
            "patches"   : List[Tensor[B, H_p*W_p, embed_dim]]
                          One tensor per requested layer, in input order.
            "grid"      : (H_p, W_p) — spatial patch grid dimensions.

        Raises
        ------
        ValueError
            If ``x`` dimensions are wrong, or a requested layer lacks a hook.
        RuntimeError
            If a hooked block's output was not captured during the forward pass.
        """
        # ── Validate inputs ───────────────────────────────────────────────────
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D input [B,3,H,W], got shape {x.shape}")
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dims ({H}, {W}) must be divisible by "
                f"patch_size={self.patch_size}."
            )

        H_p, W_p = H // self.patch_size, W // self.patch_size

        # Resolve which layers to return
        if layers is None:
            layers = self.tap_layers
        layers_list: List[int] = list(layers)

        # Verify all requested layers have hooks
        unhooked = set(layers_list) - set(self.tap_layers)
        if unhooked:
            raise ValueError(
                f"Requested layers {sorted(unhooked)} have no forward hooks.  "
                f"Only tap_layers={self.tap_layers} are hooked.  "
                f"Re-instantiate DINOv3Backbone with tap_layers covering all needed indices."
            )

        # ── Single forward pass ────────────────────────────────────────────────
        # Clear the tap cache before the pass so stale data from a prior call
        # cannot bleed through.
        self._taps.clear()
        self.model.eval()

        # Prefer forward_features() when available: it returns a richer dict
        # with pre-separated CLS / register / patch tensors AND still fires all
        # the registered block hooks (because the hooks are on block modules,
        # not on the model's top-level forward).  This avoids a second pass.
        cls_token: torch.Tensor
        reg_tokens: torch.Tensor
        R = self.num_register_tokens

        ff_dict = self._run_forward_features(x)

        if ff_dict is not None:
            # DINOv3 native API: x_norm_clstoken [B,D], x_norm_regtokens [B,R,D],
            # x_norm_patchtokens [B,N,D].  All layer-normed by the final norm.
            cls_token = ff_dict["x_norm_clstoken"]       # [B, D]
            reg_tokens = ff_dict["x_norm_regtokens"]     # [B, R, D]
        else:
            # Fallback (non-standard forward_features API): use the deepest hook.
            # The hook fires regardless of which forward API is called, so _taps
            # is already populated.
            if not self._taps:
                # Neither API produced output — call plain forward() to fire hooks.
                _ = self.model(x)

            deepest = max(self.tap_layers)
            tapped = self._taps.get(deepest)
            if tapped is None:
                raise RuntimeError(
                    "No hook output captured.  The model's forward pass may not "
                    "have fired, or block indices are wrong.  "
                    f"Installed hook indices: {sorted(self.tap_layers)}.  "
                    f"Available tap keys: {sorted(self._taps.keys())}."
                )
            # Token layout: [CLS, reg_0..reg_{R-1}, patch_0..patch_{N-1}]
            cls_token = tapped[:, 0, :]
            reg_tokens = tapped[:, 1 : 1 + R, :]

        # ── Collect patch tensors from hooks ──────────────────────────────────
        patch_list: List[torch.Tensor] = []
        for idx in layers_list:
            tap = self._taps.get(idx)
            if tap is None:
                raise RuntimeError(
                    f"Hook for block {idx} did not fire.  "
                    f"Check that model.blocks[{idx}] exists and the hook is installed."
                )
            # Token layout: [CLS, reg_0..reg_{R-1}, patch_0..patch_{H_p*W_p-1}]
            patches = tap[:, 1 + R :, :]  # [B, H_p*W_p, D]
            patch_list.append(patches)

        # Promptly free the tap cache to release activation memory
        self._taps.clear()

        return {
            "cls": cls_token,
            "registers": reg_tokens,
            "patches": patch_list,
            "grid": (H_p, W_p),
        }

    def _run_forward_features(
        self, x: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Call forward_features() if the model exposes it; return its dict.

        DINOv3's DinoVisionTransformer.forward_features() returns a dict with:
            "x_norm_clstoken"    : [B, D]
            "x_norm_regtokens"   : [B, R, D]
            "x_norm_patchtokens" : [B, N, D]
            "x_prenorm"          : [B, N_total, D]  (pre-final-norm full seq)

        This call also fires all registered block-level forward hooks because
        the hooks are attached to individual block modules that are invoked
        inside forward_features.  Thus a single call populates both self._taps
        AND gives us the final CLS/register tensors with no second pass.

        Returns None if the model does not have this API or if it fails.
        """
        if not hasattr(self.model, "forward_features"):
            return None
        try:
            out = self.model.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out  # type: ignore[return-value]
        except Exception as exc:
            log.debug("forward_features() failed: %s — will fall back to hooks.", exc)
        return None

    # ── Diagnostic: block-name inspection ─────────────────────────────────────

    def inspect_block_names(self) -> Dict[str, object]:
        """Print and return the submodule names inside ``model.blocks[0]``.

        The printed names show the exact module paths that
        ``apply_dora_to_block`` and ``apply_mona_to_backbone`` target.

        Returns
        -------
        dict with keys:
            "block_cls"       : str — class name of block[0]
            "submodule_names" : List[str] — all flat dotted paths
            "direct_children" : List[Tuple[str, str]] — (name, classname)
            "attn_children"   : List[Tuple[str, str]] — under block.attn
            "mlp_children"    : List[Tuple[str, str]] — under block.mlp
        """
        blocks = self._get_blocks()
        if blocks is None or len(blocks) == 0:
            print("[DINOv3Backbone] WARNING: cannot locate model.blocks — inspection skipped.")
            return {}

        block = blocks[0]
        block_cls = type(block).__name__

        all_names: List[str] = [name for name, _ in block.named_modules() if name]

        direct: List[Tuple[str, str]] = [
            (n, type(m).__name__) for n, m in block.named_children()
        ]

        attn_children: List[Tuple[str, str]] = []
        if hasattr(block, "attn"):
            attn_children = [
                (n, type(m).__name__) for n, m in block.attn.named_children()
            ]

        mlp_children: List[Tuple[str, str]] = []
        if hasattr(block, "mlp"):
            mlp_children = [
                (n, type(m).__name__) for n, m in block.mlp.named_children()
            ]

        result: Dict[str, object] = {
            "block_cls": block_cls,
            "submodule_names": all_names,
            "direct_children": direct,
            "attn_children": attn_children,
            "mlp_children": mlp_children,
        }

        # ── Pretty print ──────────────────────────────────────────────────────
        sep = "=" * 70
        print(sep)
        print("DINOv3Backbone.inspect_block_names()  —  model.blocks[0]")
        print(f"  Block class : {block_cls}")
        print()
        print("  Direct children:")
        for name, cls in direct:
            print(f"    {name:<20} {cls}")
        print()
        print("  attn children (block.attn.*):")
        if attn_children:
            for name, cls in attn_children:
                print(f"    attn.{name:<16} {cls}")
        else:
            print("    (no attn submodule found)")
        print()
        print("  mlp children (block.mlp.*):")
        if mlp_children:
            for name, cls in mlp_children:
                print(f"    mlp.{name:<17} {cls}")
        else:
            print("    (no mlp submodule found)")
        print()
        print("  All named submodule paths (dotted):")
        for name in all_names:
            print(f"    {name}")
        print(sep)

        return result

    # ── Standard nn.Module overrides ─────────────────────────────────────────

    def train(self, mode: bool = True) -> "DINOv3Backbone":
        """Override train() to keep the frozen backbone permanently in eval.

        The outer DINOv3Backbone wrapper can switch modes (e.g., for any
        Dropout in adapters added by later phases), but the inner DINOv3 model
        must always remain in eval to prevent BatchNorm/Dropout activation in
        the frozen layers.
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Convenience alias: forward_intermediates with default tap_layers."""
        return self.forward_intermediates(x)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"DINOv3Backbone("
            f"embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, "
            f"num_register_tokens={self.num_register_tokens}, "
            f"tap_layers={self.tap_layers}, "
            f"n_params={n_params:,}"
            f")"
        )
