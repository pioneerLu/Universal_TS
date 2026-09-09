"""Load Lightning or raw checkpoints."""

import os

import torch
from pytorch_lightning.utilities import rank_zero_info


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _strip_prefix(key):
    if key.startswith("module."):
        return key[7:]
    if key.startswith("_forward_module."):
        return key[16:]
    if key.startswith("model."):
        return key[6:]
    return key


def load_weights(model, path, strict=False):
    if not path:
        raise ValueError("checkpoint path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = _torch_load(path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cleaned = {}
    for k, v in state.items():
        k = _strip_prefix(k)
        if "smooth.conv.weight" in k:
            continue
        cleaned[k] = v

    model_keys = set(model.state_dict().keys())
    if any(k.startswith("model.model.") for k in model_keys):
        remapped = {}
        for k, v in cleaned.items():
            candidates = [
                k,
                "model.model." + k,
                "model.model." + k.replace(".weight", ".original_layer.weight"),
                "model.model." + k.replace(".bias", ".original_layer.bias"),
            ]
            hit = next((c for c in candidates if c in model_keys), k)
            remapped[hit] = v
        cleaned = remapped

    incompatible = model.load_state_dict(cleaned, strict=strict)
    rank_zero_info(f"Loaded weights from {path}")
    if getattr(incompatible, "missing_keys", None):
        rank_zero_info(f"missing_keys={incompatible.missing_keys[:8]}")
    if getattr(incompatible, "unexpected_keys", None):
        rank_zero_info(f"unexpected_keys={incompatible.unexpected_keys[:8]}")
    return model
