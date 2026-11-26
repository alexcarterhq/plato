"""Frozen causal LMs and a centered, scaled Procrustes map."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Bridge:
    source_mean: np.ndarray
    target_mean: np.ndarray
    left: np.ndarray
    right: np.ndarray
    scale: float

    @classmethod
    def fit(cls, X, Y):
        x, y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Fit requires two paired matrices and at least two rows")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Calibration states must be finite")
        mx, my = x.mean(axis=0), y.mean(axis=0)
        x, y = x - mx, y - my
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if min(nx, ny) == 0:
            raise ValueError("Calibration states must vary in both models")
        # For a small calibration set, avoid a hidden-width-square decomposition.
        if len(x) < min(x.shape[1], y.shape[1]):
            qx, rx = np.linalg.qr(x.T, mode="reduced")
            qy, ry = np.linalg.qr(y.T, mode="reduced")
            u, s, vt = np.linalg.svd(rx @ ry.T, full_matrices=False)
            left, right = qx @ u, vt @ qy.T
        else:
            left, s, right = np.linalg.svd(x.T @ y, full_matrices=False)
        # Centered calibration can be rank deficient. Use its identified subspace.
        keep = s > np.finfo(s.dtype).eps * max(x.shape[1], y.shape[1]) * s[0]
        if not keep.any():
            raise ValueError("Paired calibration states have zero cross-covariance")
        return cls(mx.astype("f"), my.astype("f"), left[:, keep].astype("f"),
                   right[keep].astype("f"), float(ny / nx))

    def apply(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.shape[-1] != len(self.source_mean):
            raise ValueError("Donor state has the wrong width for this bridge")
        return self.target_mean + self.scale * (((v - self.source_mean) @ self.left) @ self.right)

    def save(self, path):
        path = Path(path)
        with path.with_name(path.name + ".partial").open("wb") as file:
            np.savez(file, source_mean=self.source_mean, target_mean=self.target_mean,
                     left=self.left, right=self.right, scale=self.scale)
        path.with_name(path.name + ".partial").replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["source_mean"], data["target_mean"], data["left"],
                       data["right"], float(data["scale"]))


class FrozenLM:
    """Layer numbers are zero-based decoder block outputs. char_end is exclusive."""
    def __init__(self, repo, revision, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.repo = str(repo)
        self.revision = str(revision)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype=dtype, attn_implementation="eager",
        ).eval().requires_grad_(False).to(self.device)
        self.blocks = self.model.base_model.layers
        self.depth = len(self.blocks)

    def _encode(self, texts, ends):
        if len(texts) != len(ends) or not texts:
            raise ValueError("Each input requires one marker position")
        encoded = self.tokenizer(list(texts), padding=True, return_tensors="pt",
                                 return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded.pop("offset_mapping").tolist()
        positions = []
        for text, end, spans in zip(texts, ends, offsets):
            if end is None:
                indices = [i for i, (a, b) in enumerate(spans) if a < b]
            else:
                if not 0 < end <= len(text):
                    raise ValueError("Marker falls outside the input")
                indices = [i for i, (a, b) in enumerate(spans) if a < end <= b]
            if not indices:
                raise ValueError("Marker did not align with a text token")
            positions.append(indices[-1])
        # These decoders use token IDs, masks and positions, never segment IDs.
        return {k: encoded[k].to(self.device) for k in ("input_ids", "attention_mask")}, positions

    def _capture_forward(self, encoded):
        pos = (encoded["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
        return self.model(**encoded, position_ids=pos, use_cache=False)

    @staticmethod
    def _hidden(out):
        return out[0] if isinstance(out, tuple) else out

    def _block(self, layer):
        if not 0 <= layer < self.depth:
            raise ValueError(f"Decoder block index {layer} outside 0..{self.depth-1}")
        return self.blocks[layer]

    @contextmanager
    def _patch(self, layer, positions, vectors):
        import torch

        used = False
        def hook(_module, _inputs, out):
            nonlocal used
            if used:
                return None
            hidden = self._hidden(out)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            new = torch.as_tensor(np.stack(vectors), dtype=hidden.dtype, device=hidden.device)
            if new.shape != hidden[batch, positions].shape:
                raise ValueError("Patch batch or hidden width disagrees with recipient")
            changed = hidden.clone()
            changed[batch, positions] = new
            used = True
            return (changed, *out[1:]) if isinstance(out, tuple) else changed
        handle = self._block(layer).register_forward_hook(hook)
        try:
            yield
            if not used:
                raise RuntimeError("Registered patch never ran")
        finally:
            handle.remove()

    def capture_many(self, texts, layers, ends):
        import torch

        encoded, positions = self._encode(texts, ends)
        captures, handles = {}, []
        def make_hook(layer):
            def hook(_module, _inputs, out):
                hidden = self._hidden(out)
                values = hidden[torch.arange(len(texts), device=hidden.device), positions]
                captures[layer] = values.detach().float().cpu().numpy()
            return hook
        try:
            for layer in layers:
                handles.append(self._block(layer).register_forward_hook(make_hook(layer)))
            with torch.inference_mode():
                self._capture_forward(encoded)
        finally:
            for handle in handles:
                handle.remove()
        return captures

    def capture(self, text, layer, char_end=None):
        return self.capture_many([text], [layer], [char_end])[layer][0]

    def answer_many(self, texts, patches=None, *, max_new_tokens=12):
        from contextlib import nullcontext
        import torch

        ends = [p[1] for p in patches] if patches is not None else [None] * len(texts)
        encoded, positions = self._encode(texts, ends)
        intervention = nullcontext()
        if patches is not None:
            layers = {p[0] for p in patches}
            if len(layers) != 1:
                raise ValueError("A decode batch requires a common patch layer")
            intervention = self._patch(layers.pop(), positions, [p[2] for p in patches])
        with torch.inference_mode(), intervention:
            ids = self.model.generate(**encoded, do_sample=False, num_beams=1, use_cache=True,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        tail = ids[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(tail, skip_special_tokens=True)

    def answer(self, text, patch=None, *, max_new_tokens=12):
        return self.answer_many([text], [patch] if patch is not None else None,
                                max_new_tokens=max_new_tokens)[0]

    def identity_error(self, text, layer, char_end=None):
        import torch

        state = self.capture(text, layer, char_end)
        encoded, positions = self._encode([text], [char_end])
        with torch.inference_mode():
            baseline = self._capture_forward(encoded).logits
            with self._patch(layer, positions, [state]):
                patched = self._capture_forward(encoded).logits
        return (baseline - patched).abs().max().item()
