"""Verify the released checkpoints and live intervention path on a benchmark world."""
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from plato.config import MODELS
from plato.models import FrozenLM
from plato.trials import decode_packet, encode_packet
from plato.worlds import exact_answer, make_world


def main():
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    world = make_world("bench", 9)
    variants = (0, 1)
    report = {"purpose": "checkpoint integration, benchmark world 9", "device": "cpu", "models": {}}
    for model_id in ("qwen", "smol"):
        repo, revision = MODELS[model_id]
        model = FrozenLM(repo, revision, device="cpu")
        layer = round(.32 * (model.depth - 1))
        sources = [world.sender_text(v) for v in variants]
        canonical = model.capture_many(sources, [layer], [None] * len(sources))[layer]
        if not np.isfinite(canonical).all():
            raise ValueError("Nonfinite captured states")
        prompt, marker = world.receiver_text(0)
        error = model.identity_error(prompt, layer, marker)
        if error > 1e-4:
            raise ValueError(f"Identity check failed for {model_id}: {error}")
        obs = {"repo": repo, "revision": revision, "identity_max_abs": error, "layer": layer,
               "width": canonical.shape[1], "answers": []}
        if model_id == "qwen":
            prompts = [world.retrieval_text(v) for v in variants]
            for v, raw in zip(variants, model.answer_many(prompts)):
                target = world.category(v)
                obs["answers"].append({"condition": "retrieval", "variant": v, "answer": raw,
                                       "target": target, "correct": exact_answer(raw, target)})
        else:
            packets = [encode_packet(vec) for vec in canonical]
            for book in (0, 1):
                prompt, marker = world.receiver_text(book)
                direct = [world.receiver_text(book, evidence=world.evidence(v))[0] for v in variants]
                self_outputs = model.answer_many([prompt]*len(variants),
                    [(layer, marker, decode_packet(packet)) for packet in packets])
                for v, full, patched, buf in zip(variants, model.answer_many(direct), self_outputs, packets):
                    target = world.destination(book, v)
                    obs["answers"].append({"rulebook": book, "variant": v, "target": target,
                                           "full_answer": full, "full_correct": exact_answer(full, target),
                                           "own_state_answer": patched, "own_state_correct": exact_answer(patched, target),
                                           "packet_sha256": hashlib.sha256(buf).hexdigest(), "packet_bytes": len(buf)})
        report["models"][model_id] = obs
        del model
        gc.collect()
    result = json.dumps(report, indent=2, ensure_ascii=False)
    print(result, flush=True)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as file:
            file.write("## Frozen checkpoint integration\\n\\n".replace("\\n", "\n"))
            file.write("Two benchmark variants on CPU. Held-out worlds remain sealed.\n\n")
            file.write("\x60\x60\x60json\n" + result + "\n\x60\x60\x60\n")


if __name__ == "__main__":
    main()
