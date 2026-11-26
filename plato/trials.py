"""Run paired interventions, with the same transmitted bytes under both rulebooks."""
import hashlib
import time

import numpy as np

from .worlds import exact_answer, first_answer


def _timed(call, model=None):
    gpu = getattr(getattr(model, "device", None), "type", None) == "cuda"
    if gpu:
        import torch
        torch.cuda.synchronize(model.device)
    before = time.perf_counter()
    result = call()
    if gpu:
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - before
    return result, elapsed, elapsed if gpu else 0.0


def encode_packet(vector):
    """Round to nearest-even BF16 and serialize little-endian words."""
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid packet")
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7fff) + ((bits >> 16) & 1)
    return ((bits + rounding) >> 16).astype("<u2").tobytes()


def decode_packet(packet):
    return (np.frombuffer(packet, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def evaluate_world(sender_model, receiver, world, sender_id, ls, lr, bridges, lock_sha256):
    private = [world.sender_text(i) for i in range(8)]
    sender_states, capture_time, capture_gpu = _timed(
        lambda: sender_model.capture_many(private, [ls], [None]*8)[ls], sender_model)
    own, self_time, self_gpu = _timed(
        lambda: receiver.capture_many(private, [lr], [None]*8)[lr], receiver)
    retrieval = [world.retrieval_text(i) for i in range(8)]
    generated, word_time, word_gpu = _timed(lambda: sender_model.answer_many(retrieval), sender_model)
    words = [first_answer(s) for s in generated]

    def pack(values):
        return [encode_packet(v) for v in values]

    messages = {}
    for name in ("factual", "contextual"):
        buffers, bridge_time, _ = _timed(lambda: pack(bridges[name].apply(sender_states)))
        messages[name] = (buffers, capture_time + bridge_time, capture_gpu)
    own_buffers, packing, _ = _timed(lambda: pack(own))
    messages["self"] = (own_buffers, self_time + packing, self_gpu)
    rows = []
    for book in range(2):
        plain, marker_end = world.receiver_text(book)
        for arm in ("factual", "factual_wrong", "contextual", "contextual_wrong",
                    "self", "full", "word", "none", "code", "sender"):
            base = arm.removesuffix("_wrong")
            prompts = [plain]*8
            transmitted = [b""]*8
            send_time = send_gpu = decode_time = decode_gpu = 0.0
            patches = None
            if base in messages:
                packets, send_time, send_gpu = messages[base]
                indices = world.wrong if arm.endswith("_wrong") else range(8)
                transmitted = [packets[j] for j in indices]
                patches, unpack_time, _ = _timed(
                    lambda: [(lr, marker_end, decode_packet(buf)) for buf in transmitted])
                send_time += unpack_time
            elif arm == "full":
                transmitted = [world.evidence(i).encode() for i in range(8)]
                prompts = [world.receiver_text(book, evidence=b.decode())[0] for b in transmitted]
            elif arm == "word":
                send_time, send_gpu = word_time, word_gpu
                transmitted = [w.encode() for w in words]
                prompts = [world.receiver_text(book, word=b.decode())[0] for b in transmitted]
            if arm == "sender":
                answers = generated
                send_time, send_gpu = word_time, word_gpu
                prompts = [""]*8
            elif arm == "code":
                # A dictionary is the optimal conventional receiver for a known table.
                transmitted = [bytes([i]) for i in range(8)]
                answers, decode_time, _ = _timed(
                    lambda: [str(world.books[book][b[0]]) for b in transmitted])
            else:
                answers, decode_time, decode_gpu = _timed(
                    lambda: receiver.answer_many(prompts, patches), receiver)
            if len(answers) != 8:
                raise RuntimeError("Each condition must return exactly eight answers")
            for variant, answer in enumerate(answers):
                target = world.category(variant) if arm == "sender" else world.destination(book, variant)
                alternate = world.destination(book, world.wrong[variant])
                rows.append({
                    "split": world.split, "sender": sender_id, "world": world.index,
                    "variant": variant, "rulebook": book, "condition": arm, "category": world.category(variant),
                    "wrong_category": world.category(world.wrong[variant]), "wrong_target": alternate,
                    "wrong_correct": exact_answer(answer, alternate) if arm.endswith("_wrong") else None,
                    "answer": answer, "target": target, "correct": exact_answer(answer, target),
                    "sender_prompt": (retrieval[variant] if arm in {"word","sender"} else
                                      private[world.wrong[variant] if arm.endswith("_wrong") else variant]),
                    "receiver_prompt": prompts[variant],
                    "packet_bytes": len(transmitted[variant]),
                    "packet_sha256": hashlib.sha256(transmitted[variant]).hexdigest(),
                    "lock_sha256": lock_sha256, "batch_size": 8,
                    "latency_s": send_time + decode_time, "gpu_s": (send_gpu + decode_gpu)/8,
                    "latency_definition": "sum of synchronized warm batch stages, in-process transport",
                })
    return rows
