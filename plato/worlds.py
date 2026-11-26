"""Synthetic assignments created independently of model weights and data order."""
from dataclasses import dataclass
import hashlib
import random
import re

CATEGORIES = ("amber", "blue", "coral", "green", "ivory", "lilac", "red", "silver")
SPLITS = ("train", "validation", "test", "robustness", "bench", "smoke")
MARKER = " |"


def first_answer(raw):
    """The response ends at the first newline after a nonempty line."""
    return next((line.strip() for line in raw.splitlines() if line.strip()), "")


def exact_answer(raw, target):
    return re.sub(r"[.!]$", "", first_answer(raw)).strip().casefold() == str(target).casefold()


@dataclass(frozen=True)
class World:
    split: str
    index: int
    entity: str
    decoys: tuple
    slot: int
    books: tuple
    order: tuple
    wrong: tuple

    def category(self, variant):
        return CATEGORIES[variant]

    def destination(self, book, variant):
        return str(self.books[book][variant])

    def evidence(self, variant):
        facts = list(self.decoys)
        facts.insert(self.slot, (self.entity, self.category(variant)))
        if self.split == "robustness":
            return "\n".join(f"The label for {name} is {cat}." for name, cat in facts)
        return "\n".join(f"{name} is {cat}." for name, cat in facts)

    def sender_text(self, variant):
        if self.split == "robustness":
            return f"{self.evidence(variant)}\nRead the label for {self.entity}.{MARKER}"
        return f"{self.evidence(variant)}\nRead {self.entity}'s category.{MARKER}"

    def retrieval_text(self, variant):
        return self.sender_text(variant) + "\nCategory:"

    def receiver_text(self, book, *, word=None, evidence=None):
        """Returns raw prompt and exclusive end offset of the neutral carrier."""
        table = "\n".join(f"{CATEGORIES[i]} -> {self.books[book][i]}" for i in self.order)
        extra = ""
        if evidence is not None:
            extra = f"\nRecords:\n{evidence}"
        if word is not None:
            extra += f"\n{self.entity}'s category: {word}."
        if self.split == "robustness":
            prefix = f"Routing table:\n{table}{extra}\nLook up the label for {self.entity}.{MARKER}"
            suffix = "\nRoute number:"
        else:
            prefix = f"Category -> destination:\n{table}{extra}\nRead {self.entity}'s category.{MARKER}"
            suffix = "\nDestination:"
        return prefix + suffix, len(prefix)


def make_world(split, index):
    if split not in SPLITS or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid world split or index")
    seed = hashlib.sha256(f"plato-2026-11-26-v1/{split}/{index}".encode()).digest()
    rng = random.Random(int.from_bytes(seed, "big"))
    entities = [f"ID-{v:05X}" for v in rng.sample(range(16**5), 4)]
    a = rng.sample(range(1, 9), 8)
    while True:
        b = rng.sample(range(1, 9), 8)
        if all(x != y for x, y in zip(a, b)):
            break
    cycle = rng.sample(range(8), 8)
    wrong = [0] * 8
    for i, left in enumerate(cycle):
        wrong[left] = cycle[(i + 1) % 8]
    return World(split, index, entities[0], tuple((e, rng.choice(CATEGORIES)) for e in entities[1:]),
                 rng.randrange(4), (tuple(a), tuple(b)), tuple(rng.sample(range(8), 8)), tuple(wrong))
