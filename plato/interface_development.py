"""Four prespecified development interfaces for the private-fact transfer experiment."""
from dataclasses import dataclass

from .worlds import CATEGORIES, World


FORMATS = ("original", "question", "table", "python")
WORLD_IDS = tuple(range(10000, 10008))
FRACTIONS = (0.22, 0.32, 0.40)
# Demonstrations are public, fixed, and independent of experimental entities and rulebooks.
DEMONSTRATIONS = (
    ("NIA", (("LEO", "blue"), ("NIA", "green"), ("MAX", "amber"), ("IVA", "red")),
     (2, 7, 1, 5, 6, 8, 4, 3), "green", "5"),
    ("ZED", (("LIN", "lilac"), ("BEN", "silver"), ("JOY", "green"), ("ZED", "ivory")),
     (3, 6, 8, 4, 2, 5, 1, 7), "ivory", "2"),
)


def render(fmt, entity, records, table=None):
    """Return prefix ending at the neutral carrier, and the answer suffix."""
    if fmt == "question":
        lines = "".join(f"{name} is {color}.\n" for name, color in records)
        if table is None:
            return lines + f"What color is {entity}?\nAnswer:", ""
        rules = "".join(f"{color} means {number}.\n" for color, number in table)
        # Both source and recipient use the fixed ':' after the answer cue.
        prefix = rules + lines + f"What number does the color of {entity} mean?\nAnswer:"
        return prefix, ""
    if fmt == "table":
        lines = "".join(f"{name} | {color}\n" for name, color in records)
        if table is None:
            return f"ID | COLOR\n{lines}Color of {entity} =", ""
        rules = "".join(f"{color} | {number}\n" for color, number in table)
        return f"COLOR | NUMBER\n{rules}ID | COLOR\n{lines}Color of {entity} =", "\nNumber for that color ="
    if fmt == "python":
        data = ", ".join(f'"{name}": "{color}"' for name, color in records)
        colors = f">>> colors = {{{data}}}\n"
        if table is None:
            return colors + f'>>> print(colors[ "{entity}" ]', " )"
        mapping = ", ".join(f'"{color}": {number}' for color, number in table)
        return (f">>> routes = {{{mapping}}}\n" + colors
                + f'>>> print(routes[ colors[ "{entity}" ]', " ] )")
    raise ValueError(f"Unknown development format {fmt}")


def demonstrations(fmt, routing):
    examples = []
    for entity, records, numbers, color, destination in DEMONSTRATIONS:
        rules = list(zip(CATEGORIES, numbers)) if routing else None
        prefix, suffix = render(fmt, entity, records, rules)
        label = destination if routing else color
        divider = "\n" if fmt == "python" else " "
        examples.append(prefix + suffix + divider + label)
    return "\n\n".join(examples) + "\n\n"


@dataclass(frozen=True)
class FormattedWorld(World):
    fmt: str

    def _records(self, variant):
        records = list(self.decoys)
        records.insert(self.slot, (self.entity, self.category(variant)))
        return records

    def sender_text(self, variant):
        body, _ = render(self.fmt, self.entity, self._records(variant))
        return demonstrations(self.fmt, False) + body

    def retrieval_text(self, variant):
        _, suffix = render(self.fmt, self.entity, self._records(variant))
        return self.sender_text(variant) + suffix

    def receiver_text(self, book, *, word=None, evidence=None):
        if evidence is not None and word is not None:
            raise ValueError("Give either the records or the single-word message")
        # Reformat only the supplied bytes. All records keep their original order.
        records = []
        if evidence is not None:
            for line in evidence.splitlines():
                if line.startswith("The label for "):
                    line = line.removeprefix("The label for ")
                name, color = line.removesuffix(".").split(" is ", 1)
                records.append((name, color))
        if word is not None:
            records.append((self.entity, word))
        rules = [(CATEGORIES[i], str(self.books[book][i])) for i in self.order]
        body, suffix = render(self.fmt, self.entity, records, rules)
        prefix = demonstrations(self.fmt, True) + body
        return prefix + suffix, len(prefix)


def adapt(world, fmt):
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt}")
    if fmt == "original":
        return world
    return FormattedWorld(**vars(world), fmt=fmt)

