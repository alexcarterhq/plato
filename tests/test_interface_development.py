from dataclasses import replace

from plato.interface_development import FORMATS, adapt
from plato.worlds import make_world


def test_private_markers_do_not_depend_on_either_rulebook():
    original = make_world("bench", 10000)
    reversed_books = replace(original, books=original.books[::-1])
    for fmt in FORMATS:
        a, b = (adapt(w, fmt) for w in (original, reversed_books))
        for variant in range(8):
            assert a.sender_text(variant) == b.sender_text(variant)
            assert a.retrieval_text(variant) == b.retrieval_text(variant)
        assert a.receiver_text(0) == b.receiver_text(1)
        assert a.receiver_text(0)[0] != a.receiver_text(1)[0]


def test_bare_receiver_does_not_render_the_hidden_records():
    original = make_world("bench", 10001)
    alternate = replace(original, decoys=tuple((name, "red") for name, _ in original.decoys),
                        slot=(original.slot + 1) % 4)
    for fmt in FORMATS:
        a, b = (adapt(w, fmt) for w in (original, alternate))
        for book in (0, 1):
            assert a.receiver_text(book) == b.receiver_text(book)
            prompt, marker = a.receiver_text(book)
            assert 0 < marker <= len(prompt)
            assert all(other_id not in prompt for other_id, _ in original.decoys)
            assert len({a.receiver_text(book, evidence=a.evidence(i))[0] for i in range(8)}) == 8

