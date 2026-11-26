from copy import deepcopy
import io
import tarfile

import pytest

from plato import control_release as release
from plato.worlds import make_world


def recorded_patch():
    world = make_world("smoke", 0)
    rows = []
    for v in range(8):
        for b in (0,1):
            target = world.destination(b,v)
            rows.append(dict(world=0,variant=v,condition="self",rulebook=b,layer=5,
                             target=target,prompt=world.receiver_text(b)[0],source_prompt=world.sender_text(v),
                             category=world.category(v),answer=f" {target}!\nnoise",correct=True,
                             packet_sha256=f"{v:064x}",packet_bytes=4096))
    return world,rows


def verify(world,rows):
    return release._verify_rows(rows,lambda _:world,lambda _:world,[0],
                                [("self",0,5),("self",1,5)])


def test_verifier_reconstructs_both_books_before_scoring():
    world,rows = recorded_patch()
    result=verify(world,rows)
    assert result["self_1_5"]["correct"] == 8
    for k,v in [("packet_sha256","f"*64),("target","invalid"),
                ("correct",False),("prompt","changed"),("source_prompt","changed")]:
        bad=deepcopy(rows)
        bad[0][k]=v
        with pytest.raises(ValueError):
            verify(world,bad)
    with pytest.raises(ValueError):
        verify(world, rows + rows[:1])
    with pytest.raises(ValueError):
        verify(world,rows[:-1])


def test_archive_is_reproducible_and_excludes_unrelated_files(tmp_path):
    raw=tmp_path/"input"
    folder=raw/"gpu-controls/qwen";folder.mkdir(parents=True)
    (folder/"receipt.json").write_text("{}\n")
    (folder/"auth.txt").write_text("unrelated")
    for name in ("first","second"):
        out=tmp_path/name
        out.mkdir()
        release.archive(raw,out)
        with tarfile.open(out/release.BUNDLE) as tar:
            assert tar.getnames()==["gpu-controls/qwen/receipt.json"]
    assert (tmp_path/"first"/release.BUNDLE).read_bytes()==(tmp_path/"second"/release.BUNDLE).read_bytes()
    bad=tmp_path/"bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        member=tarfile.TarInfo("../example");member.size=1
        tar.addfile(member,io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        release.unpack(bad,tmp_path/"unpacked")
