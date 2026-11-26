import json

import numpy as np
import pytest

from plato.facts import _correct, _load_facts, _part, run_facts
from plato.models import Bridge, FrozenLM


def test_centered_rectangular_procrustes(tmp_path):
    rng = np.random.default_rng(390)
    x = rng.normal(size=(100, 6))
    rotation = np.linalg.qr(rng.normal(size=(8, 6)))[0].T
    y = (x - 1) @ rotation * 3.5 + 7
    bridge = Bridge.fit(x, y)
    np.testing.assert_allclose(bridge.apply(x), y, atol=4e-6)
    bridge.save(tmp_path / "bridge.npz")
    np.testing.assert_array_equal(Bridge.load(tmp_path/"bridge.npz").apply(x), bridge.apply(x))
    sub = Bridge.fit(x[:4], y[:4])
    np.testing.assert_allclose(sub.apply(x[:4]), y[:4], atol=4e-6)
    assert sub.left.shape[1] == 3


def record(i):
    return {"case_id":i, "requested_rewrite":{"subject":f"person{i}",
           "relation_id":"home", "prompt":"{} lives in", "target_true":{"str":f"city{i}"},
           "target_new":{"str":"irrelevant"}}, "paraphrase_prompts":[f"The hometown of person{i} is"]}


class KnownFacts:
    repo,revision,depth = "toy","immutable",2
    def __init__(self, n, seed):
        self.states = np.random.default_rng(122).normal(size=(n,4))
        self.rotation = np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))[0]
        self.states = self.states @ self.rotation
    def _id(self, text):
        import re
        return int(re.search(r"person(\d+)",text).group(1))
    def answer_many(self, texts, patches=None):
        if patches is None:
            return [f"city{self._id(t)}." for t in texts]
        return [f"city{np.linalg.norm(self.states-p[2],axis=1).argmin()}." for p in patches]
    def capture_many(self, texts, layers, ends):
        return {l:self.states[[self._id(t) for t in texts]].astype("f") for l in layers}
    def identity_error(self, *args):
        return 0


def test_factual_evaluation_and_restart(tmp_path):
    data = tmp_path/"cf.json"
    data.write_text(json.dumps([record(i) for i in range(240)]))
    s, r = KnownFacts(240,12), KnownFacts(240,13)
    options = dict(quick=True, splits=(40,15,30),sender_layers=[0],receiver_layers=[0])
    result = run_facts(s,r,data,tmp_path/"out",**options)
    assert result["gate_passed"] and result["accuracy"]["mapped"] == 1
    assert result["accuracy"]["wrong_real"] == 0 and result["accuracy"]["none"] == 0
    assert result["test_clusters"] == 10 and result["test_items"] == 30
    assert len((tmp_path/"out"/"factual-trials.jsonl").read_text().splitlines()) == 30
    assert run_facts(s,r,data,tmp_path/"out",**options) == result
    with pytest.raises(ValueError,match="different factual"):
        run_facts(s,r,data,tmp_path/"out",seed=3433,**options)


def test_alias_groups_and_complete_answers(tmp_path):
    records = [record(1), record(2)]
    records[0]["requested_rewrite"]["subject_aliases"] = ["Human One"]
    records[1]["requested_rewrite"]["subject"] = "Human One"
    path = tmp_path/"cf.json"
    path.write_text(json.dumps(records))
    first,second = _load_facts(path)
    assert first.group == second.group and _part(first.group,0) == _part(second.group,0)
    assert _correct(" New York. Next sentence",["New York"])
    assert not _correct(" New York City.",["New York"])


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "gpt_neox"])
def test_tiny_transformer_offsets_identity_and_cached_generation(tmp_path,monkeypatch,architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM,
                              Qwen2Config, Qwen2ForCausalLM,
                              PreTrainedTokenizerFast)

    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM","false")
    torch.manual_seed(71)
    vocab = {word:i for i,word in enumerate(("[UNK]","[PAD]","[EOS]","Mira","amber","key", "home","blue"))}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tk = PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",pad_token="[PAD]",eos_token="[EOS]")
    ckpt = tmp_path/"tiny"
    tk.save_pretrained(ckpt)
    config_cls, model_cls = {"llama": (LlamaConfig, LlamaForCausalLM),
                             "qwen2": (Qwen2Config, Qwen2ForCausalLM),
                             "gpt_neox": (GPTNeoXConfig, GPTNeoXForCausalLM)}[architecture]
    model = model_cls(config_cls(vocab_size=len(vocab),hidden_size=16,intermediate_size=24,
                              num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
                              max_position_embeddings=64,bos_token_id=None,eos_token_id=2,pad_token_id=1))
    model.save_pretrained(ckpt)
    lm = FrozenLM(ckpt, revision="local", device="cpu")
    text = "Mira amber key home"
    marker = len("Mira amber key")
    x = lm.capture(text,0,marker)
    x2 = lm.capture_many([text,"blue key"],[0,1],[marker,len("blue key")])
    np.testing.assert_allclose(x,x2[0][0],atol=1e-6)
    assert lm.identity_error(text,0,marker) == 0
    clean = lm.answer(text,max_new_tokens=3)
    assert clean == lm.answer(text,(0,marker,x),max_new_tokens=3)
    assert len(lm.model.base_model.layers[0]._forward_hooks) == 0
    assert not lm.model.training and not any(p.requires_grad for p in lm.model.parameters())
    with pytest.raises(ValueError,match="width"):
        lm.answer(text,(0,marker,np.zeros(3)))
    assert len(lm.model.base_model.layers[0]._forward_hooks) == 0
