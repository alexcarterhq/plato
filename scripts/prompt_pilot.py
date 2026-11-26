"""Prespecified CPU engineering pilot using benchmark world 9 and two worked examples."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
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
from plato.worlds import CATEGORIES, exact_answer, make_world


# These illustrative records are fixed before the pilot and do not depend on the world seed.
EXAMPLES = (
    ("NIA", "NIA is green.\nRAI is silver.", "green", "green -> 3\nsilver -> 6", "3"),
    ("ZED", "LOR is red.\nZED is ivory.", "ivory", "red -> 8\nivory -> 1", "1"),
)
FORMATS = ("current", "two_examples")


def private_prompt(entity, records):
    return f"Records:\n{records}\nRequested ID: {entity}\nRead its category. |"


def route_prompt(table, entity, records=""):
    evidence = f"\nRecords:\n{records}" if records else ""
    prefix = f"Routes:\n{table}{evidence}\nRequested ID: {entity}\nRead its category. |"
    return prefix + "\nRoute number:", len(prefix)


def demonstrations():
    retrieval, routing = [], []
    for entity, records, color, table, destination in EXAMPLES:
        retrieval.append(private_prompt(entity,records) + "\nCategory: " + color)
        routing.append(route_prompt(table,entity,records)[0] + " " + destination)
    return ("Write only the category word.\n\n" + "\n\n".join(retrieval) + "\n\n",
            "Write only the route number.\n\n" + "\n\n".join(routing) + "\n\n")


def inputs(world, fmt, book=0):
    if fmt == "current":
        private = [world.sender_text(v) for v in range(8)]
        retrieval = [world.retrieval_text(v) for v in range(8)]
        prompt, end = world.receiver_text(book)
        full = [world.receiver_text(book,evidence=world.evidence(v))[0] for v in range(8)]
        return private,retrieval,prompt,end,full
    retrieve_demo, route_demo = demonstrations()
    private = [retrieve_demo + private_prompt(world.entity,world.evidence(v)) for v in range(8)]
    retrieval = [p+"\nCategory:" for p in private]
    table = "\n".join(f"{CATEGORIES[i]} -> {world.books[book][i]}" for i in world.order)
    body, end = route_prompt(table, world.entity)
    full = [route_demo + route_prompt(table,world.entity,world.evidence(v))[0] for v in range(8)]
    return private,retrieval,route_demo+body,len(route_demo)+end,full


def option_scores(lm, prompts, options, patches=None):
    """Exact log probability of the full canonical suffix, one leading space + label.

    Read the prefix once, then fork its KV cache along the small label-token trie.
    The intervention happens once on the prompt prefill. No stop token is scored.
    """
    tokenizer = lm.tokenizer
    suffixes = [tokenizer.encode(" "+label, add_special_tokens=False) for label in options]
    for prompt in set(prompts):
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for option,suffix in zip(options,suffixes):
            if tokenizer.encode(prompt+" "+option,add_special_tokens=False) != prefix+suffix:
                raise ValueError("Option tokenization crosses the prompt boundary")
    ends = [p[1] for p in patches] if patches is not None else [None]*len(prompts)
    inputs, positions = lm._encode(prompts,ends)
    mask = inputs["attention_mask"]
    pos = (mask.long().cumsum(-1)-1).clamp_min(0)
    if patches is not None:
        intervention = lm._patch(patches[0][0],positions,[p[2] for p in patches])
    else:
        intervention = nullcontext()
    # The published checkpoints have no logit softcapping or tied-output scaling.
    backbone = lm.model.base_model
    lm_head = lm.model.get_output_embeddings()
    with torch.inference_mode():
        with intervention:
            root = backbone(**inputs,position_ids=pos,use_cache=True,return_dict=True)
        scores = torch.empty(len(prompts),len(options),dtype=torch.float32,device=lm.device)
        def step(state, active_mask, ids, accumulated, depth):
            distribution = lm_head(state.last_hidden_state[:,-1,:]).float().log_softmax(-1)
            groups = defaultdict(list)
            for i in ids:
                groups[suffixes[i][depth]].append(i)
            for token, members in groups.items():
                total = accumulated + distribution[:,token]
                nonterminal = []
                for i in members:
                    if depth+1 == len(suffixes[i]):
                        scores[:,i] = total
                    else:
                        nonterminal.append(i)
                if nonterminal:
                    new_mask = torch.cat([active_mask,torch.ones_like(active_mask[:,:1])],dim=1)
                    new_pos = active_mask.long().sum(-1,keepdim=True)
                    token_ids = torch.full_like(new_pos,token)
                    child = backbone(input_ids=token_ids,attention_mask=new_mask,position_ids=new_pos,
                                     past_key_values=deepcopy(state.past_key_values),
                                     use_cache=True,return_dict=True)
                    step(child,new_mask,nonterminal,total,depth+1)
        step(root,mask,list(range(len(options))),torch.zeros(len(prompts),device=lm.device),0)
    return scores.float().cpu().numpy()


def evaluate(lm, fmt, condition, prompts, targets, options, *, book=None, patches=None, packets=None):
    free = lm.answer_many(prompts,patches,max_new_tokens=12)
    likelihood = option_scores(lm,prompts,options,patches)
    output = []
    for v,(answer,target,logps) in enumerate(zip(free,targets,likelihood)):
        pred = options[int(np.argmax(logps))]
        packet = packets[v] if packets is not None else None
        output.append(dict(format=fmt, condition=condition, book=book, variant=v, target=target,
                           prompt=prompts[v], prompt_sha256=hashlib.sha256(prompts[v].encode()).hexdigest(),
                           free_answer=answer, free_correct=exact_answer(answer,target),
                           choice=pred, choice_correct=pred==target,
                           option_logp={k:float(v) for k,v in zip(options,logps)},
                           packet_sha256=hashlib.sha256(packet).hexdigest() if packet is not None else None,
                           packet_bytes=len(packet) if packet is not None else None))
    return output


def run():
    torch.set_num_threads(min(os.cpu_count() or 1,4))
    world = make_world("bench",9)
    output = []
    for key in ("qwen","smol"):
        lm = FrozenLM(*MODELS[key], device="cpu")
        layer = round((lm.depth-1)*.32)
        for fmt in FORMATS:
            private, retrieval, _, _, _ = inputs(world,fmt)
            if key == "qwen":
                output += evaluate(lm,fmt,"retrieval",retrieval,list(CATEGORIES),list(CATEGORIES))
                continue
            lexical = []
            for v,text in enumerate(private):
                binding = f"{world.entity} is {world.category(v)}."
                assert text.count(binding) == 1
                lexical.append(text.index(binding)+len(binding)-1)
            states = {}
            for site,ends in (("neutral",[None]*8),("lexical",lexical)):
                array = lm.capture_many(private,[layer],ends)[layer]
                states[site] = [encode_packet(x) for x in array]
            for book in (0,1):
                _,_,plain,end,full = inputs(world,fmt,book)
                targets = [world.destination(book,v) for v in range(8)]
                options = [str(i) for i in range(1,9)]
                output += evaluate(lm,fmt,"full",full,targets,options,book=book)
                # Without a message, all variants share one input and one prediction.
                first = evaluate(lm,fmt,"none",[plain],targets[:1],options,book=book)[0]
                for v,target in enumerate(targets):
                    output.append(dict(first,variant=v,target=target,
                                       free_correct=exact_answer(first["free_answer"],target),
                                       choice_correct=first["choice"]==target))
                for site,packets in states.items():
                    for suffix,order in (("",range(8)),("_wrong",world.wrong)):
                        buffers = [packets[idx] for idx in order]
                        patches = [(layer,end,decode_packet(buf)) for buf in buffers]
                        output += evaluate(lm,fmt,site+suffix,[plain]*8,targets,options,book=book,
                                           patches=patches,packets=buffers)
        del lm
        gc.collect()
    counts = defaultdict(lambda: [0,0,0])
    for row in output:
        data = counts[(row["format"],row["condition"],row["book"])]
        data[0] += row["free_correct"]
        data[1] += row["choice_correct"]
        data[2] += 1
    for fmt in FORMATS:
        for book in (0,1):
            other = [r for r in output if r["format"]==fmt and r["book"]==book and r["condition"]!="full"]
            assert len({r["prompt_sha256"] for r in other}) == 1
        for site in ("neutral","lexical"):
            rows = [r for r in output if r["format"]==fmt and r["condition"]==site]
            keyed = {(r["book"],r["variant"]):r["packet_sha256"] for r in rows}
            assert all(keyed[0,v]==keyed[1,v] for v in range(8))
    return dict(purpose="Engineering pilot on benchmark world 9; one world, eight dependent variants.",
                source_commit=os.environ.get("GITHUB_SHA"),models=MODELS,examples=EXAMPLES,
                free_metric="Exact first nonempty line; 12 greedy tokens.",
                choice_metric="Sum of token log probabilities of one leading space + candidate; no EOS or length normalization.",
                summaries=[dict(format=fmt,condition=arm,book=book,free_correct=a,choice_correct=b,items=n)
                           for (fmt,arm,book),(a,b,n) in counts.items()],rows=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summaries"],indent=2))
