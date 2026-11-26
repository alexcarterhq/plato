# Subject identity table

Each tab-separated row contains a CounterFact case ID and its Wikidata subject QID. The table was built by joining the original case-sensitive subject label, Wikidata relation ID and true object QID against the T-REx records used by LAMA. It contains no prompts or evidence sentences.

The source has 21,919 cases. Exactly one subject QID matched for 21,495 cases, representing 20,028 distinct QIDs. The table excludes 403 unmatched cases and 21 cases with multiple candidate QIDs. Twenty label strings refer to more than one included QID. Each included QID has one observed spelling in these records. The experiment splits on the QID.

## Provenance

- Kevin Meng and colleagues, *Locating and Editing Factual Associations in GPT* (2022), supplied CounterFact. Appendix D describes the derivation from ParaRel. [Original data](https://rome.baulab.info/data/dsets/counterfact.json) · [Paper supplement](https://papers.nips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Supplemental-Conference.pdf)
- Yanai Elazar and colleagues, *Measuring and Improving Consistency in Pretrained Language Models* (2021), describe using the LAMA T-REx input in the [ParaRel repository](https://github.com/yanaiela/pararel/tree/cb5554678457beb5ac163d888f1ce8cf174b3f0b).
- Fabio Petroni and colleagues, *Language Models as Knowledge Bases?* (2019), supplied the subject IDs through [LAMA T-REx](https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz). Attribution and source license for the derived identity table follow the [LAMA repository](https://github.com/facebookresearch/LAMA/blob/main/LICENSE) and [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

## Reproduction

From the project root, run the following command with both original downloads.

```sh
python scripts/build_subject_ids.py counterfact.json negated_data.tar.gz plato/assets/counterfact-subjects.tsv
```

| File | SHA-256 |
| --- | --- |
| counterfact.json | d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f |
| negated_data.tar.gz | 1a151058e6608e47983ea4c99c50bb69248c1c0763a04a3793b0a0b657aa0b61 |
| counterfact-subjects.tsv | 331bc7161e47f105f1de89b13530b257bb286b2d8aa20dc732bc74d32e773677 |
