# Plato

Can one frozen model tell another a fact only it saw?

One model reads that an entity has the category *amber*. A second model gets a private table mapping categories to destinations. Only a single mapped activation crosses between them. The decisive check gives the receiver a different table and sends the very same bytes again. We measure whether its answer follows the new table, how well the wrong packet does, and how much it costs to send one word.

## GPU results

Qwen recovered **487 of 512 private categories (95.1%)** with a worked-example table on fresh validation worlds. Given the complete records and a routing table, SmolLM2 returned a valid destination on all 1,024 cases in the Qwen configuration. **Exactly 512 destinations were correct.** On the original prompts, the same receiver scored zero on both rulebooks. The original and amended prompts were evaluated on separate sets of 64 worlds.

In fresh world 10000, the secret was amber. The first rulebook routed amber to 3. SmolLM2 answered 1. The second routed amber to 2 and the model answered 2. The replay lets you inspect both prompts and completions, change the secret, or compare them with any other recorded validation world.

| Fresh validation control | Qwen sender / L40S | Pythia sender / L4 | Required |
|---|---:|---:|---:|
| Sender names the category | 487/512 | 250/512 | 95% |
| Receiver uses full evidence, first rulebook | 258/512 | 258/512 | 95% |
| Receiver uses full evidence, second rulebook | 254/512 | 252/512 | 95% |
| Receiver uses its own evidence vector, range across books and layers | 60–67/512 | 62–65/512 | 80% on both books at one layer |

Identity replacement preserved logits exactly on both GPUs. Neither model pair qualified for the registered 250-world test or the 2,500-world confirmation. The experiment therefore leaves cross-model private-fact transfer unresolved. The release contains 18,432 validation observations and 2,560 development observations. Modal reported $0.4054 in gross usage for these five GPU apps and $0.7292 across all tagged project usage at the time of the billing export.

[Interactive replay](https://alexcarterhq.github.io/plato/) · [Tagged results, raw recordings and 25-second clip](https://github.com/alexcarterhq/plato/releases/tag/v0.1.0) · [GPU preregistration](GPU_CONFIRMATION.md)

The two senders are Qwen2.5-1.5B on L40S and Pythia-1.4B-deduped on L4. They share the same frozen SmolLM2-1.7B receiver and the same world IDs. The hardware also changes between configurations. All intervals in the replay resample whole worlds, keeping the eight categories and both rulebooks together.

[Original experimental protocol](plato/protocol.md) · [Pinned model revisions](plato/config.py) · [CounterFact subject linkage](plato/assets/counterfact-subjects.md) · [Earlier CPU integration check](https://github.com/alexcarterhq/plato/actions/runs/45867697648)

## Reproduce an observation

Requires Python 3.11 or later. From a checkout:

```sh
python -m pip install .
plato verify
plato reproduce --quick
```

The last command downloads the checksum and observations from the public release, verifies the recordings and regenerates the report, figure and replay under `release/`. Open `release/index.html` in a browser. For an offline rebuild of these GPU controls, run `plato reproduce --quick --archive plato-gpu-controls.tar.gz`. Replaying the observations requires no GPU or Modal account.

## Run the experiment on Modal

Install the local controller with `python -m pip install '.[modal]'` and authenticate with Modal. Run `python modal_app.py init --help` for the workspace, credit expiration and billing inputs. The local ledger reserves a bounded worst-case amount before each submission, accounts for unresolved attempts and compares L4 and L40S on complete worlds. Keep that ledger for the life of the run. The two GPUs are timed on the same workload. On the worker the models run in BF16, and a CPU function prepares pinned weights and verifies the CounterFact dataset checksum.

To rerun the released GPU controls, use the shared ledger to prepare both pinned model pairs, then run the following commands from the repository root. The development run nominates the single prompt used for fresh validation. The amended runner checks each pair's original outcome before admitting a new GPU job.

```sh
python modal_app.py prepare --sender qwen
python modal_app.py prepare --sender pythia
python gpu_controls.py --sender qwen
python gpu_controls.py --sender pythia
python gpu_development.py
python gpu_amended.py --sender qwen
python gpu_amended.py --sender pythia
```

A qualifying interface can enter the original factual-transfer pipeline.

```sh
python modal_app.py prepare --sender qwen
python modal_app.py bench --sender qwen
python modal_app.py run --sender qwen --stage facts
python modal_app.py run --sender qwen --stage calibrate
python modal_app.py run --sender qwen --stage validate
python modal_app.py run --sender qwen --stage test --shards 8
python modal_app.py run --sender qwen --stage robustness
python modal_app.py download --sender qwen --out results/qwen
plato reproduce --quick --input results
```

Repeat with `--sender pythia` for the second pair. The validation gate keeps the test worlds sealed if the positive controls fail. The diagnostics, failed checks, eligible factual counts and per-world rows support the report for either outcome. The test split has 250 independent worlds. All eight counterfactual variants and both rulebooks stay in the same bootstrap cluster.

Two controls can be measured before fitting either map. The CPU readiness runner measures Qwen retrieval and SmolLM2 with full text on the 64 validation worlds using FP32. Each `python readiness_runner.py --shard I --shards 4` invocation processes one of four disjoint sets. Its report names the hardware and precision. Modal requires a valid payment method on file to register GPU functions, including for funded accounts. [Modal GPU guide](https://modal.com/docs/guide/gpu)

For local GPU execution install `.[models]` with Python 3.11 and use `plato stage --help`. Keep separate output directories for each pair and for quick and full runs. The local stage command bypasses Modal billing entirely. For a warm benchmark, use `plato stage bench --sender qwen --out results/bench-qwen`.

## Release artifacts

`plato bundle --input results --out plato-observations.tar.gz` produces the replay input archive and checksum. The downloadable input includes the scientific manifests and original completions. Attach the frozen bridges independently if distributing the maps. Reconciled charges belong in `results/cost.json` as `actual_usd` and `source`; the report distinguishes them from estimates. The figure compares factual packets, wrong packets, the receiver's own state and generated words. The replay reads recorded observations without a GPU server. A 25-second clip can be rendered from a measured replay with `python scripts/clip.py --help` (Pillow and FFmpeg).

## Data and attribution

The experiment uses the true targets from [ROME CounterFact](https://rome.baulab.info/) and public Wikidata IDs linked through [LAMA](https://github.com/facebookresearch/LAMA). The mapping has its [own source and licensing notice](plato/assets/counterfact-subjects.md). Model weights and the full CounterFact and LAMA downloads are obtained from their respective providers. See the [protocol](plato/protocol.md) for related work and the preregistered interpretation rules.
