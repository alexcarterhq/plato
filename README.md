# Plato

Can one frozen model tell another a fact only it saw?

One model reads that an entity has the category *amber*. A second model gets a private table mapping categories to destinations. Only a single mapped activation crosses between them. The decisive check gives the receiver a different table and sends the very same bytes again. We measure whether its answer follows the new table, how well the wrong packet does, and how much it costs to send one word.

## Study status

This repository publishes the dated protocol and measurement code. Primary results are available once a measured release appears in [Releases](https://github.com/alexcarterhq/plato/releases). Quick tests and examples are instrumentation checks. The baseline receiver is SmolLM2-1.7B. The primary donor is Qwen2.5-1.5B and the replication donor is Pythia-1.4B-deduped.

[Experimental protocol](plato/protocol.md) · [Model revisions](plato/config.py) · [CounterFact subject linkage](plato/assets/counterfact-subjects.md)

The first public integration check used the exact Qwen and SmolLM2 weights on one benchmark world. Identity patching preserved both models’ logits exactly. Qwen repeated the correct category in a sentence for each of two variants. SmolLM2 also returned a category when the rulebook required a destination, scoring 0/4 with the full evidence. These are CPU prompt checks. The cross-model estimate requires the private-world experiment. [Recorded completions and packet hashes](https://github.com/alexcarterhq/plato/actions/runs/45867697648)

## Reproduce an observation

Requires Python 3.11 or later. From a checkout:

```sh
python -m pip install .
plato verify
plato reproduce --quick
```

The last command fetches the checksum and recorded evidence from the first measured release and regenerates the report, figures, and an offline replay under `release/`. Open `release/index.html` in a browser. Both the split and the denominator appear beside the result. For a downloaded experiment, use `plato reproduce --quick --input results`; each model has its own subdirectory. To use a downloaded release asset, pass `--archive plato-observations.tar.gz`.

## Run the experiment on Modal

Install the local controller with `python -m pip install '.[modal]'` and authenticate with Modal. Run `python modal_app.py init --help` for the workspace, credit expiration and billing inputs. The local ledger reserves a bounded worst-case amount before each submission, accounts for unresolved attempts and compares L4 and L40S on complete worlds. Keep that ledger for the life of the run. The two GPUs are timed on the same workload. On the worker the models run in BF16, and a CPU function prepares pinned weights and verifies the CounterFact dataset checksum.

After initializing the ledger:

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
