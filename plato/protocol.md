# Plato experimental protocol · 26 November 2026

## Question

Can a map fitted on familiar facts carry a newly randomized category between two frozen model families? The primary pair is Qwen2.5-1.5B → SmolLM2-1.7B. The replication replaces the donor with Pythia-1.4B-deduped. Immutable upstream revisions are in config.py. All generations are greedy and all weights frozen. This is an independent extension of [Plato’s Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key). Related work includes [StateBridge](https://arxiv.org/abs/2608.13317) and [the mismatched-cache audit](https://arxiv.org/abs/2608.04893).

## Factual transfer

Use the original CounterFact array (21,919 records, SHA-256 d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f). Take target_true. Split subject groups before fitting. Screen both frozen models by the same full-phrase scoring rule. Target 3,000 training, 500 validation and 1,000 eligible test facts, disclosing exclusions, available sample sizes and relation coverage. Use same-relation swaps with different answers. Fit means, global Frobenius scale and orthogonal Procrustes on paired post-block last-subject-token states. With rank-deficient training data, use only identified directions and report rank. Select from 22%, 32% and 40% of depth in each model by validation intervention accuracy. Controls include own alternate state, no intervention, shuffled training pairs, mean, random, and wrong-subject packets as implemented in facts.py. Check identities, known-answer substitution and held-out paraphrases. Report failures explicitly.

## Private-information audit

Generate 512 contextual training worlds, 64 validation worlds, 250 held-out worlds and 64 fresh robustness worlds with a changed prompt template. Each world has four random identifiers, eight possible private categories, and two randomly drawn receiver rulebooks constrained to give different answers for every category. The sender sees the query and evidence, the receiver sees the query and one rulebook. The sender never sees either rulebook. All eight variations change only the queried record. Read one vector at the fixed trailing marker after the evidence. Patch the recipient marker before its answer suffix. Reuse identical BF16 packet bytes across both rulebooks.

Compare the untouched factual map with a separate Procrustes map fitted to paired representations of identical private training evidence. Hold the factual layer choices fixed. Evaluate correct and wrong packets for both maps. The wrong condition uses a seeded fixed-point-free permutation within each world. Include no message, receiver's own oracle state, full evidence, the sender's generated category and a one-byte public category code with a deterministic lookup. The lookup is a conventional systems baseline. Dense packets use little-endian BF16. Retain original completions. Match the whole first nonempty answer line after case normalization and removal of a terminal full stop or exclamation mark.

## Decision rules

Require maximum absolute identity-logit error at most 0.0001, sender and full-evidence accuracy each at least 95%, and oracle-state accuracy at least 80% under each validation rulebook before evaluating held-out worlds. Failed validation triggers reporting and diagnosis using validation data. Publish factual-gate outcomes independently. Check the controls again in the held-out results.

The primary strong-result threshold is factual-map accuracy of at least 75% on each rulebook, a correct-minus-wrong difference of at least 20 percentage points with a positive paired 95% interval on each book, and at least 60% answering both changed rulebooks correctly with the same packet. The 60% joint endpoint operationalizes the second-rulebook check. Report the separate contextual diagnostic using the same measures. Use 10,000 percentile bootstrap draws over whole worlds with seed 73639. Variants and rulebooks share their world cluster. Report estimates and intervals for weaker results. A failed positive control is a diagnosis; a failed effect-size threshold alone does not establish absence of information transfer.

## Measurement and release

Time all experimental lanes, with CUDA synchronization, in batches of eight counterfactuals. Per-case warm latency sums the measured sender, map, in-process handoff and receiver batch stages where applicable. Per-case GPU-seconds allocate measured GPU-stage wall time across the batch. The complete-world benchmark separately times wall-clock execution. These measurements describe colocated inference. Record hardware, dependency versions, packet bytes, model and code hashes, and account-backed charges separately from local cost estimates. Use the L40S when its measured cost per complete world is lower than the L4. Quick-mode outputs are engineering checks. Publish a result or a diagnosed control failure, with the data supporting that statement.
