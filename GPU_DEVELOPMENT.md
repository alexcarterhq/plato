# GPU development protocol · 26 November 2026

Use eight development worlds from the existing generator, split `bench`, indices
10000 through 10007. Use all eight color interventions and both books. These
worlds are disjoint from the original validation, fresh validation and held-out
cohorts. The donor never sees the current rulebook. The bare receiver prompt is
identical across all eight variants. Demonstrations have fixed, unrelated records
and book assignments. No demonstration uses an experimental entity.

Evaluate the original format as a reference and exactly three amendments in
declared order: question and answer, two lookup tables, and a simulated Python
console. Give each amended format two fixed worked demonstrations. The Python
version uses the same greedy model decoder as the others. No interpreter,
constrained decoding, external lookup or target selection is used to answer.
Freeze all parameters and exact full-first-line scoring from the main study.
Measure both donor retrieval rates, full-evidence receiver accuracy by book and
the same-model BF16 packet patch by book at every preregistered candidate
recipient layer (22%, 32%, 40% of depth, round((depth - 1) × fraction)).
The receiver-only measurements are shared by the two donor comparisons.

For each amended format, calculate a score per donor equal to the minimum of
its retrieval accuracy / .95, both full-evidence accuracies / .95, and the two
oracle patch accuracies / .80 at the best *common* layer across books. Rank
formats by number of donors passing all these rate gates, then the larger
donor score, then the sum of donor scores, then the minimum full-evidence
accuracy across books, then the original format order given above. Report
the whole grid. The best format is diagnostic when all candidates fail.
Only an independent validation run may qualify a pair. The development layer
maximum is optimistic; the full study selects the actual layer using only
familiar factual swaps. An identity intervention must preserve logits on
unseen validation for either pair to qualify.

Keep raw prompts, completions, rulebook, original donor prompt and BF16 packet
hashes. Reserve one L40S with a twenty-minute worker ceiling in the shared
diagnostic allocation, and use no held-out data. Store one complete world per
remote compressed shard and aggregate them on download. The study output
contains model revisions, software versions, GPU receipt and source hashes.

