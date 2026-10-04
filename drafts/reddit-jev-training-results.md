# Platform: Reddit (r/MachineLearning) — [P] flair

**Post this after the architecture post, not alongside it. Space by a few days.**

**Title:** [P] What a 3M-param schema-agnostic head can learn on a frozen encoder: 0.86 macro-F1 over 77 classes, 0.62 on binary irony

---

Follow-up to my rebuild of TypeSafe's Jev architecture (schema arrives at
inference time, options encoded as text, one parallel pass). Last time the head
was randomly initialised and I made no accuracy claim. I have now trained it, so
here are the numbers and the one result I did not expect.

**Setup.** `answerdotai/ModernBERT-base` (149M) **frozen**, trained only the 3M
sampler on top: one `MultiheadAttention`, a linear probe, a masked softmax over
the option axis. Four questions of three different types answered in a single
pass — banking77 intent (77 options, `Choice`), emotion (6, `Choice`),
tweet_eval sentiment (3, `Score`), tweet_eval irony (2, `Noul`).

**The multi-task bit that makes this work.** Every state is asked all four
questions, but a state only carries a gold label for the dataset it came from.
So the loss gathers the row of the question that owns the label and leaves the
other three untouched. The four questions share the sampler, so they still train
each other. Padded option slots arrive as `-inf`, which makes cross-entropy over
the full 77-wide option axis provably identical to cross-entropy over a 2-option
question — that is a unit test in the repo, not an assumption.

**Results** (test splits, 18,148 states, best checkpoint by mean macro-F1 across
the four questions, early stopped at epoch 7 with epoch 4 selected):

| question | options | accuracy | macro-F1 | support |
| --- | --- | --- | --- | --- |
| intent | 77 | 0.859 | 0.860 | 3080 |
| emotion | 6 | 0.853 | 0.797 | 2000 |
| sentiment | 3 | 0.658 | 0.645 | 12284 |
| irony | 2 | 0.634 | 0.624 | 784 |

Mean macro-F1 0.732.

**The surprise is the ordering.** I expected difficulty to track the number of
options. It does the opposite: the 77-way question is the *easiest* and the
balanced binary one is the hardest. Banking intents are lexically separable, so
a cross-attention read of frozen features finds them and a dot product against
77 text embeddings is enough to rank them. Irony needs tone, and frozen
ModernBERT features apparently do not carry tone — 0.62 on a balanced two-class
task is close to the floor that matters. Sentiment sits in the same bucket at
0.65.

So the ceiling here is the frozen encoder, not the head, and it is task-shaped
rather than arity-shaped. Which is mildly encouraging for the architecture: the
part people assume is the hard part (scoring an option set you have never seen,
at arity 77) is not where it fails.

**Things worth knowing if you reproduce it.** Selection on mean macro-F1 across
questions rather than pooled accuracy, otherwise the 12k-row sentiment split
drowns out the 784-row irony one. And on Apple `mps` the first epoch cost 20
minutes against ~7 for every later one, purely Metal kernel warmup — I quoted an
ETA off a short benchmark and was 3x wrong.

**Still not a calibration claim.** No ECE, no reliability diagram, no
temperature scaling. Calibration is Jev's actual product claim and this measures
accuracy only. Also still no RLCD — plain cross-entropy on gold labels, where
Jev's own recipe is a preference method over decisions.

Repo: https://github.com/piotrpersona/jev-poc

The open question I would most like opinions on: if the frozen features really
are the binding constraint for tone-shaped tasks, is a LoRA on the backbone the
right next step, or does unfreezing break the thing that makes this
architecture cheap — the schema embedding cache that is only valid *because*
the backbone never moves?
