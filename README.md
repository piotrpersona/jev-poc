# jev-poc

A proof of concept of a **System One** model in the shape of TypeSafe's Jev:
unstructured program state goes in, typed probabilistic decisions come out, in
a single parallel pass. No token generation, no output parsing, no way to emit
a value that is not in the schema.

The real Jev is a closed API with no open weights, so this rebuilds the
*architecture* around a small HuggingFace encoder
(`answerdotai/ModernBERT-base`, 149M params).

The head trains: `make train` fits it on all four datasets at once, supervised
on their gold labels, and tracks the run in MLflow. The backbone stays frozen,
runs are canary-sized by default, and calibration is still out of scope - **the
numbers a short run reports are not an accuracy claim**. See
[Training](#training) and [Not implemented](#not-implemented).

## The architecture

The interesting constraint is that the question schema arrives **at inference
time**: option sets differ per request, and a question may carry up to 255
options. Fixed classification heads cannot express that, so options are encoded
as *text* and scored against a question-conditioned read of the state.

```
program state (text)                     question schema (QuestionSet)
        |                                   |                |
        |                              prompts           option labels
        v                                   v                v
 ┌──────────────┐                     ┌─────────────────────────────┐
 │  backbone    │  per token          │  backbone, mean-pooled      │  cached per schema
 │  [B, T, H]   │                     │  [Q, H]        [Q, O, H]    │
 └──────┬───────┘                     └──────┬──────────────┬───────┘
        │  keys / values                     │ queries      │
        └──────────────┬─────────────────────┘              │
                       v                                    │
              cross-attention: every question               │
              reads the state once, in parallel             │
                       │  [B, Q, H]                         │
                       └────────────┬───────────────────────┘
                                    v
                         dot product + per-question
                         masked softmax   [B, Q, O]
                                    v
                      question.decide(probs) -> typed decision
```

`src/jev/head.py` is the whole sampler: one `MultiheadAttention`, one linear
probe, one `einsum`, one masked softmax. 3M parameters on top of the backbone.

### One primitive, three types

Jev exposes `Choice`, `Score` and `Noul`. They collapse into a single mechanism
- *masked softmax over a finite set of labelled options* - and differ only in
how the probability vector is read back:

| type | options | decision |
| --- | --- | --- |
| `Choice` | 2-255 labels | argmax label + confidence + distribution |
| `Score` | 2-10 ordered tiers | tier + **ordinal expectation** over tiers |
| `Noul` | `no`, `yes` | bool + `p_yes` |

Each question class owns its own `labels` (what the encoder sees) and
`decide()` (how probabilities become a typed value), so adding a fourth
primitive touches one file and no branch elsewhere. `src/jev/schema.py`.

### Why this is type-safe

The sampler can only produce indices into the option set it was handed, and
only the question that owns those options reads them back. There is no string
to parse and no schema to re-validate after the fact, so a type error is not
"unlikely", it is unrepresentable. Pydantic rejects an invalid *schema* up
front (duplicate options, >255 options, tiers outside 2-10, rubric that does
not name every tier).

Padded option slots are masked to `-inf` before the softmax, so a 2-option
question in a batch whose widest question has 77 options still normalises over
exactly 2.

### Why the schema cache matters

Jev charges for input tokens and nothing for output. That works because the
expensive half - the schema - is fixed per deployment: question prompts and
option labels are encoded once and reused for every state. `JevModel` keys
that cache on the serialised question set.

## Run it

```sh
make install                  # uv sync
make test                     # offline suite, no weights downloaded
make demo                     # 100-sample canary on banking77
make train                    # 100-sample canary per task per split, tracked in MLflow
make e2e                      # same paths over real ModernBERT weights
make gen                      # export JSON Schema of the typed API
```

```sh
uv run python -m jev.cli demo --task sentiment --limit 10% --show 2
uv run python -m jev.cli train --limit all --epochs 20
uv run python -m jev.cli demo --task sentiment --checkpoint out/<run-id>/best_head.pt
```

`--limit` takes a count (`100`), a percentage (`10%`) or `all`, and defaults to
a 100-sample canary. Samples are drawn by a seeded round-robin over labels, so
even a small canary sees every class.

Config lives in `.env` (copy `.env.example`): backbone, device
(`auto` picks mps/cuda/cpu and the run logs which), token limits, batch size,
and `MLFLOW_TRACKING_URI` (defaults to a local `sqlite:///mlflow.db`, which is
also what the model registry needs).

A demo run reports, per request: schema arities, the stratified class
breakdown, parameter counts, the `[B, Q, O]` logit shape, ms per state, the
mass/padding invariant per question, and the type-error count.

```
schema      arities=[77, 6, 3, 2] types=['choice', 'choice', 'score', 'noul']
forward     logits=(8, 4, 77) ms_per_state=77.4
invariant   'intent: mass=1.000000..1.000000 padding_max=0.0e+00 options=77'
invariant   'irony: mass=1.000000..1.000000 padding_max=0.0e+00 options=2'
typesafe    type_errors=0 verdicts=24
```

Four questions, four different types, 77 to 2 options, one forward pass.

## Data

Real HuggingFace classification sets, each re-cast as the question type its
label column already is. Every state is asked **all four** questions at once -
which is the Jev shape - while gold labels only exist for the set a state came
from.

| key | dataset | type | options | train | validation | test |
| --- | --- | --- | --- | --- | --- | --- |
| `intent` | `legacy-datasets/banking77` | `choice` | 77 banking intents | 9003 | 1000 * | 3080 |
| `emotion` | `dair-ai/emotion` | `choice` | 6 emotions | 16000 | 2000 | 2000 |
| `sentiment` | `cardiffnlp/tweet_eval` (sentiment) | `score` | 3 tiers | 45615 | 2000 | 12284 |
| `irony` | `cardiffnlp/tweet_eval` (irony) | `noul` | yes / no | 2862 | 955 | 784 |

\* banking77 ships no validation split, so 10% of train is carved out by the
same seeded stratified round-robin used for sampling. The two sides never
overlap and the carve is reproducible from `--seed` alone.

These sets carry no record date, so there is no time range to scope and no
date stratification to check; stratification here is by label only. Every
split is drawn label-stratified, so even a 100-sample canary covers every
class it can - and `intent` is the imbalanced one by construction: 77 classes
over a 100-sample canary leaves 1-2 states per class, which is why its canary
f1 stays near the floor.

## Training

```sh
uv run python -m jev.cli train --limit 100          # canary, the default
uv run python -m jev.cli train --limit all --epochs 20
```

`src/jev/training.py`. One optimiser over the 3M-parameter sampler; the
backbone is frozen, which is what keeps the schema embeddings - and therefore
the schema cache - constant across the whole run, and keeps the autograd graph
off the 149M-parameter encoder.

Every state is asked all four questions in one pass, but a state only carries
a gold label for the task it came from, so the loss supervises that one
question's row and leaves the other three untouched. The four questions share
the sampler, so they still train each other. Padded option slots arrive as
`-inf`, which makes cross-entropy over the full 77-wide option axis identical
to cross-entropy over a 2-option question (`tests/test_training.py`).

- **Selection**: early stopping and the best checkpoint both go by
  `validation/mean_f1_macro` - per-task macro-F1, averaged over the four
  questions, so the 77-way `intent` question cannot dominate and class
  imbalance does not flatter the binary ones. `--patience` epochs without an
  improvement ends the run; the best head is reloaded before the test pass.
- **Checkpoint**: `out/<run-id>/best_head.pt` holds the head weights, the
  backbone id and the question set they were trained against. `jev demo
  --checkpoint` refuses a checkpoint trained on a different schema, because a
  head only means something against the option embeddings it learned to score.
- **Tracking**: every run goes to MLflow - params (full config, seed, limit,
  arities, parameter counts, selection criterion), per-epoch train and
  validation metrics (loss, accuracy, precision/recall/f1 macro, f1 weighted,
  support) per question, tags (canary vs full, git commit, datasets, device),
  and artifacts: the dataset breakdown, per-class precision/recall/f1/support
  as JSON for every epoch and split, a plotly confusion matrix per question
  over the test split, the checkpoint itself, and the head under the MLflow
  pytorch flavor, registered as `jev-head`.
- **Canary first**: `--limit` defaults to 100 samples per task per split and
  the run logs that it is a canary; `--limit all` is the full run.

```
device      device=mps torch=2.14.1
canary      limit=100 detail='per task per split, rerun with --limit all'
schema      arities=[77, 6, 3, 2] types=['choice', 'choice', 'score', 'noul']
split       split=train task=intent samples=100 classes=77 min_class=1 max_class=2
train       epoch=4 task=irony loss=0.2564 accuracy=0.91 f1_macro=0.91 support=100.0
validation  epoch=4 task=irony loss=0.9821 accuracy=0.52 f1_macro=0.4956 support=100.0
no improvement best=0.2927 epoch=4 patience_left=1
```

That is a 100-sample canary over four epochs: enough to show the loop learns
and then overfits, not enough to mean anything. Run `--limit all` for numbers.

## Not implemented

- **RLCD / preference training.** The head is trained with plain
  cross-entropy on gold labels. Jev's own recipe is a preference method over
  decisions, which is a different objective and a different data shape.
- **Calibration.** No ECE, no reliability diagram, no temperature scaling.
  Calibration is Jev's actual product claim and cannot be assessed here.
- **Ordinal loss for `Score`.** The expectation is computed and trained
  through cross-entropy, but nothing teaches the model that tier 4 is closer
  to 5 than to 1.
- **Backbone finetuning.** Frozen by design here, so the ceiling is whatever a
  cross-attention read of fixed ModernBERT features can reach. No DDP either -
  one device, one process.
- **Batched heterogeneous schemas.** One question set per request, as the API
  shape implies; `Q` is never padded.

## Sources

- [Jev, System One models (DataCamp)](https://www.datacamp.com/blog/system-one-models-jev)
- [TypeSafe Jev: decision model vs auto-regressive LLMs](https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model)
- [Introducing Jev (YourStory)](https://yourstory.com/ai-story/what-is-jev-typesafe-ai-decision-model)
