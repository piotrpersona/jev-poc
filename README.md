# jev-poc

A proof of concept of a **System One** model in the shape of TypeSafe's Jev:
unstructured program state goes in, typed probabilistic decisions come out, in
a single parallel pass. No token generation, no output parsing, no way to emit
a value that is not in the schema.

The real Jev is a closed API with no open weights, so this rebuilds the
*architecture* around a small HuggingFace encoder
(`answerdotai/ModernBERT-base`, 149M params).

**The head is randomly initialised. This repo makes no accuracy claim** - it
demonstrates the architecture, its shapes, and the invariants that hold by
construction. Training and calibration are deliberately out of scope (see
[Not implemented](#not-implemented)).

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
make e2e                      # same path over real ModernBERT weights
make gen                      # export JSON Schema of the typed API
```

```sh
uv run python -m jev.cli demo --task sentiment --limit 10% --show 2
```

`--limit` takes a count (`100`), a percentage (`10%`) or `all`, and defaults to
a 100-sample canary. Samples are drawn by a seeded round-robin over labels, so
even a small canary sees every class.

Config lives in `.env` (copy `.env.example`): backbone, device
(`auto` picks mps/cuda/cpu and the run logs which), token limits, batch size.

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

| key | dataset | split | type | options |
| --- | --- | --- | --- | --- |
| `intent` | `legacy-datasets/banking77` | test (3080) | `choice` | 77 banking intents |
| `emotion` | `dair-ai/emotion` | test (2000) | `choice` | 6 emotions |
| `sentiment` | `cardiffnlp/tweet_eval` (sentiment) | test (12284) | `score` | 3 tiers |
| `irony` | `cardiffnlp/tweet_eval` (irony) | test (784) | `noul` | yes / no |

These sets carry no record date, so there is no time range to scope and no
date stratification to check; stratification here is by label only.

## Not implemented

- **RLCD / any training.** The head is random init, so confidences are
  uninformative - `intent` sits near 1/77. Training with a proper scoring rule
  is what would make them mean anything.
- **Calibration.** No ECE, no reliability diagram, no temperature scaling.
  Calibration is Jev's actual product claim and cannot be assessed here.
- **Ordinal loss for `Score`.** The expectation is computed, but nothing
  teaches the model that tier 4 is closer to 5 than to 1.
- **Batched heterogeneous schemas.** One question set per request, as the API
  shape implies; `Q` is never padded.

## Sources

- [Jev, System One models (DataCamp)](https://www.datacamp.com/blog/system-one-models-jev)
- [TypeSafe Jev: decision model vs auto-regressive LLMs](https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model)
- [Introducing Jev (YourStory)](https://yourstory.com/ai-story/what-is-jev-typesafe-ai-decision-model)
