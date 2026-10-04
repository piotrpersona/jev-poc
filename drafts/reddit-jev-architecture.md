# Platform: Reddit (r/MachineLearning) — [P] flair

**Title:** [P] Rebuilding TypeSafe's "Jev" System One architecture with a 149M encoder — what the schema-agnostic head has to look like

---

TypeSafe shipped Jev as a closed API: unstructured program state in, typed
probabilistic decisions out (`Choice` / `Score` / `Noul`), one parallel pass, no
token generation. No weights, no paper. So I rebuilt the *architecture* around
`answerdotai/ModernBERT-base` to see what the design is actually forced into.

The constraint that decides everything: **the question schema arrives at
inference time.** Option sets differ per request and a `Choice` can carry up to
255 options. That rules out fixed classification heads — you cannot pre-allocate
a logit layer for an option set you have not seen.

What falls out:

1. Encode the options **as text**, not as head rows. Option labels become
   embeddings like anything else.
2. Keep the state at token resolution and let each question read it through
   cross-attention — questions are the queries, state tokens are keys/values.
   Every question reads the state once, in parallel, and no question can see
   another's answer.
3. Score by dot product between the question's read and its own option
   embeddings, then softmax **per question**, with padded option slots masked
   to `-inf`.

The whole sampler is one `MultiheadAttention`, one linear probe, one `einsum`,
one masked softmax — 3M params on top of 149M.

Two things I did not expect going in:

**All three output primitives are the same mechanism.** `Choice`, `Score` and
`Noul` are all *masked softmax over a finite set of labelled options*. They
differ only in readback: `Score` adds an ordinal expectation over tiers, `Noul`
is arity-2. Putting `labels` and `decide(probs)` on each question class means a
fourth primitive touches one file and zero branches elsewhere.

**Type safety is structural, not probabilistic.** The sampler can only emit
indices into the option list it was handed, and only the question owning those
options reads them back. There is no string to parse, so a type error is not
"unlikely", it is unrepresentable. That is a genuinely different guarantee from
constrained decoding, which still has to be enforced during generation.

**The pricing makes architectural sense too.** Jev charges for input tokens and
nothing for output. The expensive half — prompts and option labels — is fixed
per deployment, so you encode the schema once and reuse it for every state. One
cache keyed on the serialised question set.

Measured on an M-series GPU, 4 questions of 3 different types in one pass
(77 / 6 / 3 / 2 options): logits `[8, 4, 77]`, ~77 ms per state, probability
mass `1.000000` per question, exactly zero on padded slots, zero type errors.

**What this is not:** the head is randomly initialised. No training, no RLCD, no
calibration numbers. Calibration ("85% confidence means 85% correct") is Jev's
actual product claim and this PoC cannot speak to it — confidences sit near
1/77 on the 77-way question, exactly as an untrained head should. The point was
the architecture, not the accuracy.

Questions I would like opinions on: is ordinal structure for `Score` better
taught through the loss, or by making tier embeddings themselves ordered? And
does anyone have a good reason to prefer a bilinear scorer over the plain
scaled dot product here?
