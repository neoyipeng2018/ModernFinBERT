# ModernFinBERT v2 training recipe (per-stage decisions)

Compact reference for the design choices in `01a_train_short`, `01c_finetune_medium`, `01b_finetune_long`. For full rationale see `research.md`; for the rollout plan see `plan.md`.

## Loss function

| Stage  | Class proportion (NEG/NEU/POS) | Loss              | Why                                                                     |
|--------|--------------------------------|-------------------|-------------------------------------------------------------------------|
| short  | 28 / 38 / 34                   | Cross-entropy     | Balanced — no reweighting needed.                                       |
| medium | 8  / 29 / 63                   | **Focal (γ=2.0)** | Heavy POS skew; inverse-freq weights `[4.18, 1.15, 0.53]` over-correct (NEG precision drops to 0.35). Focal targets *hard examples* (NEU↔POS confusion) without flipping the decision boundary. |
| long   | 22 / 26 / 52                   | Cross-entropy     | Closer to balanced. Use focal only if Stage 2 needed γ≥3 to pass.       |

**Fallback escalation if focal underperforms** at Stage 2: γ=3.0 first; then sqrt-inverse weights `[2.05, 1.07, 0.73]` + label-smoothing 0.05. **Never** use the original `[4.18, 1.15, 0.53]` weights — that configuration produced the v2 regression.

## Best-checkpoint selection

| Stage  | `metric_for_best_model`  | `load_best_model_at_end` | Why                                                                |
|--------|--------------------------|--------------------------|--------------------------------------------------------------------|
| short  | `eval_loss`              | `True`                   | Balanced labels — eval_loss tracks F1 well.                        |
| medium | `eval_macro_f1`          | `True`                   | Skewed labels + focal loss distort eval_loss; F1 is the goal.      |
| long   | `eval_macro_f1`          | `LOAD_BEST_FALLBACK` toggle | Toggle off if the unsloth classifier-reload-after-load-best bug fires; manual best-checkpoint selection from `save_steps=25` artifacts then runs in cell-11. |

## Eval cadence and subset sizing

Full-validation eval at MAX_LENGTH≥4096 takes ~10 minutes per pass on T4; intra-epoch eval requires a subset.

| Stage  | Total optimizer steps | `eval_steps` | Eval subset (`stratified_subset` n_per_class) | Subset rows | Per-eval cost |
|--------|----------------------:|-------------:|------------------------------------------------:|------------:|--------------:|
| short  | 813                   | (per epoch — fast)  | full val                                | 4,325       | ~16 s         |
| medium | 461                   | 50           | 167                                             | ~500        | ~70 s         |
| long   | 250                   | 10           | 100                                             | ~300        | ~50 s         |

Then re-eval on full validation once at end of training (`trainer.eval_dataset = tokenized_*["validation"]; trainer.evaluate()`).

## LoRA capacity

| Stage  | r  | α   | Trainable params |
|--------|---:|----:|-----------------:|
| short  | 16 | 32  | ~3.4M (2.21%)    |
| medium | 32 | 64  | ~6.7M (4.4%)     |
| long   | 32 | 64  | ~6.7M (4.4%)     |

Keep α/r = 2 across stages. Don't bump short — it works.

## Context length

| Stage  | MAX_LENGTH | Notes                                                                                |
|--------|-----------:|--------------------------------------------------------------------------------------|
| short  | 512        | Median ~44 tokens; never the bottleneck.                                             |
| medium | 4096       | 0% truncation against 500–3072-token chunks (median 2613).                           |
| long   | **4096**   | Bumped down from 6144 — at 6144 the prior recipe truncated 52.8% of training rows.   |

Hard assertion in `cell-8`: `assert trunc_pct < 0.10`. If this fires, fix the chunker before raising MAX_LENGTH.

## Batch and grad-accum

| Stage  | per-device batch | grad-accum | Effective batch | Optimizer steps / epoch |
|--------|-----------------:|-----------:|----------------:|------------------------:|
| short  | 32 (×2 GPU)      | 2          | 128             | 271                     |
| medium | 16               | 2          | 32 (note: was 64 in v2 — kept) | 461 (calculated from log) |
| long   | 8                | 4          | 32              | 250 (was 125 at MAX=6144 with batch=4×8) |

Don't drop the effective batch below 32 at 4096+ tokens; gradient noise dominates. If OOM at rank=32, drop per-device batch first, raise grad-accum to compensate.

## T4-specific gotchas

- `attn_implementation="sdpa"` — must be set explicitly at long stage. ModernBERT's FA2 path produces NaN losses on sm_75; the silent fallback to xformers is fine at short/medium but the cost of a silent bug at 4K+ tokens is large.
- `UNSLOTH_DISABLE_FAST_GENERATION=1` — required for classification (Unsloth's fast-generation path interferes with `AutoModelForSequenceClassification`).
- `TORCHDYNAMO_DISABLE=1` — required at medium/long stages. Without it, dynamo recompiles per variable-length batch, which combined with `group_by_length=True` is catastrophic.
- `optim="adamw_8bit"` + `use_gradient_checkpointing="unsloth"` — required to fit 4096+ tokens × batch ≥ 8 in 14.5 GiB.

## Stage-1 → Stage-2 → Stage-3 boundary

Each stage merges its LoRA into the backbone (`model.merge_and_unload()`) and pushes the merged model to HF Hub. The next stage attaches a *fresh* LoRA on top. This is the "merge-then-push-then-fresh-LoRA" continued-fine-tuning pattern. Catastrophic forgetting concentrates at the merge point — always run a regression check on prior-stage test sets after each new stage.

## When to stop and ship Stage 2 only

If after Phase 6 (long-stage rerun), the long-test macro F1 fails to exceed Stage 2's medium F1 by ≥ 0.02, do not push the long-stage model. The long stage has not earned its slot in the pipeline. Push the Stage 2 model as canonical `neoyipeng/ModernFinBERT-v2` and document the long-stage ablation in `research.md`.
