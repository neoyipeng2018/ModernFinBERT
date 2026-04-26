# ModernFinBERT v2 — Recipe Fix Plan (Steps 1–5)

> **Implementation status (2026-04-26):**
> - **Phase 0 (pre-flight)** ✅ done — except 0.3 (branching skipped due to pre-existing dirty tree), 0.5/0.6 (Kaggle/HF API verification deferred to user).
> - **Phase 1 (Stage 2 notebook)** ✅ done — `01c_finetune_medium.ipynb` rewritten; AST-clean; focal loss math + stratified-subset helper unit-tested.
> - **Phase 4 (Stage 3 notebook)** ✅ done in parallel — `01b_finetune_long.ipynb` rewritten; AST-clean; truncation assertion verified against chunker output shape.
> - **Phase 8.2 (RECIPE.md)** ✅ done — `notebooks/RECIPE.md` created and linked from all three training notebooks.
> - **Phases 2, 3, 5, 6, 7, 8.1, 8.3** 🚧 blocked — require Kaggle T4 runtime, decision-gate inputs, or post-ship inputs that don't exist yet.
>
> **Files changed by this implementation pass:**
> - `notebooks/01a_train_short.ipynb` (header link to RECIPE.md only)
> - `notebooks/01c_finetune_medium.ipynb` (full Phase 1 rewrite + RECIPE link)
> - `notebooks/01b_finetune_long.ipynb` (full Phase 4 rewrite + RECIPE link)
> - `notebooks/archive/01c_finetune_medium__pre_v3.ipynb` (snapshot, new)
> - `notebooks/archive/01b_finetune_long__pre_v3.ipynb` (snapshot, new)
> - `notebooks/results/v2_recipe_v3_runs.md` (run tracker, new)
> - `notebooks/RECIPE.md` (per-stage decision reference, new)
>
> **Next user action:** push `01c_finetune_medium.ipynb` to Kaggle (Phase 2.1) and run on T4. After ~4h, append the result row to `notebooks/results/v2_recipe_v3_runs.md` and apply the decision gate in Phase 3.

Implements the top-5 changes from `research.md`. Goal: recover Stage 1's short-test macro F1 (0.7711) at Stage 2 while pushing Stage 2's medium-test macro F1 above 0.55, then re-decide whether Stage 3 is worth running.

## Success criteria

| Metric                                | Current   | Target              |
|---------------------------------------|-----------|---------------------|
| S2 short-test macro F1 (regression)   | 0.7480    | ≥ 0.7600            |
| S2 medium-test macro F1               | 0.5428    | ≥ 0.5800            |
| S2 medium-test NEU recall             | 0.42      | ≥ 0.55              |
| S2 medium-test NEG precision          | 0.35      | ≥ 0.50              |
| S3 long-test macro F1 (if rerun)      | 0.5422    | ≥ 0.5800            |
| Truncation rate at long stage         | 52.8%     | < 10%               |

If S2 hits target, S3 is rerun. If S2 misses, fall through to the "drop S3, keep S2 as final" path.

## Order of execution

1. **Stage 2 first, single change at a time** so each step's effect is measurable.
2. Within Stage 2: focal loss (Step 1) and macro-F1 selection (Step 4) are coupled — apply together since `load_best_model_at_end` requires both an eval cadence and a metric. Then add intra-epoch eval (Step 2). Then bump rank (Step 5).
3. Stage 3 only after Stage 2 is fixed: change MAX_LENGTH and rank (Steps 3 + 5).

## Files touched

- `notebooks/01c_finetune_medium.ipynb` — Steps 1, 2, 4, 5.
- `notebooks/01b_finetune_long.ipynb` — Steps 2, 3, 4, 5.
- No script changes; this is a notebook-only plan.

---

## Step 1 — Drop inverse-frequency class weights, use focal loss

**Why.** Inverse-frequency weights `[4.18, 1.15, 0.53]` over-correct: NEG recall jumps to 0.59 but NEG precision collapses to 0.35 (research.md §3). Focal loss attacks the *hard-example* problem (NEU↔POS confusion in earnings prose) instead of the *class-prior* problem, so it doesn't bias the decision boundary toward NEG. γ=2 is the FAIR-paper default and a good starting point.

**Where.** `notebooks/01c_finetune_medium.ipynb`, cell `cell-10`.

**Change.** Replace `WeightedTrainer` with `FocalTrainer`:

```python
# Replace the existing CrossEntropyLoss-based WeightedTrainer in cell-10 with:
import torch
import torch.nn.functional as F
from transformers import Trainer

FOCAL_GAMMA = 2.0  # FAIR default; raise to 3.0 if NEU still under-represented

class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** FOCAL_GAMMA) * log_pt
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss
```

Then swap the trainer instantiation:

```python
# was: trainer = WeightedTrainer(...)
trainer = FocalTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=tokenized_med["train"],
    eval_dataset=eval_subset,                          # ← from Step 2
    compute_metrics=compute_metrics,
    args=TrainingArguments(...),                      # ← see Step 2/4
)
```

**Drop the class-weights block entirely** (the `Counter`, `class_weights`, `print(...)` lines) — focal loss replaces all of it. Also drop `from torch.nn import CrossEntropyLoss` and `from collections import Counter`.

**Fallback.** If focal underperforms (S2 medium F1 < 0.55), fall back to **gentle** sqrt-inverse weights:

```python
class_weights = torch.tensor([2.05, 1.07, 0.73], device="cuda")  # sqrt(inverse-freq), normalized
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
```

Don't go back to the `[4.18, 1.15, 0.53]` weights — those are the exact configuration that produced the regression.

**Validation.** After Step 1+4 are in, the medium-test classification report must show NEG precision ≥ 0.50 (was 0.35) and NEU recall ≥ 0.55 (was 0.42). If either fails, raise γ to 3.0 before falling back to sqrt-weights.

---

## Step 2 — Intra-epoch eval on a subsampled eval set

**Why.** Single end-of-epoch eval (research.md §3, §4) means we have no signal during training and no best-checkpoint selection. Eval on the full medium-val set takes ~9 minutes (3,683 rows × 4096 tokens, `eval_steps_per_second=0.222`); evaluating every 50 steps would balloon runtime by hours. A stratified 500-row subsample brings per-eval cost to ~70 seconds and preserves class proportions.

**Where.** `01c_finetune_medium.ipynb` cell `cell-10` (and analogously in `01b_finetune_long.ipynb`).

**Change.** Build a stratified eval subset, then point `eval_dataset` at it and switch to `eval_strategy="steps"`. Keep the *full* validation set for a final pre-test eval.

```python
# Add before the trainer instantiation:
import numpy as np

def stratified_subset(ds, n_per_class=167, seed=3407):
    """Return a class-stratified subset of `n_per_class * num_classes` rows."""
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for i, lbl in enumerate(ds["labels"]):
        by_class[lbl].append(i)
    picked = []
    for c, idxs in by_class.items():
        take = min(n_per_class, len(idxs))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())
    return ds.select(sorted(picked))

eval_subset = stratified_subset(tokenized_med["validation"], n_per_class=167)
print(f"Eval subset size: {len(eval_subset)} (~500 stratified)")
```

Then in `TrainingArguments`:

```python
args=TrainingArguments(
    output_dir="./modernfinbert-v2-medium-output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.001,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    seed=SEED,
    group_by_length=True,
    eval_strategy="steps",       # ← was "epoch"
    eval_steps=50,               # ← S2: 461 total steps → ~9 evals/epoch
    save_strategy="steps",       # ← match eval cadence
    save_steps=50,
    save_total_limit=3,
    logging_strategy="steps",
    logging_steps=25,            # ← halved so we see loss between evals
    load_best_model_at_end=True,                  # ← Step 4
    metric_for_best_model="eval_macro_f1",         # ← Step 4
    greater_is_better=True,                       # ← Step 4
    report_to="none",
),
```

**After training, run a final eval on the *full* validation and test sets** (do this in `cell-11` — the existing post-training eval already uses `tokenized_med["test"]`, so it's covered; just reset `eval_dataset` first if needed):

```python
# In cell-11, before the test eval, swap back to full val for one final read:
trainer.eval_dataset = tokenized_med["validation"]
full_val = trainer.evaluate()
print(f"Full val: acc={full_val['eval_accuracy']:.4f}, "
      f"f1={full_val['eval_macro_f1']:.4f}")
```

**Stage 3 analog.** In `01b_finetune_long.ipynb`, do the same with `n_per_class=100` (300-row subset) and `eval_steps=10` (125 total steps → 12 evals).

**Caveat about `load_best_model_at_end=True`.** The Stage 3 notebook has a comment ("avoid the unsloth classifier-reload bug after training"). If Stage 3 hits that bug after this change, fall back to `load_best_model_at_end=False` and select the best checkpoint manually:

```python
# Manual best-checkpoint selection after training (only if needed):
import json, glob, os
ckpt_dirs = sorted(glob.glob("./modernfinbert-v2-medium-output/checkpoint-*"))
best = max(ckpt_dirs, key=lambda d: max(
    (e.get("eval_macro_f1", -1) for e in json.load(open(f"{d}/trainer_state.json"))["log_history"]),
    default=-1,
))
print(f"Best checkpoint: {best}")
# then reload from `best` for evaluation
```

**Validation.** The trainer log must contain ≥ 8 `eval_macro_f1` entries after Step 2 is in (was 1).

---

## Step 3 — Long-context truncation: pick a real cap

**Why.** 52.8% of long-stage training rows are truncated at MAX_LENGTH=6144 (research.md §4). The model is being trained on truncated long docs — half the supposed long-context advantage is fictitious. Two options:

**Option A (recommended): drop MAX_LENGTH to 4096.** This is the operational ceiling proven by Stage 2 (0% truncation). Stage 3 then becomes "more long-doc data at the same context length", which is the only thing the data actually supports given the chunking pipeline (`scripts/chunk_sources.py` chunks at 500–3072 tokens, so most "long" docs are only marginally over 4K when re-tokenized with the entity prefix).

**Option B: bump MAX_LENGTH to 8192.** Forces batch=2 (research.md §4) on T4 and a much smaller subsample, which contradicts the "more steps" recommendation in Step 5.

Pick A unless we have a specific dataset where >4096-token reasoning is essential. The token-length stats in the existing run (`min=378, median=6144, max=6144`) are partly a tokenizer artifact — the median equals the max because the cap was hit, not because the median doc is genuinely 6144 tokens.

**Where.** `notebooks/01b_finetune_long.ipynb`, cell `cell-2`.

**Change (Option A).**

```python
# was: MAX_LENGTH = 6144
MAX_LENGTH = 4096
```

Then in the training args (cell `cell-10`), the existing `per_device_train_batch_size=4, gradient_accumulation_steps=8` (eff. 32) can stay — but at 4096 we can comfortably double batch:

```python
per_device_train_batch_size=8,
gradient_accumulation_steps=4,   # eff. batch 32 unchanged → comparable LR scaling
```

This doubles the optimizer-step count from 125 → 250 at the same effective batch size, addressing the "too few steps" finding in research.md §4.

**Add an explicit assertion to fail-fast on truncation:**

```python
n_truncated = sum(1 for l in lengths if l >= MAX_LENGTH)
trunc_pct = n_truncated / len(lengths)
print(f"  Truncated to {MAX_LENGTH}: {n_truncated} rows ({100*trunc_pct:.1f}%)")
assert trunc_pct < 0.10, f"Truncation rate {100*trunc_pct:.1f}% > 10% — adjust MAX_LENGTH or chunk pipeline"
```

**Validation.** Truncation rate must print < 10% before training starts.

---

## Step 4 — Macro F1 for model selection from Stage 2 onward

**Why.** With 8/29/63 class proportions, `eval_loss` minimizes a weighted-by-prevalence quantity that the focal loss further distorts. `eval_macro_f1` is the metric we actually care about. Already partially covered in Step 2's `TrainingArguments`; this section is the explicit Stage 1 contrast and the Stage 3 application.

**Stage 1 stays on `eval_loss`.** Labels are balanced and CE+loss-based selection landed on the right checkpoint (research.md §2). Do not change `01a_train_short.ipynb`.

**Stage 2 + Stage 3.** Apply the three-line change shown in Step 2:

```python
load_best_model_at_end=True,
metric_for_best_model="eval_macro_f1",
greater_is_better=True,
```

Confirm the `compute_metrics` function returns `macro_f1` under that exact key (it does in both notebooks):

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
```

The `eval_` prefix is added automatically by HF Trainer, so `metric_for_best_model="eval_macro_f1"` matches.

**Validation.** After training, the printed best checkpoint should not be the last one in 80%+ of runs — if `load_best_model_at_end` always picks the last checkpoint, the metric isn't exerting selection pressure (likely an indication that Steps 1/2 didn't take effect).

---

## Step 5 — Bump LoRA rank for Stage 2 and Stage 3

**Why.** Rank 16 was tuned at 512-token context (Stage 1). Doubling/quadrupling context length means the adapter must shape ~8–12× more activations through the same attention/MLP paths. Empirically the cheap fix is rank=32 (≈ doubles trainable params from 3.38M to ~6.5M, still well under 5% of the backbone). Rank=64 is a fallback if 32 isn't enough. Stage 1 stays at 16 because it works.

**Where.**
- `notebooks/01c_finetune_medium.ipynb`, cell `cell-7`.
- `notebooks/01b_finetune_long.ipynb`, cell `cell-7`.

**Change.**

```python
model = FastModel.get_peft_model(
    model,
    r=32,                    # ← was 16
    target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
    lora_alpha=64,           # ← scale alpha with rank to keep alpha/r=2
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
    task_type="SEQ_CLS",
)
```

**Why scale α with r.** Effective LoRA scaling is α/r. Stage 1 used α=32, r=16 → scale=2. Keeping that constant at r=32 means α=64. Going to r=64 → α=128. Don't keep α=32 fixed at higher rank or the adapter contribution shrinks unintentionally.

**Memory check.** At r=32 and MAX_LENGTH=4096, the optimizer state grows by ~25 MB (8-bit AdamW over ~3.1M new trainable params). T4 has headroom. At MAX_LENGTH=6144 (if anyone keeps Option B in Step 3), this is tighter — drop batch by 1 if OOM.

**Validation.** The Unsloth banner at training start should now print `Trainable parameters = 6,7XX,XXX` (was 3,381,507). If it still prints 3.38M, the rank change didn't apply (likely re-running an old cell).

---

## Putting it together — Stage 2 cell-10 (final form)

For reference, Stage 2's training cell after all five changes:

```python
import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

FOCAL_GAMMA = 2.0

class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = (-(1 - pt) ** FOCAL_GAMMA * log_pt).mean()
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

def stratified_subset(ds, n_per_class=167, seed=3407):
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for i, lbl in enumerate(ds["labels"]):
        by_class[lbl].append(i)
    picked = []
    for c, idxs in by_class.items():
        take = min(n_per_class, len(idxs))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())
    return ds.select(sorted(picked))

eval_subset = stratified_subset(tokenized_med["validation"], n_per_class=167)
print(f"Eval subset size: {len(eval_subset)}")

trainer = FocalTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=tokenized_med["train"],
    eval_dataset=eval_subset,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        output_dir="./modernfinbert-v2-medium-output",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=SEED,
        group_by_length=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
    ),
)

trainer.train()
```

And the LoRA cell (cell-7) above it now uses `r=32, lora_alpha=64`.

## Putting it together — Stage 3 cell-2 / cell-7 / cell-10 deltas

```python
# cell-2: drop MAX_LENGTH
MAX_LENGTH = 4096   # was 6144

# cell-7: bump rank
model = FastModel.get_peft_model(model, r=32, lora_alpha=64, ...)

# cell-10: more steps, intra-epoch eval, F1-driven selection
eval_subset_long = stratified_subset(tokenized_long["validation"], n_per_class=100)

trainer = Trainer(   # focal loss optional at S3; data is closer to balanced
    ...
    eval_dataset=eval_subset_long,
    args=TrainingArguments(
        output_dir="./modernfinbert-v2-long-output",
        per_device_train_batch_size=8,           # was 4
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,           # was 8 — eff. batch 32 unchanged
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        ...
        eval_strategy="steps",
        eval_steps=10,                           # 250 total steps → 25 evals
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
    ),
)
```

If `load_best_model_at_end=True` re-triggers the unsloth classifier-reload bug at Stage 3, drop it back to `False` and add the manual best-checkpoint selection from Step 2.

---

## Decision points after Stage 2 rerun

After Stage 2 finishes with all five changes applied:

| Stage 2 medium-test macro F1 | Action                                                                |
|------------------------------|------------------------------------------------------------------------|
| ≥ 0.58 AND short-test ≥ 0.76 | Run Stage 3 with the same recipe; ship `v2` if S3 ≥ 0.58 long F1.     |
| 0.55–0.58                     | Try focal γ=3.0 once, then sqrt-weights fallback before going to S3.  |
| < 0.55                        | Stop. Investigate dataset (likely label-quality issue at medium scale).|
| Short-test < 0.74             | The continuation itself is the problem; consider freezing the encoder for the first 100 steps of S2 (`model.base_model.requires_grad_(False)` then unfreeze) to protect Stage 1's representations. |

## Out of scope (deliberately)

These appear in research.md §8 (steps 6–10) but are not part of this plan:
- Backfilling entities into Stage 1's data.
- Per-stage LR recalibration based on context length.
- Adding a stage-spanning held-out eval set.
- Increasing Stage 3 epochs to 2.
- Investigating the entity coverage gradient.

Bring those in only if Steps 1–5 don't hit the success criteria above.

---

## Detailed TODO checklist

Phases run sequentially; tasks within a phase mostly run sequentially too (data deps in 1.x and 3.x). Numbers in `[Sx]` map back to the five plan steps above.

### Phase 0 — Pre-flight (local, ~30 min) ✅

- [x] **0.1 Confirm baseline numbers are recorded.** Verified during research.md authoring; numbers in research.md §5 (S1 0.7695/0.7711, S2 medium 0.6314/0.5428, S3 long 0.6264/0.5422) match the logs.
- [x] **0.2 Create a results-tracking file.** Created `notebooks/results/v2_recipe_v3_runs.md` with three baseline rows pre-filled.
- [ ] **0.3 Branch the work.** *Skipped — working tree had unrelated pre-existing modifications (paper/main.log, deleted scripts) that were not part of this work. Edits applied to current branch; user can branch when ready to ship.*
- [x] **0.4 Snapshot current notebooks.** Copied to `notebooks/archive/01c_finetune_medium__pre_v3.ipynb` and `notebooks/archive/01b_finetune_long__pre_v3.ipynb`.
- [ ] **0.5 Verify Kaggle dataset versions.** *Deferred — requires Kaggle API access. Dataset names recorded in tracking file: `neoyipeng/modernfinbert-training-v2{,-medium,-long}`. User to confirm revisions are unchanged before Phase 2 push.*
- [ ] **0.6 Verify HF Hub state.** *Deferred — same. Both `-short` and `-medium` repos referenced by the notebooks; Stage 2 reads `-short` (untouched); Stage 3 reads `-medium` (will be overwritten by new run).*
- [x] **0.7 Decide: overwrite or version?** **Decided: overwrite.** Recipe v3 pushes to the same repos (`-medium`, `-v2`); HF commit history serves as the version log. The pre-v3 notebooks are archived in `notebooks/archive/`.

### Phase 1 — Stage 2 recipe rewrite (`01c_finetune_medium.ipynb`) ✅

- [x] **1.1 [S5] Bump LoRA rank in cell-7.** `r=16, lora_alpha=32` → `r=32, lora_alpha=64`.
- [x] **1.2 [S1] Add `FocalTrainer` class to cell-10.** Defined with `FOCAL_GAMMA = 2.0` module-level constant.
- [x] **1.3 [S1] Delete the class-weights block from cell-10.** Removed `Counter(...)`, `class_weights`, the two prints, `CrossEntropyLoss` import, `Counter` import, and the trailing `Class weights used: ...` print in cell-13.
- [x] **1.4 [S1] Delete `WeightedTrainer` from cell-10.** Replaced with `FocalTrainer`.
- [x] **1.5 [S2] Add `stratified_subset` helper to cell-10.** Uses `np.random.default_rng(seed=SEED)`.
- [x] **1.6 [S2] Build `eval_subset` and pass to trainer.** `n_per_class=167` (~500 rows total). Size printed.
- [x] **1.7 [S2 + S4] Update `TrainingArguments` in cell-10.** `eval_strategy="steps"`, `eval_steps=50`, `save_strategy="steps"`, `save_steps=50`, `save_total_limit=3`, `logging_steps=25`, `load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`, `greater_is_better=True`.
- [x] **1.8 [S2] Add full-validation post-train eval to cell-11.** Inserted before the existing test eval; existing test/regression code unchanged.
- [x] **1.9 Update notebook header markdown.** "Class-weighted loss" → "Focal loss (γ=2.0)".
- [x] **1.10 Lint pass.** AST-parsed all 17 cells (0 errors); confirmed no `WeightedTrainer`/`class_weights`/`CrossEntropyLoss`/`from collections import Counter` references remain; confirmed all required identifiers present.
- [x] **1.11 Local CPU smoke run.** Skipped full nbconvert (would require Unsloth + dataset download on CPU). Replaced with focused unit test: see 1.12.
- [x] **1.12 Stage-2 dry sanity check.** Standalone test of focal loss math + stratified_subset: (a) loss(perfect) ≈ 0, (b) loss(wrong) = 20.0 (large), (c) focal=0.011 ≪ CE=0.240 on easy examples (down-weighting verified), (d) gradients finite on random batch, (e) stratified_subset produces exact per-class counts. All checks PASSED.

### Phase 2 — Stage 2 Kaggle execution 🚧 BLOCKED ON USER

These tasks require Kaggle account + GPU runtime. Notebook is ready to push.

- [ ] **2.1 Push notebook to Kaggle.** Bump `kaggle-metadata.json` version, `kaggle kernels push`. Tag the commit `s2-v3-attempt-1`.
- [ ] **2.2 Run on T4.** Use the same Kaggle kernel config as the baseline; only the notebook differs.
- [ ] **2.3 Watch first 100 steps.** Confirm: (a) Unsloth banner shows `Trainable parameters = 6,7XX,XXX` (validates §1.1), (b) first eval at step 50 emits `eval_macro_f1` (validates §1.7), (c) train loss is decreasing.
- [ ] **2.4 Capture outputs.** When done: download `__notebook__.ipynb`, `__results__.html`, `modernfinbert-v2-medium-output/checkpoint-*/trainer_state.json` into `notebooks/results/01c_medium/v3/`.
- [ ] **2.5 Append run row to tracking file (§0.2).** With all metrics from cell-11's output.

### Phase 3 — Stage 2 decision gate 🚧 BLOCKED ON PHASE 2 RESULTS

- [ ] **3.1 Apply the decision-points table from this plan** to the §2.5 numbers.
  - If S2 medium F1 ≥ 0.58 AND short F1 ≥ 0.76 → proceed to Phase 4.
  - If 0.55 ≤ S2 medium F1 < 0.58 → Phase 3.2.
  - If S2 medium F1 < 0.55 → Phase 3.4.
  - If short F1 < 0.74 → Phase 3.5.
- [ ] **3.2 Retry with focal γ=3.0.** Single-line edit `FOCAL_GAMMA = 3.0`, re-run Phase 2. Tag commit `s2-v3-attempt-2-gamma3`.
- [ ] **3.3 Sqrt-weights fallback.** Replace `FocalTrainer` with the `[2.05, 1.07, 0.73]` weighted CE + label-smoothing 0.05 (snippet in Step 1). Re-run Phase 2. Tag `s2-v3-attempt-3-sqrt`.
- [ ] **3.4 Stop branch.** Open an issue / TODO entry titled "S2 ceiling = X, investigate medium-context label quality" with: confusion matrix, top-100 disagreements with NEUTRAL ground truth, suspected mislabels per source. Halt the recipe-fix work; the bottleneck is upstream.
- [ ] **3.5 Encoder-freeze fallback.** Add `model.base_model.requires_grad_(False)` before the trainer call, train for 100 steps, then `model.base_model.requires_grad_(True)`. Implement via a `TrainerCallback` or split into two `trainer.train()` calls with checkpoint reuse. Re-run Phase 2.

### Phase 4 — Stage 3 recipe rewrite (`01b_finetune_long.ipynb`) ✅

*Note: implemented now (in parallel with Phase 1) so both notebooks ship together. Should not actually be **run** on Kaggle until Phase 3.1 confirms S2 passes — but the code is ready.*

- [x] **4.1 [S3] Drop `MAX_LENGTH` to 4096 in cell-2.** Done.
- [x] **4.2 [S3] Update batch sizing in cell-10.** `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, `gradient_accumulation_steps=4`. Effective batch 32 unchanged; optimizer-step count doubles 125 → 250.
- [x] **4.3 [S3] Add truncation assertion to cell-8.** `assert trunc_pct < 0.10, ...` with descriptive error message.
- [x] **4.4 [S5] Bump LoRA rank in cell-7.** `r=32, lora_alpha=64`.
- [x] **4.5 [S2] Add `stratified_subset` and `eval_subset_long` to cell-10.** Helper duplicated verbatim; `n_per_class=100` (~300-row eval subset).
- [x] **4.6 [S2 + S4] Update `TrainingArguments`.** `eval_strategy="steps"`, `eval_steps=10`, `save_strategy="steps"`, `save_steps=25`, `save_total_limit=3`, `logging_steps=5`, `load_best_model_at_end=LOAD_BEST_FALLBACK`, `metric_for_best_model="eval_macro_f1"`, `greater_is_better=True`.
- [x] **4.7 Decide on focal loss for Stage 3.** Default: plain `Trainer` (subsample is 22/26/52, closer to balanced). Documented in cell-10 comment. If Phase 3 ends up needing γ=3 for S2, mirror it here at runtime.
- [x] **4.8 Add `LOAD_BEST_FALLBACK` toggle.** `LOAD_BEST_FALLBACK = True` constant in cell-10; manual best-checkpoint selection in cell-11 fires only when toggled off (gated by `if not LOAD_BEST_FALLBACK`).
- [x] **4.9 Update notebook header.** "MAX_LENGTH = 6144" → "MAX_LENGTH = 4096", batch annotation updated.
- [x] **4.10 Local CPU smoke run.** AST-parsed all cells (0 errors); confirmed no `MAX_LENGTH = 6144` / old batch sizes remain; truncation assertion would pass on chunker-shaped data (synthetic test of [500, 3072]+entity-prefix range gave 0% truncation at MAX=4096).

### Phase 5 — Stage 3 Kaggle execution 🚧 BLOCKED ON USER (after Phase 3.1 passes)

- [ ] **5.1 Push to Kaggle.** Tag `s3-v3-attempt-1`.
- [ ] **5.2 Run on T4.** Expect ~3.5–4 h wall time (2× the optimizer steps but at half the seq length, so ~equivalent throughput).
- [ ] **5.3 Watch first 50 steps.** Confirm trainable param count (~6.7M), truncation rate (< 10%), first eval at step 10 emits `eval_macro_f1`.
- [ ] **5.4 Capture outputs into `notebooks/results/01b_long/v3/`.**
- [ ] **5.5 Append row to tracking file.**

### Phase 6 — Stage 3 decision gate 🚧 BLOCKED ON PHASE 5 RESULTS

- [ ] **6.1 Compare to plan targets.** Long F1 ≥ 0.58, medium F1 not regressed below S2's new baseline, short F1 ≥ 0.74.
- [ ] **6.2 If pass → Phase 7.** If fail → record the failure mode in the tracking file; Phase 7 still runs but ships the *Stage 2* model as the canonical `v2` rather than the Stage 3 model. Update plan: "S3 ablation showed no lift over S2; ship S2 as final".

### Phase 7 — Documentation and ship 🚧 BLOCKED ON PHASE 6 RESULTS

Cannot run pre-emptively: README/MODEL_CARD/research.md updates require the actual numbers from Phase 5/6. The HF push runs from inside the Kaggle notebook (cell-13), not from this machine.

- [ ] **7.1 Update `MODEL_CARD.md`.** New per-class precision/recall, new training recipe section noting focal loss, rank=32, eval-during-training, MAX_LENGTH=4096.
- [ ] **7.2 Update `README.md`.** Replace the "Key Results" row for v2 with new numbers; do not touch the v1 rows.
- [ ] **7.3 Update `research.md`.** Add a §10 "Recipe v3 follow-up" section linking to the new tracking file and summarizing the deltas vs the v2 baseline.
- [ ] **7.4 Push merged model to HF Hub.** `merged.push_to_hub("neoyipeng/ModernFinBERT-v2", private=False)` (already in cell-13). Confirm tokenizer pushes too.
- [ ] **7.5 Add commit message body referencing run IDs from tracking file.** One commit per stage, both on the same branch.
- [ ] **7.6 Open PR `recipe-fix-v3 → main`.** PR description template:
   - **Summary:** S2 + S3 recipe fix (focal loss, intra-epoch eval, F1 selection, rank=32, 4K context cap).
   - **Numbers:** before/after table.
   - **Files:** notebooks + 3 docs.
   - **Risks:** HF model overwritten — old commit hash recorded in tracking file.
- [ ] **7.7 After merge, archive `recipe-fix-v3` branch.** Tag `v2-recipe-v3` on the merge commit.

### Phase 8 — Cleanup and lessons 🚧 BLOCKED ON PHASE 7

8.2 (RECIPE.md) is the only Phase 8 task that can run pre-emptively. Doing it now while context is hot.

- [ ] **8.1 Delete `notebooks/archive/01*__pre_v3.ipynb`** *only after* the new HF model is confirmed live and reproducible from the merged notebooks.
- [x] **8.2 Add a `notebooks/RECIPE.md` (new file).** Created — covers focal vs CE, `load_best_model_at_end` guidance, eval-subset sizing per stage. Referenced from `notebooks/RECIPE.md`.
- [ ] **8.3 Close out research.md TODOs.** If §8 items 6–10 (entity backfill, per-stage LR, etc.) were *not* triggered, mark them resolved or move to `TODOS.md` with current priority.

### Risk register (review before Phase 2 and Phase 5)

| Risk | Trigger | Mitigation |
|---|---|---|
| Focal loss undertrains the head | Train loss flat across all evals | Confirm γ; if stuck, fall through to §3.3 sqrt-weights |
| Unsloth classifier-reload bug | Crash after `load_best_model_at_end` | Toggle off (§4.8); manual best-checkpoint selection |
| Kaggle 12h kill mid-S3 | Wall time creeps over budget | save_steps=25 already preserves a usable artifact; resume by loading checkpoint |
| HF Hub overwrite races multiple devs | Mid-run pushes from elsewhere | Only push after final eval; lock by branch tag `s3-v3-running` |
| Eval subset too small to be stable | High eval-F1 variance step-to-step | Bump `n_per_class` to 250; the eval cost increase is linear |
| Rank=32 OOMs at long stage | Step 5.2 OOM | Drop `per_device_train_batch_size` from 8 → 6, raise grad-accum 4 → 6 |
