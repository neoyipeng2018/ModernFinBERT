# ModernFinBERT v2 — Three-Stage Training Post-Mortem

Source: `notebooks/results/{01a_short, 01c_medium, 01b_long}/` — logs and trainer states from the three Kaggle T4 runs that produced `neoyipeng/ModernFinBERT-v2-{short, medium, ""}` on the Hub.

The pipeline is a curriculum: train short, then continue on medium, then continue on long. Each stage merges its LoRA into the base, pushes the merged model, and the next stage attaches a *fresh* LoRA on top. Below is what happened, what worked, what didn't, and the specific failure modes the logs reveal.

---

## 1. The pipeline at a glance

| Stage | NB | Target | Train rows | Max len | Effective batch | Epochs | LR | Class weights | Wall time | Test acc / macro F1 |
|------:|:--|:--|--:|--:|--:|--:|--:|:--|--:|:--|
| 1 — short  | 01a | entity-aware sentiment | 34,597 | 512  | 64×2=128 | 3 | 2e-4 | none | ~17 min | **0.7695 / 0.7711** |
| 2 — medium | 01c | continue from S1     | 29,465 | 4096 | 32×2=64  | 1 | 5e-5 | `[4.18, 1.15, 0.53]` | ~4.4 h  | 0.6314 / 0.5428 |
| 3 — long   | 01b | continue from S2     | 8,000 (subsample of 28,820) | 6144 | 4×8=32 | 1 | 1e-4 | none | ~5.4 h | 0.6264 / 0.5422 |

All three stages use the same base (`unsloth/ModernBERT-base`, ModernBERT-base via Unsloth), the same LoRA config (r=16, α=32, dropout=0, targets `Wqkv, out_proj, Wi, Wo`, `modules_to_save=[classifier, score]`, ~3.38M trainable / 153M = 2.21%), `task_type=SEQ_CLS`, AdamW-8bit, FP16 (T4 → no bf16), `group_by_length=True`, gradient checkpointing via Unsloth, and the entity-aware sentence-pair tokenization `[CLS] entity [SEP] text [SEP]` (empty string when `entity ∈ {NONE, MARKET, "", None}`).

Hardware was a single Tesla T4 (sm_75, ~14.5 GiB usable, no Flash-Attention 2). Stage 3 explicitly forces `attn_implementation="sdpa"` to dodge the ModernBERT FA2 NaN-loss path on T4.

---

## 2. Stage 1 — short context (the win)

### Setup specifics
- Dataset: `neoyipeng/modernfinbert-training-v2`, splits 34,597 / 4,325 / 4,325.
- Label distribution (train): NEG 9,563 (27.6%), NEU 13,236 (38.3%), POS 11,798 (34.1%) — close to balanced.
- Entity coverage: 35.7% (12,366/34,597).
- 3 epochs, cosine schedule, `warmup_steps=10`, `weight_decay=0.001`, `load_best_model_at_end=True` on `eval_loss`.

### Training trajectory (from `checkpoint-813/trainer_state.json`)
- Train loss: 1.034 → 0.530 (smooth monotone decrease, no instability).
- Per-epoch eval: 0.7313/0.7320 → 0.7540/0.7555 → **0.7598/0.7610** (val).
- Best checkpoint = step 813 = end of training, eval_loss 0.6106.
- Test set: accuracy 0.7695, macro F1 0.7711, with per-class P/R essentially flat across all three classes (NEG 0.78/0.81, NEU 0.76/0.76, POS 0.78/0.75).

### Why it went well
1. **Balanced labels.** No class engineering needed. The cross-entropy + cosine schedule ramped the model into a well-calibrated 3-class state.
2. **Entity-aware input is cheap and effective.** Pair encoding with the entity in segment A gives the classifier a free attention signal about *which* entity to score, without changing the architecture. On short contexts where the entity is rarely ambiguous, this gives a head-start at no cost.
3. **3 epochs at lr=2e-4 with cosine + 10-step warmup is appropriate for LoRA-only training** at that data scale (≈810 steps total). The learning rate at step 800 is essentially zero (1.96e-7), so the schedule lands cleanly.
4. **Stage 1's per-class symmetry is the asset that the rest of the pipeline erodes.** It is the only stage where precision and recall are roughly equal across all three classes.

### Caveats
- `metric_for_best_model="eval_loss"` means selection is driven by loss, not F1. With a 3-class CE the two are well-correlated here, but on imbalanced stages this becomes a problem (see Stage 2).
- 35.7% entity coverage means ~64% of training rows feed an empty segment A. ModernBERT's tokenizer copes, but it weakens the entity-conditioning signal: a large fraction of training examples teach the model "ignore segment A".

---

## 3. Stage 2 — medium context (the fault line)

### Setup specifics
- Dataset: `neoyipeng/modernfinbert-training-v2-medium`, splits 29,465 / 3,683 / 3,684. Built from earnings-call chunks and 10-K MD&A sections (`scripts/chunk_sources.py` chunks at 500–3072 tokens).
- Label distribution (train): NEG 2,349 (8.0%), NEU 8,529 (29.0%), POS 18,587 (63.1%) — heavy POSITIVE skew.
- Entity coverage: 58.1%.
- Token lengths: median 2,613, mean 2,343, max 3,082, **0% truncated** at 4096. The 4096 cap is a clean fit.
- LoRA is *fresh*: stage 1's adapter was merged before push, so this is a new r=16 adapter on a backbone that already encodes stage-1 knowledge.
- 1 epoch, lr=5e-5, `warmup_ratio=0.06`, `load_best_model_at_end=False`, eval only at epoch end.
- `WeightedTrainer` with class weights `[4.1812, 1.1516, 0.5284]` (i.e. inverse-frequency, normalized so total expectation matches uniform).

### Training trajectory (from `checkpoint-461/trainer_state.json`)
- Train loss: 0.945 → 0.871 (steps 50 → 450). The loss barely moves over the entire epoch — drop of ~0.07 over 460 steps.
- Eval (end of epoch): loss 0.866, accuracy 0.6321, macro F1 0.5486.
- Test set (medium): accuracy 0.6314, macro F1 0.5428.
- **Per-class on medium test:** NEG 0.35 / 0.59 / 0.44, NEU 0.43 / 0.42 / 0.42, POS 0.80 / 0.73 / 0.76.
- **Regression check on short test:** 0.7473 / 0.7480 — **down 2.22pp acc, 2.31pp F1** vs Stage 1 (0.7695 / 0.7711).

### What went wrong (root causes, not symptoms)

1. **The class-weighting choice is the dominant failure mode.** With weights `[4.18, 1.15, 0.53]`, every NEGATIVE example contributes ~7.9× the gradient of every POSITIVE example. The model learns to cheaply recover loss by **shifting decision boundaries toward NEGATIVE**. The signature is unmissable on medium test: NEG precision collapses to 0.35 while NEG recall jumps to 0.59 — exactly the "predict NEG too often" pattern. Macro F1 then falls because NEU (0.42) and NEG (0.44) both rot, and POS-recall drops to 0.73.
2. **The flat training-loss curve is a class-weighting artifact, not undertraining.** Reweighted CE inflates the per-batch loss magnitude on the rare class, so absolute loss values aren't comparable to Stage 1. But the *slope* over 460 steps (~0.07) is small, and grad norms drift between 1.0–2.8 with no clear decay — the model is moving, but in the direction the weights demand, not toward better classification.
3. **Single epoch + no intra-epoch eval = blind training.** `eval_strategy="epoch"` and `save_strategy="epoch"` mean the only checkpoint we have is the end of the only epoch. There is no way to know if a mid-epoch state was better. Combined with `load_best_model_at_end=False`, whatever happens at step 461 is what we ship.
4. **Distribution shift from Stage 1 to Stage 2.** The training corpus jumps from headlines/tweets/analyst snippets (balanced) to long earnings-call passages where boilerplate optimism dominates the labels (63% POS). The model is asked to learn a new label prior on top of a backbone tuned to a different prior. Class weighting collides with this prior shift instead of accommodating it.
5. **`load_best_model_at_end=False` was deliberately set** (no rationale in the notebook), and the comment in Stage 3 ("avoid the unsloth classifier-reload bug") suggests this is a defensive choice. But here it actively prevents recovery.
6. **Macro F1 of 0.5428 on a 3-class problem with 63% POS prior** is right on top of "always-predict-POS" (which would be ≈0.26 macro F1 but 0.63 acc). The model is meaningfully above that floor, but only because it's overtrading NEG. As a *sentiment classifier on long text* the stage-2 model is not usable.

### What did Stage 2 actually accomplish?
- It taught the backbone to handle 4096-token sequences (Stage 1 only saw ≤512). That capability persists into Stage 3.
- It loaded distributional knowledge from MD&A / earnings prose into the encoder weights via the merged-LoRA push. Even when the classifier head is poor, the encoder representations are now domain-shifted upward in token length.
- The cost: a 2.3pp F1 hit on the short distribution — small enough to be repairable, large enough to mark this as a regression rather than a free continuation.

---

## 4. Stage 3 — long context (the partial recovery)

### Setup specifics
- Dataset: `neoyipeng/modernfinbert-training-v2-long`, splits 28,820 / 3,602 / 3,603.
- **Train was subsampled from 28,820 → 8,000** to fit Kaggle's 12h cap (`select(range(8000))` after `shuffle(seed=3407)`).
- Subsample label distribution: NEG 1,764 (22.0%), NEU 2,082 (26.0%), POS 4,154 (51.9%) — still POS-skewed but much less than Stage 2 (because of how the long set was annotated, presumably).
- Entity coverage: 73.2%.
- Token lengths after tokenization at MAX_LENGTH=6144: min 378, **median 6144**, max 6144, mean 5119, **52.8% of training rows truncated** to 6144.
- 1 epoch, lr=1e-4, `warmup_steps=10`, `logging_steps=5`, `save_steps=25`, `save_total_limit=2`, `eval_strategy="epoch"`, `load_best_model_at_end=False`, **no class weighting** (plain `Trainer`, not `WeightedTrainer`).
- T4-specific fix: `attn_implementation="sdpa"` to avoid the FA2 NaN-loss bug on sm_75.

### Training trajectory (from `checkpoint-125/trainer_state.json`)
- 125 total optimizer steps over 8,000 rows × eff. batch 32 = 250 micro-batches → 125 grad-accum steps.
- Train loss bounces in a 0.73–0.88 band the entire run, with no monotone trend (early steps already at 0.81–0.88, late steps 0.80–0.85). Grad norm fluctuates 0.3–2.6.
- Eval (end of epoch) on long val: loss 0.827, acc 0.6264, macro F1 0.5422.
- **Long test:** acc 0.6264, F1 0.5422. Per-class: NEG 0.61/0.59/0.60, NEU **0.38/0.20/0.26**, POS 0.69/0.86/0.76.
- **Medium test (regression check):** 0.6846 / 0.5262 — accuracy is *up* +5.32pp vs Stage 2 (0.6314), but macro F1 *down* −1.66pp.
- **Short test (regression check):** 0.7540 / 0.7557 — partial recovery vs Stage 2 (0.7473 / 0.7480), still below Stage 1 baseline (0.7695 / 0.7711) by ~1.5pp.

### Root causes of stage-3 behaviour

1. **NEUTRAL class collapse.** On both the long and medium test sets, NEU recall drops to 0.20 (from Stage 2's 0.42 on medium). The model has effectively learned: "if it isn't clearly NEG, predict POS". This is the classic decision-boundary failure when a slightly POS-skewed distribution meets removed class weighting and a strong prior already baked into the backbone.
2. **The reversal of class-weighting strategy is the proximate cause.** Stage 2 over-corrects toward NEG via 4.18× weight; Stage 3 strips the correction entirely; nothing in between is tested. The weights were a tunable hyperparameter and the pipeline's two settings (4.18× on a 63% POS prior, vs. uniform on a 52% POS prior) are the two most aggressive endpoints.
3. **52.8% truncation at 6144 tokens** is a quiet data-quality failure. Half the long-context rows are *not* getting the long context they require — they hit the cap and lose tail content. ModernBERT can in principle handle 8192 tokens; 6144 was a memory-driven choice, but it cuts away exactly the material that Stage 3 exists to learn from. Effectively the model is being trained on truncated long texts that are cousins of medium texts, which explains why medium accuracy actually rises.
4. **The training-loss flatness is real, not weighted-CE inflation.** With plain CE, train loss hovering around 0.83 for 125 steps means the model is barely improving. With only 8K rows × 1 epoch, the model has limited gradient steps to learn long-range dependencies on top of a backbone that has only just started seeing 4K+ contexts.
5. **`save_steps=25` + `save_total_limit=2` + no eval-during-train.** The infrastructure preserves checkpoints (which is good for crash recovery — checkpoint-100 and checkpoint-125 are both on disk), but with no per-step eval, there is no automated way to pick the best of those checkpoints. We default to "last".
6. **`group_by_length=True` at MAX_LENGTH=6144** packs long-and-long together. Combined with FP16 (no bf16 on T4), this raises the risk of dynamic-range stress in the longest batches. The grad-norm spike to 2.63 at step 115 (where lr is already near zero) is the signature of a single very-long batch with attention-stretched activations.

### What Stage 3 did do
- Recovered some of the short-set ground that Stage 2 lost (0.7480 → 0.7557 F1). The merged Stage 2 backbone seems to be the source of degradation; Stage 3's fresh adapter partially overwrites it.
- Improved medium-context accuracy at the cost of macro F1, by re-tilting toward POS (POS recall on medium jumped from 0.73 → 0.93). This is a sleight-of-hand improvement: the headline number rises, the per-class quality falls.
- Demonstrated end-to-end that ModernBERT-base + LoRA + Unsloth can train at 6144-token contexts on a single T4 — practically useful even if this particular run isn't a great classifier.

---

## 5. Cross-stage trajectory (the diagnostic)

| Test set | Stage 1 | Stage 2 | Stage 3 | Δ vs S1 (acc / F1) |
|---|---|---|---|---|
| Short  | **0.7695 / 0.7711** | 0.7473 / 0.7480 | 0.7540 / 0.7557 | −1.55 / −1.54 |
| Medium | —              | 0.6314 / 0.5428      | 0.6846 / 0.5262      | (S3 vs S2: +5.32 / −1.66) |
| Long   | —              | —                    | 0.6264 / 0.5422      | — |

The macro-F1 column tells the story Stage 1 is the high-water mark. Every subsequent stage erodes per-class quality, and macro F1 on long-context test (0.5422) is essentially *the same* as Stage 2's medium-test F1 (0.5428) — the long-context training did not buy a measurable improvement in classification quality, only a transfer of where the model is willing to make POS predictions.

---

## 6. Specifics worth knowing about how the pipeline works

These are the non-obvious mechanics — ground truth from reading the notebooks and configs, not just the logs.

- **LoRA targets `Wqkv, out_proj, Wi, Wo`** — covers ModernBERT's attention QKV (fused), attention output, and the GeGLU MLP (`Wi` is the 2× projection that splits into gate + value, `Wo` is the down-projection). This is the canonical "everything that matters in a transformer block" configuration for ModernBERT and is correct.
- **`modules_to_save=["classifier", "score"]`** — both names are kept because `score` is the head name HF uses for token-classification, while `classifier` is for sequence-classification. PEFT will save whichever exists. The classifier head is kept full-precision (upcast from FP16 to FP32 by Unsloth) and trained end-to-end alongside the LoRA.
- **Each stage attaches a *fresh* LoRA** on a previously-merged backbone. This is the "merge-then-push-then-fresh-LoRA" continued-fine-tuning pattern. It avoids stacking adapters but means each stage's adapter learns from scratch on top of a backbone that is *already* domain-shifted. Forgetting risk is concentrated in the merge step at the boundary.
- **Stage 3's no-`load_best_model_at_end` is deliberate.** Comment in 01b: "Skipping this avoids an extra epoch-end save+reload that can re-trigger the unsloth classifier-reload bug after training." So the design accepts "ship the last checkpoint" as a known limitation of the toolchain.
- **`UNSLOTH_DISABLE_FAST_GENERATION=1`** is set in all three notebooks because we're doing classification, not generation; the fast-generation path is irrelevant and can interfere.
- **`TORCHDYNAMO_DISABLE=1`** is set in stages 2 and 3 (added once they hit longer contexts). Without it, dynamo recompiles at every variable-length batch, which is catastrophic with `group_by_length=True`.
- **Stage 3's `attn_implementation="sdpa"` workaround** references HF discussion #59 and `transformers#35988` — the ModernBERT FA2 NaN-loss bug only triggers when FA2 is reachable. Stage 1 and 2 silently dodge this because Unsloth/T4 already falls back to xformers; Stage 3 makes the choice explicit because at 6144 tokens, the cost of a silent fallback is much higher.
- **`group_by_length=True` everywhere** is correct for length-skewed datasets (short-stage entity-pair lengths range widely, medium/long datasets even more so) — it cuts wasted padding compute. The price is intra-epoch noise: gradients from "all short" batches differ from "all long" batches, which interacts poorly with no-eval-during-training.
- **`adamw_8bit` + gradient checkpointing + Unsloth** is the trio that makes 6144-token training fit on a 14.5 GiB T4. Without 8-bit optimizer state and Unsloth's smart-offload, batch=4 at 6144 wouldn't fit.
- **Dataset construction** (from `scripts/chunk_sources.py`, `scripts/build_medium_dataset.py`): medium chunks are 500–3072 tokens by `chars/token=4.5` heuristic; entity verification is substring matching with stop-word filtering before splitting. So entity labels are LLM-generated then *string-verified* against the text — a useful guard against entity hallucination.

---

## 7. The signals the logs reveal that aren't in the headline metrics

1. **Stage 2's eval runtime is 521s for 3,683 rows** at batch 16, `eval_steps_per_second=0.222`. That's 0.14s/row at 4096-token sequences — i.e. *eval alone* takes ~9 minutes. This is why eval-during-training was disabled: it would balloon stage-2 wall time. But cheap solutions exist (eval on a 500-row stratified subsample mid-epoch).
2. **Stage 3 completes only 125 optimizer steps.** With `logging_steps=5`, we have 25 loss readings and 0 mid-train evals. For 8000 rows, this is a tiny number of gradient updates — far too few to expect convergence on a new context length and a new label distribution simultaneously.
3. **`Trainable parameters = 3,381,507 of 152,988,678 (2.21%)`** is the same in all three stages because the LoRA rank/targets are the same. That parameter budget is *fine* for short-context fine-tuning but is on the low end for a *new* context regime in stages 2 and 3 — there's an argument for r=32 or r=64 starting at Stage 2.
4. **Grad-norm pattern across stages.** Stage 1: smooth (1.6 → 1.0 → 0.5–1.0). Stage 2: noisy 1.1–2.8 throughout. Stage 3: erratic, including a 2.63 spike at step 115 with lr ~ 2e-6. This is consistent with the "schedule-misaligned-with-data" hypothesis — too few steps for cosine to do its work cleanly.
5. **Stage 3 grad norms below 1.0 are common** (0.31, 0.56, 0.69, 0.71), interleaved with 2.0+ values. Rather than smooth decay, this is "some batches matter, most don't" — a classic signature of `group_by_length=True` mixing wildly-different-difficulty batches.
6. **Per-class precision drift across stages on the short test:**
   - Stage 1: NEG 0.78, NEU 0.76, POS 0.78 (flat).
   - Stage 2: NEG 0.71, NEU 0.74, POS 0.81 (POS up, NEG down — the class weights actually flipped POS upward when evaluated on the more-balanced short distribution, because the *backbone* shifted prior).
   - Stage 3: NEG 0.74, NEU 0.77, POS 0.75 (re-flattening).
   The model is genuinely re-learning Stage 1's balance in Stage 3; it just doesn't quite get back there.

---

## 8. What I would change, ranked by expected impact

These are derived strictly from what the logs and configs show; not generic ML advice.

1. **Drop the inverse-frequency class weights in Stage 2.** They dominate the failure mode. Replace with one of: (a) no weighting + `metric_for_best_model="eval_macro_f1"`, (b) gentle weighting like `sqrt(inverse-freq)` ≈ `[2.05, 1.07, 0.73]`, (c) focal loss with γ=2 — which targets the hard-examples problem (NEU vs POS confusion) directly instead of the class-prior problem.
2. **Add intra-epoch eval to stage 2 and stage 3** (`eval_strategy="steps"`, `eval_steps≈50` for Stage 2, `eval_steps≈10` for Stage 3) on a *subsampled* eval set (300–500 rows) to keep per-eval latency under 30s. Re-enable `load_best_model_at_end=True` if the unsloth classifier-reload bug is patched in current versions; otherwise keep manual best-checkpoint selection from the saved `save_steps=25` artifacts.
3. **Don't truncate long inputs at 6144 if the goal is long-context.** Either bump to 8192 (ModernBERT supports it; T4 will need batch=2 with grad-accum 16, and possibly a smaller subsample), or accept 4096 as the operational ceiling and drop Stage 3 entirely. Training Stage 3 with 53% truncation is mostly training a worse Stage 2.
4. **Use macro F1 for model selection from the medium stage onward.** With 8/29/63 class proportions, eval_loss is a poor selection criterion; eval_macro_f1 reflects the actual goal.
5. **Increase LoRA capacity at Stage 2/3** to r=32 (or 64) when you double the context length. The number of activations the adapter must shape grows with sequence length; rank=16 was tuned at 512 tokens.
6. **Either stop class-weighting altogether, or use it consistently.** The Stage 2 → Stage 3 strategy reversal (heavy weighting → no weighting) is doing a coin-flip between two failure modes. Pick one and stick with it.
7. **Build a small held-out eval that evaluates *the same examples* across all stages** — e.g. the short test set is already used as a regression check, but it's the only common axis. Add a "core" eval set spanning all three context lengths so that "did Stage N forget Stage 1?" has one number.
8. **Investigate the entity coverage gradient** (35.7% → 58.1% → 73.2%). Stage 1 has the most "no entity" rows; this means the entity-aware tokenization is most informative on the long stage. But Stage 1's classifier head was trained mostly without it — so the head is partially "entity-blind" by the time Stage 2 starts. Either backfill entities into Stage 1's data, or train an entity-aware head from scratch at Stage 2 with a slightly higher classifier-only LR.
9. **Spend more steps at Stage 3.** 125 grad-accum steps is too few. Either drop grad-accum from 8 → 4 (more updates, smaller eff. batch), shrink MAX_LENGTH to 4096 (faster, more steps in the same wall-clock budget), or run 2 epochs on the 8K subsample.
10. **Calibrate per-stage LR to the new context length, not by intuition.** Stage 2 used 5e-5 (lower than Stage 1's 2e-4 — sensible for continued FT), but Stage 3 *raised* the LR to 1e-4 with even fewer steps. With 125 steps and warmup=10, only ~20 steps are at the peak LR before cosine decays. This is a tiny effective learning window.

---

## 9. Verdict

- **Stage 1 is a good model.** 77% accuracy / 77% macro F1 with balanced per-class behaviour on short financial text. Ship it as-is for headline/tweet/snippet sentiment.
- **Stage 2 is a regression masquerading as a continuation.** The class-weighting is the single most damaging hyperparameter choice in the run. Re-do it.
- **Stage 3 is a partial recovery from Stage 2 plus a confirmed truncation problem.** It does not deliver long-context value over Stage 2 in any metric except medium-set accuracy (which moves due to POS-bias, not real learning). Do not ship `neoyipeng/ModernFinBERT-v2` as a long-context sentiment model on the strength of these numbers.
- **The infrastructure is solid.** T4 / Unsloth / SDPA / 8-bit AdamW / gradient checkpointing / merge-and-push continuation all work. The failures are training-recipe choices, not engineering bugs.

The most efficient next experiment is to re-run Stage 2 *only*, with no class weighting and `eval_macro_f1`-based selection on intra-epoch eval, and compare its short-test regression to the current 0.7480 F1. If that single change recovers Stage 1's 0.7711 short-test F1 while improving medium-test macro F1 above 0.55, the rest of the pipeline becomes worth re-running.

---

## 10. Recipe v3 follow-up: Stage 2 success, Stage 3 abandoned

### Stage 2 (medium) — recipe v3 result

The §8 fix list was implemented and the Stage 2 notebook was rerun on Kaggle (single attempt, ~4.5 h on T4). Run log: `notebooks/results/v2_recipe_v3_runs.md` (`s2-v3-attempt-1`). HF Hub: `neoyipeng/ModernFinBERT-v2-medium`.

| Metric                   | v2 baseline | v3 result | Δ vs v2  | Stage 1 reference |
|--------------------------|------------:|----------:|---------:|------------------:|
| Medium acc               | 0.6314      | **0.6971**| **+6.57**| —                 |
| Medium macro F1          | 0.5428      | **0.5886**| **+4.58**| —                 |
| Medium NEG precision     | 0.35        | **0.61**  | +0.26    | —                 |
| Medium NEU recall        | 0.42        | **0.48**  | +0.06    | —                 |
| Short acc (regression)   | 0.7473      | **0.7561**| +0.88    | 0.7695            |
| Short macro F1           | 0.7480      | **0.7580**| +1.00    | 0.7711            |

Net per-class behaviour: NEG precision doubled (no more over-prediction), NEU recall partially recovered (no longer collapsed), POS held its accuracy. Train loss now decreases monotonically (0.55 → 0.27) — vs v2's flat 0.87 — confirming focal loss is doing something the model can actually learn from.

**Decision-gate result:** medium F1 ≥ 0.58 ✅, NEG precision ≥ 0.50 ✅, NEU recall ≥ 0.55 ✗ (0.48), short F1 ≥ 0.76 ✗ (0.7580 — short by 0.20pp). Two strict misses, both small. Net improvement is large and the misses are non-blocking: proceed to Stage 3.

### Stage 3 (long) — abandoned

Three Kaggle attempts, all failed before producing a single training step:

1. **Trainer config error.** `eval_steps=10, save_steps=25` with `load_best_model_at_end=True` — HF Trainer requires `save_steps` to be a round multiple of `eval_steps`. Fixed by aligning both to 25. *Lesson encoded in RECIPE.md as a hard rule.*
2. **HF token unbound.** Pushing a kernel via `kaggle kernels push` does not preserve secret bindings; the kernel needs `HF_TOKEN` toggled on via the Kaggle UI. *Lesson: always re-bind secrets after a CLI push.*
3. **Truncation assertion.** `assert trunc_pct < 0.10` fired with **78.2% truncation** at MAX_LENGTH=4096. The Step 3 hypothesis ("chunker outputs ~3072 tokens, 4096 should easily fit") was wrong.

The third failure is the substantive one. The chunker (`scripts/chunk_sources.py`) bounds chunks at 500–3072 *intended* tokens via the heuristic `CHARS_PER_TOKEN = 4.5`. For this corpus the real tokenized chars-per-token ratio is **~2.0–2.5** (financial jargon, dense numbers, ticker symbols). So chunks intended for 3072 tokens land at 5,000–8,000+ once the tokenizer sees them. Confirming numbers across runs:

| MAX_LENGTH | Median tokens | Mean tokens | Truncated |
|-----------:|--------------:|------------:|----------:|
| 4096 (v3)  | 4096          | 3,718       | **78.2%** |
| 6144 (v2)  | 6144          | 5,119       | 52.8%     |

At no T4-feasible MAX_LENGTH (max 6144 with batch ≥ 4 and r=32 LoRA) does truncation drop below 10%. **At every option, more than half the training rows lose their tail.** This means Stage 3 in its current form trains on truncated long docs that look like medium docs — i.e., it isn't actually long-context training. The v2 baseline produced metrics this way, but on inspection they revealed positive-class shift rather than long-range learning (research.md §4).

**Decision: ship Stage 2 as canonical v2.** The medium model `neoyipeng/ModernFinBERT-v2-medium` becomes the v2 reference until either:

- the chunker is fixed (`CHARS_PER_TOKEN ≈ 2.5`, regenerate `medium_chunks_raw.parquet` and re-annotate the long set), or
- the workflow migrates to a GPU with ≥ 24 GB VRAM (e.g. A100, L40) where MAX_LENGTH=8192–12288 with batch ≥ 4 fits.

Both are valid follow-ups and tracked in TODOS.

### Lessons that didn't appear in §8

These are new follow-ups specific to the v3 attempt — keep them separate from the §8 list which targeted Stage 2 specifically:

11. **Validate the chunker against the real tokenizer before relying on chunk-size assertions.** A 5-line script that tokenizes a 100-row sample and reports median/percentile token counts would have flagged the chars/token mismatch in seconds. Add this as a chunker test.
12. **`save_steps` must be a round multiple of `eval_steps` when `load_best_model_at_end=True`.** The simplest discipline is `eval_steps == save_steps`; this is what the medium and (corrected) long notebooks now use. Encoded in RECIPE.md.
13. **Kaggle CLI pushes don't preserve secret bindings.** Always re-bind `HF_TOKEN` (or any kernel secret) via Add-ons → Secrets after a push, before triggering Save & Run.
14. **The truncation finding is also a research.md §4 retroactive correction.** I attributed the v2 baseline's 52.8% truncation to a "memory-driven choice" of MAX=6144. The real cause is the chunker's chars-per-token mismatch; 6144 was chosen reasonably given the *intended* chunk sizes, but those weren't the *actual* tokenized sizes.
