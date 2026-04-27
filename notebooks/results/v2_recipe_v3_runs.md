# ModernFinBERT v2 — Recipe v3 run log

One row per training run. Columns:

- `run_id` — `s{stage}-v{recipe}-{attempt}`
- `stage` — short / medium / long
- `change_set` — short tag for what changed vs prior row
- `short_*`, `med_*`, `long_*` — accuracy / macro_f1 on respective test sets
- `NEU_recall`, `NEG_precision` — diagnostic per-class metrics on this stage's *own* test set
- `wall_time` — total Kaggle kernel runtime
- `hf_repo` — repo pushed to (commit hash if overwriting)
- `kaggle_url` — Kaggle kernel URL for the run
- `notes` — anything material (focal γ, rank, MAX_LENGTH, OOMs, etc.)

## Baselines (recipe v2 — prior run)

| run_id | stage  | change_set     | short_acc | short_f1 | med_acc | med_f1  | long_acc | long_f1 | NEU_recall | NEG_precision | wall_time | hf_repo                            | notes |
|--------|--------|----------------|-----------|----------|---------|---------|----------|---------|------------|---------------|-----------|------------------------------------|-------|
| s1-v2  | short  | baseline       | 0.7695    | 0.7711   | —       | —       | —        | —       | 0.76       | 0.78          | ~17 min   | neoyipeng/ModernFinBERT-v2-short   | r=16, α=32, MAX=512, 3 epochs, CE. Stage 1 baseline; not retraining. |
| s2-v2  | medium | baseline       | 0.7473    | 0.7480   | 0.6314  | 0.5428  | —        | —       | 0.42       | 0.35          | ~4.4 h    | neoyipeng/ModernFinBERT-v2-medium  | r=16, α=32, MAX=4096, 1 epoch, weighted CE [4.18, 1.15, 0.53]. POSITIVE-skewed train set. |
| s3-v2  | long   | baseline       | 0.7540    | 0.7557   | 0.6846  | 0.5262  | 0.6264   | 0.5422  | 0.20       | 0.61          | ~5.4 h    | neoyipeng/ModernFinBERT-v2         | r=16, α=32, MAX=6144 (52.8% truncated), 1 epoch, plain CE. NEU collapse. |

## Recipe v3 runs

| run_id | stage  | change_set     | short_acc | short_f1 | med_acc | med_f1  | long_acc | long_f1 | NEU_recall | NEG_precision | wall_time | hf_repo                            | kaggle_url | notes |
|--------|--------|----------------|-----------|----------|---------|---------|----------|---------|------------|---------------|-----------|------------------------------------|------------|-------|
| s2-v3-attempt-1 | medium | focal γ=2.0, r=32/α=64, eval@steps=50 on stratified-500 subset, F1-based best-ckpt, MAX=4096 | 0.7561 | 0.7580 | 0.6971 | 0.5886 | — | — | 0.48 | 0.61 | ~4.5 h | neoyipeng/ModernFinBERT-v2-medium (v3) | https://www.kaggle.com/code/neoyipeng2018/modernfinbert-v2-medium | Big win vs v2 baseline: medium F1 +4.58pp, medium acc +6.57pp, NEG precision doubled (0.35→0.61), NEU recall +6pp (0.42→0.48), short test partially recovered (+1.00pp F1 but still 1.31pp below S1). Train loss actually decreased (0.55→0.27, vs v2's flat 0.87). Decision gate: medium F1 ≥0.58 ✅, NEG precision ≥0.50 ✅, NEU recall <0.55 ✗ (short by 7pp), short F1 <0.76 ✗ (short by 0.2pp). Proceeding to Phase 4 — net improvement is clear. |
| s3-v3-attempt-1-FAILED | long | r=32/α=64, MAX=4096, batch 8×4, eval/save_steps=10/25 | — | — | — | — | — | — | — | — | <30 s | — | https://www.kaggle.com/code/neoyipeng2018/modernfinbert-v2-finetune-long?scriptVersionId=314693366 | Trainer construction failed: `load_best_model_at_end=True` requires `save_steps` to be a round multiple of `eval_steps`; eval_steps=10, save_steps=25, 25%10=5≠0. Fix: align both to 25. RECIPE.md updated with hard-constraint note. |
| s3-v3-attempt-2 | long | as above + eval_steps=25 (aligned with save_steps) | _(pending Kaggle execution — version 28)_ | | | | | | | | | | https://www.kaggle.com/code/neoyipeng2018/modernfinbert-v2-finetune-long | |
