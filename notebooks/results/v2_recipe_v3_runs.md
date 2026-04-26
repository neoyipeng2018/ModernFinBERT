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
| _(pending Kaggle execution — Phase 2)_ | | | | | | | | | | | | | | |
