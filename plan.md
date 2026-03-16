# Notebook Cleanup Plan

## Goal

Remove or archive all notebooks and directories not directly used in the paper (`paper/main.tex`). Keep the repository lean with only the experiments that back published claims.

---

## 1. Audit: What the Paper References

The paper cites 7 experiments across 10 notebooks. Here is the mapping from paper sections to notebook files:

| Paper Section | Experiment | Notebook(s) | Status |
|---|---|---|---|
| §4.1 (Exp 1) | Held-out evaluation | `01_architecture_comparison.ipynb` | **KEEP** |
| §4.2 (Exp 2) | DataBoost (VS-CoT) | `02A_databoost_vs.ipynb` | **KEEP** |
| §4.3 (Exp 3) | Claude comparison | `03_claude_comparison.ipynb` | **KEEP** |
| §4.4 (Exp 4) | 10-fold CV on FPB | `04_kfold_cv.ipynb` | **KEEP** |
| §4.5 (Exp 5) | Dedup audit | `09a_dedup_audit.ipynb` | **KEEP** |
| §4.5 (Exp 5) | Head-to-head CV | `09b_fpb_crossval.ipynb` | **KEEP** |
| §4.5 (Exp 5) | Clean held-out baselines | `09c_clean_holdout.ipynb` | **KEEP** |
| §4.6 (Exp 6) | Multi-seed robustness | `06_multi_seed.ipynb` | **KEEP** |
| §4.7 (Exp 7) | Self-training | `07_self_training.ipynb` | **KEEP** |
| Table 1 | Data provenance audit | `11_data_provenance_audit.ipynb` | **KEEP** |

**Total: 10 notebooks to keep.**

---

## 2. What to Remove

### Notebooks to Archive

| Notebook | Reason for removal |
|---|---|
| `02_databoost.ipynb` | Prototype of NB02A; paper uses VS-CoT method (NB02A), not runtime API paraphrasing |
| `03A_test_evaluation.ipynb` | Cross-notebook utility; no results cited in paper |
| `09d_sample_efficiency.ipynb` | Mentioned in §6 Limitations as future work, but no results in paper |
| `09e_full_finetune.ipynb` | Mentioned in §6 Limitations as future work, but no results in paper |
| `10_finbert_tone_deep_dive.ipynb` | finbert-tone results in paper come from NB09c, not NB10 |

### Archive Directories to Remove

| Directory | Reason |
|---|---|
| `notebooks/archive/` | Old exploratory notebooks (Data.ipynb, old kfold attempts) |
| `kaggle_push/` | Kaggle export of NB01 |
| `kaggle_push_02/` | Kaggle export of NB02 (superseded) |
| `kaggle_push_02A/` | Kaggle export of NB02A |
| `kaggle_push_03A/` | Kaggle export of NB03A (removed notebook) |
| `kaggle_push_04/` | Kaggle export of NB04 |
| `kaggle_push_06/` | Kaggle export of NB06 |
| `kaggle_push_07/` | Kaggle export of NB07 |
| `kaggle_push_09b/` | Kaggle export of NB09b |
| `kaggle_push_09c/` | Kaggle export of NB09c |
| `kaggle_push_09d/` | Kaggle export of NB09d (removed notebook) |
| `kaggle_push_09e/` | Kaggle export of NB09e (removed notebook) |
| `kaggle_push_10/` | Kaggle export of NB10 (removed notebook) |

### Kaggle Output Directories

| Directory | Reason |
|---|---|
| `kaggle_output/` | Generic NB01 outputs |
| `kaggle_output_01/` | NB01 outputs |
| `kaggle_output_02/` | NB02 (superseded) |
| `kaggle_output_03A/` | NB03A (not in paper) |
| `kaggle_output_04/` | NB04 outputs |
| `kaggle_output_09b/` | NB09b outputs |
| `kaggle_output_09c/` | NB09c outputs |

---

## 3. Final Repository Structure

```
notebooks/
├── 01_architecture_comparison.ipynb   # Exp 1: Held-out eval
├── 02A_databoost_vs.ipynb             # Exp 2: DataBoost (VS-CoT)
├── 03_claude_comparison.ipynb         # Exp 3: Claude comparison
├── 04_kfold_cv.ipynb                  # Exp 4: 10-fold CV
├── 06_multi_seed.ipynb                # Exp 6: Multi-seed robustness
├── 07_self_training.ipynb             # Exp 7: Self-training
├── 09a_dedup_audit.ipynb              # Exp 5a: Data contamination check
├── 09b_fpb_crossval.ipynb             # Exp 5b: Head-to-head CV
├── 09c_clean_holdout.ipynb            # Exp 5c: Clean held-out baselines
├── 11_data_provenance_audit.ipynb     # Data provenance (Table 1)
└── archive/
    └── not_in_paper/
        ├── 02_databoost.ipynb         # Prototype (superseded by 02A)
        ├── 03A_test_evaluation.ipynb  # Utility notebook
        ├── 09d_sample_efficiency.ipynb # Future work
        ├── 09e_full_finetune.ipynb    # Future work
        └── 10_finbert_tone_deep_dive.ipynb  # Not cited
```

---

## 4. Risk Checklist

- [x] Archive branch created before any deletions
- [x] No kaggle_push directory contains unique code not in `notebooks/`
- [x] Paper compiles successfully after cleanup (`cd paper && bash build.sh`)
- [x] All 10 kept notebooks can be opened without import errors
- [x] `results/data_provenance_audit.json` and `results/fair_comparison_results.json` preserved
- [x] `results/nb10_parts_ab.json` deleted (belongs to NB10)

---

## 5. Detailed TODO List

### Phase 0: Pre-Flight Safety

- [x] **0.1** Ensure working tree is clean or all changes are stashed
- [x] **0.2** Create archive branch as a full snapshot of current state
  - Branch: `archive/pre-cleanup` at commit `ca17ce6`
- [x] **0.3** Verify the archive branch exists and contains everything

---

### Phase 1: Audit & Cross-Reference (Read-Only)

- [x] **1.1** Confirm which notebooks the paper cites
  - Result: `NB01 NB02 NB03 NB04 NB05 NB06 NB07 NB09` — all map to kept notebooks
- [x] **1.2** Verify each kaggle_push directory is a copy of its notebook source
  - Result: 11 of 12 IDENTICAL; kaggle_push/ (NB01) DIFFERS slightly (cell reordering, no unique code)
- [x] **1.3** Identify which scripts belong to which notebooks
  - Remove: `build_nb10.py`, `run_parts_ab.py` (NB10)
  - Keep: `data_provenance_audit.py`, `gen_audit_json.py` (NB11), `fair_claude_comparison.py` (NB03)
- [x] **1.4** Identify which results files belong to which notebooks
  - Remove: `nb10_parts_ab.json`, `confusion_ft_vs_mfb.png`, `kaggle_10_cpu/`, `kaggle_10_v2/`, `kaggle_10_v3/`
  - Keep: `data_provenance_audit.json`, `data_provenance_figure.png`, `fair_comparison_results.json`, `source8_truncation.png`
- [x] **1.5** Verify skills/ and reference/ directories are used by kept notebooks
  - All used: `financial-sentiment-engine/` (NB03), `verbalized-sampling-augment/` (NB02A), `fpb_benchmarks.md` (NB01/NB04)
- [x] **1.6** Verify data/ files are used by kept notebooks
  - All used: FPB zip, vs_augmented_errors.csv, cleaned/, processed/, raw/

---

### Phase 2: Archive Superseded Notebooks

- [x] **2.1** Create the archive subdirectory
- [x] **2.2** Move NB02 (superseded by NB02A)
- [x] **2.3** Move NB03A (not cited in paper)
- [x] **2.4** Move NB09d (future work, no results in paper)
- [x] **2.5** Move NB09e (future work, no results in paper)
- [x] **2.6** Move NB10 (not cited in paper)
- [x] **2.7** Verify exactly 10 notebooks remain — **confirmed: 10**

---

### Phase 3: Remove Old Archive Notebooks

- [x] **3.1** Delete old exploratory notebooks (`archive/exploratory/`)
- [x] **3.2** Delete old kfold prototypes (`archive/kfold/`)
- [x] **3.3** Delete old train_more_data experiments (`archive/train_more_data/`)

---

### Phase 4: Remove Kaggle Push Directories (12 dirs)

- [x] **4.1** Remove kaggle_push directories for kept notebooks (7 dirs)
- [x] **4.2** Remove kaggle_push directories for removed notebooks (5 dirs)

---

### Phase 5: Remove Kaggle Output Directories (7 dirs)

- [x] **5.1** Remove kaggle_output directories for kept notebooks (5 dirs)
- [x] **5.2** Remove kaggle_output directories for removed notebooks (2 dirs)

---

### Phase 6: Clean Results Artifacts

- [x] **6.1** Remove NB10-related results files (`nb10_parts_ab.json`, `confusion_ft_vs_mfb.png`)
- [x] **6.2** Remove NB10-related results directories (`kaggle_10_cpu/`, `kaggle_10_v2/`, `kaggle_10_v3/`)
- [x] **6.3** Verify remaining results files all map to kept notebooks
  - Confirmed: `data_provenance_audit.json`, `data_provenance_figure.png`, `fair_comparison_results.json`, `source8_truncation.png`

---

### Phase 7: Clean Scripts

- [x] **7.1** Remove NB10-specific scripts (`build_nb10.py`, `run_parts_ab.py`)
- [x] **7.2** Remove empty scaffold directories (`evaluation/`, `preprocessing/`, `training/`)
- [x] **7.3** Verify remaining scripts all map to kept notebooks
  - Confirmed: `data_provenance_audit.py`, `gen_audit_json.py`, `fair_claude_comparison.py`

---

### Phase 8: Clean Empty Top-Level Directories

- [x] **8.1** Remove empty `docs/` directory
- [x] **8.2** Remove empty `logs/` directory
- [x] **8.3** Remove `models/` (only contained `.gitkeep`)

---

### Phase 9: Update .gitignore

- [x] **9.1** Add kaggle directory patterns to prevent re-accumulation
- [x] **9.2** Clean up stale references (old `kaggle_output/trainer_output/` pattern, `.gitkeep` refs for deleted dirs)

---

### Phase 10: Verification

- [x] **10.1** Confirm exactly 10 notebooks in notebooks/ — **10**
- [x] **10.2** Confirm no kaggle directories remain at top level — **0**
- [x] **10.3** Confirm paper still compiles — **13 pages, 234691 bytes**
- [x] **10.4** Confirm all paper NB references map to existing notebooks
- [x] **10.5** Confirm results/ has only files referenced in paper — **4 files**
- [x] **10.6** Confirm scripts/ has no orphaned files — **3 files**
- [x] **10.7** Spot-check the archive — **5 notebooks in not_in_paper/**

---

### Phase 11: Commit

- [x] **11.1** Stage all changes
- [x] **11.2** Review what will be committed — 50 files changed, 191 insertions, 45,590 deletions
- [x] **11.3** Commit with descriptive message — commit `40ebc36`

---

### Phase 12: Post-Cleanup Housekeeping (Optional)

- [ ] **12.1** Consider deleting archive branch if cleanup is confirmed good
- [ ] **12.2** Consider renumbering notebooks to match paper sections
- [ ] **12.3** Update CLAUDE.md to reflect new notebook inventory
- [ ] **12.4** Remove `plan.md` and `research.md` if no longer needed, or commit them as documentation

---

## Summary: Deletion Inventory

| Category | Count | Status |
|---|---|---|
| Notebooks archived | 5 | Done |
| Old archive notebooks deleted | 3 dirs (~5 files) | Done |
| kaggle_push directories deleted | 12 | Done |
| kaggle_output directories deleted | 7 | Done |
| NB10 results files/dirs deleted | 5 | Done |
| NB10 scripts deleted | 2 | Done |
| Empty directories removed | 5 | Done |
| .gitignore updated | 1 | Done |
| **Total items removed** | **~39** | **All complete** |
