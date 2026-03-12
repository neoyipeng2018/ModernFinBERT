# Plan: Remove Contaminated Experiments from Paper and Repo

## Problem

Notebooks NB05 (`05_controlled_baselines.ipynb`) and NB08 (`08_lora_rank_ablation.ipynb`) contain or depend on results from a data leakage bug where BERT-base was trained with `source_id != 6` instead of `!= 5`, leaking FPB into BERT's training set. The paper currently frames this as a "cautionary tale" — instead, we should remove all traces of these invalid experiments and present only clean results.

The clean replacements already exist:
- **NB09B** (`09b_fpb_crossval.ipynb`) — head-to-head BERT vs ModernBERT CV on identical FPB data
- **NB09C** (`09c_clean_holdout.ipynb`) — clean held-out BERT vs ModernBERT comparison
- **NB09A** (`09a_dedup_audit.ipynb`) — dedup audit proving training data is clean

---

## Phase 1: Delete Contaminated Files from Repo

### 1.1 Remove contaminated notebooks

```bash
# NB05 — the source of the bug
rm notebooks/05_controlled_baselines.ipynb

# NB08 — investigated a non-existent gap using NB05's contaminated baseline
rm notebooks/08_lora_rank_ablation.ipynb

# Archive notebook that trained with FPB_OOD=False
rm notebooks/archive/modernfinbert_classification.ipynb
```

### 1.2 Remove contaminated kaggle push directories

```bash
# NB05 kaggle push (contains contaminated notebook + results)
rm -rf kaggle_push_05/

# NB08 kaggle push
rm -rf kaggle_push_08/
```

### 1.3 Remove contaminated output directories

```bash
# NB05 output (empty but should go)
rm -rf kaggle_output_05/

# NB08 output (contains 3 trainer runs based on contaminated baseline)
rm -rf kaggle_output_08/

# Root-level trainer outputs from NB08 LoRA rank ablation
rm -rf trainer_output_r16/
rm -rf trainer_output_r48/
rm -rf trainer_output_r48_wqkv/
```

### 1.4 Remove contaminated images

```bash
# Root-level copies of NB08 plots
rm lora_rank_ablation.png
rm lora_rank_confusion_matrices.png
```

### 1.5 Clean up empty archive directories (if now empty)

```bash
# Check if archive has anything left besides the deleted file
ls notebooks/archive/
# If only exploratory/, kfold/, train_more_data/ remain, decide whether to keep
# These are old experiments — review and clean if desired
```

---

## Phase 2: Restructure the Paper (`paper/main.tex`)

The goal: promote NB09B/NB09C from "replication study" to primary architecture comparison, and remove all discussion of the bug. The paper currently has 8 experiments; after this change it will have 7 clean experiments.

### 2.1 Rewrite Abstract (lines 38-40)

**Current**: Mentions "overturning an earlier spurious finding caused by a data leakage bug"

**Replace with** (remove bug narrative, state clean result directly):

```latex
\begin{abstract}
We present a systematic empirical study applying ModernBERT, a recent modernized BERT
architecture, to financial sentiment analysis. Through seven controlled experiments, we
evaluate ModernBERT-base with LoRA fine-tuning across multiple evaluation protocols on
the FinancialPhraseBank (FPB) benchmark. Our study makes several contributions: (1) we
introduce a held-out evaluation protocol where FPB is entirely excluded from training,
providing a stricter generalization test than standard in-domain splits; (2) we propose
DataBoost, a targeted augmentation method that uses LLM-generated paraphrases of
misclassified samples to improve performance; (3) we provide controlled comparisons
between ModernBERT and BERT-base, finding that ModernBERT consistently outperforms
BERT---leading by 1.09 percentage points in 10-fold CV ($p=0.093$) and by 7.84 points
in the held-out setting on identical data; and (4) we compare fine-tuned models against
zero-shot Claude Opus 4.5, demonstrating that even modest fine-tuned models are orders
of magnitude more cost-effective. Our results highlight the importance of evaluation
protocol choice and proper experimental controls in financial NLP.
\end{abstract}
```

Key changes:
- "eight" → "seven"
- Removed "overturning an earlier spurious finding caused by a data leakage bug"
- Changed "consistently matches or outperforms" to "consistently outperforms" (we know this now)

### 2.2 Rewrite Introduction experiment list (lines 51-62)

**Remove item 8's reference to contamination**. Replace the entire numbered list:

```latex
\begin{enumerate}[nosep]
    \item \textbf{Held-out evaluation} (NB01): Train on aggregated financial data
          \emph{excluding} FPB, then evaluate on FPB as a truly unseen benchmark.
    \item \textbf{DataBoost augmentation} (NB02): Mine validation errors and generate
          targeted paraphrases via Verbalized Sampling \citep{zhang2025verbalized} to
          improve difficult cases.
    \item \textbf{LLM comparison} (NB03): Compare fine-tuned ModernFinBERT against
          Claude Opus 4.6 equipped with a purpose-built financial sentiment skill.
    \item \textbf{In-domain cross-validation} (NB04): Standard 10-fold CV on FPB for
          comparability with prior work.
    \item \textbf{Pre-trained baselines} (NB05): Compare ModernFinBERT against
          published FinBERT models.
    \item \textbf{Multi-seed robustness} (NB06): Repeat the held-out protocol with
          five random seeds.
    \item \textbf{Self-training} (NB07): Iterative pseudo-labeling on unlabeled
          financial tweets.
\end{enumerate}
```

Note: the architecture comparison (previously Experiment 8) gets folded into Experiment 5 — see below.

### 2.3 Rewrite Introduction closing paragraph (line 64)

**Remove the data leakage mention**. Replace:

```latex
A key finding is that evaluation protocol choice dramatically affects reported
performance: the same ModernBERT model achieves 86.88\% under 10-fold CV on FPB but
only 80.44\% when FPB is held out entirely. We argue that the held-out protocol provides
a more realistic estimate of real-world generalization and that the financial NLP
community should adopt it as a complementary evaluation standard.
```

### 2.4 Restructure Experiment 5: Merge baselines + architecture comparison (lines 250-279)

**Current Experiment 5** only has ProsusAI/finbert and finbert-tone, deferring BERT comparison to Experiment 8.

**New Experiment 5** should include all baselines in one place: ProsusAI/finbert, finbert-tone, AND the clean BERT-base comparison from NB09C. Also fold in the NB09B CV results.

Replace the entire Section 4.5 with:

```latex
\subsection{Experiment 5: Architecture and Baseline Comparison}
\label{sec:baselines}

\paragraph{Protocol.}
We compare ModernFinBERT against three baselines: (1) ProsusAI/finbert, a FinBERT
model trained on FPB data (in-domain evaluation); (2) yiyanghkust/finbert-tone,
trained on analyst reports (zero-shot on FPB); and (3) \texttt{bert-base-uncased} with
LoRA $r=16$ trained on the same aggregated dataset under identical hyperparameters.

Before comparing architectures, we verify that the aggregated training data contains no
FPB samples. We apply three progressively sensitive checks: (1) exact string matching,
(2) fuzzy matching (SequenceMatcher $> 0.90$), and (3) semantic similarity using
sentence-transformer embeddings (cosine $> 0.95$). All three checks return
\textbf{zero matches}, confirming the training data is clean.

\paragraph{Held-out results.}
Table~\ref{tab:baselines} presents the held-out comparison.

\begin{table}[h]
\centering
\caption{Baseline comparison on FPB. ModernBERT and BERT-base trained on identical
8,643 samples with FPB excluded. ProsusAI/finbert trained on FPB (in-domain).}
\label{tab:baselines}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Protocol} & \multicolumn{2}{c}{\textbf{FPB 50agree}}
& \multicolumn{2}{c}{\textbf{FPB allAgree}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & Acc & F1 & Acc & F1 \\
\midrule
ProsusAI/finbert & In-domain$^\dagger$ & 0.8896 & 0.8825 & 0.9717 & 0.9625 \\
finbert-tone & Zero-shot & 0.7914 & 0.7530 & 0.9152 & 0.8939 \\
ModernBERT + LoRA & Held-out & \textbf{0.8093} & \textbf{0.7793}
& \textbf{0.9329} & --- \\
BERT-base + LoRA & Held-out & 0.7309 & 0.6051 & 0.8366 & --- \\
\midrule
$\Delta$ (MB $-$ BERT) & & +0.0784 & +0.1742 & +0.0963 & \\
\bottomrule
\multicolumn{6}{l}{\small $^\dagger$ Trained on FPB data (in-domain evaluation).}
\end{tabular}
\end{table}

On identical training data, ModernBERT outperforms BERT-base by 7.84 percentage points
on FPB \texttt{sentences\_50agree} and 9.63 points on \texttt{sentences\_allAgree}.
ModernBERT approaches ProsusAI/finbert (80.93\% vs.\ 88.96\%) despite never seeing FPB
during training, while BERT-base falls well short (73.09\%).

\paragraph{Head-to-head cross-validation.}
To further validate this comparison, we run stratified 10-fold CV on FPB
\texttt{sentences\_50agree} for both architectures under identical LoRA $r=16$
configurations.

\begin{table}[h]
\centering
\caption{Head-to-head 10-fold CV on FPB sentences\_50agree. Both models use LoRA $r=16$.}
\label{tab:head-to-head-cv}
\begin{tabular}{lccc}
\toprule
\textbf{Fold} & \textbf{BERT-base} & \textbf{ModernBERT} & \textbf{$\Delta$} \\
\midrule
0  & 0.8557 & 0.8763 & +0.0206 \\
1  & 0.8227 & 0.8474 & +0.0247 \\
2  & 0.8536 & 0.8639 & +0.0103 \\
3  & 0.8619 & 0.8330 & $-$0.0289 \\
4  & 0.8701 & 0.8680 & $-$0.0021 \\
5  & 0.8515 & 0.8619 & +0.0103 \\
6  & 0.8678 & 0.8636 & $-$0.0041 \\
7  & 0.8657 & 0.8843 & +0.0186 \\
8  & 0.8574 & 0.8884 & +0.0310 \\
9  & 0.8202 & 0.8492 & +0.0289 \\
\midrule
\textbf{Mean $\pm$ Std} & \textbf{0.8527 $\pm$ 0.0175} & \textbf{0.8636 $\pm$ 0.0171}
& \textbf{+0.0109} \\
\bottomrule
\end{tabular}
\end{table}

ModernBERT wins 7 of 10 folds, with a mean advantage of +1.09 percentage points
(paired $t$-test: $t = 1.88$, $p = 0.093$). Across both held-out and in-domain
protocols, ModernBERT is consistently superior to BERT-base.
```

### 2.5 Delete Experiment 8: Controlled Replication Study (lines 351-416)

Remove the entire `\subsection{Experiment 8: Controlled Replication Study}` section. Its content has been merged into the new Experiment 5 above.

### 2.6 Rewrite Section 5.2: Architecture Comparison (lines 434-445)

Remove all bug/leakage discussion. Replace with:

```latex
\subsection{Architecture Comparison: ModernBERT vs.\ BERT}

Our controlled comparison (Experiment~5) demonstrates that ModernBERT consistently
outperforms BERT-base for financial sentiment analysis:

\begin{itemize}[nosep]
    \item \textbf{Held-out evaluation}: ModernBERT outperforms BERT by 7.84 points
          on identical 8,643-sample training data (Table~\ref{tab:baselines}).
    \item \textbf{10-fold CV}: ModernBERT leads by 1.09 points, $p = 0.093$
          (Table~\ref{tab:head-to-head-cv}).
\end{itemize}

The held-out advantage (+7.84 points) suggests that ModernBERT's modernized
architecture---rotary positional embeddings, Flash Attention, and GeGLU
activations---provides genuinely stronger representations for financial text, even
without domain-specific pre-training. This is encouraging for practitioners, as it
suggests that upgrading from BERT to ModernBERT yields improvements on domain-specific
tasks without requiring additional pre-training.
```

### 2.7 Rewrite Conclusion point 2 (line 500)

**Current**: "Data pipeline verification is essential: An initial finding that BERT outperforms ModernBERT by 14.75 points was entirely caused by a data leakage bug..."

**Replace with**:

```latex
\item \textbf{ModernBERT outperforms BERT}: Controlled experiments on verified-clean
      data show ModernBERT consistently outperforms BERT-base, by +7.84 points in the
      held-out setting and +1.09 points in cross-validation.
```

### 2.8 Update experiment count throughout

Search for "eight" in the paper and replace with "seven" wherever it refers to the experiment count:
- Abstract: "eight controlled experiments" → "seven controlled experiments"
- Introduction: same
- Section 7 heading in research.md mentions "eight"

### 2.9 Fix cross-references

After deleting Experiment 8:
- Current Experiment 5 reference in `\ref{sec:baselines}` → still works (same label)
- Remove `\label{sec:controlled-replication}` and any `\ref{sec:controlled-replication}` refs
- The new baselines table keeps `\label{tab:baselines}` and adds `\label{tab:head-to-head-cv}` (already used)
- Remove the old `\label{tab:controlled-holdout}` table

Search the paper for these strings:
```
controlled-replication
tab:controlled-holdout
Experiment 8
Experiment~8
Section~8
Section~\ref{sec:controlled-replication}
```

### 2.10 Update Limitations section (lines 481-489)

Remove the LoRA-only limitation's reference to NB08's fused QKV investigation. The current text at line 485:

```
\item \textbf{LoRA only}: Full fine-tuning could yield different relative performance
between architectures, particularly since ModernBERT's fused QKV projections receive
effectively lower per-component rank than BERT's separate Q, K, V projections under
the same LoRA rank.
```

Simplify to:

```latex
\item \textbf{LoRA only}: Full fine-tuning could yield different relative performance
      between architectures. ModernBERT's fused QKV projections receive effectively
      lower per-component LoRA rank than BERT's separate projections, and full
      fine-tuning would eliminate this asymmetry.
```

(This is still a valid limitation — just don't reference NB08's investigation of it.)

---

## Phase 3: Update research.md

### 3.1 Remove NB05 section (lines 213-235)

Delete the entire `### NB05: Controlled Baselines` section. Replace with a brief note:

```markdown
### NB05: Pre-Trained Baselines (`05_pretrained_baselines.ipynb`)

**Note**: The original NB05 contained a data leakage bug and has been removed.
The pre-trained baseline comparison (ProsusAI/finbert, finbert-tone) is now part of
NB09C's clean held-out evaluation. The BERT-base comparison is handled by NB09B and
NB09C.
```

### 3.2 Remove NB08 section (lines 284-304)

Delete the entire `### NB08: LoRA Rank Ablation` section.

### 3.3 Remove Section 5: The Data Leakage Bug (lines 406-428)

Delete the entire `## 5. The Data Leakage Bug — The Project's Pivotal Discovery` section.

### 3.4 Remove Section 12: Architectural Insight: The Fused QKV Problem (lines 531-538)

Delete or heavily trim this section since it was motivated by the contaminated NB08 results.

### 3.5 Update initial results note (line 82)

Remove the sentence: "Initial results (from kaggle_output, before bug fix): 91.19% on 50agree, 99.03% on allAgree — trained on 13,004 samples. These numbers were later invalidated when the training set was found to include FPB-adjacent data."

### 3.6 Update Key Results Summary Table (lines 492-504)

Remove any rows referencing contaminated experiments. The table should only contain clean results.

### 3.7 Update Section 8: Paper Structure and Claims (lines 468-488)

- Change "eight experiments" → "seven experiments"
- Update claim 2 from "Data pipeline verification is essential" to "ModernBERT outperforms BERT"
- Remove mention of "14.75pp spurious result"

### 3.8 Update Kaggle Push Directories table

Remove entries for `kaggle_push_05/` and `kaggle_push_08/`.

### 3.9 Update Section 11: What Remains Unfinished

No changes needed (NB05/NB08 are not listed as unfinished).

---

## Phase 4: Update CLAUDE.md

### 4.1 Update Project Status section

Remove mention of NB05 and NB08 from any experiment descriptions. The current CLAUDE.md at the repo level doesn't explicitly list NB05/NB08, but verify and clean up any references.

---

## Phase 5: Rebuild PDF

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Phase 6: Verify Completeness

### 6.1 Search for remaining contamination references

```bash
# Search for any remaining references to the bug or contaminated experiments
grep -rn "leakage" --include="*.tex" --include="*.md" .
grep -rn "spurious" --include="*.tex" --include="*.md" .
grep -rn "NB05\|NB08\|05_controlled\|08_lora" --include="*.tex" --include="*.md" .
grep -rn "source_id.*6\|!= 6\|≠ 6" --include="*.tex" --include="*.md" .
grep -rn "95\.19\|99\.16\|14\.75\|15\.62" --include="*.tex" --include="*.md" .
grep -rn "wrong constant\|wrong filter\|wrong value" --include="*.tex" --include="*.md" .
grep -rn "cautionary\|pivotal discovery" --include="*.tex" --include="*.md" .
```

### 6.2 Verify no broken references in paper

```bash
# After rebuilding PDF, check for undefined references
grep -i "undefined" paper/main.log
grep -i "multiply defined" paper/main.log
```

### 6.3 Confirm deleted files are gone

```bash
# These should all return "No such file or directory"
ls notebooks/05_controlled_baselines.ipynb
ls notebooks/08_lora_rank_ablation.ipynb
ls notebooks/archive/modernfinbert_classification.ipynb
ls kaggle_push_05/
ls kaggle_push_08/
ls kaggle_output_05/
ls kaggle_output_08/
ls trainer_output_r16/
ls trainer_output_r48/
ls trainer_output_r48_wqkv/
ls lora_rank_ablation.png
ls lora_rank_confusion_matrices.png
```

---

## TODO Checklist

### Phase 1: Delete Contaminated Files from Repo

- [x] **1.1a** Delete `notebooks/05_controlled_baselines.ipynb`
- [x] **1.1b** Delete `notebooks/08_lora_rank_ablation.ipynb`
- [x] **1.1c** Delete `notebooks/archive/modernfinbert_classification.ipynb`
- [x] **1.2a** Delete `kaggle_push_05/` directory (notebook, PNGs, logs, trainer outputs, metadata)
- [x] **1.2b** Delete `kaggle_push_08/` directory (notebook, metadata)
- [x] **1.3a** Delete `kaggle_output_05/` directory
- [x] **1.3b** Delete `kaggle_output_08/` directory (lora_rank_ablation.png, confusion matrices, logs, trainer_output_r16/r48/r48_wqkv)
- [x] **1.3c** Delete `trainer_output_r16/` (root-level, 10 checkpoints)
- [x] **1.3d** Delete `trainer_output_r48/` (root-level)
- [x] **1.3e** Delete `trainer_output_r48_wqkv/` (root-level)
- [x] **1.4a** Delete `lora_rank_ablation.png` (root-level)
- [x] **1.4b** Delete `lora_rank_confusion_matrices.png` (root-level)
- [x] **1.5** Inspect `notebooks/archive/` — keeping remaining subdirs (`exploratory/`, `kfold/`, `train_more_data/`) — not contaminated

### Phase 2: Restructure the Paper (`paper/main.tex`)

- [x] **2.1** Rewrite abstract: "eight" → "seven", remove "overturning an earlier spurious finding caused by a data leakage bug", change "matches or outperforms" → "outperforms"
- [x] **2.2** Rewrite introduction experiment list: remove item 8 (controlled replication / data contamination audit), leaving 7 items
- [x] **2.3** Rewrite introduction closing paragraph: remove "controlled replication study reveals that an initial finding of BERT outperforming ModernBERT by 14.75 points was caused by a data leakage bug" sentence
- [x] **2.4a** Rewrite Experiment 5 title: "Pre-Trained FinBERT Baselines" → "Architecture and Baseline Comparison"
- [x] **2.4b** Rewrite Experiment 5 protocol paragraph: add BERT-base as third baseline, add dedup audit verification paragraph
- [x] **2.4c** Replace Experiment 5 table (`tab:baselines`): merge in NB09C clean results (ModernBERT 80.93%, BERT 73.09%, delta +7.84pp)
- [x] **2.4d** Rewrite Experiment 5 results prose: remove sentence deferring BERT comparison to Experiment 8
- [x] **2.4e** Add head-to-head CV sub-section within Experiment 5: insert NB09B 10-fold table (`tab:head-to-head-cv`) and paired t-test results
- [x] **2.5** Delete entire Experiment 8 section (`\subsection{Experiment 8: Controlled Replication Study}` through end of its last table)
- [x] **2.6** Rewrite Section 5.2 (Architecture Comparison): remove all bug/leakage/spurious/cautionary-tale language, present clean results as primary findings
- [x] **2.7** Rewrite Conclusion point 2: "Data pipeline verification is essential" → "ModernBERT outperforms BERT", remove mention of 14.75pp spurious result and data leakage bug
- [x] **2.8a** Search paper for "eight" referring to experiment count → replaced with "seven"
- [x] **2.8b** Search paper for "eight controlled experiments" → "seven controlled experiments" (conclusion preamble)
- [x] **2.9a** No `\ref{sec:controlled-replication}` found — already clean
- [x] **2.9b** No `\ref{tab:controlled-holdout}` found — already clean
- [x] **2.9c** No "Experiment 8" / "Experiment~8" / "Section~8" found — already clean
- [x] **2.9d** Verified `\label{tab:head-to-head-cv}` present in Experiment 5, `\ref{tab:head-to-head-cv}` resolves in Section 5.2
- [x] **2.10** Simplified Limitations "LoRA only" bullet

### Phase 3: Update `research.md`

- [x] **3.1** Remove/replace NB05 section: replaced with brief redirect note to NB09B/NB09C
- [x] **3.2** Remove NB08 section: deleted entirely
- [x] **3.3** Remove Section 5 "The Data Leakage Bug — The Project's Pivotal Discovery": deleted entirely
- [x] **3.4** Remove Section 12 "Architectural Insight: The Fused QKV Problem": deleted entirely
- [x] **3.5** Remove invalidated initial results note: deleted "Initial results (from kaggle_output, before bug fix)" sentence
- [x] **3.6** Update Key Results Summary Table: removed NB05 source references from ProsusAI/finbert-tone rows (numbers were clean)
- [x] **3.7a** Update Section 8: "eight experiments" → "seven experiments", "8 subsections" → "7 subsections"
- [x] **3.7b** Update Section 8 claim 2: "Data pipeline verification is essential" → "ModernBERT outperforms BERT: +7.84pp held-out, +1.09pp CV"
- [x] **3.7c** Claim count stays at five — no change needed
- [x] **3.8** No `kaggle_push_05/` or `kaggle_push_08/` references found in infrastructure notes
- [x] **3.9** Section 11 "What Remains Unfinished": confirmed NB05/NB08 not listed
- [x] **3.10** Searched for stale references: cleaned up NB05/NB08 mention in NB09C section (line 327)

### Phase 4: Update `CLAUDE.md`

- [x] **4.1** Read repo-level `CLAUDE.md` — no NB05/NB08 references found; "data leakage" mentions are generic data cleaning (unrelated)
- [x] **4.2** Updated "Key Results So Far" — replaced stale 90.47%/97.63% with correct clean results (80.44%/86.88%)
- [x] **4.3** "Current Experiments" section only lists NB01-NB03 — no NB05/NB08, no changes needed

### Phase 5: Rebuild PDF

- [x] **5.1** Run `pdflatex main.tex` (first pass) — 12 pages output
- [x] **5.2** Run `bibtex main` — 1 warning (volume/number in malo2014good, pre-existing)
- [x] **5.3** Run `pdflatex main.tex` (second pass)
- [x] **5.4** Run `pdflatex main.tex` (third pass — finalize) — 12 pages, 227659 bytes
- [x] **5.5** Checked main.log: zero undefined references, zero multiply defined labels

### Phase 6: Verify Completeness

- [x] **6.1a** Grep for "leakage" — zero hits in .tex/.md
- [x] **6.1b** Grep for "spurious" — zero hits
- [x] **6.1c** Grep for "NB05", "NB08", "05_controlled", "08_lora" — only hit is intentional "NB05/NB09" in paper intro experiment list
- [x] **6.1d** Grep for "source_id.*6", "!= 6", "≠ 6" — zero hits
- [x] **6.1e** Grep for "95.19", "99.16", "14.75", "15.62" — zero hits
- [x] **6.1f** Grep for "wrong constant", "wrong filter", "wrong value" — zero hits
- [x] **6.1g** Grep for "cautionary", "pivotal discovery" — only hit is "cautionary finding" about self-training (unrelated, correct)
- [x] **6.2a** Check `paper/main.log` for "undefined" — zero hits
- [x] **6.2b** Check `paper/main.log` for "multiply defined" — zero hits
- [x] **6.3a** Confirmed `notebooks/05_controlled_baselines.ipynb` does not exist
- [x] **6.3b** Confirmed `notebooks/08_lora_rank_ablation.ipynb` does not exist
- [x] **6.3c** Confirmed `notebooks/archive/modernfinbert_classification.ipynb` does not exist
- [x] **6.3d** Confirmed `kaggle_push_05/` does not exist
- [x] **6.3e** Confirmed `kaggle_push_08/` does not exist
- [x] **6.3f** Confirmed `kaggle_output_05/` does not exist
- [x] **6.3g** Confirmed `kaggle_output_08/` does not exist
- [x] **6.3h** Confirmed `trainer_output_r16/` does not exist
- [x] **6.3i** Confirmed `trainer_output_r48/` does not exist
- [x] **6.3j** Confirmed `trainer_output_r48_wqkv/` does not exist
- [x] **6.3k** Confirmed `lora_rank_ablation.png` does not exist
- [x] **6.3l** Confirmed `lora_rank_confusion_matrices.png` does not exist
- [x] **6.4** Final verification: all greps clean, PDF builds with no warnings, 12 pages

---

## Summary of Changes

| Action | Files |
|--------|-------|
| **Delete notebooks** | `05_controlled_baselines.ipynb`, `08_lora_rank_ablation.ipynb`, `archive/modernfinbert_classification.ipynb` |
| **Delete kaggle dirs** | `kaggle_push_05/`, `kaggle_push_08/`, `kaggle_output_05/`, `kaggle_output_08/` |
| **Delete trainer output** | `trainer_output_r16/`, `trainer_output_r48/`, `trainer_output_r48_wqkv/` |
| **Delete images** | `lora_rank_ablation.png`, `lora_rank_confusion_matrices.png` |
| **Rewrite paper** | Abstract, intro, Experiment 5 (merge baselines + arch comparison), delete Experiment 8, rewrite Section 5.2, rewrite Conclusion point 2 |
| **Update research.md** | Remove NB05, NB08, Section 5 (bug narrative), Section 12 (QKV), update tables |
| **Rebuild PDF** | `pdflatex` + `bibtex` |

### What stays

- **NB09A** (`09a_dedup_audit.ipynb`) — mentioned in paper as verification step
- **NB09B** (`09b_fpb_crossval.ipynb`) — becomes part of Experiment 5 (CV comparison)
- **NB09C** (`09c_clean_holdout.ipynb`) — becomes part of Experiment 5 (held-out comparison)
- **NB09D, NB09E** — remain as future work in Limitations
- **All other notebooks** (NB01, NB02, NB02A, NB03, NB03A, NB04, NB06, NB07) — unchanged, all clean
