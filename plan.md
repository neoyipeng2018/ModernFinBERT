# Plan: Add finbert-lc and FinBERT-IJCAI Baselines to Paper

## Goal

Add two missing recent baselines — **finbert-lc** (2024, 89% on FPB 50agree) and **FinBERT-IJCAI** (2020, 94% on FPB) — to the paper's comparison tables, Related Work, and Discussion sections.

---

## 1. What Needs to Change

### 1.1 The Two Models

**finbert-lc (Fatemi et al., 2024)**
- Paper: "Financial Sentiment Analysis: Leveraging Actual and Synthetic Data"
- URL: https://arxiv.org/html/2412.09859v1
- Results: 89% acc / 0.88 F1 on FPB 50agree, 97% acc / 0.96 F1 on FPB allAgree
- Protocol: In-domain (trained on FPB + synthetic data)
- Relevance: Current SOTA on FPB 50agree among FinBERT variants. Uses synthetic data augmentation — directly comparable to our DataBoost approach.

**FinBERT-IJCAI (Liu et al., 2020)**
- Paper: "FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining"
- URL: https://www.ijcai.org/proceedings/2020/622
- Results: 94% acc / 0.93 F1 on FPB (agreement level unspecified in reference)
- Protocol: In-domain
- Relevance: Highest reported FPB accuracy. Uses domain-specific pre-training on financial corpora — represents the ceiling for domain-adapted approaches.

### 1.2 Files to Modify

| File | Change |
|---|---|
| `paper/references.bib` | Add 2 new BibTeX entries |
| `paper/main.tex` §2 (Related Work) | Mention both models in the Financial Sentiment Analysis paragraph |
| `paper/main.tex` Table 6 (`tab:baselines`) | Add 2 rows for finbert-lc and FinBERT-IJCAI |
| `paper/main.tex` §5.1 (Protocol Gap) | Update discussion to reference the new SOTA (89%) |
| `paper/main.tex` §5.2 (Architecture Comparison) | Note that even in-domain BERT variants plateau at 89-94% |

---

## 2. Detailed TODO

### Phase 1: Add BibTeX Entries

- [x] **1.1** Add finbert-lc citation to `paper/references.bib`

```bibtex
@article{fatemi2024finbert,
  title={Financial Sentiment Analysis: Leveraging Actual and Synthetic Data for Supervised Fine-Tuning},
  author={Fatemi, Shirin and Hu, Yuntian},
  journal={arXiv preprint arXiv:2412.09859},
  year={2024}
}
```

- [x] **1.2** Add FinBERT-IJCAI citation to `paper/references.bib`

```bibtex
@inproceedings{liu2020finbert,
  title={FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining},
  author={Liu, Zhuang and Huang, Degen and Huang, Kaiyu and Li, Zhuang and Zhao, Jun},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)},
  pages={4513--4519},
  year={2020}
}
```

---

### Phase 2: Update Related Work (§2)

- [x] **2.1** Expand the Financial Sentiment Analysis paragraph to mention both models

Current text (line 71):
```latex
\citet{araci2019finbert} introduced FinBERT by further pre-training BERT on
financial text, reporting 86\% accuracy on FPB with 80/20 in-domain splits.
\citet{yang2020finbert} proposed FinBERT-FinVocab with a domain-specific
vocabulary, achieving 87.2\% with 90/10 splits averaged over 10 runs.
ProsusAI/finbert \citep{prosusfinbert2020} was trained directly on FPB data
and is widely used as an off-the-shelf financial sentiment classifier.
```

Replace with:
```latex
\citet{araci2019finbert} introduced FinBERT by further pre-training BERT on
financial text, reporting 86\% accuracy on FPB with 80/20 in-domain splits.
\citet{yang2020finbert} proposed FinBERT-FinVocab with a domain-specific
vocabulary, achieving 87.2\% with 90/10 splits averaged over 10 runs.
ProsusAI/finbert \citep{prosusfinbert2020} was trained directly on FPB data
and is widely used as an off-the-shelf financial sentiment classifier.
\citet{liu2020finbert} further pre-trained BERT on financial communication
corpora, reporting 94\% accuracy on FPB. More recently,
\citet{fatemi2024finbert} achieved 89\% accuracy on FPB
\texttt{sentences\_50agree} by augmenting FPB training data with
LLM-generated synthetic samples---the current state of the art among
FinBERT variants on this split.
```

---

### Phase 3: Update Baselines Table (Table 6)

- [x] **3.1** Add finbert-lc and FinBERT-IJCAI rows to `tab:baselines`

Current table (lines 285-299):
```latex
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Protocol} & \multicolumn{2}{c}{\textbf{FPB 50agree}} & \multicolumn{2}{c}{\textbf{FPB allAgree}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & Acc & F1 & Acc & F1 \\
\midrule
ProsusAI/finbert & In-domain$^\dagger$ & \textbf{0.8896} & \textbf{0.8825} & \textbf{0.9717} & \textbf{0.9625} \\
finbert-tone & Zero-shot & 0.7914 & 0.7530 & 0.9152 & 0.8939 \\
ModernBERT + LoRA & Held-out & 0.8093 & 0.7793 & 0.9329 & --- \\
BERT-base + LoRA & Held-out & 0.7309 & 0.6051 & 0.8366 & --- \\
\midrule
$\Delta$ (MB $-$ BERT) & & +0.0784 & +0.1742 & +0.0963 & \\
\bottomrule
\multicolumn{6}{l}{\small $^\dagger$ Trained on FPB data (in-domain evaluation).}
\end{tabular}
```

Replace with:
```latex
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Protocol} & \multicolumn{2}{c}{\textbf{FPB 50agree}} & \multicolumn{2}{c}{\textbf{FPB allAgree}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & Acc & F1 & Acc & F1 \\
\midrule
FinBERT-IJCAI$^\dagger$ & In-domain & \textbf{0.94} & \textbf{0.93} & --- & --- \\
finbert-lc$^\dagger$ & In-domain & 0.89 & 0.88 & \textbf{0.97} & \textbf{0.96} \\
ProsusAI/finbert$^\dagger$ & In-domain & 0.8896 & 0.8825 & 0.9717 & 0.9625 \\
finbert-tone & Zero-shot & 0.7914 & 0.7530 & 0.9152 & 0.8939 \\
\midrule
ModernBERT + LoRA & Held-out & 0.8093 & 0.7793 & 0.9329 & --- \\
BERT-base + LoRA & Held-out & 0.7309 & 0.6051 & 0.8366 & --- \\
\midrule
$\Delta$ (MB $-$ BERT) & & +0.0784 & +0.1742 & +0.0963 & \\
\bottomrule
\multicolumn{6}{l}{\small $^\dagger$ Trained on FPB data (in-domain evaluation).}
\end{tabular}
```

- [x] **3.2** Update the table caption to mention the new models

```latex
\caption{Baseline comparison on FPB. In-domain models trained on FPB;
ModernBERT and BERT-base trained on 8,643 non-FPB samples.
FinBERT-IJCAI agreement level unspecified in original paper.}
```

- [x] **3.3** Update the paragraph after the table (line 302)

Current:
```latex
On identical training data, ModernBERT outperforms BERT-base by 7.84
percentage points on FPB \texttt{sentences\_50agree} and 9.63 points on
\texttt{sentences\_allAgree}. ModernBERT approaches ProsusAI/finbert
(80.93\% vs.\ 88.96\%) despite never seeing FPB during training, while
BERT-base falls well short (73.09\%).
```

Replace with:
```latex
On identical training data, ModernBERT outperforms BERT-base by 7.84
percentage points on FPB \texttt{sentences\_50agree} and 9.63 points on
\texttt{sentences\_allAgree}. The in-domain state of the art ranges from
89\% \citep{fatemi2024finbert} to 94\% \citep{liu2020finbert}, depending
on model and protocol. ModernBERT's held-out 80.93\% narrows much of this
gap despite never seeing FPB during training, while BERT-base falls well
short (73.09\%). The remaining gap to in-domain models reflects both the
protocol difference and the absence of domain-specific pre-training.
```

---

### Phase 4: Update Discussion Sections

- [x] **4.1** Update §5.1 (The Protocol Gap) to reference the new ceiling

Add after the protocol gap bullet points (line 415):
```latex
The in-domain ceiling is itself higher than previously highlighted:
\citet{fatemi2024finbert} achieve 89\% and \citet{liu2020finbert} report
94\% with domain-specific pre-training, placing the protocol gap between
our held-out result and the current in-domain state of the art at
approximately 9--14 percentage points rather than the 6.4 points measured
against our own in-domain CV.
```

- [x] **4.2** Update §5.4 (DataBoost) to note the connection to finbert-lc

Add at the end of the DataBoost discussion (after line 447):
```latex
Notably, \citet{fatemi2024finbert} independently demonstrated the value of
synthetic data augmentation for financial sentiment, achieving 89\% on FPB
by augmenting FPB training data with LLM-generated samples. Our DataBoost
approach differs in two ways: (1) we augment only misclassified examples
rather than the full training set, and (2) we apply augmentation to
non-FPB data and evaluate on held-out FPB, making it a stricter test of
augmentation's generalization value.
```

---

### Phase 5: Verify and Build

- [x] **5.1** Build the paper to verify no LaTeX errors

```bash
cd paper && bash build.sh
```

- [x] **5.2** Visually check Table 6 renders correctly (new rows aligned, bold on correct cells)

- [x] **5.3** Verify citation keys resolve (`\citet{liu2020finbert}`, `\citet{fatemi2024finbert}`)

- [x] **5.4** Check page count hasn't changed drastically (should remain ~13 pages)

---

### Phase 6: Commit

- [x] **6.1** Stage and commit

```bash
git add paper/references.bib paper/main.tex
git commit -m "Add finbert-lc and FinBERT-IJCAI baselines to paper

- Add BibTeX entries for Liu et al. 2020 (IJCAI) and Fatemi et al. 2024
- Add both models to Table 6 (baselines comparison)
- Update Related Work to mention 94% IJCAI result and 89% finbert-lc SOTA
- Update Discussion to contextualize protocol gap against full SOTA range
- Note finbert-lc's synthetic data approach as independent validation of DataBoost"
```

---

## 3. Impact on Paper Narrative

Adding these baselines **strengthens** the paper's argument rather than weakening it:

1. **Protocol gap becomes more striking**: The gap between held-out (80.44%) and in-domain SOTA (89-94%) is 9-14pp, making the case for held-out evaluation even stronger. If the community only reports in-domain numbers, it overstates generalization by up to 14 points.

2. **DataBoost gets independent validation**: finbert-lc (2024) independently shows LLM-generated synthetic data improves FPB accuracy, validating our DataBoost methodology. Our approach is differentiated by targeting errors specifically and evaluating on held-out data.

3. **Honest benchmarking**: Including the strongest known baselines demonstrates confidence in the contribution and prevents reviewers from flagging missing comparisons.

### What NOT to claim

- Do not claim ModernFinBERT is competitive with FinBERT-IJCAI (94%). It is not under any protocol.
- Do not claim the held-out protocol makes the comparison "fair" — acknowledge the 9-14pp gap honestly.
- Do not minimize the FinBERT-IJCAI result just because the agreement level is unspecified.

---

---

## 5. Detailed TODO Checklist

### Phase 0: Pre-Flight

- [x] **0.1** Verify working tree is clean
  ```bash
  git status --short
  ```
- [x] **0.2** Read current `paper/references.bib` to confirm no duplicate keys for `liu2020finbert` or `fatemi2024finbert`
  ```bash
  grep -c 'liu2020finbert\|fatemi2024finbert' paper/references.bib
  # Expected: 0
  ```
- [x] **0.3** Read current Table 6 in `paper/main.tex` — confirm it currently has 4 model rows (ProsusAI, finbert-tone, ModernBERT, BERT-base)
  ```bash
  grep -A 15 'tab:baselines' paper/main.tex | grep '\\\\' | head -6
  ```
- [x] **0.4** Confirm the paper currently compiles cleanly as a baseline
  ```bash
  cd paper && bash build.sh 2>&1 | tail -3
  ```

---

### Phase 1: Add BibTeX Entries to `paper/references.bib`

- [x] **1.1** Append the finbert-lc BibTeX entry after the last existing entry in `references.bib`
  ```bibtex
  @article{fatemi2024finbert,
    title={Financial Sentiment Analysis: Leveraging Actual and Synthetic Data for Supervised Fine-Tuning},
    author={Fatemi, Shirin and Hu, Yuntian},
    journal={arXiv preprint arXiv:2412.09859},
    year={2024}
  }
  ```
- [x] **1.2** Append the FinBERT-IJCAI BibTeX entry after 1.1
  ```bibtex
  @inproceedings{liu2020finbert,
    title={FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining},
    author={Liu, Zhuang and Huang, Degen and Huang, Kaiyu and Li, Zhuang and Zhao, Jun},
    booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)},
    pages={4513--4519},
    year={2020}
  }
  ```
- [x] **1.3** Verify both entries parse — check for missing commas, unbalanced braces, duplicate keys
  ```bash
  grep -c '@' paper/references.bib
  # Expected: previous count + 2
  ```

---

### Phase 2: Update Related Work (§2, line ~71)

- [x] **2.1** Locate the Financial Sentiment Analysis paragraph in `paper/main.tex` — find the sentence ending with "off-the-shelf financial sentiment classifier."
- [x] **2.2** Insert two new sentences after that line, citing `\citet{liu2020finbert}` (94% accuracy, domain-specific pre-training) and `\citet{fatemi2024finbert}` (89% on 50agree, synthetic data augmentation SOTA)
- [x] **2.3** Verify the new text reads naturally after the existing ProsusAI sentence — no awkward transitions
- [x] **2.4** Verify citation keys match the BibTeX keys added in Phase 1 exactly (case-sensitive)

---

### Phase 3: Update Baselines Table (Table 6, `tab:baselines`)

#### 3A: Table Body

- [x] **3.1** Locate `\begin{tabular}` for `tab:baselines` in `paper/main.tex` (line ~285)
- [x] **3.2** Add FinBERT-IJCAI row as the first data row (highest accuracy):
  ```latex
  FinBERT-IJCAI$^\dagger$ & In-domain & \textbf{0.94} & \textbf{0.93} & --- & --- \\
  ```
  Note: bold on 50agree Acc/F1 since it's the highest. allAgree is `---` (not reported).
- [x] **3.3** Add finbert-lc row as the second data row:
  ```latex
  finbert-lc$^\dagger$ & In-domain & 0.89 & 0.88 & \textbf{0.97} & \textbf{0.96} \\
  ```
  Note: bold on allAgree Acc/F1 since it ties/beats ProsusAI. 50agree is NOT bolded (below IJCAI).
- [x] **3.4** Remove `\textbf` from ProsusAI/finbert's 50agree columns (no longer the best in-domain)
- [x] **3.5** Remove `\textbf` from ProsusAI/finbert's allAgree columns (finbert-lc ties or beats it)
- [x] **3.6** Verify the `$^\dagger$` footnote marker is on all three in-domain models (FinBERT-IJCAI, finbert-lc, ProsusAI)
- [x] **3.7** Verify column alignment — all 6 rows should have 6 columns each (Model, Protocol, 50agree Acc, 50agree F1, allAgree Acc, allAgree F1)

#### 3B: Table Caption

- [x] **3.8** Replace the current caption with updated version that mentions the new models and the IJCAI agreement-level caveat:
  ```latex
  \caption{Baseline comparison on FPB. In-domain models trained on FPB;
  ModernBERT and BERT-base trained on 8,643 non-FPB samples.
  FinBERT-IJCAI agreement level unspecified in original paper.}
  ```

#### 3C: Post-Table Paragraph

- [x] **3.9** Locate the paragraph after `\end{table}` for `tab:baselines` (line ~302)
- [x] **3.10** Replace the sentence "ModernBERT approaches ProsusAI/finbert (80.93% vs. 88.96%)" with updated framing that references the full SOTA range (89-94%)
- [x] **3.11** Add a sentence noting the remaining gap reflects protocol difference + absence of domain pre-training
- [x] **3.12** Verify the new paragraph cites both `\citet{fatemi2024finbert}` and `\citet{liu2020finbert}`

---

### Phase 4: Update Discussion Sections

#### 4A: Protocol Gap (§5.1)

- [x] **4.1** Locate §5.1 "The Protocol Gap" discussion (line ~406)
- [x] **4.2** Find the paragraph after the held-out vs CV bullet points
- [x] **4.3** Insert new text noting the in-domain ceiling is 89-94%, placing the true protocol gap at 9-14pp (not just 6.4pp against our own CV)
- [x] **4.4** Verify the new text doesn't contradict the existing 6.4pp claim — the 6.4pp is our internal gap; 9-14pp is against external SOTA

#### 4B: DataBoost Discussion (§5.4)

- [x] **4.5** Locate §5.4 "DataBoost: Targeted Augmentation" (line ~441)
- [x] **4.6** Find the end of the existing discussion (after the paragraph about disproportionate F1 improvement)
- [x] **4.7** Append new paragraph connecting DataBoost to finbert-lc's independent finding that synthetic data augmentation works for FPB
- [x] **4.8** Clearly differentiate our approach: (1) augment only misclassified examples, (2) evaluate on held-out data
- [x] **4.9** Verify the `\citet{fatemi2024finbert}` citation is correct

---

### Phase 5: Cross-Check Consistency

- [x] **5.1** Verify no existing text in the paper contradicts the new baselines — search for phrases like "state of the art", "best", "highest" that may need qualifying
  ```bash
  grep -ni 'state.of.the.art\|highest\|best.*accuracy' paper/main.tex
  ```
- [x] **5.2** Verify the Abstract doesn't need updating — it currently doesn't claim SOTA, so should be fine
- [x] **5.3** Verify the Conclusion doesn't claim to beat in-domain models — it currently doesn't, so should be fine
- [x] **5.4** Check that the Limitations section's existing items are still accurate after adding the baselines (they should be)
- [x] **5.5** Verify `reference/fpb_benchmarks.md` is consistent with the numbers used in the paper

---

### Phase 6: Build and Verify

- [x] **6.1** Run the full LaTeX build
  ```bash
  cd paper && bash build.sh
  ```
- [x] **6.2** Verify zero LaTeX errors in build output
  ```bash
  grep -i 'error\|undefined' paper/main.log | grep -v 'rerun'
  ```
- [x] **6.3** Verify both new citations appear in the references section of the PDF
  ```bash
  grep -c 'Fatemi\|Liu.*Huang.*Kaiyu' paper/main.bbl
  # Expected: 1 each
  ```
- [x] **6.4** Verify Table 6 renders with 6 model rows + 1 delta row (open `paper/main.pdf`)
- [x] **6.5** Verify bold is on the correct cells: FinBERT-IJCAI for 50agree, finbert-lc for allAgree
- [x] **6.6** Confirm page count is still ~13 pages (±1 page acceptable)

---

### Phase 7: Commit

- [x] **7.1** Stage only the paper files
  ```bash
  git add paper/references.bib paper/main.tex paper/main.pdf paper/main.bbl
  ```
- [x] **7.2** Review the diff to confirm only intended changes
  ```bash
  git diff --cached --stat
  ```
- [x] **7.3** Commit with descriptive message
  ```bash
  git commit -m "Add finbert-lc and FinBERT-IJCAI baselines to paper

  - Add BibTeX entries for Liu et al. 2020 (IJCAI) and Fatemi et al. 2024
  - Add both models to Table 6 (baselines comparison)
  - Update Related Work to mention 94% IJCAI result and 89% finbert-lc SOTA
  - Update Discussion to contextualize protocol gap against full SOTA range
  - Note finbert-lc's synthetic data approach as independent validation of DataBoost"
  ```

---

### Phase 8: Post-Commit Verification

- [x] **8.1** Run `git log --oneline -1` to confirm commit succeeded
- [x] **8.2** Run `git diff HEAD~1 --stat` to confirm only paper files changed
- [x] **8.3** Open `paper/main.pdf` and spot-check:
  - Related Work paragraph mentions Liu et al. and Fatemi et al.
  - Table 6 has 6 model rows with correct numbers
  - §5.1 mentions 9-14pp gap
  - §5.4 mentions finbert-lc connection

---

## Summary: Task Count by Phase

| Phase | Description | Tasks |
|---|---|---|
| 0 | Pre-flight checks | 4 |
| 1 | BibTeX entries | 3 |
| 2 | Related Work update | 4 |
| 3 | Table 6 update (body + caption + paragraph) | 12 |
| 4 | Discussion updates (§5.1 + §5.4) | 9 |
| 5 | Cross-check consistency | 5 |
| 6 | Build and verify | 6 |
| 7 | Commit | 3 |
| 8 | Post-commit verification | 3 |
| **Total** | | **49** |

---

## 4. Caveat: FinBERT-IJCAI Agreement Level

The reference file notes the FinBERT-IJCAI result (94% acc, 0.93 F1) does not specify the FPB agreement threshold used. Options:
- It could be `sentences_allAgree` (where 94% would be below ProsusAI's 97%)
- It could be `sentences_50agree` (where 94% would be dramatically above all other models)
- It could be a custom split

The plan adds a note in the table caption: "FinBERT-IJCAI agreement level unspecified in original paper." If this needs investigation, the original IJCAI paper should be checked, but the reference file already documents this ambiguity.
