# Major Revision Checklist — OGR / MFC-A Paper

Target file: `springer_format/overleaf_project/main.tex`
Legend: R1/R2/R3 = Reviewer, #n = comment number. 🧪 = needs new experiments.

## STATUS (2026-10-17) — revision complete except the items marked ❌ / 👤

Deliverables (springer_format/):
- overleaf_project/main.tex + main.pdf ........ revised manuscript (15 pp, now incl. AVE)
- overleaf_project/main_diff.pdf .............. changes marked vs. first submission (22 pp)
- overleaf_project/response_letter.pdf ........ point-by-point reply, all 27 comments (7 pp)
- revision_submission.zip ..................... everything above, compiles standalone
- overleaf_project/main_original_submission.tex  backup of the submitted version

Code & data (revision/): bench.py, extract_crema_av.py, extract_degraded.py, prep_crema_spec.py,
make_grid.py, select_gamma.py, aggregate.py; results/ = 1,000+ per-seed JSON logs; out/summary.md = all numbers.
GPU server copy: gpu:~/ogr_revision (~1.3 GB on the nearly full system disk — delete when done); AVE videos in gpu:/mnt/hdd/ogr_ave (~12 GB, HDD has 8 TB free).

Done: Phases 0–9 (10 seeds, 4 disjoint splits, actor-disjoint audio–video CREMA-D, 5 baselines,
component ablation, sensitivity, capacity, blur/occlusion/low-light robustness, cost breakdown,
γ selected without test data, method/notation fixes, 8 new references, full rewrite, response letter, latexdiff).
✅ Second real AV dataset: AVE (4,134 clips, 28 classes, official test split) — MFC-A: acc kept, OGR gap −63% (p=0.03), robustness up (p<0.04); OGM-GE best on AVE.
❌ Not done: AGM / PMR / OLM baselines (stated as limitation).
👤 For you: native-speaker / Grammarly proofread; verify author bios; decide on public code repo URL.

---


## Phase 0 — Fix the experiment code (do this first)

- [ ] Write one clean script (`revision_benchmark.py`) that runs every method with the same model, splits, optimizer and seeds.
- [ ] Use 4 separate splits: **train / controller-val / tuning-val / test**. The controller sees only controller-val. Report every final number on test. (R3 #2)
- [ ] Seeds: 5 runs (e.g. 0, 1, 2, 3, 4) for every method on every dataset. (R1 #12, R2, R3 #3)
- [ ] Save per-epoch logs to CSV for each run: train/val loss per modality, OGR_v, OGR_a, λ_v, λ_a, accuracy. (R3 #7)
- [ ] Measure overhead honestly: time and FLOPs of the extra per-epoch train/val loss pass, counted separately from the controller arithmetic. (R1 #13, #14)

## Phase 1 — Datasets 🧪

- [ ] **Synthetic benchmark:** describe it truthfully as an sklearn `make_classification` benchmark (6 classes, 64-d "vision", 32-d "audio", noise 0.2 / 0.9). Remove the claims about CIFAR-10 images, `Linear(3072,512)` and VGGish. Give enough detail to reproduce it. (R2, R3 #5)
- [ ] Remove or rewrite Table 1 (CIFAR-10 / CREMA-D specs) and the CIFAR/spectrogram sample figures if they no longer describe what was run.
- [ ] **Real audio–video:** download CREMA-D video + audio (same clip → genuinely paired). Video = face frames (pretrained CNN features); Audio = mel/MFCC features. Split by **actor** so no speaker appears in both train and test. (R3 #1, #5)
- [ ] Keep the spectrogram two-stream experiment only as a secondary "feature-view" test and say clearly that it is not true multimodal. (R3 #1)
- [ ] Optional second real dataset (e.g. AVE or Kinetics-Sounds subset) if time allows.

## Phase 2 — Baselines 🧪 (R1 #10, R3 #6)

- [ ] Baseline (equal weights)
- [ ] Gradient Blending (also on CREMA-D; currently missing there)
- [ ] OGM-GE
- [ ] At least one recent method: AGM / PMR / MMPareto / OLM (pick 1–2 that are easy to reimplement)
- [ ] Same architecture, epochs, optimizer, splits and seeds for all
- [ ] Add a short paragraph explaining why these baselines were chosen

## Phase 3 — Ablations 🧪

- [ ] **Component ablation** (R1 #9):
  - [ ] OGR (ratio) vs raw generalization gap (val − train) as the signal (also answers R1 #4)
  - [ ] Dynamic weights vs fixed weights
  - [ ] Throttle rule only
  - [ ] Amplify rule only
  - [ ] Threshold MFC vs inverse-OGR MFC-A
- [ ] **Hyperparameter sensitivity** (R1 #8): sweep window {3,5,10}, EMA {0.5,0.7,0.9}, warm-up {0,3,5}, clip {[0.2,1.8],[0.1,5.0]} on **both** datasets, using tuning-val only
- [ ] Re-run the hidden-dimension table with seeds (it currently contradicts the main table: 512-dim gives 82.5/84.8 vs 83.3/83.0)
- [ ] Statistical test (paired t-test or Wilcoxon over seeds) for every claim of "better" (R1 #12, R2)

## Phase 4 — Robustness 🧪 (R1 #7)

- [ ] Gaussian noise on vision at σ ∈ {0, 0.5, 1.0, 2.0, 3.0}, on the **test** set, mean ± std
- [ ] Add at least one more degradation on the real video stream: blur, occlusion or brightness/fog
- [ ] Discussion paragraph on adverse visual conditions + cite SUSTechGAN

## Phase 5 — Method section fixes (R1 #1, #4, #5, #6, #13, #14, R2)

- [ ] Fix equation typo: define $\Delta Gen_m = \mathcal{L}^{val}_m - \mathcal{L}^{train}_m$ (currently says $\mathcal{L}^{val} = \mathcal{L}^{train} - \mathcal{L}^{train}$)
- [ ] Make notation consistent everywhere ($\mathcal{L}^{train}_m$, $\mathcal{L}^{val}_m$, $OGR_m(t)$, $\lambda_m(t)$)
- [ ] Justify OGR vs the raw gap: scale invariance argument + the empirical ablation from Phase 3 (R1 #4)
- [ ] Replace Algorithm 1 with **MFC-A** pseudocode (inverse-OGR + EMA + warm-up + clip) — the one actually evaluated
- [ ] Move the threshold MFC to a clearly labelled variant (appendix or "MFC-T"), and state which variant every table uses (R1 #6, R2)
- [ ] Fix "Training Dynamics" section — it still describes τ = 1.2 throttling and ×0.95 steps
- [ ] Fix "Qualitative Observations" — λ_a "reaching 3.5" is impossible with clip [0.2, 1.8]
- [ ] Add a "Novelty / Difference from prior work" paragraph vs OGM-GE, GradNorm, Gradient Blending, AGM (R1 #1)
- [ ] Clarify the separation between the OGR metric (contribution 1) and the MFC-A controller (contribution 2) (R1 #1)
- [ ] Redefine "zero additional FLOPs": state what is included (controller arithmetic) and excluded (extra loss-evaluation pass), and report both (R1 #13)
- [ ] Rewrite the complexity analysis: training $O(N_{tr} E)$ + per-epoch evaluation $O((N_{tr}+N_{val}) E)$ forward passes + $O(MH)$ controller (R1 #14)
- [ ] Explain the data-split protocol in the method/setup section (R3 #2)

## Phase 6 — Results rewrite

- [ ] Replace every table with mean ± std from the new runs
- [ ] Update abstract, contributions and conclusion numbers to match the new tables
- [ ] Remove any claim the new results don't support (e.g. "+3.1% at 1024-dim" if it doesn't hold)
- [ ] Add per-epoch λ / OGR plots (mean ± std bands) as intermediate evidence (R3 #7)
- [ ] Put the full per-epoch CSV logs + code in supplementary material / GitHub link (R3 #7)
- [ ] Fix the conclusion sentence "MFC at σ=1.0 – modality imbalance gap is reduced…" (mixes up two different results)

## Phase 7 — Related work + citations

Add to `references.bib` and cite where they fit:
- [ ] R1 #2 — Vision-based human pose estimation via deep learning: A survey, IEEE THMS 53(1), 2023 → visual representation paragraph
- [ ] R1 #3 — Time efficiency in optimization with a Bayesian-Evolutionary algorithm, SWEVO 69, 2022, 10.1016/j.swevo.2021.100970 → adaptive optimization / efficiency paragraph
- [ ] R1 #7 — SUSTechGAN, IEEE T-IV 10(11), 2025, 10.1109/TIV.2024.3504566 → robustness discussion
- [ ] R1 #11 — Class binarization to neuroevolution for multiclass classification, NCAA 34, 2022, 10.1007/s00521-022-07525-6 → adaptive/evolutionary learning paragraph
- [ ] R2 — 10.1109/EUROCON56442.2023.10198928 (look up title)
- [ ] R2 — 10.1007/978-3-030-58542-6_8 (look up title)
- [ ] R2 — 10.3390/app131911110 (look up title)
- [ ] R3 #4 — Deepfake image detection using different combinations of vision transformers and CNNs → multimodal/vision paragraph
- [ ] New subsection: "Adaptive and Evolutionary Optimization" (R1 #3, #11)
- [ ] Expand visual representation discussion (R1 #2)
- [ ] Verify every new reference's metadata (authors, year, pages) before submitting

## Phase 8 — English revision (R1 #15, R2)

- [ ] Rewrite the abstract from scratch (currently garbled: "the Modality is presented. Fairness Controller…")
- [ ] Rewrite the Introduction; delete the repeated "visual features are compact" paragraph (appears 3×)
- [ ] Fix: "Proposed experiments/approach" → "our experiments/approach"; "MFC(the aproch)"; "Noisey"; "Pefrani et al." → "Pezeshki et al."; "The world is tasted…"; "paper proposed framework"
- [ ] One consistent name: decide "MFC-A" for the method everywhere (not MFC / MFC-A / Proposed(MFC))
- [ ] Consistent capitalization (Baseline, Gradient Blending, OGR Gap, etc.)
- [ ] Remove sentence fragments in Related Work and Discussion
- [ ] Full proofread pass (Grammarly or a native speaker) at the end
- [ ] Optional: change the title to "OGR-Driven Dynamic Loss Balancing for Mitigating Gradient Starvation in Multimodal Learning"

## Phase 9 — Response letter

- [ ] Point-by-point reply to every comment (R1: 15, R2: 5, R3: 7)
- [ ] For each: quote the comment → our response → exact change + page/line in the revised manuscript
- [ ] Openly acknowledge the synthetic-benchmark description error and how it was corrected
- [ ] Produce a highlighted (colored) version of the manuscript showing changes (`latexdiff old.tex new.tex > diff.tex`)
- [ ] Final compile check: no undefined refs, all figures present, tables fit the column

---

## Suggested order
1. Phase 0 → 1 → 2 → 3 → 4 (experiments; everything else depends on the numbers)
2. Phase 5 → 6 → 7 → 8 (manuscript)
3. Phase 9 (response letter last, once page numbers are final)
