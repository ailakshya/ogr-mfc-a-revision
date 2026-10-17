# OGR-Driven Dynamic Loss Balancing for Mitigating Gradient Starvation

Code, data, per-seed logs and manuscript for the major revision of the paper by
Lakshya, Amit Kumar Bairwa and Varun Tiwari (Manipal University Jaipur), submitted to a Springer Nature journal.

The paper proposes the **Overfitting-to-Generalization Ratio** (OGR = held-out loss / training loss, per modality)
and **MFC-A**, a closed-loop controller that re-weights the unimodal auxiliary losses of a late-fusion
audio–visual network by the inverse of each modality's smoothed OGR.

This repository is organised so that every number, table and figure in the revised manuscript can be traced to
raw per-seed logs and regenerated with one command.

## Repository layout

| Path | Contents |
|---|---|
| `mfc/mfc_a.py` | **Standalone implementation of the method** (OGR + MFC-A controller, any number of modalities) for use in your own PyTorch training loop; verified to reproduce the weights of all 560 logged MFC-A runs |
| `manuscript/` | Revised paper (`main.tex`, `main.pdf`), marked-up changes against the reviewed version (`main_diff.pdf`), response to reviewers (`response_letter.pdf`), the version the reviewers saw (`reviewed_manuscript.pdf`, source `main_reviewed_version.tex`) |
| `experiments/` | All code for the revision: feature extraction, benchmark, aggregation, tests |
| `experiments/data/` | Extracted features (`*.npz`, stored with Git LFS) |
| `experiments/results/` | 1,120 per-seed JSON logs of the main grid (4 datasets × methods/ablations × 10 seeds) |
| `experiments/results_v2/` | 160 reruns with per-modality gradient-norm logging and multi-severity degradations |
| `experiments/out/` | Generated tables (`tables/*.tex`), figures (`figs/*.png`) and full numeric summaries (`summary.md`, `summary_v2.md`) |
| `experiments/grid*.txt` | The exact job lists that produced every log |
| `docs/` | Revision checklist mapping each reviewer comment to the change made |
| `literature/` | Reading notes on related work (third-party PDFs are not redistributed) |
| `legacy/first_submission/` | Material from the first submission (superseded; see "Legacy material") |

## Using MFC-A in your own training

```python
from mfc.mfc_a import MFCA

ctl = MFCA(num_modalities=2, gamma=3.0)      # defaults: window 5, EMA 0.7, warm-up 3, clip [0.2, 1.8]
for epoch in range(epochs):
    lam = ctl.weights
    for xv, xa, y in train_loader:
        out, logit_v, logit_a = model(xv, xa)
        loss = ce(out, y) + lam[0] * ce(logit_v, y) + lam[1] * ce(logit_a, y)
        ...
    # mean unimodal losses on the training data and on a held-out controller split (never the test set)
    ctl.step(train_losses=[Ltr_v, Ltr_a], heldout_losses=[Lcv_v, Lcv_a])
```

`python mfc/mfc_a.py` runs a self-check of the properties stated in the paper.

## Quick inspection (no training needed)

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy scipy scikit-learn matplotlib
git lfs pull                                            # fetch experiments/data/*.npz
.venv/bin/python experiments/tests/test_consistency.py
```

The test regenerates every table and figure from the raw logs, checks that each of the nine tables printed in the
manuscript matches `manuscript/main.tex` line for line, checks that all 1,120 grid jobs have a log, checks that the standalone
`mfc/mfc_a.py` reproduces the loss weights of every logged MFC-A run, and checks that the 160 gradient-norm reruns
reproduce the reported results exactly. Expected output ends with `ALL CHECKS PASSED`.

## Where every table and figure comes from

| Manuscript item | Generated file | Script | Logs (under `experiments/`) |
|---|---|---|---|
| Table 3 (CREMA-D audio–video) | `out/tables/main_crema_av.tex` | `aggregate.py` | `results/crema_av/*` |
| Table 4 (AVE) | `out/tables/main_ave.tex` | `aggregate.py` | `results/ave/*` |
| Table 5 (synthetic) | `out/tables/main_synthetic.tex` | `aggregate.py` | `results/synthetic/*` |
| Table 6 (spectrogram views) | `out/tables/main_crema_spec.tex` | `aggregate.py` | `results/crema_spec/*` |
| Table 7 (component ablation) | `out/tables/ablation_joint.tex` | `aggregate.py` | `results/{crema_av,ave,synthetic}/{baseline,static__*,mfc_a*}` |
| Table 8 (sensitivity) | `out/tables/sensitivity.tex` | `aggregate.py` | `results/*/mfc_a__{gamma,window,ema,warmup,clip}=*` |
| Table 9 (capacity) | `out/tables/capacity.tex` | `aggregate.py` | `results/synthetic/{baseline,mfc_a}__hv=*` |
| Table 10 (graded degradations, AVE) | `out/tables/severity_ave.tex` | `aggregate_v2.py` | `results_v2/ave/*` |
| Table 11 (cost) | `out/tables/cost.tex` | `aggregate.py` | `time` and `flops` fields of `results/{synthetic,crema_av}/*` |
| Figures 1–3 (training dynamics) | `out/figs/dynamics_{crema_av,ave,synthetic}.png` | `aggregate.py` | per-epoch `log` rows: `tune_acc, ogr_v, ogr_a, lam_v, lam_a` |
| Figure 4 (gradient norms, AVE) | `out/figs/gradnorm_ave.png` | `aggregate_v2.py` | per-epoch `gj_v, gj_a, gu_v, gu_a` in `results_v2/ave/*` |
| Figure 5 (robustness curves) | `out/figs/robustness.png` | `aggregate.py` | `robustness` field of `results/*/*` |
| Every number quoted in the text | `out/summary.md`, `out/summary_v2.md` | both | all of the above |

Each log file `results/<dataset>/<method>[__<option>=<value>]/seed<k>.json` contains the configuration, the
selected epoch, all test metrics (fused and unimodal accuracy, per-modality OGR, OGR gap, every robustness
condition), wall-clock and FLOP counts per component, and a per-epoch list with every split loss, both OGRs,
both loss weights (and, in `results_v2`, the four gradient norms).

The γ used for MFC-A in the main tables is chosen without test data by `experiments/select_gamma.py`
(lowest controller-split OGR gap among γ ∈ {1, 2, 3} whose tuning accuracy is within 0.5 points of the baseline).

## Protocol

Every method shares the architecture, objective, optimiser (SGD, lr 0.01, momentum 0.9, weight decay 5e-4,
batch 64), epochs, splits and seeds. Data are split into four disjoint parts: training (60%), controller
validation (15%, the only held-out data any controller may read), tuning (10%, checkpoint and hyper-parameter
selection) and test (15%, used once after training). CREMA-D is split by actor; AVE keeps its official test list
and splits the rest by video ID. Seeds 0–9 fix the split, the initialisation and the batch order, and are shared by
all methods, so comparisons use paired t-tests.

## Re-running the experiments

Feature extraction (GPU recommended) and the full grid (CPU, parallel) were run on a 24-core workstation with an
RTX 5070 Ti; the full grid takes about 30 minutes there.

```bash
# features (download CREMA-D and AVE first; see manuscript, "Data availability")
.venv/bin/python experiments/extract_crema_av.py <crema_dir>       # then extract_degraded.py
.venv/bin/python experiments/extract_ave.py <ave_dir>
.venv/bin/python experiments/extract_severity.py <crema_dir> <ave_dir>
.venv/bin/python experiments/prep_crema_spec.py

# one run, or a whole grid
.venv/bin/python experiments/bench.py ave mfc_a 0 gamma=3.0
JOBS=20 .venv/bin/python experiments/bench.py --grid experiments/grid.txt

# tables, figures, summaries
.venv/bin/python experiments/aggregate.py && .venv/bin/python experiments/aggregate_v2.py
```

Training runs use single-threaded CPU PyTorch and are bit-for-bit deterministic on the same machine.
The `run_*.sh` and `finish_*.sh` helpers contain the paths used on the authors' workstation.

## Honest summary of the results

MFC-A never reduced accuracy significantly. On AVE it reduced the OGR gap by 63% (p = 0.03) and improved
robustness to noise, blur and low light over the baseline; on the synthetic benchmark it reduced the gap by 41%.
On CREMA-D its effects were not significant. OGM-GE was the strongest method on AVE and Gradient Blending on
CREMA-D. Gradient-norm logging shows that MFC-A changes the auxiliary gradients as designed (−55% to the
overfitting video encoder, +45–66% to audio) but leaves the joint-loss gradient share almost unchanged.
The limitations are listed in the manuscript's Discussion.

## Legacy material

`legacy/first_submission/` holds the first submission's sources, scripts, figures and presentation material for
reference only. Several of its claims did not survive the multi-seed protocol and are withdrawn in the revision
(see the response letter); in particular its "synthetic CIFAR-10/CREMA-D benchmark" was in fact generated with
scikit-learn's `make_classification`. Nothing in the revised manuscript depends on that folder.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences (CREMA-D: Open Database License; AVE: see the
original release).
