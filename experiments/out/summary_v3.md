# v3 results: JPEG-compression degradation and memory overhead
(mean ± std over 10 seeds; p = two-sided paired t-test over seeds)

JPEG: every one of the 8 face/scene frames is JPEG-encoded/decoded with Pillow (libjpeg defaults, 4:2:0 chroma subsampling) at quality 30 (s1) / 15 (s2) / 5 (s3) before the frozen ResNet-18; test split only, audio clean, models are the unchanged trained checkpoints (same seeds as results/).


## CREMA-D (audio-video) — selected MFC-A: `mfc_a__gamma=2.0`

Cells: mean±std [Δ vs sel MFC-A, p] [Δ vs baseline, p]; Δ = row − reference, points.


### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 49.67±5.81 | +0.90 (p=0.0515) | — |
| Gradient Blending | 10 | 52.12±3.56 | +3.35 (p=0.00192) | +2.45 (p=0.02) |
| Uncertainty weighting | 10 | 49.43±5.44 | +0.66 (p=0.228) | -0.24 (p=0.692) |
| MMPareto | 10 | 50.14±3.52 | +1.37 (p=0.361) | +0.47 (p=0.779) |
| OGM-GE (alpha=0.3) | 10 | 50.59±3.90 | +1.82 (p=0.0763) | +0.92 (p=0.39) |
| MFC-T (threshold) | 10 | 49.17±5.59 | +0.40 (p=0.394) | -0.50 (p=0.18) |
| MFC-A (gamma=1) | 10 | 49.31±5.38 | +0.53 (p=0.26) | -0.37 (p=0.475) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 48.77±5.32 | — | -0.90 (p=0.0515) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 43.43±3.27 | +0.24 (p=0.539) | — |
| Gradient Blending | 10 | 47.32±2.59 | +4.13 (p=0.000548) | +3.89 (p=0.000171) |
| Uncertainty weighting | 10 | 43.98±3.50 | +0.79 (p=0.285) | +0.55 (p=0.31) |
| MMPareto | 10 | 44.25±2.54 | +1.06 (p=0.266) | +0.82 (p=0.343) |
| OGM-GE (alpha=0.3) | 10 | 44.37±2.56 | +1.18 (p=0.104) | +0.94 (p=0.193) |
| MFC-T (threshold) | 10 | 43.38±3.32 | +0.19 (p=0.539) | -0.05 (p=0.916) |
| MFC-A (gamma=1) | 10 | 43.78±3.32 | +0.59 (p=0.389) | +0.35 (p=0.511) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 43.19±3.52 | — | -0.24 (p=0.539) |

### jpeg_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 36.25±2.91 | +1.08 (p=0.0951) | — |
| Gradient Blending | 10 | 39.61±2.65 | +4.44 (p=0.00194) | +3.36 (p=0.00427) |
| Uncertainty weighting | 10 | 35.13±3.21 | -0.03 (p=0.953) | -1.12 (p=0.204) |
| MMPareto | 10 | 35.29±3.41 | +0.12 (p=0.875) | -0.96 (p=0.153) |
| OGM-GE (alpha=0.3) | 10 | 34.58±3.01 | -0.59 (p=0.409) | -1.68 (p=0.0195) |
| MFC-T (threshold) | 10 | 34.94±3.25 | -0.23 (p=0.701) | -1.31 (p=0.00983) |
| MFC-A (gamma=1) | 10 | 35.40±3.22 | +0.23 (p=0.702) | -0.86 (p=0.0436) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 35.17±3.38 | — | -1.08 (p=0.0951) |

### jpeg_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.36±3.45 | +0.61 (p=0.187) | — |
| Gradient Blending | 10 | 37.21±4.00 | +3.46 (p=0.0203) | +2.85 (p=0.0437) |
| Uncertainty weighting | 10 | 33.47±3.83 | -0.29 (p=0.713) | -0.89 (p=0.169) |
| MMPareto | 10 | 33.13±5.19 | -0.62 (p=0.626) | -1.22 (p=0.316) |
| OGM-GE (alpha=0.3) | 10 | 31.90±4.29 | -1.85 (p=0.261) | -2.46 (p=0.106) |
| MFC-T (threshold) | 10 | 33.47±3.52 | -0.28 (p=0.659) | -0.88 (p=0.0508) |
| MFC-A (gamma=1) | 10 | 33.68±4.15 | -0.07 (p=0.925) | -0.68 (p=0.255) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 33.75±3.34 | — | -0.61 (p=0.187) |

### jpeg_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 23.62±4.38 | +1.47 (p=0.097) | — |
| Gradient Blending | 10 | 21.89±2.81 | -0.25 (p=0.623) | -1.73 (p=0.148) |
| Uncertainty weighting | 10 | 22.29±4.89 | +0.14 (p=0.887) | -1.33 (p=0.138) |
| MMPareto | 10 | 21.56±2.73 | -0.58 (p=0.606) | -2.06 (p=0.17) |
| OGM-GE (alpha=0.3) | 10 | 22.80±5.68 | +0.65 (p=0.743) | -0.82 (p=0.619) |
| MFC-T (threshold) | 10 | 22.98±4.05 | +0.83 (p=0.175) | -0.64 (p=0.446) |
| MFC-A (gamma=1) | 10 | 22.67±4.26 | +0.52 (p=0.598) | -0.95 (p=0.439) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 22.15±3.14 | — | -1.47 (p=0.097) |

## AVE (audio-visual events) — selected MFC-A: `mfc_a__gamma=3.0`

Cells: mean±std [Δ vs sel MFC-A, p] [Δ vs baseline, p]; Δ = row − reference, points.


### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 79.88±1.84 | -0.32 (p=0.462) | — |
| Gradient Blending | 10 | 81.04±1.34 | +0.85 (p=0.0442) | +1.17 (p=0.0392) |
| Uncertainty weighting | 10 | 77.94±0.99 | -2.26 (p=0.000834) | -1.94 (p=0.0284) |
| MMPareto | 10 | 79.35±1.86 | -0.85 (p=0.121) | -0.52 (p=0.235) |
| OGM-GE (alpha=0.8) | 10 | 82.36±1.03 | +2.16 (p=0.00303) | +2.49 (p=0.00681) |
| MFC-T (threshold) | 10 | 79.88±1.29 | -0.32 (p=0.481) | -0.00 (p=1) |
| MFC-A (gamma=1) | 10 | 80.25±1.61 | +0.05 (p=0.887) | +0.37 (p=0.512) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 80.20±1.34 | — | +0.32 (p=0.462) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 74.88±1.24 | -1.07 (p=0.0144) | — |
| Gradient Blending | 10 | 76.57±1.43 | +0.62 (p=0.279) | +1.69 (p=0.014) |
| Uncertainty weighting | 10 | 72.16±2.03 | -3.78 (p=7.63e-05) | -2.71 (p=0.00141) |
| MMPareto | 10 | 74.08±1.40 | -1.87 (p=0.00742) | -0.80 (p=0.0793) |
| OGM-GE (alpha=0.8) | 10 | 78.76±1.18 | +2.81 (p=0.000153) | +3.88 (p=3.1e-05) |
| MFC-T (threshold) | 10 | 74.70±1.58 | -1.24 (p=0.0122) | -0.17 (p=0.673) |
| MFC-A (gamma=1) | 10 | 74.80±1.81 | -1.14 (p=0.0571) | -0.07 (p=0.871) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 75.95±1.17 | — | +1.07 (p=0.0144) |

### jpeg_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 76.44±1.97 | -0.90 (p=0.0494) | — |
| Gradient Blending | 10 | 78.21±1.56 | +0.87 (p=0.0479) | +1.77 (p=0.0111) |
| Uncertainty weighting | 10 | 75.65±1.81 | -1.69 (p=0.0831) | -0.80 (p=0.446) |
| MMPareto | 10 | 76.24±1.72 | -1.09 (p=0.0655) | -0.20 (p=0.733) |
| OGM-GE (alpha=0.8) | 10 | 80.12±1.18 | +2.79 (p=0.000313) | +3.68 (p=0.000576) |
| MFC-T (threshold) | 10 | 76.82±1.38 | -0.52 (p=0.238) | +0.37 (p=0.578) |
| MFC-A (gamma=1) | 10 | 76.99±1.67 | -0.35 (p=0.0942) | +0.55 (p=0.308) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 77.34±1.67 | — | +0.90 (p=0.0494) |

### jpeg_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 70.92±1.25 | -0.40 (p=0.377) | — |
| Gradient Blending | 10 | 73.93±1.62 | +2.61 (p=0.00142) | +3.01 (p=0.000308) |
| Uncertainty weighting | 10 | 69.25±2.63 | -2.06 (p=0.104) | -1.67 (p=0.154) |
| MMPareto | 10 | 70.57±1.07 | -0.75 (p=0.033) | -0.35 (p=0.264) |
| OGM-GE (alpha=0.8) | 10 | 76.27±0.88 | +4.95 (p=1.01e-05) | +5.35 (p=1.34e-07) |
| MFC-T (threshold) | 10 | 70.85±1.74 | -0.47 (p=0.0852) | -0.07 (p=0.898) |
| MFC-A (gamma=1) | 10 | 70.65±1.81 | -0.67 (p=0.0187) | -0.27 (p=0.626) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 71.32±1.57 | — | +0.40 (p=0.377) |

### jpeg_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 40.32±2.56 | -0.75 (p=0.222) | — |
| Gradient Blending | 10 | 45.07±2.97 | +4.00 (p=0.00402) | +4.75 (p=0.00227) |
| Uncertainty weighting | 10 | 39.40±3.47 | -1.67 (p=0.231) | -0.92 (p=0.441) |
| MMPareto | 10 | 40.62±2.37 | -0.45 (p=0.486) | +0.30 (p=0.526) |
| OGM-GE (alpha=0.8) | 10 | 51.92±2.59 | +10.85 (p=3.38e-07) | +11.59 (p=5.9e-08) |
| MFC-T (threshold) | 10 | 41.32±1.44 | +0.25 (p=0.586) | +1.00 (p=0.134) |
| MFC-A (gamma=1) | 10 | 40.52±2.06 | -0.55 (p=0.0955) | +0.20 (p=0.724) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 41.07±1.81 | — | +0.75 (p=0.222) |

## Memory (seed 0, one fresh process per run; Linux server, CPU, 1 thread)

heap = tracemalloc peak (Python + NumPy allocations; torch CPU tensors are not traced); RSS = peak RSS during training+eval − RSS just before (data loading and torch lazy imports excluded), glibc MALLOC_MMAP_THRESHOLD_=65536; RSS(default malloc) = same with glibc's dynamic threshold, which swings by tens of MB between identical runs; all medians of 5 fresh-process runs; P+G+M = analytic params+grads+SGD momentum; extra = analytic method state. Δ = vs baseline of the same dataset.

| dataset | method | params | heap MB (Δ) | peak RSS Δ MB (Δ vs base) [5 runs] | RSS default malloc MB [5 runs] | P+G+M MB | extra KB |
|---|---|---|---|---|---|---|---|
| synthetic | baseline | 17426 | 0.371 (+0.000) | 1.0 (+0.0) [1.8, 0.7, 0.8, 1.1, 1.0] | 2.5 [2.1, 4.0, -0.1, 3.0, 2.5] | 0.199 | 0.00 |
| synthetic | grad_blend | 17426 | 0.371 (-0.000) | 0.7 (-0.4) [0.7, 0.5, 0.6, 1.1, 1.3] | 5.0 [2.0, 15.3, 6.4, 5.0, 5.0] | 0.199 | 204.23 |
| synthetic | uncertainty | 17426 | 0.371 (+0.000) | 1.0 (-0.1) [0.3, 1.2, 1.0, 0.3, 1.0] | 7.6 [7.6, 13.9, 8.2, 2.2, 2.4] | 0.199 | 0.02 |
| synthetic | mmpareto | 17426 | 0.371 (+0.000) | 2.7 (+1.7) [2.3, 2.8, 1.8, 3.2, 2.7] | -0.1 [-0.7, 1.4, -0.4, -0.1, 2.6] | 0.199 | 275.88 |
| synthetic | ogm_ge | 17426 | 0.371 (+0.000) | 1.0 (-0.0) [0.4, 2.6, 1.0, 1.6, 0.3] | 0.8 [0.8, -0.1, 0.8, 2.3, -0.2] | 0.199 | 35.00 |
| synthetic | mfc_t | 17426 | 0.371 (+0.000) | 1.2 (+0.2) [0.9, 1.4, 0.4, 1.3, 1.2] | 10.9 [8.1, 11.5, 12.2, 10.9, 3.6] | 0.199 | 0.95 |
| synthetic | mfc_a | 17426 | 0.371 (+0.000) | 1.3 (+0.3) [2.0, 1.3, 2.9, 0.4, 1.0] | 6.9 [6.1, 6.9, -0.3, 16.0, 17.3] | 0.199 | 0.95 |
| crema_av | baseline | 190482 | 0.371 (+0.000) | 10.3 (+0.0) [10.0, 9.8, 10.3, 11.9, 10.3] | 55.4 [63.0, 55.4, 24.8, 74.6, 20.6] | 2.180 | 0.00 |
| crema_av | grad_blend | 190482 | 0.371 (-0.000) | 10.3 (+0.0) [10.0, 10.3, 9.9, 10.8, 10.6] | 34.8 [30.4, 31.8, 34.8, 51.3, 61.2] | 2.180 | 2232.23 |
| crema_av | uncertainty | 190482 | 0.371 (+0.000) | 10.3 (+0.0) [10.3, 9.2, 10.9, 10.9, 10.1] | 64.6 [69.9, 47.6, 64.6, 95.4, 49.3] | 2.180 | 0.02 |
| crema_av | mmpareto | 190482 | 0.371 (+0.000) | 12.9 (+2.6) [11.8, 13.1, 13.0, 12.0, 12.9] | 35.5 [33.8, 35.5, 29.2, 41.6, 42.1] | 2.180 | 3207.75 |
| crema_av | ogm_ge | 190482 | 0.371 (+0.000) | 10.6 (+0.4) [10.9, 9.7, 10.6, 11.9, 9.9] | 49.0 [55.6, 39.4, 62.6, 49.0, 47.6] | 2.180 | 515.00 |
| crema_av | mfc_t | 190482 | 0.376 (+0.005) | 9.9 (-0.4) [9.9, 9.7, 9.6, 11.0, 10.6] | 100.0 [130.8, 100.0, 54.6, 108.5, 96.0] | 2.180 | 1.27 |
| crema_av | mfc_a | 190482 | 0.374 (+0.003) | 10.0 (-0.2) [10.3, 9.7, 10.6, 10.0, 10.0] | 86.2 [97.7, 86.2, 30.6, 110.3, 78.1] | 2.180 | 1.27 |
| ave | baseline | 196180 | 0.371 (+0.000) | 6.6 (+0.0) [6.8, 6.5, 6.6, 6.6, 7.5] | 37.2 [37.2, 39.8, 29.6, 17.7, 38.5] | 2.245 | 0.00 |
| ave | grad_blend | 196180 | 0.371 (-0.000) | 7.3 (+0.7) [6.4, 8.9, 7.3, 6.1, 9.1] | 30.8 [38.2, 30.8, 23.6, 35.9, 27.1] | 2.245 | 2299.00 |
| ave | uncertainty | 196180 | 0.371 (+0.000) | 7.8 (+1.2) [6.4, 8.4, 7.9, 7.8, 7.4] | 42.4 [30.8, 42.4, 47.8, 28.7, 54.7] | 2.245 | 0.02 |
| ave | mmpareto | 196180 | 0.371 (+0.000) | 9.3 (+2.7) [10.5, 9.3, 8.6, 8.8, 9.5] | 18.6 [18.6, 19.2, 18.0, 23.6, 14.1] | 2.245 | 3207.75 |
| ave | ogm_ge | 196180 | 0.371 (+0.000) | 7.3 (+0.6) [7.3, 7.4, 7.0, 7.8, 6.8] | 26.8 [38.8, 32.3, 18.3, 26.8, 18.9] | 2.245 | 526.00 |
| ave | mfc_t | 196180 | 0.374 (+0.003) | 7.0 (+0.3) [7.0, 6.8, 7.0, 7.4, 6.5] | 54.9 [70.7, 49.8, 54.9, 24.7, 72.4] | 2.245 | 1.27 |
| ave | mfc_a | 196180 | 0.372 (+0.001) | 7.0 (+0.4) [6.8, 7.0, 6.4, 7.0, 7.5] | 61.5 [61.5, 69.2, 55.2, 58.1, 65.6] | 2.245 | 1.27 |
| crema_spec | baseline | 25426 | 0.371 (+0.000) | 2.6 (+0.0) [2.5, 2.6, 3.3, 3.0, 2.6] | 22.5 [22.5, 23.1, 13.9, 23.2, 13.5] | 0.291 | 0.00 |
| crema_spec | grad_blend | 25426 | 0.371 (-0.000) | 2.9 (+0.2) [2.3, 3.4, 3.3, 2.9, 2.4] | 17.8 [17.8, 12.0, 25.4, 23.6, 14.3] | 0.291 | 297.98 |
| crema_spec | uncertainty | 25426 | 0.371 (+0.000) | 3.2 (+0.5) [3.2, 3.2, 3.6, 2.4, 2.4] | 29.1 [33.1, 29.1, 24.6, 29.5, 22.2] | 0.291 | 0.02 |
| crema_spec | mmpareto | 25426 | 0.371 (+0.000) | 5.2 (+2.6) [5.2, 5.2, 5.2, 5.0, 8.0] | 11.1 [13.3, 11.1, 13.9, 8.0, 4.4] | 0.291 | 380.75 |
| crema_spec | ogm_ge | 25426 | 0.371 (+0.000) | 2.6 (-0.1) [2.3, 2.5, 2.6, 3.2, 2.6] | 18.3 [3.4, 8.4, 20.2, 19.2, 18.3] | 0.291 | 35.00 |
| crema_spec | mfc_t | 25426 | 0.374 (+0.003) | 2.8 (+0.1) [4.2, 2.5, 2.8, 2.6, 3.4] | 30.1 [48.8, 19.3, 7.0, 45.2, 30.1] | 0.291 | 1.27 |
| crema_spec | mfc_a | 25426 | 0.374 (+0.003) | 2.8 (+0.2) [2.6, 4.9, 2.2, 2.9, 2.8] | 19.1 [11.9, 14.4, 40.3, 48.9, 19.1] | 0.291 | 1.27 |

## Provenance and checks (2026-10-18 run)

- Determinism: all 160 v3 runs reproduce results/ exactly (test_acc, ogr_gap, robustness['gauss_1.0']:
  160 exact matches, 0 mismatches), so the JPEG numbers are the same trained checkpoints as every other table.
- npz: vis_jpeg_s1/s2/s3 added; every pre-existing key of crema_av.npz (17) and ave.npz (19) is byte-identical.
  Re-encoded clean frames vs stored `vis`: max |diff| 0.0 on AVE, 0.320 on CREMA-D (identical to the value
  extract_severity.py reported in the v2 run, i.e. a pre-existing small CUDA/decoder difference, not misalignment).
- Memory runs train with the unchanged bench.run; their seed-0 test_acc equals results/ for crema_av and ave
  (all 7 methods) and for synthetic baseline/grad_blend/uncertainty/mmpareto, but NOT for synthetic
  ogm_ge/mfc_t/mfc_a or any crema_spec method. Plain `bench.py synthetic mfc_a 0` on the server gives the same
  0.8533 (stored: 0.8456), so this is not caused by the measurement: the stored synthetic/crema_spec result files
  carry whole-second mtimes (copied from another machine) and were apparently computed elsewhere.

## Interpretation

- JPEG, CREMA-D: selected MFC-A (gamma=2) is statistically indistinguishable from the baseline at every quality
  (-1.08 / -0.61 / -1.47 points, p=0.095 / 0.19 / 0.097). Gradient Blending is the only method significantly
  better than both baseline and MFC-A at q=30 and q=15 (+3.4 / +2.9 vs baseline); at q=5 nothing differs
  significantly and the baseline has the highest mean. MFC-T and MFC-A(gamma=1) are slightly but significantly
  below the baseline at q=30 (-1.3, p=0.010; -0.9, p=0.044).
- JPEG, AVE: OGM-GE (alpha=0.8) is by far the most robust (+3.7 / +5.4 / +11.6 vs baseline, all p<0.001), followed
  by Gradient Blending (+1.8 / +3.0 / +4.8, all p<0.05). Selected MFC-A (gamma=3) is +0.9 over the baseline at q=30
  (p=0.049) and not significantly different at q=15 / q=5 (+0.4, p=0.38; +0.75, p=0.22).
- The JPEG ranking matches the blur/occlusion/low-light severities of v2: MFC-A does not buy visual-corruption
  robustness beyond the baseline; OGM-GE (AVE) and Gradient Blending (both datasets) do.
- Memory: MFC-A's analytic extra state is 1.27 KB (40 epochs x 2 modalities x (L_train, L_val) float64 + 2 weights;
  the algorithm itself needs only the last W=5 entries, 20 floats) against 2.2 MB of params+grads+momentum on
  CREMA-D/AVE (0.06%). Measured, it is invisible: the traced Python heap rises by <=0.005 MB and the median peak RSS
  delta is within the run-to-run spread of the baseline on all four datasets. The only method with a measurable RSS
  increase is MMPareto (+1.7 to +2.7 MB, from its transient per-step gradient copies, analytic ~3.1 MB). These are
  CPU measurements of the small fusion heads on frozen features; they do not include the backbone.

