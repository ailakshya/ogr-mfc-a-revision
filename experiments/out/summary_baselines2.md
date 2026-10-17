# AGM / PMR / OLM baselines under the shared protocol
(10 seeds; mean ± std; p = two-sided paired t-test over seeds, scipy.stats.ttest_rel; Δ = row − reference; accuracies in %, ogr_gap unitless; hyper-parameters selected by mean tune-split accuracy only)

## Synthetic

### Hyper-parameter selection (mean tune acc, %)

| method | config | n | tune acc | test acc |
|---|---|---|---|---|
| AGM | alpha=0.1 | 10 | 86.00 | 83.93 |
| AGM | alpha=0.5 | 10 | 85.83 | 84.10 |
| AGM | alpha=1.0 | 10 | 85.63 | 83.63 |
| PMR | alpha=1.0__mu=0.001 | 10 | 85.62 | 83.86 |
| PMR | alpha=1.0__mu=0.01 | 10 | 85.78 | 84.12 |
| PMR | alpha=2.0__mu=0.001 | 10 | 85.87 | 84.13 |
| PMR | alpha=2.0__mu=0.01 | 10 | 85.82 | 84.19 |
| OLM | eta=0.01 | 10 | 80.88 | 79.92 |
| OLM | eta=0.1 | 10 | 83.58 | 81.90 |
| OLM | eta=1.0 | 10 | 84.52 | 82.54 |

Selected: AGM `agm__alpha=0.1`, PMR `pmr__alpha=2.0__mu=0.001`, OLM `olm__eta=1.0`

### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 84.13±1.14 | -0.13 (p=0.468) | — |
| MFC-A (gamma=3.0, selected) | 10 | 84.27±1.26 | — | +0.13 (p=0.468) |
| AGM (alpha=0.1) | 10 | 83.93±0.93 | -0.33 (p=0.194) | -0.20 (p=0.518) |
| PMR (alpha=2.0, mu=0.001) | 10 | 84.13±1.61 | -0.13 (p=0.607) | -0.00 (p=1) |
| OLM (eta=1.0) | 10 | 82.54±1.90 | -1.72 (p=0.00699) | -1.59 (p=0.0169) |

### acc_v

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 83.71±1.23 | +0.04 (p=0.817) | — |
| MFC-A (gamma=3.0, selected) | 10 | 83.67±1.28 | — | -0.04 (p=0.817) |
| AGM (alpha=0.1) | 10 | 83.32±1.41 | -0.34 (p=0.396) | -0.39 (p=0.359) |
| PMR (alpha=2.0, mu=0.001) | 10 | 83.64±1.65 | -0.02 (p=0.939) | -0.07 (p=0.814) |
| OLM (eta=1.0) | 10 | 82.57±1.56 | -1.10 (p=0.0357) | -1.14 (p=0.00921) |

### acc_a

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 28.70±1.78 | -0.94 (p=0.0198) | — |
| MFC-A (gamma=3.0, selected) | 10 | 29.64±1.43 | — | +0.94 (p=0.0198) |
| AGM (alpha=0.1) | 10 | 28.12±2.15 | -1.52 (p=0.0215) | -0.58 (p=0.363) |
| PMR (alpha=2.0, mu=0.001) | 10 | 28.84±1.64 | -0.80 (p=0.0779) | +0.14 (p=0.768) |
| OLM (eta=1.0) | 10 | 28.47±1.48 | -1.18 (p=0.0191) | -0.23 (p=0.612) |

### ogr_gap

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 6.01±1.36 | +2.45 (p=0.000185) | — |
| MFC-A (gamma=3.0, selected) | 10 | 3.56±0.60 | — | -2.45 (p=0.000185) |
| AGM (alpha=0.1) | 10 | 5.39±2.17 | +1.83 (p=0.00872) | -0.62 (p=0.395) |
| PMR (alpha=2.0, mu=0.001) | 10 | 5.46±1.24 | +1.91 (p=0.00035) | -0.55 (p=0.228) |
| OLM (eta=1.0) | 10 | 3.95±1.17 | +0.39 (p=0.264) | -2.06 (p=0.000665) |

### gauss_0.5

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 70.76±1.75 | -0.50 (p=0.251) | — |
| MFC-A (gamma=3.0, selected) | 10 | 71.26±1.43 | — | +0.50 (p=0.251) |
| AGM (alpha=0.1) | 10 | 71.11±1.47 | -0.14 (p=0.703) | +0.36 (p=0.497) |
| PMR (alpha=2.0, mu=0.001) | 10 | 70.68±1.32 | -0.58 (p=0.182) | -0.08 (p=0.882) |
| OLM (eta=1.0) | 10 | 69.02±1.60 | -2.23 (p=0.000178) | -1.73 (p=0.00579) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 48.81±2.08 | -0.90 (p=0.042) | — |
| MFC-A (gamma=3.0, selected) | 10 | 49.71±2.90 | — | +0.90 (p=0.042) |
| AGM (alpha=0.1) | 10 | 49.11±1.99 | -0.60 (p=0.378) | +0.30 (p=0.484) |
| PMR (alpha=2.0, mu=0.001) | 10 | 49.18±2.39 | -0.53 (p=0.227) | +0.37 (p=0.426) |
| OLM (eta=1.0) | 10 | 47.74±2.20 | -1.97 (p=0.0038) | -1.07 (p=0.0398) |

### Method cost (wall-clock seconds per run, mean over seeds; CPU, 1 thread)

| method | train | method | monitor | ctrl | select | total |
|---|---|---|---|---|---|---|
| Baseline | 1.34 | 0.00 | 0.00 | 0.00 | 0.02 | 1.36 |
| MFC-A (gamma=3.0, selected) | 1.75 | 0.00 | 0.13 | 0.00 | 0.02 | 1.90 |
| AGM (alpha=0.1) | 3.85 | 1.43 | 0.00 | 0.00 | 0.05 | 5.32 |
| PMR (alpha=2.0, mu=0.001) | 4.44 | 0.89 | 0.00 | 0.00 | 0.04 | 5.37 |
| OLM (eta=1.0) | 4.06 | 0.62 | 0.40 | 0.00 | 0.05 | 5.12 |


## CREMA-D (audio-video)

### Hyper-parameter selection (mean tune acc, %)

| method | config | n | tune acc | test acc |
|---|---|---|---|---|
| AGM | alpha=0.1 | 10 | 51.77 | 49.05 |
| AGM | alpha=0.5 | 10 | 51.45 | 50.48 |
| AGM | alpha=1.0 | 10 | 51.54 | 49.75 |
| PMR | alpha=1.0__mu=0.001 | 10 | 51.45 | 50.39 |
| PMR | alpha=1.0__mu=0.01 | 10 | 51.59 | 48.76 |
| PMR | alpha=2.0__mu=0.001 | 10 | 51.29 | 48.13 |
| PMR | alpha=2.0__mu=0.01 | 10 | 51.44 | 49.13 |
| OLM | eta=0.01 | 10 | 53.27 | 51.68 |
| OLM | eta=0.1 | 10 | 52.64 | 50.73 |
| OLM | eta=1.0 | 10 | 51.97 | 50.36 |

Selected: AGM `agm__alpha=0.1`, PMR `pmr__alpha=1.0__mu=0.01`, OLM `olm__eta=0.01`

### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 49.67±5.81 | +0.90 (p=0.0515) | — |
| MFC-A (gamma=2.0, selected) | 10 | 48.77±5.32 | — | -0.90 (p=0.0515) |
| AGM (alpha=0.1) | 10 | 49.05±4.83 | +0.28 (p=0.63) | -0.62 (p=0.385) |
| PMR (alpha=1.0, mu=0.01) | 10 | 48.76±5.32 | -0.01 (p=0.975) | -0.91 (p=0.0334) |
| OLM (eta=0.01) | 10 | 51.68±3.10 | +2.91 (p=0.0211) | +2.01 (p=0.122) |

### acc_v

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 32.62±3.67 | +0.30 (p=0.499) | — |
| MFC-A (gamma=2.0, selected) | 10 | 32.32±3.28 | — | -0.30 (p=0.499) |
| AGM (alpha=0.1) | 10 | 32.18±3.05 | -0.14 (p=0.701) | -0.44 (p=0.201) |
| PMR (alpha=1.0, mu=0.01) | 10 | 32.39±3.36 | +0.07 (p=0.807) | -0.23 (p=0.33) |
| OLM (eta=0.01) | 10 | 32.64±2.80 | +0.32 (p=0.771) | +0.02 (p=0.982) |

### acc_a

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 47.09±4.81 | +0.69 (p=0.00818) | — |
| MFC-A (gamma=2.0, selected) | 10 | 46.40±4.63 | — | -0.69 (p=0.00818) |
| AGM (alpha=0.1) | 10 | 47.07±4.74 | +0.67 (p=0.0162) | -0.02 (p=0.939) |
| PMR (alpha=1.0, mu=0.01) | 10 | 46.97±4.74 | +0.58 (p=0.0574) | -0.11 (p=0.671) |
| OLM (eta=0.01) | 10 | 47.98±3.44 | +1.58 (p=0.0391) | +0.89 (p=0.247) |

### ogr_gap

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 3.73±3.60 | +0.84 (p=0.453) | — |
| MFC-A (gamma=2.0, selected) | 10 | 2.90±2.12 | — | -0.84 (p=0.453) |
| AGM (alpha=0.1) | 10 | 4.22±3.24 | +1.32 (p=0.197) | +0.48 (p=0.694) |
| PMR (alpha=1.0, mu=0.01) | 10 | 4.04±3.63 | +1.14 (p=0.304) | +0.30 (p=0.852) |
| OLM (eta=0.01) | 10 | 2.68±0.94 | -0.21 (p=0.701) | -1.05 (p=0.359) |

### gauss_0.5

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 47.59±4.93 | +0.51 (p=0.324) | — |
| MFC-A (gamma=2.0, selected) | 10 | 47.08±4.99 | — | -0.51 (p=0.324) |
| AGM (alpha=0.1) | 10 | 47.58±4.44 | +0.51 (p=0.414) | -0.00 (p=0.998) |
| PMR (alpha=1.0, mu=0.01) | 10 | 46.51±4.60 | -0.57 (p=0.2) | -1.08 (p=0.0778) |
| OLM (eta=0.01) | 10 | 50.53±3.38 | +3.45 (p=0.00345) | +2.94 (p=0.0162) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 43.43±3.27 | +0.24 (p=0.539) | — |
| MFC-A (gamma=2.0, selected) | 10 | 43.19±3.52 | — | -0.24 (p=0.539) |
| AGM (alpha=0.1) | 10 | 43.45±3.17 | +0.25 (p=0.682) | +0.02 (p=0.969) |
| PMR (alpha=1.0, mu=0.01) | 10 | 43.47±3.54 | +0.28 (p=0.542) | +0.04 (p=0.862) |
| OLM (eta=0.01) | 10 | 46.75±2.96 | +3.56 (p=0.000193) | +3.32 (p=0.000914) |

### blur

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 24.67±4.99 | +0.89 (p=0.0538) | — |
| MFC-A (gamma=2.0, selected) | 10 | 23.79±5.10 | — | -0.89 (p=0.0538) |
| AGM (alpha=0.1) | 10 | 23.06±3.95 | -0.73 (p=0.474) | -1.61 (p=0.0978) |
| PMR (alpha=1.0, mu=0.01) | 10 | 23.67±4.86 | -0.11 (p=0.835) | -1.00 (p=0.00962) |
| OLM (eta=0.01) | 10 | 25.58±4.13 | +1.79 (p=0.196) | +0.91 (p=0.482) |

### occlusion

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 28.35±3.86 | +0.56 (p=0.352) | — |
| MFC-A (gamma=2.0, selected) | 10 | 27.79±2.89 | — | -0.56 (p=0.352) |
| AGM (alpha=0.1) | 10 | 27.77±3.76 | -0.02 (p=0.98) | -0.58 (p=0.303) |
| PMR (alpha=1.0, mu=0.01) | 10 | 28.03±3.18 | +0.24 (p=0.499) | -0.32 (p=0.689) |
| OLM (eta=0.01) | 10 | 29.67±4.59 | +1.88 (p=0.171) | +1.31 (p=0.33) |

### lowlight

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.44±3.41 | +0.58 (p=0.507) | — |
| MFC-A (gamma=2.0, selected) | 10 | 33.86±5.26 | — | -0.58 (p=0.507) |
| AGM (alpha=0.1) | 10 | 33.79±4.24 | -0.07 (p=0.947) | -0.66 (p=0.278) |
| PMR (alpha=1.0, mu=0.01) | 10 | 33.98±5.26 | +0.12 (p=0.92) | -0.46 (p=0.619) |
| OLM (eta=0.01) | 10 | 37.64±5.07 | +3.78 (p=0.00638) | +3.20 (p=0.00361) |

### vision_missing

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 44.31±4.79 | +0.67 (p=0.433) | — |
| MFC-A (gamma=2.0, selected) | 10 | 43.64±4.37 | — | -0.67 (p=0.433) |
| AGM (alpha=0.1) | 10 | 45.29±5.37 | +1.65 (p=0.0578) | +0.98 (p=0.17) |
| PMR (alpha=1.0, mu=0.01) | 10 | 44.31±5.10 | +0.67 (p=0.327) | +0.00 (p=1) |
| OLM (eta=0.01) | 10 | 42.63±2.91 | -1.01 (p=0.367) | -1.68 (p=0.192) |

### Method cost (wall-clock seconds per run, mean over seeds; CPU, 1 thread)

| method | train | method | monitor | ctrl | select | total |
|---|---|---|---|---|---|---|
| Baseline | 14.18 | 0.00 | 0.00 | 0.00 | 0.32 | 14.50 |
| MFC-A (gamma=2.0, selected) | 14.67 | 0.00 | 2.85 | 0.01 | 0.34 | 17.87 |
| AGM (alpha=0.1) | 21.99 | 8.05 | 0.00 | 0.00 | 0.55 | 30.59 |
| PMR (alpha=1.0, mu=0.01) | 22.02 | 2.76 | 0.00 | 0.00 | 0.54 | 25.32 |
| OLM (eta=0.01) | 21.62 | 1.91 | 3.88 | 0.01 | 0.52 | 27.94 |


## AVE

### Hyper-parameter selection (mean tune acc, %)

| method | config | n | tune acc | test acc |
|---|---|---|---|---|
| AGM | alpha=0.1 | 10 | 84.23 | 79.90 |
| AGM | alpha=0.5 | 10 | 84.39 | 80.50 |
| AGM | alpha=1.0 | 10 | 84.55 | 80.47 |
| PMR | alpha=1.0__mu=0.001 | 10 | 84.12 | 79.70 |
| PMR | alpha=1.0__mu=0.01 | 10 | 84.12 | 79.48 |
| PMR | alpha=2.0__mu=0.001 | 10 | 84.07 | 78.66 |
| PMR | alpha=2.0__mu=0.01 | 10 | 83.77 | 79.13 |
| OLM | eta=0.01 | 10 | 79.91 | 74.28 |
| OLM | eta=0.1 | 10 | 81.49 | 76.92 |
| OLM | eta=1.0 | 10 | 83.24 | 78.18 |

Selected: AGM `agm__alpha=1.0`, PMR `pmr__alpha=1.0__mu=0.01`, OLM `olm__eta=1.0`

### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 79.88±1.84 | -0.32 (p=0.462) | — |
| MFC-A (gamma=3.0, selected) | 10 | 80.20±1.34 | — | +0.32 (p=0.462) |
| AGM (alpha=1.0) | 10 | 80.47±1.38 | +0.27 (p=0.645) | +0.60 (p=0.434) |
| PMR (alpha=1.0, mu=0.01) | 10 | 79.48±1.47 | -0.72 (p=0.0959) | -0.40 (p=0.464) |
| OLM (eta=1.0) | 10 | 78.18±1.68 | -2.01 (p=0.00301) | -1.69 (p=0.0736) |

### acc_v

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 73.58±2.09 | -0.82 (p=0.246) | — |
| MFC-A (gamma=3.0, selected) | 10 | 74.40±0.82 | — | +0.82 (p=0.246) |
| AGM (alpha=1.0) | 10 | 73.08±1.18 | -1.32 (p=0.0173) | -0.50 (p=0.537) |
| PMR (alpha=1.0, mu=0.01) | 10 | 73.21±2.25 | -1.19 (p=0.0826) | -0.37 (p=0.383) |
| OLM (eta=1.0) | 10 | 72.91±1.43 | -1.49 (p=0.00357) | -0.67 (p=0.361) |

### acc_a

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 48.56±5.39 | -1.87 (p=0.167) | — |
| MFC-A (gamma=3.0, selected) | 10 | 50.42±4.38 | — | +1.87 (p=0.167) |
| AGM (alpha=1.0) | 10 | 47.41±4.05 | -3.01 (p=0.155) | -1.14 (p=0.538) |
| PMR (alpha=1.0, mu=0.01) | 10 | 44.63±4.34 | -5.80 (p=0.000531) | -3.93 (p=0.061) |
| OLM (eta=1.0) | 10 | 51.29±1.87 | +0.87 (p=0.442) | +2.74 (p=0.0717) |

### ogr_gap

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 25.87±22.28 | +16.37 (p=0.0332) | — |
| MFC-A (gamma=3.0, selected) | 10 | 9.50±4.06 | — | -16.37 (p=0.0332) |
| AGM (alpha=1.0) | 10 | 12.82±13.87 | +3.32 (p=0.493) | -13.05 (p=0.126) |
| PMR (alpha=1.0, mu=0.01) | 10 | 18.88±21.04 | +9.38 (p=0.194) | -6.99 (p=0.478) |
| OLM (eta=1.0) | 10 | 19.96±16.16 | +10.46 (p=0.0423) | -5.91 (p=0.161) |

### gauss_0.5

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 78.81±1.74 | -0.82 (p=0.0932) | — |
| MFC-A (gamma=3.0, selected) | 10 | 79.63±1.08 | — | +0.82 (p=0.0932) |
| AGM (alpha=1.0) | 10 | 79.03±1.33 | -0.60 (p=0.0678) | +0.22 (p=0.625) |
| PMR (alpha=1.0, mu=0.01) | 10 | 78.53±1.94 | -1.09 (p=0.101) | -0.27 (p=0.589) |
| OLM (eta=1.0) | 10 | 77.04±2.16 | -2.59 (p=0.00644) | -1.77 (p=0.0849) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 74.88±1.24 | -1.07 (p=0.0144) | — |
| MFC-A (gamma=3.0, selected) | 10 | 75.95±1.17 | — | +1.07 (p=0.0144) |
| AGM (alpha=1.0) | 10 | 74.85±1.23 | -1.09 (p=0.0446) | -0.02 (p=0.957) |
| PMR (alpha=1.0, mu=0.01) | 10 | 75.07±1.40 | -0.87 (p=0.00858) | +0.20 (p=0.618) |
| OLM (eta=1.0) | 10 | 73.01±2.02 | -2.94 (p=0.0014) | -1.87 (p=0.00733) |

### blur

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 39.43±4.77 | -2.09 (p=0.0351) | — |
| MFC-A (gamma=3.0, selected) | 10 | 41.52±4.99 | — | +2.09 (p=0.0351) |
| AGM (alpha=1.0) | 10 | 41.12±3.91 | -0.40 (p=0.781) | +1.69 (p=0.13) |
| PMR (alpha=1.0, mu=0.01) | 10 | 40.52±4.78 | -1.00 (p=0.154) | +1.09 (p=0.205) |
| OLM (eta=1.0) | 10 | 33.98±5.09 | -7.54 (p=0.00017) | -5.45 (p=0.000323) |

### occlusion

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 65.50±2.69 | -0.30 (p=0.702) | — |
| MFC-A (gamma=3.0, selected) | 10 | 65.80±1.75 | — | +0.30 (p=0.702) |
| AGM (alpha=1.0) | 10 | 66.32±1.62 | +0.52 (p=0.343) | +0.82 (p=0.225) |
| PMR (alpha=1.0, mu=0.01) | 10 | 65.02±1.28 | -0.77 (p=0.199) | -0.47 (p=0.531) |
| OLM (eta=1.0) | 10 | 62.44±3.25 | -3.36 (p=0.00156) | -3.06 (p=0.0186) |

### lowlight

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 50.97±1.92 | -1.37 (p=0.0299) | — |
| MFC-A (gamma=3.0, selected) | 10 | 52.34±2.25 | — | +1.37 (p=0.0299) |
| AGM (alpha=1.0) | 10 | 53.41±1.62 | +1.07 (p=0.184) | +2.44 (p=0.00929) |
| PMR (alpha=1.0, mu=0.01) | 10 | 53.33±2.17 | +1.00 (p=0.361) | +2.36 (p=0.0153) |
| OLM (eta=1.0) | 10 | 48.61±3.76 | -3.73 (p=0.019) | -2.36 (p=0.0722) |

### vision_missing

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 27.21±3.92 | -1.79 (p=0.0879) | — |
| MFC-A (gamma=3.0, selected) | 10 | 29.00±3.74 | — | +1.79 (p=0.0879) |
| AGM (alpha=1.0) | 10 | 28.86±2.87 | -0.15 (p=0.91) | +1.64 (p=0.276) |
| PMR (alpha=1.0, mu=0.01) | 10 | 22.29±2.49 | -6.72 (p=0.00015) | -4.93 (p=0.00154) |
| OLM (eta=1.0) | 10 | 43.23±3.13 | +14.23 (p=1.16e-06) | +16.02 (p=9.4e-07) |

### Method cost (wall-clock seconds per run, mean over seeds; CPU, 1 thread)

| method | train | method | monitor | ctrl | select | total |
|---|---|---|---|---|---|---|
| Baseline | 9.39 | 0.00 | 0.00 | 0.00 | 0.17 | 9.57 |
| MFC-A (gamma=3.0, selected) | 5.95 | 0.00 | 1.23 | 0.01 | 0.13 | 7.32 |
| AGM (alpha=1.0) | 14.41 | 5.17 | 0.00 | 0.00 | 0.31 | 19.88 |
| PMR (alpha=1.0, mu=0.01) | 15.42 | 1.92 | 0.00 | 0.00 | 0.25 | 17.59 |
| OLM (eta=1.0) | 9.96 | 0.83 | 1.73 | 0.00 | 0.20 | 12.72 |


## CREMA-D (spectrogram)

### Hyper-parameter selection (mean tune acc, %)

| method | config | n | tune acc | test acc |
|---|---|---|---|---|
| AGM | alpha=0.1 | 10 | 52.00 | 49.13 |
| AGM | alpha=0.5 | 10 | 52.16 | 49.83 |
| AGM | alpha=1.0 | 10 | 52.07 | 49.67 |
| PMR | alpha=1.0__mu=0.001 | 10 | 52.54 | 49.25 |
| PMR | alpha=1.0__mu=0.01 | 10 | 51.94 | 49.01 |
| PMR | alpha=2.0__mu=0.001 | 10 | 52.33 | 48.84 |
| PMR | alpha=2.0__mu=0.01 | 10 | 52.19 | 49.18 |
| OLM | eta=0.01 | 10 | 48.97 | 46.61 |
| OLM | eta=0.1 | 10 | 50.99 | 48.72 |
| OLM | eta=1.0 | 10 | 51.44 | 49.16 |

Selected: AGM `agm__alpha=0.5`, PMR `pmr__alpha=1.0__mu=0.001`, OLM `olm__eta=1.0`

### acc

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 49.07±1.79 | -0.26 (p=0.669) | — |
| MFC-A (gamma=2.0, selected) | 10 | 49.33±1.45 | — | +0.26 (p=0.669) |
| AGM (alpha=0.5) | 10 | 49.83±1.32 | +0.50 (p=0.332) | +0.76 (p=0.166) |
| PMR (alpha=1.0, mu=0.001) | 10 | 49.25±1.64 | -0.08 (p=0.891) | +0.18 (p=0.297) |
| OLM (eta=1.0) | 10 | 49.16±1.50 | -0.17 (p=0.679) | +0.09 (p=0.855) |

### acc_v

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 47.06±1.64 | +0.09 (p=0.856) | — |
| MFC-A (gamma=2.0, selected) | 10 | 46.97±1.56 | — | -0.09 (p=0.856) |
| AGM (alpha=0.5) | 10 | 47.49±1.28 | +0.52 (p=0.39) | +0.43 (p=0.427) |
| PMR (alpha=1.0, mu=0.001) | 10 | 47.50±1.81 | +0.53 (p=0.276) | +0.44 (p=0.143) |
| OLM (eta=1.0) | 10 | 47.58±1.71 | +0.61 (p=0.209) | +0.52 (p=0.103) |

### acc_a

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 45.35±2.12 | +0.65 (p=0.258) | — |
| MFC-A (gamma=2.0, selected) | 10 | 44.70±2.15 | — | -0.65 (p=0.258) |
| AGM (alpha=0.5) | 10 | 45.54±2.21 | +0.84 (p=0.175) | +0.19 (p=0.533) |
| PMR (alpha=1.0, mu=0.001) | 10 | 44.54±2.49 | -0.15 (p=0.796) | -0.81 (p=0.111) |
| OLM (eta=1.0) | 10 | 44.78±1.67 | +0.09 (p=0.849) | -0.56 (p=0.349) |

### ogr_gap

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 1.63±0.96 | +0.38 (p=0.241) | — |
| MFC-A (gamma=2.0, selected) | 10 | 1.25±0.61 | — | -0.38 (p=0.241) |
| AGM (alpha=0.5) | 10 | 1.71±0.59 | +0.46 (p=0.141) | +0.08 (p=0.762) |
| PMR (alpha=1.0, mu=0.001) | 10 | 1.66±0.88 | +0.41 (p=0.157) | +0.02 (p=0.785) |
| OLM (eta=1.0) | 10 | 1.10±0.50 | -0.14 (p=0.345) | -0.53 (p=0.157) |

### gauss_0.5

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 43.96±1.39 | -0.33 (p=0.247) | — |
| MFC-A (gamma=2.0, selected) | 10 | 44.29±1.23 | — | +0.33 (p=0.247) |
| AGM (alpha=0.5) | 10 | 43.86±1.36 | -0.43 (p=0.405) | -0.10 (p=0.812) |
| PMR (alpha=1.0, mu=0.001) | 10 | 44.04±1.27 | -0.25 (p=0.302) | +0.08 (p=0.763) |
| OLM (eta=1.0) | 10 | 43.74±1.74 | -0.56 (p=0.241) | -0.22 (p=0.575) |

### gauss_1.0

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 35.77±1.52 | -0.16 (p=0.711) | — |
| MFC-A (gamma=2.0, selected) | 10 | 35.93±1.63 | — | +0.16 (p=0.711) |
| AGM (alpha=0.5) | 10 | 35.57±1.44 | -0.36 (p=0.353) | -0.20 (p=0.519) |
| PMR (alpha=1.0, mu=0.001) | 10 | 35.61±1.88 | -0.32 (p=0.277) | -0.16 (p=0.592) |
| OLM (eta=1.0) | 10 | 36.22±2.21 | +0.29 (p=0.636) | +0.45 (p=0.498) |

### Method cost (wall-clock seconds per run, mean over seeds; CPU, 1 thread)

| method | train | method | monitor | ctrl | select | total |
|---|---|---|---|---|---|---|
| Baseline | 2.49 | 0.00 | 0.00 | 0.00 | 0.03 | 2.52 |
| MFC-A (gamma=2.0, selected) | 3.09 | 0.00 | 0.22 | 0.01 | 0.04 | 3.36 |
| AGM (alpha=0.5) | 9.15 | 3.35 | 0.00 | 0.00 | 0.10 | 12.61 |
| PMR (alpha=1.0, mu=0.001) | 12.85 | 2.68 | 0.00 | 0.00 | 0.16 | 15.69 |
| OLM (eta=1.0) | 8.90 | 1.40 | 0.92 | 0.01 | 0.13 | 11.35 |


Notes:
- Wall-clock caveat: AGM/PMR/OLM ran on the shared server under heavy concurrent load (load avg ~55 on 24 cores from other jobs), so even their pure-SGD 'train' time is inflated 1.5-3x vs the baseline/MFC-A rows (crema_av/ave baseline/MFC-A were run on the same server earlier at lower load; synthetic/crema_spec baseline/MFC-A on a laptop). Treat the 'method' and 'monitor' columns (per-method overhead) and the 'flops' field in each JSON as the comparable cost measures.
- 'method' = AGM's zero-input Shapley passes + coefficient math / PMR's per-epoch prototype pass + distances, PCE and PER / OLM's per-step coefficient update; 'monitor' = OLM's per-epoch train+ctrl loss passes (MFC-A: its ctrl/train passes).
- synthetic/crema_spec baseline and MFC-A JSONs were produced on a laptop; rerunning them on the server changes results in the 3rd-4th significant digit (float/platform), so paired comparisons there mix platforms.
