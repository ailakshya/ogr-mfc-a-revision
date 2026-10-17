# v2 results: multi-severity degradations and gradient norms
(mean ± std over seeds; p = two-sided paired t-test over seeds)


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

### blur

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 24.67±4.99 | +0.89 (p=0.0538) | — |
| Gradient Blending | 10 | 23.36±4.63 | -0.43 (p=0.76) | -1.31 (p=0.319) |
| Uncertainty weighting | 10 | 24.24±4.16 | +0.45 (p=0.665) | -0.43 (p=0.665) |
| MMPareto | 10 | 23.49±3.30 | -0.29 (p=0.836) | -1.18 (p=0.364) |
| OGM-GE (alpha=0.3) | 10 | 22.67±3.39 | -1.12 (p=0.513) | -2.00 (p=0.24) |
| MFC-T (threshold) | 10 | 24.39±4.61 | +0.60 (p=0.254) | -0.28 (p=0.419) |
| MFC-A (gamma=1) | 10 | 24.05±4.77 | +0.26 (p=0.642) | -0.62 (p=0.199) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 23.79±5.10 | — | -0.89 (p=0.0538) |

### occlusion

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 28.35±3.86 | +0.56 (p=0.352) | — |
| Gradient Blending | 10 | 28.16±4.04 | +0.38 (p=0.74) | -0.19 (p=0.881) |
| Uncertainty weighting | 10 | 27.76±3.57 | -0.03 (p=0.968) | -0.60 (p=0.423) |
| MMPareto | 10 | 26.11±2.55 | -1.68 (p=0.073) | -2.24 (p=0.0576) |
| OGM-GE (alpha=0.3) | 10 | 26.61±4.99 | -1.18 (p=0.369) | -1.75 (p=0.155) |
| MFC-T (threshold) | 10 | 27.50±3.35 | -0.29 (p=0.477) | -0.85 (p=0.272) |
| MFC-A (gamma=1) | 10 | 27.54±2.64 | -0.25 (p=0.702) | -0.81 (p=0.262) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 27.79±2.89 | — | -0.56 (p=0.352) |

### lowlight

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.44±3.41 | +0.58 (p=0.507) | — |
| Gradient Blending | 10 | 36.34±6.25 | +2.48 (p=0.123) | +1.90 (p=0.189) |
| Uncertainty weighting | 10 | 34.39±5.03 | +0.53 (p=0.641) | -0.05 (p=0.955) |
| MMPareto | 10 | 33.33±4.67 | -0.53 (p=0.649) | -1.11 (p=0.339) |
| OGM-GE (alpha=0.3) | 10 | 35.58±3.03 | +1.72 (p=0.198) | +1.14 (p=0.272) |
| MFC-T (threshold) | 10 | 33.73±4.47 | -0.13 (p=0.892) | -0.71 (p=0.399) |
| MFC-A (gamma=1) | 10 | 34.71±5.30 | +0.85 (p=0.569) | +0.27 (p=0.771) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 33.86±5.26 | — | -0.58 (p=0.507) |

### blur_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.14±4.12 | -0.10 (p=0.904) | — |
| Gradient Blending | 10 | 36.94±5.53 | +2.69 (p=0.0297) | +2.79 (p=0.0351) |
| Uncertainty weighting | 10 | 33.95±4.22 | -0.30 (p=0.79) | -0.20 (p=0.877) |
| MMPareto | 10 | 33.85±3.75 | -0.39 (p=0.722) | -0.29 (p=0.811) |
| OGM-GE (alpha=0.3) | 10 | 32.97±3.37 | -1.28 (p=0.162) | -1.17 (p=0.356) |
| MFC-T (threshold) | 10 | 34.05±3.74 | -0.19 (p=0.797) | -0.09 (p=0.921) |
| MFC-A (gamma=1) | 10 | 33.57±4.93 | -0.68 (p=0.568) | -0.58 (p=0.586) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 34.25±3.18 | — | +0.10 (p=0.904) |

### blur_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 26.85±4.75 | -0.04 (p=0.941) | — |
| Gradient Blending | 10 | 27.08±4.92 | +0.19 (p=0.882) | +0.23 (p=0.857) |
| Uncertainty weighting | 10 | 26.54±4.20 | -0.35 (p=0.794) | -0.31 (p=0.822) |
| MMPareto | 10 | 26.18±3.44 | -0.71 (p=0.634) | -0.66 (p=0.655) |
| OGM-GE (alpha=0.3) | 10 | 24.92±3.98 | -1.97 (p=0.259) | -1.92 (p=0.306) |
| MFC-T (threshold) | 10 | 26.59±4.11 | -0.30 (p=0.548) | -0.25 (p=0.735) |
| MFC-A (gamma=1) | 10 | 26.20±4.41 | -0.69 (p=0.235) | -0.65 (p=0.435) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 26.89±4.44 | — | +0.04 (p=0.941) |

### blur_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 24.67±4.99 | +0.89 (p=0.0538) | — |
| Gradient Blending | 10 | 23.36±4.63 | -0.43 (p=0.76) | -1.31 (p=0.319) |
| Uncertainty weighting | 10 | 24.24±4.16 | +0.45 (p=0.665) | -0.43 (p=0.665) |
| MMPareto | 10 | 23.49±3.30 | -0.29 (p=0.836) | -1.18 (p=0.364) |
| OGM-GE (alpha=0.3) | 10 | 22.67±3.39 | -1.12 (p=0.513) | -2.00 (p=0.24) |
| MFC-T (threshold) | 10 | 24.39±4.61 | +0.60 (p=0.254) | -0.28 (p=0.419) |
| MFC-A (gamma=1) | 10 | 24.05±4.77 | +0.26 (p=0.642) | -0.62 (p=0.199) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 23.79±5.10 | — | -0.89 (p=0.0538) |

### occlusion_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 39.10±2.30 | +0.19 (p=0.622) | — |
| Gradient Blending | 10 | 40.48±3.54 | +1.57 (p=0.0925) | +1.38 (p=0.173) |
| Uncertainty weighting | 10 | 38.83±2.42 | -0.08 (p=0.892) | -0.27 (p=0.607) |
| MMPareto | 10 | 38.03±3.46 | -0.88 (p=0.0831) | -1.07 (p=0.0659) |
| OGM-GE (alpha=0.3) | 10 | 38.05±3.07 | -0.86 (p=0.15) | -1.05 (p=0.0934) |
| MFC-T (threshold) | 10 | 38.76±2.51 | -0.15 (p=0.655) | -0.34 (p=0.452) |
| MFC-A (gamma=1) | 10 | 37.95±2.29 | -0.96 (p=0.089) | -1.15 (p=0.0306) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 38.91±2.71 | — | -0.19 (p=0.622) |

### occlusion_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.03±2.90 | +0.15 (p=0.73) | — |
| Gradient Blending | 10 | 34.87±3.38 | +0.99 (p=0.295) | +0.84 (p=0.413) |
| Uncertainty weighting | 10 | 33.77±2.95 | -0.11 (p=0.885) | -0.26 (p=0.749) |
| MMPareto | 10 | 32.74±2.99 | -1.14 (p=0.16) | -1.29 (p=0.164) |
| OGM-GE (alpha=0.3) | 10 | 32.47±3.93 | -1.41 (p=0.16) | -1.56 (p=0.101) |
| MFC-T (threshold) | 10 | 33.15±2.81 | -0.73 (p=0.23) | -0.88 (p=0.188) |
| MFC-A (gamma=1) | 10 | 33.16±2.47 | -0.72 (p=0.286) | -0.87 (p=0.203) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 33.88±2.01 | — | -0.15 (p=0.73) |

### occlusion_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 28.83±4.58 | +0.62 (p=0.417) | — |
| Gradient Blending | 10 | 28.19±4.38 | -0.01 (p=0.994) | -0.63 (p=0.653) |
| Uncertainty weighting | 10 | 28.79±4.54 | +0.59 (p=0.542) | -0.04 (p=0.954) |
| MMPareto | 10 | 26.36±3.08 | -1.84 (p=0.139) | -2.47 (p=0.0798) |
| OGM-GE (alpha=0.3) | 10 | 26.69±4.87 | -1.51 (p=0.283) | -2.13 (p=0.142) |
| MFC-T (threshold) | 10 | 27.81±3.40 | -0.40 (p=0.429) | -1.02 (p=0.297) |
| MFC-A (gamma=1) | 10 | 27.61±3.11 | -0.59 (p=0.218) | -1.22 (p=0.112) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 28.20±3.14 | — | -0.62 (p=0.417) |

### lowlight_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 42.20±5.52 | +0.49 (p=0.589) | — |
| Gradient Blending | 10 | 43.20±7.33 | +1.48 (p=0.258) | +0.99 (p=0.321) |
| Uncertainty weighting | 10 | 41.65±5.08 | -0.06 (p=0.932) | -0.55 (p=0.408) |
| MMPareto | 10 | 42.16±4.94 | +0.45 (p=0.703) | -0.04 (p=0.972) |
| OGM-GE (alpha=0.3) | 10 | 42.18±5.54 | +0.47 (p=0.499) | -0.02 (p=0.985) |
| MFC-T (threshold) | 10 | 41.18±5.34 | -0.53 (p=0.569) | -1.03 (p=0.27) |
| MFC-A (gamma=1) | 10 | 42.01±4.78 | +0.29 (p=0.812) | -0.20 (p=0.784) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 41.71±5.83 | — | -0.49 (p=0.589) |

### lowlight_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 38.32±2.23 | +0.16 (p=0.885) | — |
| Gradient Blending | 10 | 40.61±5.19 | +2.45 (p=0.208) | +2.29 (p=0.0931) |
| Uncertainty weighting | 10 | 38.70±2.46 | +0.54 (p=0.567) | +0.38 (p=0.591) |
| MMPareto | 10 | 38.40±4.27 | +0.24 (p=0.815) | +0.08 (p=0.944) |
| OGM-GE (alpha=0.3) | 10 | 38.16±4.37 | -0.00 (p=0.998) | -0.16 (p=0.877) |
| MFC-T (threshold) | 10 | 37.16±4.91 | -1.00 (p=0.162) | -1.16 (p=0.324) |
| MFC-A (gamma=1) | 10 | 38.80±3.41 | +0.65 (p=0.688) | +0.48 (p=0.546) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 38.16±4.81 | — | -0.16 (p=0.885) |

### lowlight_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 34.70±4.01 | +0.78 (p=0.214) | — |
| Gradient Blending | 10 | 36.20±6.01 | +2.28 (p=0.126) | +1.50 (p=0.234) |
| Uncertainty weighting | 10 | 34.41±4.77 | +0.49 (p=0.552) | -0.30 (p=0.633) |
| MMPareto | 10 | 33.83±5.32 | -0.09 (p=0.938) | -0.87 (p=0.434) |
| OGM-GE (alpha=0.3) | 10 | 35.83±3.28 | +1.91 (p=0.161) | +1.13 (p=0.328) |
| MFC-T (threshold) | 10 | 34.16±4.56 | +0.24 (p=0.785) | -0.54 (p=0.526) |
| MFC-A (gamma=1) | 10 | 35.35±5.59 | +1.43 (p=0.307) | +0.65 (p=0.483) |
| textbf{MFC-A} (gamma=2, selected) | 10 | 33.92±4.82 | — | -0.78 (p=0.214) |

### gradient norms (probe = first 512 train samples, eval mode), crema_av

| method | quantity | ep1 | ep5 | ep10 | ep20 | ep40 | mean over epochs |
|---|---|---|---|---|---|---|---|
| Baseline | gj_v | 0.6383 | 1.7693 | 1.3986 | 0.8005 | 0.2927 | 0.8775 |
| Baseline | gj_a | 0.7437 | 0.9414 | 0.7891 | 0.5745 | 0.2522 | 0.5777 |
| Baseline | gu_v | 0.9204 | 2.5265 | 2.4234 | 1.9286 | 1.4180 | 1.9469 |
| Baseline | gu_a | 0.9146 | 1.3916 | 1.6867 | 1.8431 | 1.6730 | 1.6468 |
| Baseline | gj_a/gj_v | 1.1693 | 0.5265 | 0.5610 | 0.7234 | 0.8886 | 0.7229 |
| Baseline | gj_a/(gj_v+gj_a) | 0.5358 | 0.3428 | 0.3589 | 0.4179 | 0.4659 | 0.4130 |
| Gradient Blending | gj_v | 0.6209 | 1.9292 | 1.7694 | 1.1340 | 0.4532 | 1.1030 |
| Gradient Blending | gj_a | 0.7406 | 1.1150 | 0.7682 | 0.5492 | 0.2448 | 0.5812 |
| Gradient Blending | gu_v | 0.9009 | 1.3472 | 1.3892 | 1.3360 | 1.2304 | 1.2976 |
| Gradient Blending | gu_a | 0.9824 | 1.4482 | 1.3896 | 1.4163 | 1.1219 | 1.3262 |
| Gradient Blending | gj_a/gj_v | 1.2007 | 0.5794 | 0.4372 | 0.4817 | 0.5363 | 0.5447 |
| Gradient Blending | gj_a/(gj_v+gj_a) | 0.5435 | 0.3645 | 0.3035 | 0.3239 | 0.3470 | 0.3461 |
| Uncertainty weighting | gj_v | 0.6237 | 1.9020 | 1.5320 | 0.8604 | 0.3080 | 0.9465 |
| Uncertainty weighting | gj_a | 0.7597 | 0.9487 | 0.8807 | 0.6193 | 0.3074 | 0.6383 |
| Uncertainty weighting | gu_v | 0.8663 | 2.3510 | 2.5715 | 1.9539 | 1.2491 | 1.9386 |
| Uncertainty weighting | gu_a | 0.8459 | 1.2972 | 1.7254 | 1.8725 | 1.5150 | 1.6391 |
| Uncertainty weighting | gj_a/gj_v | 1.2235 | 0.4935 | 0.5732 | 0.7245 | 0.9948 | 0.7541 |
| Uncertainty weighting | gj_a/(gj_v+gj_a) | 0.5466 | 0.3285 | 0.3634 | 0.4196 | 0.4930 | 0.4214 |
| MMPareto | gj_v | 0.6452 | 1.7918 | 1.3954 | 0.7920 | 0.2784 | 0.8724 |
| MMPareto | gj_a | 0.7504 | 0.9523 | 0.7720 | 0.5816 | 0.2380 | 0.5753 |
| MMPareto | gu_v | 0.9176 | 2.5512 | 2.4459 | 1.9624 | 1.4384 | 1.9539 |
| MMPareto | gu_a | 0.9282 | 1.3961 | 1.6647 | 1.8682 | 1.6345 | 1.6509 |
| MMPareto | gj_a/gj_v | 1.1686 | 0.5253 | 0.5508 | 0.7362 | 0.9244 | 0.7250 |
| MMPareto | gj_a/(gj_v+gj_a) | 0.5359 | 0.3426 | 0.3545 | 0.4224 | 0.4701 | 0.4136 |
| OGM-GE (alpha=0.3) | gj_v | 0.6444 | 1.8040 | 1.4306 | 0.8289 | 0.3709 | 0.9124 |
| OGM-GE (alpha=0.3) | gj_a | 0.7615 | 0.9008 | 0.8018 | 0.5652 | 0.2868 | 0.5849 |
| OGM-GE (alpha=0.3) | gu_v | 0.9383 | 2.4907 | 2.4529 | 2.0670 | 1.4753 | 1.9528 |
| OGM-GE (alpha=0.3) | gu_a | 0.9321 | 1.3415 | 1.5928 | 1.6193 | 1.5118 | 1.5553 |
| OGM-GE (alpha=0.3) | gj_a/gj_v | 1.1874 | 0.4991 | 0.5637 | 0.6832 | 0.7639 | 0.6977 |
| OGM-GE (alpha=0.3) | gj_a/(gj_v+gj_a) | 0.5398 | 0.3321 | 0.3589 | 0.4043 | 0.4314 | 0.4053 |
| MFC-T (threshold) | gj_v | 0.6383 | 1.7895 | 1.3837 | 0.7517 | 0.2405 | 0.8629 |
| MFC-T (threshold) | gj_a | 0.7437 | 0.9626 | 0.7332 | 0.5036 | 0.2097 | 0.5389 |
| MFC-T (threshold) | gu_v | 0.9204 | 2.5410 | 2.4624 | 2.1767 | 1.8092 | 2.1482 |
| MFC-T (threshold) | gu_a | 0.9146 | 1.4188 | 1.6351 | 1.8583 | 1.6843 | 1.6607 |
| MFC-T (threshold) | gj_a/gj_v | 1.1693 | 0.5330 | 0.5298 | 0.6722 | 0.8857 | 0.6791 |
| MFC-T (threshold) | gj_a/(gj_v+gj_a) | 0.5358 | 0.3454 | 0.3457 | 0.4003 | 0.4682 | 0.3984 |
| MFC-A (gamma=1) | gj_v | 0.6383 | 1.7671 | 1.4422 | 0.8198 | 0.3096 | 0.9134 |
| MFC-A (gamma=1) | gj_a | 0.7437 | 0.9409 | 0.7677 | 0.4996 | 0.1996 | 0.5308 |
| MFC-A (gamma=1) | gu_v | 0.9204 | 2.5239 | 2.5212 | 2.1635 | 1.6787 | 2.1389 |
| MFC-A (gamma=1) | gu_a | 0.9146 | 1.3884 | 1.6533 | 1.7557 | 1.4072 | 1.5438 |
| MFC-A (gamma=1) | gj_a/gj_v | 1.1693 | 0.5271 | 0.5309 | 0.6096 | 0.6692 | 0.6104 |
| MFC-A (gamma=1) | gj_a/(gj_v+gj_a) | 0.5358 | 0.3431 | 0.3458 | 0.3773 | 0.3977 | 0.3740 |
| textbf{MFC-A} (gamma=2, selected) | gj_v | 0.6383 | 1.7762 | 1.4350 | 0.8921 | 0.3016 | 0.9474 |
| textbf{MFC-A} (gamma=2, selected) | gj_a | 0.7437 | 0.9441 | 0.7557 | 0.4871 | 0.1723 | 0.5221 |
| textbf{MFC-A} (gamma=2, selected) | gu_v | 0.9204 | 2.5332 | 2.5375 | 2.3949 | 1.9830 | 2.2983 |
| textbf{MFC-A} (gamma=2, selected) | gu_a | 0.9146 | 1.3946 | 1.6241 | 1.6480 | 1.2136 | 1.4812 |
| textbf{MFC-A} (gamma=2, selected) | gj_a/gj_v | 1.1693 | 0.5270 | 0.5290 | 0.5477 | 0.5789 | 0.5690 |
| textbf{MFC-A} (gamma=2, selected) | gj_a/(gj_v+gj_a) | 0.5358 | 0.3431 | 0.3450 | 0.3528 | 0.3638 | 0.3582 |

Epoch-averaged gradient quantities, paired over seeds vs baseline:

- ogm_ge__alpha=0.3 gj_v: 0.9124 vs baseline 0.8775 (ratio 1.04, p=2.8e-05)
- ogm_ge__alpha=0.3 gj_a: 0.5849 vs baseline 0.5777 (ratio 1.01, p=0.203)
- ogm_ge__alpha=0.3 gu_v: 1.9528 vs baseline 1.9469 (ratio 1.00, p=0.585)
- ogm_ge__alpha=0.3 gu_a: 1.5553 vs baseline 1.6468 (ratio 0.94, p=0.000267)
- ogm_ge__alpha=0.3 ratio: 0.6977 vs baseline 0.7229 (ratio 0.97, p=0.000877)
- mfc_a__gamma=2.0 gj_v: 0.9474 vs baseline 0.8775 (ratio 1.08, p=4.47e-05)
- mfc_a__gamma=2.0 gj_a: 0.5221 vs baseline 0.5777 (ratio 0.90, p=6.64e-06)
- mfc_a__gamma=2.0 gu_v: 2.2983 vs baseline 1.9469 (ratio 1.18, p=1.47e-09)
- mfc_a__gamma=2.0 gu_a: 1.4812 vs baseline 1.6468 (ratio 0.90, p=1.14e-06)
- mfc_a__gamma=2.0 ratio: 0.5690 vs baseline 0.7229 (ratio 0.79, p=1.02e-08)

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

### blur

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 39.43±4.77 | -2.09 (p=0.0351) | — |
| Gradient Blending | 10 | 43.48±6.64 | +1.97 (p=0.176) | +4.05 (p=0.00799) |
| Uncertainty weighting | 10 | 32.31±4.46 | -9.20 (p=0.00232) | -7.11 (p=0.00762) |
| MMPareto | 10 | 37.71±5.37 | -3.81 (p=0.00262) | -1.72 (p=0.0657) |
| OGM-GE (alpha=0.8) | 10 | 51.22±4.87 | +9.70 (p=3.09e-05) | +11.79 (p=8.68e-06) |
| MFC-T (threshold) | 10 | 39.75±4.13 | -1.77 (p=0.0617) | +0.32 (p=0.626) |
| MFC-A (gamma=1) | 10 | 40.22±4.66 | -1.29 (p=0.201) | +0.80 (p=0.326) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 41.52±4.99 | — | +2.09 (p=0.0351) |

### occlusion

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 65.50±2.69 | -0.30 (p=0.702) | — |
| Gradient Blending | 10 | 68.31±2.55 | +2.51 (p=0.00651) | +2.81 (p=0.00577) |
| Uncertainty weighting | 10 | 62.56±2.11 | -3.23 (p=0.00152) | -2.94 (p=0.0188) |
| MMPareto | 10 | 64.93±2.03 | -0.87 (p=0.318) | -0.57 (p=0.393) |
| OGM-GE (alpha=0.8) | 10 | 71.44±1.65 | +5.65 (p=3.46e-06) | +5.95 (p=7.01e-06) |
| MFC-T (threshold) | 10 | 65.57±1.50 | -0.22 (p=0.756) | +0.07 (p=0.932) |
| MFC-A (gamma=1) | 10 | 65.77±1.58 | -0.02 (p=0.968) | +0.27 (p=0.713) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 65.80±1.75 | — | +0.30 (p=0.702) |

### lowlight

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 50.97±1.92 | -1.37 (p=0.0299) | — |
| Gradient Blending | 10 | 57.41±3.07 | +5.07 (p=0.000495) | +6.44 (p=2.48e-05) |
| Uncertainty weighting | 10 | 49.25±2.60 | -3.08 (p=0.0204) | -1.72 (p=0.119) |
| MMPareto | 10 | 50.50±2.76 | -1.84 (p=0.00848) | -0.47 (p=0.431) |
| OGM-GE (alpha=0.8) | 10 | 61.84±2.30 | +9.50 (p=1.49e-06) | +10.87 (p=1.55e-07) |
| MFC-T (threshold) | 10 | 51.59±2.59 | -0.75 (p=0.201) | +0.62 (p=0.474) |
| MFC-A (gamma=1) | 10 | 52.06±2.91 | -0.27 (p=0.594) | +1.09 (p=0.203) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 52.34±2.25 | — | +1.37 (p=0.0299) |

### blur_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 71.17±1.81 | -1.29 (p=0.0711) | — |
| Gradient Blending | 10 | 74.00±2.28 | +1.54 (p=0.0946) | +2.84 (p=0.00203) |
| Uncertainty weighting | 10 | 68.11±1.55 | -4.35 (p=1.11e-05) | -3.06 (p=0.0012) |
| MMPareto | 10 | 70.67±1.78 | -1.79 (p=0.0203) | -0.50 (p=0.455) |
| OGM-GE (alpha=0.8) | 10 | 76.22±0.68 | +3.76 (p=5.04e-06) | +5.05 (p=6.11e-05) |
| MFC-T (threshold) | 10 | 71.97±1.89 | -0.50 (p=0.408) | +0.80 (p=0.379) |
| MFC-A (gamma=1) | 10 | 72.06±1.63 | -0.40 (p=0.339) | +0.90 (p=0.193) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 72.46±1.17 | — | +1.29 (p=0.0711) |

### blur_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 53.08±3.02 | -1.02 (p=0.218) | — |
| Gradient Blending | 10 | 56.67±4.27 | +2.56 (p=0.0182) | +3.58 (p=0.00422) |
| Uncertainty weighting | 10 | 46.64±4.03 | -7.46 (p=0.00119) | -6.44 (p=0.00237) |
| MMPareto | 10 | 50.85±3.65 | -3.26 (p=0.00426) | -2.24 (p=0.0234) |
| OGM-GE (alpha=0.8) | 10 | 61.84±3.02 | +7.74 (p=5.1e-07) | +8.76 (p=6.48e-07) |
| MFC-T (threshold) | 10 | 53.26±2.72 | -0.85 (p=0.174) | +0.17 (p=0.851) |
| MFC-A (gamma=1) | 10 | 52.89±2.60 | -1.22 (p=0.0918) | -0.20 (p=0.828) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 54.10±2.43 | — | +1.02 (p=0.218) |

### blur_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 39.43±4.77 | -2.09 (p=0.0351) | — |
| Gradient Blending | 10 | 43.48±6.64 | +1.97 (p=0.176) | +4.05 (p=0.00799) |
| Uncertainty weighting | 10 | 32.31±4.46 | -9.20 (p=0.00232) | -7.11 (p=0.00762) |
| MMPareto | 10 | 37.71±5.37 | -3.81 (p=0.00262) | -1.72 (p=0.0657) |
| OGM-GE (alpha=0.8) | 10 | 51.22±4.87 | +9.70 (p=3.09e-05) | +11.79 (p=8.68e-06) |
| MFC-T (threshold) | 10 | 39.75±4.13 | -1.77 (p=0.0617) | +0.32 (p=0.626) |
| MFC-A (gamma=1) | 10 | 40.22±4.66 | -1.29 (p=0.201) | +0.80 (p=0.326) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 41.52±4.99 | — | +2.09 (p=0.0351) |

### occlusion_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 76.39±1.89 | -0.70 (p=0.239) | — |
| Gradient Blending | 10 | 77.69±1.48 | +0.60 (p=0.283) | +1.29 (p=0.0133) |
| Uncertainty weighting | 10 | 73.81±1.18 | -3.28 (p=0.000389) | -2.59 (p=0.0139) |
| MMPareto | 10 | 75.57±1.59 | -1.52 (p=0.0187) | -0.82 (p=0.0474) |
| OGM-GE (alpha=0.8) | 10 | 79.40±1.29 | +2.31 (p=0.000812) | +3.01 (p=0.00244) |
| MFC-T (threshold) | 10 | 76.54±0.84 | -0.55 (p=0.146) | +0.15 (p=0.808) |
| MFC-A (gamma=1) | 10 | 76.54±0.96 | -0.55 (p=0.214) | +0.15 (p=0.844) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 77.09±1.27 | — | +0.70 (p=0.239) |

### occlusion_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 72.76±1.90 | -0.65 (p=0.194) | — |
| Gradient Blending | 10 | 75.30±1.19 | +1.89 (p=0.00141) | +2.54 (p=0.00086) |
| Uncertainty weighting | 10 | 70.22±1.62 | -3.18 (p=0.000343) | -2.54 (p=0.0132) |
| MMPareto | 10 | 72.26±1.51 | -1.14 (p=0.0808) | -0.50 (p=0.165) |
| OGM-GE (alpha=0.8) | 10 | 77.11±0.97 | +3.71 (p=0.000326) | +4.35 (p=0.000452) |
| MFC-T (threshold) | 10 | 73.38±1.05 | -0.02 (p=0.961) | +0.62 (p=0.232) |
| MFC-A (gamma=1) | 10 | 73.58±1.52 | +0.17 (p=0.61) | +0.82 (p=0.144) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 73.41±1.71 | — | +0.65 (p=0.194) |

### occlusion_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 65.62±1.70 | -0.82 (p=0.209) | — |
| Gradient Blending | 10 | 68.21±2.50 | +1.77 (p=0.0287) | +2.59 (p=0.00201) |
| Uncertainty weighting | 10 | 62.24±2.75 | -4.20 (p=0.00177) | -3.38 (p=0.00629) |
| MMPareto | 10 | 64.70±1.89 | -1.74 (p=0.00801) | -0.92 (p=0.0846) |
| OGM-GE (alpha=0.8) | 10 | 71.59±1.93 | +5.15 (p=2.06e-06) | +5.97 (p=1.63e-05) |
| MFC-T (threshold) | 10 | 65.87±1.53 | -0.57 (p=0.315) | +0.25 (p=0.691) |
| MFC-A (gamma=1) | 10 | 66.04±1.44 | -0.40 (p=0.359) | +0.42 (p=0.554) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 66.44±1.42 | — | +0.82 (p=0.209) |

### lowlight_s1

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 74.48±1.22 | -0.95 (p=0.0236) | — |
| Gradient Blending | 10 | 76.84±1.50 | +1.42 (p=0.00191) | +2.36 (p=0.000323) |
| Uncertainty weighting | 10 | 72.59±1.85 | -2.84 (p=0.000332) | -1.89 (p=0.00928) |
| MMPareto | 10 | 74.65±1.25 | -0.77 (p=0.00151) | +0.17 (p=0.58) |
| OGM-GE (alpha=0.8) | 10 | 78.46±1.16 | +3.03 (p=8.76e-05) | +3.98 (p=7.32e-06) |
| MFC-T (threshold) | 10 | 75.10±1.27 | -0.32 (p=0.278) | +0.62 (p=0.15) |
| MFC-A (gamma=1) | 10 | 75.10±1.27 | -0.32 (p=0.423) | +0.62 (p=0.164) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 75.42±1.19 | — | +0.95 (p=0.0236) |

### lowlight_s2

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 65.45±2.14 | -1.79 (p=0.00538) | — |
| Gradient Blending | 10 | 68.83±1.67 | +1.59 (p=0.0148) | +3.38 (p=0.000187) |
| Uncertainty weighting | 10 | 62.76±2.26 | -4.48 (p=0.000415) | -2.69 (p=0.00777) |
| MMPareto | 10 | 65.60±2.37 | -1.64 (p=0.00127) | +0.15 (p=0.782) |
| OGM-GE (alpha=0.8) | 10 | 71.87±1.64 | +4.63 (p=2.98e-05) | +6.42 (p=1.07e-06) |
| MFC-T (threshold) | 10 | 66.34±1.39 | -0.90 (p=0.00999) | +0.90 (p=0.124) |
| MFC-A (gamma=1) | 10 | 66.37±1.99 | -0.87 (p=0.0371) | +0.92 (p=0.165) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 67.24±1.57 | — | +1.79 (p=0.00538) |

### lowlight_s3

| method | n | mean±std | vs sel MFC-A | vs baseline |
|---|---|---|---|---|
| Baseline | 10 | 50.95±2.12 | -1.14 (p=0.18) | — |
| Gradient Blending | 10 | 57.04±3.04 | +4.95 (p=0.000225) | +6.09 (p=0.000128) |
| Uncertainty weighting | 10 | 49.20±2.22 | -2.89 (p=0.0191) | -1.74 (p=0.0859) |
| MMPareto | 10 | 51.12±2.77 | -0.97 (p=0.205) | +0.17 (p=0.817) |
| OGM-GE (alpha=0.8) | 10 | 61.34±2.19 | +9.25 (p=2.71e-06) | +10.40 (p=1.55e-08) |
| MFC-T (threshold) | 10 | 51.37±2.17 | -0.72 (p=0.25) | +0.42 (p=0.488) |
| MFC-A (gamma=1) | 10 | 51.32±2.82 | -0.77 (p=0.233) | +0.37 (p=0.652) |
| textbf{MFC-A} (gamma=3, selected) | 10 | 52.09±1.79 | — | +1.14 (p=0.18) |

### gradient norms (probe = first 512 train samples, eval mode), ave

| method | quantity | ep1 | ep5 | ep10 | ep20 | ep40 | mean over epochs |
|---|---|---|---|---|---|---|---|
| Baseline | gj_v | 0.9306 | 0.7403 | 0.4346 | 0.2105 | 0.1205 | 0.3235 |
| Baseline | gj_a | 0.8094 | 0.3458 | 0.1598 | 0.0580 | 0.0370 | 0.1425 |
| Baseline | gu_v | 1.3068 | 1.1390 | 0.6446 | 0.3086 | 0.1589 | 0.4714 |
| Baseline | gu_a | 1.0512 | 1.4438 | 1.5836 | 1.4182 | 1.4602 | 1.4874 |
| Baseline | gj_a/gj_v | 0.8838 | 0.4714 | 0.3806 | 0.2816 | 0.3166 | 0.3543 |
| Baseline | gj_a/(gj_v+gj_a) | 0.4621 | 0.3198 | 0.2748 | 0.2192 | 0.2388 | 0.2547 |
| Gradient Blending | gj_v | 0.7473 | 0.7660 | 0.5543 | 0.2636 | 0.1445 | 0.3696 |
| Gradient Blending | gj_a | 0.9239 | 0.4543 | 0.2418 | 0.0910 | 0.0452 | 0.1892 |
| Gradient Blending | gu_v | 1.0335 | 1.4410 | 1.1322 | 0.7343 | 0.4275 | 0.8333 |
| Gradient Blending | gu_a | 1.2297 | 1.5128 | 1.6054 | 1.4022 | 1.3233 | 1.4114 |
| Gradient Blending | gj_a/gj_v | 1.2550 | 0.5963 | 0.4607 | 0.3466 | 0.3115 | 0.4205 |
| Gradient Blending | gj_a/(gj_v+gj_a) | 0.5517 | 0.3718 | 0.3111 | 0.2559 | 0.2362 | 0.2843 |
| Uncertainty weighting | gj_v | 0.8175 | 0.8981 | 0.4568 | 0.2690 | 0.1223 | 0.3484 |
| Uncertainty weighting | gj_a | 0.7371 | 0.4727 | 0.1794 | 0.0904 | 0.0687 | 0.1756 |
| Uncertainty weighting | gu_v | 1.0298 | 1.4308 | 0.7730 | 0.6702 | 0.4190 | 0.6734 |
| Uncertainty weighting | gu_a | 0.8330 | 1.3681 | 1.4405 | 1.4773 | 1.5993 | 1.4877 |
| Uncertainty weighting | gj_a/gj_v | 0.9147 | 0.5285 | 0.3937 | 0.3730 | 0.5785 | 0.4607 |
| Uncertainty weighting | gj_a/(gj_v+gj_a) | 0.4712 | 0.3449 | 0.2810 | 0.2695 | 0.3609 | 0.3082 |
| MMPareto | gj_v | 0.9288 | 0.7465 | 0.4334 | 0.2171 | 0.1222 | 0.3241 |
| MMPareto | gj_a | 0.8156 | 0.3473 | 0.1592 | 0.0600 | 0.0369 | 0.1425 |
| MMPareto | gu_v | 1.2923 | 1.1415 | 0.6465 | 0.3064 | 0.1619 | 0.4719 |
| MMPareto | gu_a | 1.0387 | 1.4454 | 1.5839 | 1.4444 | 1.5445 | 1.4842 |
| MMPareto | gj_a/gj_v | 0.8938 | 0.4702 | 0.3778 | 0.2821 | 0.3130 | 0.3540 |
| MMPareto | gj_a/(gj_v+gj_a) | 0.4645 | 0.3193 | 0.2734 | 0.2195 | 0.2368 | 0.2545 |
| OGM-GE (alpha=0.8) | gj_v | 0.9070 | 0.9010 | 0.8492 | 0.5821 | 0.3056 | 0.6239 |
| OGM-GE (alpha=0.8) | gj_a | 0.8154 | 0.6299 | 0.4486 | 0.2755 | 0.1184 | 0.3366 |
| OGM-GE (alpha=0.8) | gu_v | 1.2873 | 1.5402 | 1.6212 | 1.3109 | 0.7087 | 1.2177 |
| OGM-GE (alpha=0.8) | gu_a | 1.0519 | 1.4194 | 1.5279 | 1.4333 | 1.4391 | 1.4628 |
| OGM-GE (alpha=0.8) | gj_a/gj_v | 0.9130 | 0.7094 | 0.5338 | 0.4768 | 0.3934 | 0.5065 |
| OGM-GE (alpha=0.8) | gj_a/(gj_v+gj_a) | 0.4703 | 0.4138 | 0.3459 | 0.3218 | 0.2818 | 0.3296 |
| MFC-T (threshold) | gj_v | 0.9306 | 0.7228 | 0.4213 | 0.2196 | 0.1297 | 0.3230 |
| MFC-T (threshold) | gj_a | 0.8094 | 0.3350 | 0.1517 | 0.0604 | 0.0378 | 0.1401 |
| MFC-T (threshold) | gu_v | 1.3068 | 1.1235 | 0.6260 | 0.2994 | 0.1809 | 0.4668 |
| MFC-T (threshold) | gu_a | 1.0512 | 1.4637 | 1.6072 | 1.3681 | 1.3476 | 1.4245 |
| MFC-T (threshold) | gj_a/gj_v | 0.8838 | 0.4677 | 0.3738 | 0.2791 | 0.2999 | 0.3490 |
| MFC-T (threshold) | gj_a/(gj_v+gj_a) | 0.4621 | 0.3181 | 0.2713 | 0.2176 | 0.2291 | 0.2519 |
| MFC-A (gamma=1) | gj_v | 0.9306 | 0.7395 | 0.4435 | 0.2319 | 0.1309 | 0.3353 |
| MFC-A (gamma=1) | gj_a | 0.8094 | 0.3454 | 0.1611 | 0.0617 | 0.0336 | 0.1434 |
| MFC-A (gamma=1) | gu_v | 1.3068 | 1.1376 | 0.6616 | 0.3338 | 0.2125 | 0.4963 |
| MFC-A (gamma=1) | gu_a | 1.0512 | 1.4456 | 1.6142 | 1.4428 | 1.4189 | 1.4802 |
| MFC-A (gamma=1) | gj_a/gj_v | 0.8838 | 0.4715 | 0.3771 | 0.2715 | 0.2679 | 0.3405 |
| MFC-A (gamma=1) | gj_a/(gj_v+gj_a) | 0.4621 | 0.3199 | 0.2730 | 0.2129 | 0.2100 | 0.2465 |
| textbf{MFC-A} (gamma=3, selected) | gj_v | 0.9306 | 0.7383 | 0.4587 | 0.2603 | 0.1347 | 0.3453 |
| textbf{MFC-A} (gamma=3, selected) | gj_a | 0.8094 | 0.3447 | 0.1637 | 0.0706 | 0.0372 | 0.1469 |
| textbf{MFC-A} (gamma=3, selected) | gu_v | 1.3068 | 1.1351 | 0.6547 | 0.4228 | 0.2574 | 0.5444 |
| textbf{MFC-A} (gamma=3, selected) | gu_a | 1.0512 | 1.4479 | 1.6392 | 1.4127 | 1.3392 | 1.4521 |
| textbf{MFC-A} (gamma=3, selected) | gj_a/gj_v | 0.8838 | 0.4712 | 0.3728 | 0.2756 | 0.2809 | 0.3445 |
| textbf{MFC-A} (gamma=3, selected) | gj_a/(gj_v+gj_a) | 0.4621 | 0.3197 | 0.2706 | 0.2154 | 0.2183 | 0.2490 |

Epoch-averaged gradient quantities, paired over seeds vs baseline:

- ogm_ge__alpha=0.8 gj_v: 0.6239 vs baseline 0.3235 (ratio 1.93, p=1.08e-10)
- ogm_ge__alpha=0.8 gj_a: 0.3366 vs baseline 0.1425 (ratio 2.36, p=3.23e-11)
- ogm_ge__alpha=0.8 gu_v: 1.2177 vs baseline 0.4714 (ratio 2.58, p=7.45e-12)
- ogm_ge__alpha=0.8 gu_a: 1.4628 vs baseline 1.4874 (ratio 0.98, p=0.068)
- ogm_ge__alpha=0.8 ratio: 0.5065 vs baseline 0.3543 (ratio 1.43, p=1.15e-07)
- mfc_a__gamma=3.0 gj_v: 0.3453 vs baseline 0.3235 (ratio 1.07, p=2.21e-06)
- mfc_a__gamma=3.0 gj_a: 0.1469 vs baseline 0.1425 (ratio 1.03, p=0.000101)
- mfc_a__gamma=3.0 gu_v: 0.5444 vs baseline 0.4714 (ratio 1.15, p=1.68e-12)
- mfc_a__gamma=3.0 gu_a: 1.4521 vs baseline 1.4874 (ratio 0.98, p=0.00038)
- mfc_a__gamma=3.0 ratio: 0.3445 vs baseline 0.3543 (ratio 0.97, p=0.0172)

## Interpretation of the gradient-norm evidence (written for the 2026-10-18 run; numbers in the tables above)

- AVE, baseline: the audio encoder's share of the joint-loss gradient, gj_a/(gj_v+gj_a), falls from 0.46 at
  epoch 1 to about 0.22-0.24 from epoch 20 on, so after the first few epochs roughly three quarters of the
  L_joint gradient reaching the encoders goes to the vision encoder. Vision's own unimodal gradient gu_v decays
  fast (1.31 -> 0.16), audio's gu_a stays near 1.4-1.6, i.e. the vision branch fits the training probe much
  faster. This is consistent with a starved audio encoder under the joint loss, but it is a norm-share
  measurement, not a causal test, and the share never reaches zero ("reduced", not "collapsed").
- AVE, MFC-A (gamma=3): epoch-averaged audio share is essentially unchanged (ratio gj_a/gj_v 0.345 vs 0.354,
  p=0.017, ~3% lower). What MFC-A does change is the unimodal signal: gu_v +15% (p<1e-11), gu_a -2% (p<0.001).
  It does not rebalance the joint-loss gradient on AVE.
- AVE, OGM-GE (alpha=0.8): roughly doubles all joint and vision-unimodal gradient norms and raises the audio
  share (ratio 0.51 vs 0.35, p~1e-7). It is the only method here that measurably shifts joint-loss gradient
  towards audio, and it is also the most robust method on every AVE severity.
- CREMA-D: baseline audio share dips to ~0.34 at epoch 5 and recovers to ~0.47 by epoch 40, so there is no
  sustained starvation. MFC-A (gamma=2) lowers the audio share (ratio 0.57 vs 0.72, p~1e-8), raises gu_v by 18%
  and lowers gu_a by 10% (both p<1e-5) - i.e. it shifts unimodal gradient towards vision and away from audio.

