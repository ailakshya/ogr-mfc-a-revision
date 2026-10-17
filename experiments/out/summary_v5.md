Zero-lower-bound follow-up (grid_v5.txt): MFC-A, MFC-J and MFC-OG with clip=[0, 2] (a stream's weight may
reach 0, as Gradient Blending's offline weights can) and gamma in {1, 3, 5}. All runs on the workstation.

Goal stated before the run: on every dataset, match or beat Gradient Blending, i.e. be statistically no worse
(two-sided paired t-test, p >= 0.05, or better) on accuracy, OGR gap and robustness at sigma=1.
Settings are selected by the pre-registered rule (tune/ctrl splits only). Writes out/summary_v5.md.

Configurations tried per dataset: 12

## mfc_a with weights allowed to reach 0
- crema_av: mfc_a__clip=0.0,2.0__gamma=5.0  acc 50.19 gap 1.64 rob1 44.29 | vs Gradient Blending: acc -1.93 (p=0.039), gap +0.47 (p=0.123), rob1 -3.02 (p=0.002) | vs baseline: acc +0.51 (p=0.263), gap -2.09 (p=0.064), rob1 +0.86 (p=0.021) -> goal NOT met
- ave: mfc_a__clip=0.0,2.0__gamma=5.0  acc 80.25 gap 4.23 rob1 75.62 | vs Gradient Blending: acc -0.80 (p=0.131), gap -1.11 (p=0.639), rob1 -0.95 (p=0.115) | vs baseline: acc +0.37 (p=0.458), gap -21.64 (p=0.012), rob1 +0.75 (p=0.069) -> meets goal
- synthetic: mfc_a__clip=0.0,2.0__gamma=5.0  acc 83.96 gap 2.56 rob1 49.53 | vs Gradient Blending: acc -0.08 (p=0.805), gap -1.68 (p=0.001), rob1 -0.32 (p=0.552) | vs baseline: acc -0.16 (p=0.588), gap -3.47 (p=0.000), rob1 +0.67 (p=0.096) -> meets goal
- crema_spec: mfc_a__clip=0.0,2.0__gamma=5.0  acc 49.47 gap 0.94 rob1 36.47 | vs Gradient Blending: acc -1.22 (p=0.030), gap +0.42 (p=0.024), rob1 -3.30 (p=0.000) | vs baseline: acc +0.40 (p=0.393), gap -0.27 (p=0.122), rob1 +0.55 (p=0.021) -> goal NOT met
**mfc_a: goal not met on every dataset**

## mfc_j with weights allowed to reach 0
- crema_av: mfc_j__clip=0.0,2.0__gamma=5.0  acc 50.39 gap 0.91 rob1 45.13 | vs Gradient Blending: acc -1.73 (p=0.050), gap -0.26 (p=0.243), rob1 -2.19 (p=0.027) | vs baseline: acc +0.72 (p=0.146), gap -2.82 (p=0.036), rob1 +1.70 (p=0.002) -> goal NOT met
- ave: mfc_j__clip=0.0,2.0__gamma=5.0  acc 80.85 gap 2.76 rob1 76.72 | vs Gradient Blending: acc -0.20 (p=0.644), gap -2.57 (p=0.311), rob1 +0.15 (p=0.716) | vs baseline: acc +0.97 (p=0.189), gap -23.10 (p=0.009), rob1 +1.84 (p=0.004) -> meets goal
- synthetic: mfc_j__clip=0.0,2.0__gamma=1.0  acc 84.74 gap 4.99 rob1 49.31 | vs Gradient Blending: acc +0.71 (p=0.020), gap +0.75 (p=0.068), rob1 -0.54 (p=0.259) | vs baseline: acc +0.63 (p=0.074), gap -1.05 (p=0.041), rob1 +0.44 (p=0.114) -> meets goal
- crema_spec: mfc_j__clip=0.0,2.0__gamma=5.0  acc 50.07 gap 0.78 rob1 36.44 | vs Gradient Blending: acc -0.62 (p=0.169), gap +0.27 (p=0.048), rob1 -3.32 (p=0.000) | vs baseline: acc +1.00 (p=0.066), gap -0.43 (p=0.058), rob1 +0.52 (p=0.306) -> goal NOT met
**mfc_j: goal not met on every dataset**

## mfc_og with weights allowed to reach 0
- crema_av: mfc_og__alpha=0.3__clip=0.0,2.0__gamma=5.0  acc 49.36 gap 1.33 rob1 43.96 | vs Gradient Blending: acc -2.76 (p=0.018), gap +0.16 (p=0.676), rob1 -3.35 (p=0.005) | vs baseline: acc -0.32 (p=0.478), gap -2.40 (p=0.037), rob1 +0.53 (p=0.257) -> goal NOT met
- ave: mfc_og__alpha=0.8__clip=0.0,2.0__gamma=5.0  acc 82.09 gap 1.20 rob1 79.38 | vs Gradient Blending: acc +1.04 (p=0.040), gap -4.13 (p=0.119), rob1 +2.81 (p=0.000) | vs baseline: acc +2.21 (p=0.007), gap -24.67 (p=0.007), rob1 +4.50 (p=0.000) -> meets goal
- synthetic: no setting passes the selection rule (tune accuracy within 0.5 of baseline) -> goal NOT met
- crema_spec: mfc_og__alpha=0.8__clip=0.0,2.0__gamma=5.0  acc 49.57 gap 0.62 rob1 38.05 | vs Gradient Blending: acc -1.12 (p=0.014), gap +0.10 (p=0.336), rob1 -1.72 (p=0.013) | vs baseline: acc +0.50 (p=0.204), gap -0.59 (p=0.020), rob1 +2.12 (p=0.002) -> goal NOT met
**mfc_og: goal not met on every dataset**

Gradient Blending reference: crema_av acc 52.12 gap 1.17 rob1 47.32; ave acc 81.04 gap 5.33 rob1 76.57; synthetic acc 84.03 gap 4.23 rob1 49.86; crema_spec acc 50.69 gap 0.51 rob1 39.77
