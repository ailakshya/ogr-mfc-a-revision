# Citation review: 8 reviewer-requested references

Manuscript: `springer_format/overleaf_project/main.tex` (bib: `references.bib`, same folder). No .tex/.bib file was edited.

Sources used: Crossref API (all 8 DOIs), Semantic Scholar API (abstracts), full texts from arXiv (2308.13872, 2005.04166, 2408.01430, 2308.13876) and ECVA open-access PDF (ECCV 2020 paper 123670120). MDPI abstract via Crossref.

**Access summary**
| Key | What I read |
|---|---|
| lan2023pose | Full text (arXiv version) |
| lan2022time | Full text (arXiv version) |
| lan2025sustechgan | Full text (arXiv version) |
| lan2022class | Full text (arXiv version) |
| barra2023multimodal | **Abstract only.** The green-OA PDF (unora.unior.it) is behind a Cloudflare challenge; IEEE Xplore did not render. Method details (fusion type, accuracy) were **not** verified. |
| sanguineti2020leveraging | Full text (ECVA open-access PDF) |
| greco2023feasibility | Abstract (MDPI, via Crossref) |
| mishra2026deepfake | **Nothing beyond title/authors/pages from Crossref.** No abstract on Crossref, Semantic Scholar (not indexed) or Taylor & Francis page (not rendered). Content is **unverified**; any claim beyond the title is inference. |

All citations occur in Related Work (lines 96, 99, 102) plus two more uses of `lan2025sustechgan` in Results (line 432) and Limitations (line 479).

---

## 1. `lan2023pose`

**Verified metadata (Crossref):** G. Lan, Y. Wu, F. Hu, Q. Hao. "Vision-Based Human Pose Estimation via Deep Learning: A Survey." *IEEE Transactions on Human-Machine Systems* 53(1):253–268, Feb 2023. doi:10.1109/THMS.2022.3219242.
**Bib discrepancies:** none.

**What the paper does**
- Survey of deep-learning human pose estimation (HPE) from images and video, covering 2-D (single- and multi-person, top-down/bottom-up) and 3-D methods.
- Summarises datasets, metrics and applications (HCI, sports analysis, tracking).
- Covers CNN-based and transformer-based HPE networks (e.g., TransPose, ViT backbones).
- A bibliometric analysis (Scopus, 2012–2022) identifies four trends: multi-person HPE, 3-D HPE, *efficient* deep-learning HPE, and *multi-modal learning*.

**Current sentence (l.96):**
> Surveys of deep visual representation, for example for human pose estimation\cite{lan2023pose}, and comparisons of vision transformers with CNNs, for example for deepfake detection\cite{mishra2026deepfake}, illustrate how backbone design and efficiency trade off.

**Verdict:** Vague but not wrong. The survey is about HPE, not about "deep visual representation" in general. It does name efficient HPE as a trend, but it is not a study of backbone trade-offs. The joint sentence also assigns a claim ("efficiency trade-off") to `mishra2026deepfake` that I could not verify (see §8).

**Proposed replacement (splits the two citations):**
```latex
In vision-only tasks the visual backbone has moved from CNNs to transformers; a survey of deep human pose estimation\cite{lan2023pose} documents this shift and identifies efficient models and multi-modal learning as open research directions.
```

**Relevance:** tangential.

---

## 2. `lan2022time`

**Verified metadata (Crossref):** G. Lan, J. M. Tomczak, D. M. Roijers, A. E. Eiben. "Time efficiency in optimization with a Bayesian-Evolutionary algorithm." *Swarm and Evolutionary Computation* 69:100970, Mar 2022. doi:10.1016/j.swevo.2021.100970.
**Bib discrepancies:** none (Crossref has lowercase "bayesian"; the bib's `{Bayesian}` is the correct form). `pages={100970}` is an article number; fine for Springer styles.

**What the paper does**
- Points out that BO and EA work is usually compared by *number of evaluations*, which is unfair because the overhead of generating a candidate differs a lot (BO's Gaussian-process cost grows cubically with data; EA overhead is roughly constant).
- Proposes measuring *time efficiency*: expected gain in objective value per unit of computation time, with computation time = evaluation time + overhead time.
- Shows the preferred algorithm can switch after some number of evaluations, and proposes BEA (start with BO, transfer knowledge to an EA, continue with the EA).
- BEA beats BO and EA in time efficiency on benchmark functions with many local optima and on nine robot-learning tasks.

**Current sentence (l.99):**
> Work on Bayesian-evolutionary optimisation\cite{lan2022time} argues that efficiency should be measured in wall-clock time, not only in the number of function evaluations, because surrogate and bookkeeping costs can dominate. We follow that advice and report wall-clock time and FLOPs for every component of every method.

**Verdict:** Essentially accurate. "Bookkeeping" is loose: the paper's point is the *overhead time to generate candidates* (surrogate fitting and acquisition), which maps well onto our monitoring-pass overhead. Minor tightening:

**Proposed replacement:**
```latex
Lan et al.\cite{lan2022time} show that comparing optimisers by the number of function evaluations is misleading when their per-iteration overhead differs, and propose measuring efficiency as objective gain per unit of computation time, split into evaluation and overhead time. Following this, we report wall-clock time and FLOPs separately for the controller arithmetic, the monitoring passes and the method-specific work of every method (Table~\ref{tab:cost}).
```

**Relevance:** moderate (it directly motivates how we report cost).

---

## 3. `lan2025sustechgan`

**Verified metadata (Crossref):** G. Lan, Y. Peng, Q. Hao, C. Xu. "SUSTechGAN: Image Generation for Object Detection in Adverse Conditions of Autonomous Driving." *IEEE Transactions on Intelligent Vehicles* 10(11):4854–4864, Nov 2025. doi:10.1109/TIV.2024.3504566.
**Bib discrepancies:** none.

**What the paper does**
- GAN for generating driving images in adverse conditions, motivated by the long-tailed nature of driving data (adverse-condition data is hard to collect).
- Architecture: dual attention modules (position and channel attention), multi-scale generators, and a new loss with an extra detection term.
- Generates **rain and night** images only; snow and fog are named explicitly as **future work** in the conclusion.
- Adds the generated images to the training set to retrain YOLOv5, which improves detection in rain and night and beats well-known GANs (e.g., CycleGAN-style baselines).

**Current sentences**
- l.102: > Generative approaches go further: SUSTechGAN\cite{lan2025sustechgan} generates images of adverse driving conditions such as fog, rain and night to improve detection. Such data generation and modality balancing address the same failure from opposite directions: augmentation makes the visual encoder robust, whereas balancing ensures that a second modality is available when the visual encoder fails.
- l.432: > Handling such inputs requires detecting or gating the corrupted modality, or training with adverse-condition data\cite{lan2025sustechgan}, which is complementary to balancing.
- l.479: > (v) The pixel-level degradations are synthetic; generative adverse-condition data\cite{lan2025sustechgan} would provide a more realistic test.

**Verdict:** l.102 is **factually wrong**: SUSTechGAN does not generate fog (only rain and night). It should also say it was evaluated by retraining YOLOv5. l.432 is accurate. l.479 is acceptable. Note that SUSTechGAN is for single-modality driving images, so "a more realistic test" means realistic visual corruption, not audio-visual data. (Unrelated: the Limitations list has items in the order (iv), (vi), (v); renumber them.)

**Proposed replacement (l.102):**
```latex
Generative augmentation goes further: SUSTechGAN\cite{lan2025sustechgan} synthesises rain and night driving images with attention-based multi-scale generators, and adding them to the training set improves YOLOv5 detection under those conditions. Augmentation and modality balancing tackle visual degradation from opposite directions: the former makes the visual encoder itself robust, whereas balancing keeps a second modality informative for when the visual encoder fails.
```
Optional tightening for l.479:
```latex
(v) The pixel-level degradations are synthetic; realistic adverse-condition imagery, e.g.\ GAN-generated rain and night scenes\cite{lan2025sustechgan}, would provide a more realistic test.
```

**Relevance:** moderate (robustness to adverse visual conditions; complementary approach).

---

## 4. `lan2022class`

**Verified metadata (Crossref):** G. Lan, Z. Gao, L. Tong, T. Liu. "Class binarization to neuroevolution for multiclass classification." *Neural Computing and Applications* 34(22):19845–19862, 2022 (online 9 Jul 2022, issue Nov 2022). doi:10.1007/s00521-022-07525-6.
**Bib discrepancies:** none.

**What the paper does**
- Applies class binarization (decomposing a multiclass problem into binary problems) to the neuroevolution algorithm NEAT.
- Proposes ECOC-NEAT, which uses Error-Correcting Output Codes, and compares it with One-vs-One and One-vs-All NEAT.
- Evaluates on the Digit, Satellite and Ecoli datasets from **four** aspects: multiclass-classification degradation, accuracy, evolutionary (training) efficiency and robustness.
- Finds that ECOC-NEAT gives high accuracy with low variance, allows a flexible number of binary classifiers, and is the most robust.

**Current sentence (l.99):**
> Neuroevolution studies of multiclass classification\cite{lan2022class} evaluate adaptive learners along three axes, accuracy, evolutionary efficiency and robustness, and we adopt the same three axes.

**Verdict:** **Inaccurate.** The paper uses four aspects, not three (it leaves out multiclass degradation), and "adaptive learners" misdescribes NEAT with class binarization. "Adopt the same three axes" is also an overclaim: our efficiency axis is wall-clock/FLOP cost, not evolutionary efficiency, and we add a balance metric (the OGR gap).

**Proposed replacement:**
```latex
Lan et al.\cite{lan2022class} evaluate class-binarization strategies for neuroevolution (NEAT) not only by accuracy but also by multiclass degradation, evolutionary efficiency and robustness; in the same spirit, we report accuracy, modality balance (OGR gap), training cost and robustness rather than accuracy alone.
```

**Relevance:** tangential.

---

## 5. `barra2023multimodal`

**Verified metadata (Crossref):** P. Barra, Z. Mnasri, D. Greco. "Multimodal Emotion Recognition from Voice and Video Signals." *IEEE EUROCON 2023 – 20th International Conference on Smart Technologies*, pp. 169–174, 6 Jul 2023. doi:10.1109/EUROCON56442.2023.10198928.
**Bib discrepancies:** none.

**What the paper does (from the abstract only; full text not accessed)**
- Proposes a method that detects emotion by combining voice analysis with facial-expression recognition from a video stream.
- Tested on the **IEMOCAP** database.
- Motivated by mental-health assessment, specifically collaborative robots (cobots) combined with ChatGPT to interact with autistic children.
- Results are presented as showing potential for healthcare use.
- *Not verified:* the fusion strategy (early/late), architecture and accuracy numbers.

**Current sentence (l.96):**
> Barra et al.\cite{barra2023multimodal} combine voice and video for emotion recognition, the task of our real-data experiment.

**Verdict:** Accurate in substance but imprecise. The paper has **two** real-data benchmarks; emotion recognition is the CREMA-D task only (AVE is event classification). It is also more informative to name IEMOCAP.

**Proposed replacement:**
```latex
Barra et al.\cite{barra2023multimodal} combine voice analysis and facial-expression recognition for emotion recognition on IEMOCAP, targeting assistive healthcare settings; audio--visual emotion recognition is also the task of our CREMA-D benchmark.
```

**Relevance:** moderate (same task as CREMA-D; the abstract gives no sign that it addresses modality imbalance, so do not imply it does).

---

## 6. `sanguineti2020leveraging`

**Verified metadata (Crossref + ECVA PDF):** V. Sanguineti, P. Morerio, N. Pozzetti, D. Greco, M. Cristani, V. Murino. "Leveraging Acoustic Images for Effective Self-supervised Audio Representation Learning." In *Computer Vision – ECCV 2020*, LNCS, Springer, pp. 119–135, 2020. doi:10.1007/978-3-030-58542-6_8.
**Bib discrepancies:** missing `volume={12367}` (inferred from the ECVA file id 123670120, i.e. LNCS 12367; not shown in the Crossref record) and the editors. Otherwise correct.

**What the paper does**
- Introduces *acoustic images* as a modality: a 128-microphone planar array plus a camera produces multispectral (36×48×512) beamformed images that are aligned with RGB frames in both time and space.
- Releases ACIVW, about 5 hours of outdoor video in 10 classes (drone, train, drill, …) with RGB, acoustic images and monaural audio.
- Self-supervised learning via an audio-visual correspondence pretext task (triplet loss). An acoustic-image stream is trained with RGB and then *distilled* as privileged information into a monaural audio network, so no microphone array is needed at test time.
- The distilled audio and visual features beat features trained without acoustic images on classification and cross-modal retrieval, including on other datasets (DCASE 2018 urban scenes, AVIA).

**Current sentence (l.96):**
> Acoustic imaging extends the audio side with spatial structure, both for self-supervised audio representation learning\cite{sanguineti2020leveraging} and in hand-held acoustic cameras\cite{greco2023feasibility}.

**Verdict:** Accurate but thin. It misses the point that connects to our work: the paper strengthens the *audio* representation using a richer training-time signal.

**Proposed replacement (for the Sanguineti half; see §7 for Greco):**
```latex
Acoustic images, i.e.\ spatially resolved spectra from a microphone array aligned with the camera, have been used as training-time privileged information: Sanguineti et al.\cite{sanguineti2020leveraging} distil them, through a self-supervised audio--visual correspondence task, into monaural audio networks that then outperform audio models trained without them on classification and cross-modal retrieval.
```

**Relevance:** moderate (audio-visual representation learning that strengthens the audio stream; it is not a modality-imbalance method).

---

## 7. `greco2023feasibility`

**Verified metadata (Crossref):** D. Greco. "A Feasibility Study for a Hand-Held Acoustic Imaging Camera." *Applied Sciences* 13(19):11110, 9 Oct 2023 (MDPI). doi:10.3390/app131911110.
**Bib discrepancies:** none (`pages={11110}` is the article number).

**What the paper does (abstract)**
- Studies how to miniaturise acoustic cameras, whose bulk limits adoption.
- Starting point: a 128-microphone, 50×50 cm² beamforming array (the type of sensor used to collect the acoustic images in §6).
- Optimises an irregular (aperiodic) planar array layout with tailored spatial-filter synthesis to concentrate sensitivity and avoid aliasing.
- In simulation, a 32-element array of about 20×20 cm² beats the original in directivity and noise suppression below 4 kHz, a 4× reduction in surface area; hardware validation is ongoing.

**Current sentence:** same l.96 sentence as §6 ("...and in hand-held acoustic cameras\cite{greco2023feasibility}").

**Verdict:** Accurate as far as it goes, but "in hand-held acoustic cameras" overstates it slightly: this is a *feasibility/simulation* study, not a built hand-held device. The link to our work is weak; keep it short and factual.

**Proposed replacement (follows the §6 sentence):**
```latex
Making such sensors portable is an active hardware problem: Greco\cite{greco2023feasibility} shows in simulation that an optimised 32-microphone array of about $20{\times}20$\,cm can match or exceed a 128-microphone, $50{\times}50$\,cm array below 4\,kHz, which would bring spatial audio closer to everyday audio--visual capture.
```

**Relevance:** tangential.

---

## 8. `mishra2026deepfake`

**Verified metadata (Crossref):** S. R. Mishra, H. Mohapatra. "Deepfake image detection using different combinations of vision transformers and CNNs." In *Connecting Intelligence*, CRC Press, pp. 44–50, 17 Feb 2026. doi:10.1201/9781003773504-8 (book chapter).
**Bib discrepancies:** the booktitle subtitle "Trends in Computation and Data Communication" does **not** appear in Crossref, which gives only "Connecting Intelligence"; check it against the publisher page. The editors are missing. Authors, pages and year match.

**What the paper does:** **Could not access** an abstract or full text (not indexed by Semantic Scholar; no Crossref abstract; the Taylor & Francis page did not render). From the title alone: it compares combinations of vision-transformer and CNN architectures for detecting deepfake images. Datasets, results and any efficiency analysis are **unknown**.

**Current sentence (l.96):** same as §1:
> ...and comparisons of vision transformers with CNNs, for example for deepfake detection\cite{mishra2026deepfake}, illustrate how backbone design and efficiency trade off.

**Verdict:** **Unverified claim.** Nothing I could access supports the claim that this chapter studies an *efficiency* trade-off. Limit the sentence to what the title supports, or get the chapter and confirm first.

**Proposed replacement (title-level claim only):**
```latex
Hybrid and comparative studies of vision transformers and CNNs, for example for deepfake image detection\cite{mishra2026deepfake}, show that the choice of visual backbone remains an open design question. We use a frozen ImageNet ResNet-18\cite{he2016deep} as a deliberately modest visual encoder; ...
```
(The rest of l.96 stays as it is.)

**Relevance:** tangential.

---

## Summary

| Key | Relevance | Sentence OK? | Bib issues |
|---|---|---|---|
| lan2023pose | tangential | vague; reword | none |
| lan2022time | moderate | yes (minor tightening) | none |
| lan2025sustechgan | moderate | **no**: "fog" is wrong (rain + night only) | none |
| lan2022class | tangential | **no**: four aspects, not three; "adaptive learners" is wrong | none |
| barra2023multimodal | moderate | mostly; "our real-data experiment" should be CREMA-D | none (full text not read) |
| sanguineti2020leveraging | moderate | OK but thin | add LNCS volume 12367 / editors |
| greco2023feasibility | tangential | slight overstatement (simulation study) | none |
| mishra2026deepfake | tangential | unverifiable "efficiency" claim | subtitle not in Crossref; no editors |
