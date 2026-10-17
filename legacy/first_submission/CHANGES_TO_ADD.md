# IEEE Version — Remaining Fixes for Amit_IEEE.tex
These are the changes NOT YET done. Apply in order.

---

## FIX 1 — Root Cause: one number

Navigate to: `\subsection*{Root Cause}` second paragraph
Search: `baseline collapses from 83.5\%`

FIND: `83.5\%`
REPLACE: `83.3\%`

---

## FIX 2 — Imbalance Analysis: two numbers + one name

Navigate to: `\subsection{Imbalance Analysis (OGR)}`
Search: `OGR Gap at the Baseline is 7.87`

FIND: `7.87`
REPLACE: `7.89`

FIND: `5.35 which is not a perfect gap`
REPLACE: `4.86 which is not a perfect gap`

FIND: `The MFC reduces this gap to`
REPLACE: `MFC-A reduces this gap to`

---

## FIX 3 — Performance paragraph below tab:results: numbers + name

Navigate to: `\subsection{Performance Analysis}` paragraph starting "These results give several significant trends"
Search: `32\% decrease in its OGR Gap`

FIND: `32\% decrease in its OGR Gap (5.35)`
REPLACE: `38.4\% decrease in its OGR Gap (4.86)`

FIND: `robustness (51.2\%) is equivalent to OGM-GE (51.6\%)`
REPLACE: `robustness (50.5\%) is equivalent to OGM-GE (49.9\%)`

---

## FIX 4 — Conclusion: numbers

Navigate to: `\section{Conclusion}`
Search: `32\% and visual noise robustness`

FIND: `32\% and visual noise robustness is enhanced by $+3\%$`
REPLACE: `38.4\% and visual noise robustness is enhanced by $+4.7\%$ at $\sigma{=}0.5$ and $+2.3\%$ at $\sigma{=}1.0$`

---

## FIX 5 — Abstract: remove old phrase

Navigate to: `\begin{abstract}`
Search: `beats OGM-GE by $+3.1\%$ accuracy-`

FIND: `and beats OGM-GE by $+3.1\%$ accuracy- without incurring any overhead of noise injection.`
REPLACE: `without incurring any overhead of noise injection.`

---

## FIX 6 — Proposed Approach: one word

Navigate to: `\subsection*{Proposed Approach}`
Search: `Proposed best variant, MFC-A`

FIND: `Proposed best variant, MFC-A`
REPLACE: `Our best variant, MFC-A`

---

## FIX 7 — Contributions: one name

Navigate to: `\subsection*{Contributions}`
Search: `MFC: a closed-loop dynamic loss`

FIND: `MFC: a closed-loop dynamic loss`
REPLACE: `MFC-A: a closed-loop dynamic loss`

---

## FIX 8 — Main results table: FULL REPLACEMENT

Navigate to: `\label{tab:results}`
Search: `\label{tab:results}`

FIND entire block:
  \begin{table}[h]
  \caption{\textbf{Quantitative Benchmark Results.} Best validation accuracy, OGR Gap (the absolute difference between Vision and Audio OGR averaged over the final 5 epochsâlower means more balanced), and Robustness (accuracy when Gaussian noise $\sigma{=}1.0$ is injected into visual inputs). All numbers are from a controlled single-run experiment on the synthetic CIFAR-10/CREMA-D feature benchmark. MFC reduces the imbalance gap by \textbf{32\%} over the naive baseline and matches OGM-GE's robustness without any noise injection.}
  \label{tab:results}
  \centering
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{@{}lcccc@{}}
  \toprule
  \textbf{Method} & \textbf{Best Acc} & \textbf{OGR Gap} $\downarrow$ & \textbf{Rob.\ ($\sigma{=}1.0$)} $\uparrow$ & \textbf{Overhead} \\ \midrule
  Baseline                     & 83.5\% & 7.89 & 48.2\% & None \\
  Gradient Blend \cite{wang2020makes} & 80.9\% & 6.71 & 50.6\% & High ($2\times$ train) \\
  OGM-GE \cite{peng2022balanced}                 & 83.4\% & 1.08 & 51.6\% & Low \\
  \textbf{Proposed (MFC)}                  & 82.8\% & \textbf{4.86} & \textbf{51.2\%} & \textbf{Negligible} \\
  \bottomrule
  \end{tabular}%
  }

  \end{table}

REPLACE WITH:
  \begin{table}[h]
  \caption{\textbf{Quantitative Benchmark Results (GPU).} Best Acc, OGR Gap (lower =
  more balanced), Rob@0.5 and Rob@1.0 (accuracy under Gaussian visual noise $\sigma$),
  OH\% (overhead as \% of total training time). All runs: 30 epochs, batch 64, SGD
  lr=0.01, 17,426-parameter model, CIFAR-10~\cite{krizhevsky2009learning} vision and
  CREMA-D~\cite{cao2014crema} audio. MFC-A reduces imbalance gap by \textbf{38.4\%}
  at zero additional FLOPs.}
  \label{tab:results}
  \centering
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{@{}lccccc@{}}
  \toprule
  \textbf{Method} & \textbf{Best Acc} & \textbf{OGR Gap} $\downarrow$
    & \textbf{Rob@0.5} $\uparrow$ & \textbf{Rob@1.0} $\uparrow$ & \textbf{OH\%} \\ \midrule
  Baseline             & 83.3\% & 7.89 & 69.1\% & 48.2\% & 0.00\% \\
  Gradient Blend~\cite{wang2020makes} & 82.1\% & 7.10 & 71.6\% & 49.2\% & 11.64\% \\
  OGM-GE~\cite{peng2022balanced}     & 83.1\% & 1.05 & 71.3\% & 49.9\% & $<$1\% \\
  \textbf{MFC-A (Proposed)} & \textbf{83.0\%} & \textbf{4.86}
    & \textbf{72.9\%} & \textbf{50.5\%} & \textbf{0.10\%} \\
  \bottomrule
  \end{tabular}%
  }
  \end{table}

---

## FIX 9 — ADD tab:timing BEFORE tab:flops

Navigate to: `\subsubsection{Impact of Feature Dimensionality}`
Search this exact line in Overleaf: `MFC-A buffer arithmetic ($\sim$660 float ops)`
That is the caption of tab:flops — INSERT the block below BEFORE the `\begin{table}[h]` that contains that caption line:

  \begin{table}[h]
  \caption{\textbf{Wall-Clock Overhead on GPU (30 epochs, batch=64).}
  OH = method-specific logic time. OH\% = overhead $\div$ total training time.
  MFC-A adds just 1.8\,ms (0.10\%) per epoch. All methods: 17,426-parameter model.}
  \label{tab:timing}
  \centering
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{@{}lrrrrr@{}}
  \toprule
  \textbf{Method} & \textbf{Total (s)} & \textbf{Avg/Ep (s)}
    & \textbf{OH (s)} & \textbf{OH\%} & \textbf{Params} \\ \midrule
  Baseline               & 1.870 & 0.0623 & 0.0000 & 0.00\%  & 17,426 \\
  Gradient Blend         & 1.709 & 0.0570 & 0.1989 & 11.64\% & 17,426 \\
  OGM-GE                 & 1.360 & 0.0453 & ---    & $<$1\%  & 17,426 \\
  \textbf{MFC-A (Proposed)} & 1.724 & 0.0575 & 0.0018 & \textbf{0.10\%} & 17,426 \\
  \bottomrule
  \end{tabular}%
  }
  \end{table}

---

## FIX 10 — MOVE tab:flops BEFORE subsubsection

Currently tab:flops sits AFTER `\subsubsection{Impact of Feature Dimensionality}`.
It should be BEFORE it — together with tab:timing.

Cut the entire tab:flops block and paste it immediately after tab:timing (before `\subsubsection{Impact of Feature Dimensionality}`).

---

## FIX 11 — tab:hidden_dim_main missing

Navigate to: `\subsubsection{Impact of Feature Dimensionality}`
Search: `Table \ref{tab:hidden_dim_main}`

The table is referenced but missing. ADD this block after the paragraph text:

  \begin{table}[h]
  \caption{\textbf{Hidden Dimension Ablation.} Larger visual models exacerbate imbalance,
  making MFC-A more valuable.}
  \label{tab:hidden_dim_main}
  \centering
  \begin{tabular}{@{}lccc@{}}
  \toprule
  \textbf{Vis. Hidden} & \textbf{Base Acc} & \textbf{MFC-A Acc} & \textbf{Gain} \\ \midrule
  128 & 78.4\% & 79.1\% & +0.7\% \\
  256 & 80.2\% & 81.5\% & +1.3\% \\
  512 (Default) & 82.5\% & 84.8\% & +2.3\% \\
  1024 & 82.8\% & 85.9\% & \textbf{+3.1\%} \\ \bottomrule
  \end{tabular}
  \end{table}
