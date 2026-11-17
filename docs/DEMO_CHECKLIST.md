# Project Triathlon · 2-Minute Live Demo Checklist & Viva Guide

**Date**: 17 November 2026  
**Audience**: Evaluation Committee & Prof. Kishor P. Upla (SVNIT Surat)  
**Host URL**: `http://localhost:8501`  
**Command to Launch**: `python3 apps/web/app.py --port 8501`

---

## 1. Two-Minute Live Demonstration Click Path

```mermaid
flowchart LR
    Start([Open http://localhost:8501]) --> S1[Sample 1: HELD-OUT Scene 547]
    S1 --> S2[Sample 2: TRAIN Scene 101]
    S2 --> S3[Sample 3: EXTERNAL-NEAR-DUP Scene 690]
    S3 --> S4[Section 4: Analytics & Occam's Razor]
    S4 --> End([Conclusion & Q&A])
```

### Stage 1: Zero-Delay First Paint & Deterministic Held-Out Baseline (0:00 - 0:40)
1. **Initial Screen Load**:
   - The app boots instantaneously (< 50 ms) displaying **LOL-v1 held-out eval15 Scene 547**.
   - **Point out to Committee**:
     - *Deterministic Selection*: Scene 547 was chosen strictly because its mean brightness ($15.67/255$) is the exact median of the 15 held-out test scenes (documented in the footnote).
     - *Preview Boost*: Notice that the degraded input is clearly visible because the **"Preview Boost (Display Only)"** toggle is active ($\gamma = 0.4$, elevating display luminance from $15.7$ to $\approx 83/255$).
     - *Scientific Integrity*: Emphasize that all 8-D physical feature extractions and ML inference execute strictly on the uncompressed raw pixels ($15.7/255$).
     - *Model Decision*: Model recommends `HIST_EQ` ($14.72\text{ dB}$, regret $6.74\text{ dB}$).
   - **Click Action**: Click **"Interactive Slider"** to demonstrate the side-by-side split wiper across the desk, ink bottle, and stapler.

### Stage 2: In-Sample Training Illustration (0:40 - 1:10)
2. **Click `Scene 101 [TRAIN]` under LOL-v1 In-Sample**:
   - **Point out to Committee**:
     - The badge explicitly states `[TRAIN]`, acknowledging that this sample was in the 485-scene training fold.
     - *Selected Enhancer*: `HIST_EQ` ($14.88\text{ dB}$, PSNR match with Oracle).
     - *Regret Badge*: Shows green `Zero Regret` ($0.000\text{ dB}$ regret, perfect Oracle agreement).
     - *Downstream Keypoints*: Click **"Keypoints Overlay"** to reveal 400 ORB keypoints extracted on CPU in $< 5\text{ ms}$.
     - *Classical Object Detector*: Click **"Classical Detector"** to show 11 MSER/contour bounding boxes discovered on enhanced features.

### Stage 3: Cross-Dataset External Hardware Verification (1:10 - 1:40)
3. **Click `Scene 690 [EXTERNAL-NEAR-DUP]` under LOL-v2 Real**:
   - **Point out to Committee**:
     - The badge displays `[EXTERNAL-NEAR-DUP]`, honestly reflecting our dHash audit finding that 91 of 100 LOL-v2 Real test images share physical scene geometry with LOL-v1 (Hamming distance $\le 4$).
     - *Generalization*: The linear selector demonstrates zero-shot transfer across distinct camera hardware, achieving $+2.02\text{ dB}$ gain over static Retinex.

### Stage 4: Quantitative Headroom, Regret & Model Face-Off (1:40 - 2:00)
4. **Scroll to Section 4: Performance Graphs & Model Selection Analytics**:
   - **KPI Cards**:
     - `44.9%` Headroom Closed across 5-fold grouped cross-validation.
     - `1.50 dB` Mean Decision Regret ($95\%\text{ CI } [1.28, 1.83]\text{ dB}$).
     - `+1.28 dB` Empirical win over the best-fixed single method (Retinex: $2.78\text{ dB}$).
   - **Regret Bar Chart**:
     - Point out the high-contrast bars: Oracle ($0.00\text{ dB}$), Logistic Regression ($1.50\text{ dB}$), Random Forest ($1.76\text{ dB}$), HistGradientBoosting ($2.01\text{ dB}$), Best-Fixed Retinex ($2.78\text{ dB}$), Random Choice ($7.06\text{ dB}$).
     - Click **"🔍 Zoom (Drop Random Choice)"** to drop the uniform random baseline and magnify the competitive $0 - 3\text{ dB}$ frontier.
   - **Click Tab 2 ("⚖️ 2. Model Face-Off")**:
     - Highlight the **Empirical TIE**: Paired difference $95\%\text{ CI } [-0.27, +0.06]\text{ dB}$ spans zero ($p = 0.22$).
     - State the engineering principle: Random Forest failed to satisfy the pre-registered burden of proof ($\Delta \ge 0.05\text{ dB}$). Therefore, by Occam's razor, the parsimonious linear model ($2.4\text{ KB}$, $< 0.05\text{ ms}$) is deployed.

---

## 2. Five Critical Viva Questions Answered by the UI

| Question | Where the UI Answers It | Exact Technical Defense |
| :--- | :--- | :--- |
| **Q1: Why did you deploy Logistic Regression instead of Random Forest?** | Section 4, Tab 2 (*Model Face-Off*) | In 5-fold scene-grouped cross-validation, Logistic Regression achieves $1.50\text{ dB}$ mean regret vs. $1.76\text{ dB}$ for Random Forest. The paired difference 95% CI is $[-0.27, +0.06]\text{ dB}$, confirming a statistical tie. Under our pre-registered engineering protocol, the burden of proof rests on the complex model. Random Forest failed to outperform LR by $\ge 0.05\text{ dB}$, requires $1,458\times$ more memory ($3.5\text{ MB}$ vs. $2.4\text{ KB}$), and is $5\times$ slower. |
| **Q2: Does your cross-validation suffer from data leakage between frames?** | Header & Section 4 Subheader Badges (`scene-grouped CV (dHash)`) | Folds are partitioned by reference image perceptual hash (`dHash` at Hamming distance $\le 10$). An explicit runtime assertion (`assert len(set(groups[tr]).intersection(set(groups[va]))) == 0`) guarantees that images of the same physical scene never appear in both training and validation folds. |
| **Q3: How do you know the selector improves downstream machine vision tasks?** | Mode 4 & Mode 5 Viewers (*Inlier Matching & Classical Detector*) | Enhancement is verified through downstream ORB keypoint matching and RANSAC inlier recovery. On average across the benchmark, model-selected enhancement delivers a $+24.8\%$ increase in geometric inlier count and enables classical region detectors to localize $3\times$ more structural features. |
| **Q4: Why does LOL-v2 Real carry an `[EXTERNAL-NEAR-DUP]` badge?** | Sidebar Presets & KPI Card 4 | Our perceptual hash audit revealed that 91 of the 100 images in the LOL-v2 Real Test set are near-duplicate captures of the same physical scenes in LOL-v1. To maintain scientific honesty, we label them as near-duplicates and report performance both with and without exclusion ($n=9$ strictly disjoint scenes). |
| **Q5: What happens if a user submits a normal daylight or overexposed image?** | Sidebar Checkbox (*OOD Exposure Safeguard*) | The 8-D physical feature extractor detects out-of-distribution photon flux ($\mu > 60$, dark channel $> 25$). When the safeguard is toggled, it suppresses overexposure blowout by clamping the decision to `IDENTITY` or `CLAHE` with a clear explanation banner. |

---

## 3. Known Limitations & Honest Disclosures

1. **Class Skew in Portfolio**: In the LOL-v1 training split ($485$ images), Retinex ($163$ wins) and HistEq ($158$ wins) dominate the Oracle distribution, while Identity achieves only $1$ win ($0.2\%$). The linear selector's L2 regularization prevents class starvation, but minor classes remain under-represented.
2. **Single-Image Context**: The selector operates purely on single-frame static statistics ($8\text{ features}$). Temporal coherence across video streams requires the Phase 2 exponential moving average (EMA) smoother.
3. **Reference-Free Evaluation**: PSNR and SSIM are only available when reference ground-truth images exist. For real-world webcam frames, the selector relies on confidence probabilities and no-reference naturalness proxies.
