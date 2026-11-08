# Phase 0 Project Proposal: Fidelity-Driven Image Enhancement Selection

**Project Title**: Fidelity-Driven Algorithm Selection for Low-Light Image Enhancement via Handcrafted Physical Statistics  
**Project Identifier**: Project Triathlon — AIML Capstone Phase 0  
**Authors**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan  
**Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE*  
**Department**: Department of Electronics Engineering, SVNIT Surat  
**Academic Year**: 2026–2027 (5th Semester B.Tech)  
**Date**: November 2026  

---

## 1. Problem Statement & Motivation
Digital images captured under adverse lighting conditions suffer from heavy photon shot noise, contrast loss, and color degradation. While numerous restoration operators exist (Gamma correction, CLAHE, Histogram Equalization, Retinex, edge-preserving bilateral filtering), **no single method is optimal for all images**.

Current deployed systems apply one static algorithm to every frame. At inference time, ground truth reference images are unavailable, preventing direct computation of full-reference metrics (PSNR, SSIM). Furthermore, standard No-Reference Quality Assessment metrics (BRISQUE, NIQE) fail catastrophically at ranking enhancement algorithms (achieving 0% top-1 fidelity selection accuracy per Lyu et al., 2026).

## 2. Proposed Research & Methodology
We formulate per-image enhancement selection as an **algorithm selection problem** (Rice, 1976). Rather than executing expensive neural networks or running all candidate filters, we extract **seven computationally cheap physical degradation descriptors** directly from the degraded input in milliseconds:
1. Brightness ($\mu$): Global mean intensity.
2. Contrast ($\sigma$): Standard deviation of luminance.
3. Shannon Entropy ($H$): Information distribution across 256 bins.
4. Noise ($\sigma_n$): Laplacian Median Absolute Deviation in the flattest decile.
5. Dark Channel Prior ($D$): Mean of $15 \times 15$ eroded local minimum across color channels.
6. Edge Statistics: Mean and 90th percentile of Sobel gradient magnitudes.
7. Colourfulness ($C$): Hasler-Süsstrunk opponent color chromatic metric.

Supervised machine learning classifiers (Random Forest, Logistic Regression, XGBoost) are trained against an offline Oracle defined by full-reference PSNR (with SSIM tie-breaking within $0.05\text{ dB}$).

## 3. Evaluation Framework & Decision Regret
We evaluate the decision quality through **non-negative decibel regret**:
$$r(x, \pi) \triangleq \text{PSNR}_{\text{oracle}}(x) - \text{PSNR}_{\pi(x)}(x) \ge 0\text{ dB}$$
Performance is validated using strict **5-fold GroupKFold cross-validation** grouped by physical scene ID across 3 random seeds (15 folds per model) to prevent scene duplicate leakage.

## 4. Work Breakdown & Team Responsibilities
- **Ananya Kasaudhan**: Dataset acquisition (LOL v1 and v2), scene grouping splits, integrity verification, data audit.
- **Suyash Srivastav**: Enhancement operators bank implementation, PSNR/SSIM cache computation, oracle labeling, tie-breaking logic.
- **Logi Vasan**: Handcrafted physical feature extraction, ML classification pipeline, cross-validation protocols, demo interface, integration.

## 5. Phased Research Roadmap
- **Phase 0 (Current - 5th Sem)**: Prove cheap physical features can predict the PSNR winner and slash regret (+21.85% accuracy, 57% regret reduction).
- **Phase 1 (6th Sem)**: Downstream object detection evaluation (evaluating detection mAP with YOLOv10 / RT-DETR on night frames).
- **Phase 2 (7th Sem)**: Multi-degradation transferability (composite low-light, haze, motion blur).
- **Phase 3 (8th Sem)**: Hybrid portfolios incorporating lightweight neural enhancers (Zero-DCE, SCI) alongside classical filters.

