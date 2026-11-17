# Project Triathlon: Multi-Year Engineering & Research Roadmap
**Institution**: Gati Shakti Vishwavidyalaya (GSV), Vadodara & SVNIT Surat  
**Team**: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan (B.Tech ECE)  
**Supervisor**: Prof. Kishor P. Upla (Senior Member, IEEE)  
**Document Status**: Active / Canonical Strategic Plan  

---

## 1. Overview & Project Progression

```
+-----------------------------------------------------------------------------------+
| PHASE 0 (Semester 5) · COMPLETED & FROZEN                                         |
| "Per-Image Enhancement Selection: Can Handcrafted Image Statistics Predict       |
|  the Best Enhancer?"                                                              |
| - Pre-committed decision rule (docs/DECISION_RULE.md)                             |
| - Pre-registered experiments E0-E5 settling LR vs. RF (Verdict: TIE -> LogReg)    |
| - 22/22 unit tests, zero leakage (dHash grouping), reproducible CLI suite         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 1 (Semester 6) · COMPLETED & VERIFIED                                       |
| "Edge Feasibility, Conformal Risk-Control, and Cross-Degradation Generalization"  |
| - Split-Conformal Prediction with finite-sample 90%/95% coverage guarantees       |
| - Confidence-gated abstention frontier (tau in [0.0, 0.70])                       |
| - RQ2 PSNR vs. SSIM oracle study (63.3% agreement, rho = 0.8823 rank correlation) |
| - Zero-shot transfer to LOL-v2 Real & Synthetic (+2.01 dB on synthetic)          |
| - Leave-One-Enhancer-Out (LOEO) bank sensitivity & 10D physical feature benchmark |
| - 28/28 unit tests passing, repository integrity audit 100% green                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 2 (Semester 7) · PLANNED SCOPE                                              |
| "Multimodal Foundation, Compound Hazard Cascades & Continuous Metric Optimization"|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 3 (Semester 8) · CAPSTONE SCOPE                                             |
| "Edge Silicon Co-Design, INT8 SIMD Acceleration & Closed-Loop Autonomous Robotics"|
+-----------------------------------------------------------------------------------+
```

---

## 2. Phase 2: Semester 7 Detailed Research & Engineering Plan
**Target Window**: July 2027 – November 2027  
**Theme**: Multimodal Foundation, Compound Hazard Cascades & Continuous Metric Optimization  

### Milestone 2.1: Compound Multi-Hazard Degradation Taxonomy
- **Problem**: Natural low-light imagery rarely presents pure photon starvation; real-world cameras simultaneously encounter atmospheric haze, extreme Poisson-Gaussian sensor noise, and motion/optical blur.
- **Objectives**:
  1. Expand benchmark from pure low-light (LOL-v1/v2) to compound multi-hazard datasets:
     - Compound Low-Light + Haze: Dark-Haze benchmark & RESIDE-Beta.
     - Compound Low-Light + Noise: Smartphone Image Denoising Dataset (SIDD) under low lux.
     - Compound Low-Light + Blur: Real-world night video blur from GoPro/REDS.
  2. Implement a 3-tier diagnostic selector:
     - Tier 1: Dominant hazard classification ($\text{Low-Light} \times \text{Haze} \times \text{Noise} \times \text{Blur}$).
     - Tier 2: Physical severity parameter estimation ($\mu_{\text{lux}}, \sigma_{\text{noise}}, \beta_{\text{haze}}$).
     - Tier 3: Compound operator routing.
- **Success Criteria**: Selector achieves $>80\%$ classification accuracy on dominant hazard identification and reduces compound distortion regret below $1.20$ dB.

### Milestone 2.2: Continuous Predict-Then-Optimize (PTO) & Multi-Metric Frontier
- **Problem**: Discrete multi-class selection discards relative numerical utility between sub-optimal candidates. If Candidate A is 0.05 dB worse than the oracle while Candidate B is 6.0 dB worse, classification penalizes both identically.
- **Objectives**:
  1. Mature continuous multi-head regression architectures predicting expected utility vector $\hat{\mathbf{u}}(x) \in \mathbb{R}^{|\mathcal{A}|}$.
  2. Multi-objective objective formulation:
     $$\max_{a \in \mathcal{A}} \left[ w_{\text{psnr}} \cdot \widehat{\text{PSNR}}_a(x) + w_{\text{ssim}} \cdot \widehat{\text{SSIM}}_a(x) - \lambda \cdot \text{Cost}_a \right]$$
  3. Calibrate continuous conformal intervals $[\hat{u}_{a}^{\text{lo}}, \hat{u}_{a}^{\text{hi}}]$ for each operator.
- **Success Criteria**: PTO selector reduces regret on LOL-v1 to $< 1.35$ dB under scene-grouped CV without hyperparameter leakage.

### Milestone 2.3: Adaptive Operator Cascades & Micro-Pipelines
- **Problem**: Monolithic enhancement operators cannot resolve conflicting physical degradations simultaneously.
- **Objectives**:
  1. Author a DAG-based micro-pipeline generator: allowing dynamic 2-stage compositions:
     $$\text{Input} \longrightarrow \mathcal{O}_1 \longrightarrow \mathcal{O}_2 \longrightarrow \text{Output}$$
     Examples: $\text{Fast-BM3D} \to \text{Retinex}$ or $\text{Optical Invert-Dehaze} \to \text{High-Boost Sharpen}$.
  2. Prune search space using dynamic programming and Pareto latency constraints ($< 25$ ms).
- **Success Criteria**: Two-stage composite cascade outperforms the best single operator by at least $+1.50$ dB on compound low-light/haze scenes.

### Milestone 2.4: Self-Supervised Lightweight Embeddings vs. Physical Statistics
- **Problem**: Do self-supervised edge vision foundation representations capture degradation semantics better than handcrafted physics?
- **Objectives**:
  1. Extract embeddings from tiny edge backbones (MobileNetV4-Small, EdgeNeXt-Micro, FastViT) with $< 1.5$M parameters.
  2. Evaluate feature extraction latency vs. accuracy tradeoff against the 8D/10D physical features.
  3. Design a hybrid representation: $\mathbf{f}_{\text{hybrid}} = [\mathbf{f}_{\text{physical}}^{10D} \,\|\, \mathbf{f}_{\text{latent}}^{16D}]$.
- **Success Criteria**: Strict profiling on CPU; feature extraction must execute within $\le 4.0$ ms while demonstrating statistically significant regret reduction ($p < 0.01$).

---

## 3. Phase 3: Semester 8 Detailed Capstone Roadmap
**Target Window**: January 2028 – May 2028  
**Theme**: Edge Silicon Co-Design, INT8 SIMD Acceleration & Closed-Loop Autonomous Systems  

### Milestone 3.1: Bare-Metal Silicon Profiling & Hardware Matrix
- **Problem**: Desktop CPU benchmarks do not capture thermal throttling, memory bandwidth constraints, and cache hierarchy bottlenecks on embedded systems.
- **Objectives**:
  1. Benchmarking suite on real hardware boards:
     - Board 1: **Raspberry Pi 5** (Quad-core ARM Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X).
     - Board 2: **NVIDIA Jetson Orin Nano** (6-core ARM Cortex-A78AE + 1024-core Ampere GPU, 8 GB).
     - Board 3: **Google Coral Edge TPU** (4 TOPS peak INT8 accelerator via PCIe/USB).
  2. Measure: End-to-end latency (ms), frame rate (FPS), power consumption (Watts via USB power meter), and thermal equilibrium temperature ($^\circ$C).
- **Success Criteria**: End-to-end pipeline operates at $> 60$ FPS at $< 5.0$ Watts power consumption on Raspberry Pi 5.

### Milestone 3.2: INT8 Quantization & SIMD Vectorization
- **Problem**: Python interpreter overhead and float64 matrix operations limit microsecond responsiveness.
- **Objectives**:
  1. Rewrite feature extraction and inference in clean C++20 with zero third-party dependencies (beyond OpenCV core).
  2. Leverage ARM NEON and x86 AVX2 vector intrinsics for patch operations, dark channel min-filter, and Immerkaer noise filter.
  3. Implement post-training INT8 quantization for selector weights with verified numerical tolerance ($|\Delta \text{PSNR}| < 0.01$ dB).
- **Success Criteria**: C++/NEON implementation achieves $> 300$ FPS (latency $< 3.3$ ms) on single-core ARM Cortex-A76.

### Milestone 3.3: Closed-Loop Perception Integration (Robotics & Autonomous Driving)
- **Problem**: Enhancement quality must ultimately serve downstream edge perception tasks (object detection, visual odometry, obstacle avoidance) rather than human visual aesthetics.
- **Objectives**:
  1. Integrate the Triathlon selector into real-time perception pipelines:
     - Object Detection: ORB/Classical Vision / RT-DETR-Tiny on night traffic datasets (BDD100K Night, ExDark).
     - Visual SLAM: ORB-SLAM3 in low-light indoor corridors.
  2. Benchmark Task Utility: mAP@50, tracking loss rate, and feature matching inlier count across raw, fixed, and selected streams.
- **Success Criteria**: Triathlon selector delivers $\ge +10\%$ mAP improvement over raw low-light video and $> 50\%$ reduction in SLAM tracking loss incidents.

### Milestone 3.4: Autonomous Telemetry Swarm & Environmental Adaptation
- **Problem**: Static selectors fail during abrupt environmental transitions (entering tunnels, sudden glare).
- **Objectives**:
  1. Couple image statistics with ambient hardware telemetry: photo-diode lux measurements, camera exposure time, and IMU frame jerk.
  2. Multi-rate control loop: 1 kHz hardware sensor sampling feeding a 60 Hz selector update loop with adaptive temporal hysteresis.
- **Success Criteria**: Zero visual flickering on 1000-frame transition sequences; adaptive selector switches modes in $< 2$ frames upon entering darkness.

---

## 4. Phase-by-Phase Deliverables & Verification Matrix

| Phase | Milestone | Deliverable Artifact | Verification Standard | Target Completion |
| :---: | :---: | :--- | :--- | :---: |
| **Phase 0** | Core ML | `models/selector.joblib`, `results/decision/verdict.json` | 22/22 unit tests green, pre-committed decision rule | **COMPLETED** |
| **Phase 1** | Edge & Risk | `src/conformal.py`, `results/phase1/*.csv`, Figures 12-16 | 28/28 unit tests green, finite coverage $\ge 90\%$ | **COMPLETED** |
| **Phase 2** | Compound | `src/compound_diagnostics.py`, compound hazard benchmark | Classify 4 hazards $> 80\%$, regret $< 1.20$ dB | Month 14 (Sem 7) |
| **Phase 2** | Regressors | `src/pto_multimetric.py`, multi-objective Pareto solver | PTO regret $< 1.35$ dB under Grouped CV | Month 16 (Sem 7) |
| **Phase 2** | Cascades | `src/operator_cascade.py`, 2-stage dynamic pipeline | $+1.50$ dB advantage on compound distortions | Month 18 (Sem 7) |
| **Phase 3** | Embedded | C++ library `triathlon-core/`, ARM NEON intrinsics | $> 300$ FPS on ARM Cortex-A76, INT8 parity | Month 21 (Sem 8) |
| **Phase 3** | Robotics | ROS 2 perception node `triathlon_enhancement_node` | Night BDD100K mAP gain $\ge +10\%$, 60 FPS | Month 23 (Sem 8) |
| **Phase 3** | Capstone | IEEE Transactions Camera-Ready + Live Field Demo | Full bachelor thesis defense & hardware prototype | Month 24 (Sem 8) |

---

## 5. Risk Assessment & Mitigation Strategy

| Risk Factor | Probability | Severity | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Compound Degradation Scarcity** | Medium | High | Construct calibrated synthetic blends of low-light (LOL-v2) + synthetic haze (Koschmieder model) + real sensor noise (SIDD) while strictly documenting synthesis seeds. |
| **Embedded Hardware Stockouts** | Low | Medium | Containerize edge runtime using QEMU ARM64 emulation for CI/CD; reserve physical testbed in GSV ECE Embedded Systems Laboratory. |
| **Intrinsics Numerical Divergence** | Low | High | Enforce unit test parity gates comparing double-precision Python NumPy against INT8 C++ NEON outputs; reject any assembly optimization exceeding $0.05$ dB error. |
| **Temporal Flickering in SLAM** | Medium | Medium | Maintain the Phase 0 exponential moving average (EMA) filter and 5-frame hysteresis buffer (`src/video_pipeline.py`) across all downstream robotics perception nodes. |
