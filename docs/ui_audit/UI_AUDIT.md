# Project Triathlon · Comprehensive UI Audit & Defect Root-Cause Analysis

**Date**: 17 November 2026  
**Auditor**: Antigravity Assistant & Engineering Review  
**Branch**: `ui-audit` (`git tag pre-ui`)  
**Application Entrypoint**: `apps/web/app.py` (`http://localhost:8501`)  
**Evidence Artifacts**: `docs/ui_audit/team_1_first_load.png`, `docs/ui_audit/team_2_analytics.png`, `docs/ui_audit/before/`

---

## 1. Executive Summary

This audit rigorously reproduces, root-causes, and documents all visual, empirical, architectural, and accessibility defects identified in the Project Triathlon interactive web application. In accordance with live demo requirements (Deadline: 20 Nov 2026), all findings are cataloged with concrete file/line citations and verifiable remediation criteria.

---

## 2. Defect Matrix (Severity, Root Cause, Fix, and Verification)

| ID | Severity | Defect Description | Evidence (Screenshot / Location) | Technical Root Cause (File:Line) | Engineering Fix | Verified By |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | **Blocker** | **First Load Solid Black Image**: Low-light degraded input renders as a pitch-black box without contrast or detail. | `docs/ui_audit/team_1_first_load.png` | `apps/web/app.py:718-725`: `img#imgDegraded` lacks a visible border frame; low-light images (mean pixel intensity ~15-22 / 255, 6-9% of dynamic range) are physically dark and invisible against the dark card theme without preview gamma compensation. | 1. Deterministic default sample set to LOL-v1 held-out eval15 Scene `547` (brightness 15.67/255, exact eval15 median).<br>2. Add a 1px solid border frame (`#cbd5e1`).<br>3. Add a "Preview Boost (Display Only)" toggle (gamma 0.4, default ON when brightness < 40/255, boosting visual display to ~83/255 while keeping ML computations strictly on raw pixels).<br>4. Display true mean brightness beside image. | `tests/test_ui_audit.py:test_r1_default_sample_and_preview` |
| **DEF-02** | **Blocker** | **Broken Image Icon Below Viewer**: Missing image placeholder glyph visible between panels. | `docs/ui_audit/team_1_first_load.png` (center below viewer) | `apps/web/app.py:734`: Hidden image tags (`#matchImg`, `#detectorDeg`, `#diffHeatmap`) initialized with `src=""` or empty strings; browsers render broken-link icon before DOM load. | 1. Initialize all images with transparent 1x1 SVG data URIs (`data:image/svg+xml,...`).<br>2. Enforce CSS suppression: `img:not([src]), img[src=""], img[src="#"] { display: none !important; }`.<br>3. Only reveal image elements when valid base64 payload is populated. | Automated DOM inspection & screenshot verification |
| **DEF-03** | **Major** | **Stuck Modal Backdrop & Overlapping Toasts**: Viewport dimmed with floating `&times;` close icon; overlapping toasts at bottom right. | `docs/ui_audit/team_1_first_load.png` (entire screen) | `apps/web/app.py:990, 1044`: `#figureModal` lacked explicit initial `display: none !important` isolation. Toast manager appended multiple concurrent toasts without clearing existing timeouts, producing overlapping toasts with cryptic text `Loaded v1/79`. | 1. Modal backdrop strictly constrained to `display: none !important` until triggered, with Escape key and backdrop-click handlers.<br>2. Single-toast FIFO manager with 3.5s timeout and human-readable text (e.g., `Loaded LOL-v1 Scene 547 (eval15 held-out)`). | Visual inspection & unit test |
| **DEF-04** | **Blocker** | **Invisible Bars in Analytics Chart**: Random Forest (2.06 dB) and HistGradientBoosting (2.49 dB) bars are invisible, leaving faint white labels on light background. | `docs/ui_audit/team_2_analytics.png` (Section 4) | `apps/web/app.py:562-563`: Bar fills used linear gradients with light stops (`#64748b` to `#94a3b8`) containing white text (`#ffffff`), producing contrast ratios < 1.8:1 against `#f1f5f9` (violating WCAG AA 3:1 graphical / 4.5:1 text standard). | 1. High-contrast solid fills: LR `#16a34a` (green), RF `#475569` (slate), HGB `#64748b` (medium slate), Retinex `#d97706` (amber), Random `#334155` (dark charcoal), Oracle `#0284c7` (sky blue).<br>2. Value labels placed outside bars in dark text (`var(--text)`).<br>3. Add "Zoom (drop Random Choice)" toggle to expand 0-3 dB range. | `tests/test_ui_audit.py:test_r2_chart_contrast` |
| **DEF-05** | **Major** | **Hard-Coded Headline Numbers & Unsupported P-Values**: Headline claims `42.8% HEADROOM CLOSED (P < 10^-10)`, `+1.20 dB LOL-V2 REAL HARDWARE GAIN`, badge `GroupKFold Verified` hardcoded in markup. | `apps/web/app.py:816-830` | `apps/web/app.py` contained literal hardcoded strings from legacy unverified runs without dynamic data linkage. `P < 10^-10` was statistically unsubstantiated. | 1. Implement `apps/web/facts.py` to dynamically load all numbers from `results/<run>` and `models/selector.json`.<br>2. Replace `P < 10^-10` with verified 95% bootstrap CI `[1.28, 1.83] dB`.<br>3. Replace `GroupKFold Verified` with `scene-grouped CV (dHash)`.<br>4. Gate LOL-v2 card with sample size $n=9$, audit status (91% near-duplicates excluded), and 95% CI. | `tests/test_ui_audit.py:test_r5_facts_parity` |
| **DEF-06** | **Major** | **Portfolio Parity Violation (8 vs 6 Enhancers)**: Header claims "All 8 Multi-Hazard Enhancers" without separating Phase 0 frozen 6-enhancer bank from Phase 2 extensions. | `docs/ui_audit/team_1_first_load.png` | `apps/web/app.py:780`: Mixed Phase 0 canonical methods with multi-degradation extensions (`high_boost_sharpen`, `dcp_dehaze`). | 1. Split portfolio into primary "Evaluated Enhancers (Phase 0 Canonical)" (6 methods: Identity, Gamma, CLAHE, HistEq, Retinex, Bilateral+Unsharp).<br>2. Highlight `★ SELECTED` (blue) and `★ ORACLE` (gold).<br>3. Place multi-degradation extensions in separate secondary row labeled *"Phase 2 Extension (Preview only, not in reported benchmark metric)"*. | `tests/test_ui_audit.py:test_r7_bank_parity` |
| **DEF-07** | **Minor** | **Safeguard Active by Default & Confusing Preset Labels**: "OOD Exposure Safeguard" checked on startup; sample buttons spoil enhancer choice (`102 (Gamma)`). | `docs/ui_audit/team_1_first_load.png` (sidebar) | `apps/web/app.py:633`: Checkbox hardcoded `checked`. Preset sample names included winner names, compromising blind demonstration. | 1. Safeguard OFF by default, labeled *"OOD Exposure Safeguard (Demo-only heuristic)"*.<br>2. Dynamic badge alerting when active: *"Selector said X, safeguard changed to Y"*.<br>3. Provenance badging on all sample buttons: `[TRAIN]`, `[HELD-OUT]`, `[EXTERNAL]`, `[EXTERNAL-NEAR-DUP]`.<br>4. Added seeded `🎲 Random Curated Sample` button. | Browser manual inspection |
| **DEF-08** | **Minor** | **Mobile Responsiveness Defect (390x844)**: KPI cards and buttons overflow horizontally on mobile screens. | `docs/ui_audit/before/first_load_390x844.png` | `apps/web/app.py:550`: Media query threshold at 800px; no layout rules for <= 480px viewports. | 1. Add `@media (max-width: 480px)` styles.<br>2. Single-column KPI cards and button flex-wrapping with zero horizontal overflow. | Headless Chrome 390x844 screenshot |
| **DEF-09** | **Minor** | **Heavy Synchronous Computation on Initial Load**: ORB matching, RANSAC, and classical detection ran on every request, delaying initial paint. | `apps/web/app.py:260-330` | Non-visual downstream tasks executed synchronously in image processing loop. | 1. Cached default payload for Scene 547 (`DEFAULT_PAYLOAD_CACHE`) for instantaneous (< 50ms) initial paint.<br>2. Downstream tasks gated behind view selectors. | Performance timing audit (< 50ms first paint) |
| **DEF-10** | **Minor** | **Missing File Crash Vulnerability**: Missing CSV or model crashed server with HTTP 500. | Code review of `apps/web/app.py` | Unhandled file IO in request handlers. | Wrap all file loads in `facts.py` and `app.py` with safe defaults, displaying actionable troubleshooting instructions (e.g. `Run python3 scripts/train_deploy_model.py`) instead of stack traces. | Unit test |

---

## 3. Root-Cause Analysis of Primary Reported Defects

### Finding (a): The "Black Image" Phenomenon
- **Reproduction**: When loading LOL-v1 eval15 Scene `79` (or `547`), the degraded low-light input has a mean pixel intensity of $15.6 - 22.2$ out of $255$ ($6.1\% - 8.7\%$ of full luminance scale). In standard sRGB color space on typical uncalibrated LCD displays, luminance values below $25/255$ are indistinguishable from true black without gamma compensation.
- **Verification of Processing Integrity**: We verified that `cv2.imread` correctly decodes the 3-channel 8-bit unsigned integer array (`dtype=uint8`, values in $[0, 255]$). There was **no uint8/255 double division** and **no float-to-int truncation bug**.
- **The Core Issue**: It was purely a **presentation failure**:
  1. The image had no enclosing 1px light border frame, merging seamlessly into dark container margins.
  2. The UI offered no display-only preview gamma boost.
  3. The app did not inform the viewer of the true physical brightness statistic.
- **The Solution**: 
  - Switch the deterministic default sample to **held-out eval15 Scene 547** (the exact median brightness of eval15).
  - Add a 1px border (`#cbd5e1`).
  - Introduce a **"Preview Boost (Display Only)"** toggle applying a gamma curve of $\gamma = 0.4$ ($I' = 255 \cdot (I/255)^{0.4}$) that elevates the visual presentation from $15.7/255$ to $\approx 83/255$ while keeping ML inference strictly on the uncompressed raw pixels.

### Finding (g): The "Lost Data" Phenomenon in Section 4 Bar Charts
- **Reproduction**: In the Average Decision Regret chart, the bars for Random Forest ($2.06\text{ dB}$) and HistGradientBoosting ($2.49\text{ dB}$) appeared completely invisible.
- **The Core Issue**: It was a **CSS styling failure**, not missing data. The labels were positioned at the exact pixel geometry corresponding to $2.06\text{ dB}$ and $2.49\text{ dB}$ on the $7.03\text{ dB}$ scale. However:
  1. The bar fill classes `.chart-bar-fill.rf` and `.chart-bar-fill.hgb` utilized subtle light-gray/slate linear gradients that blended into the `#f1f5f9` container background.
  2. The text label inside the bar was styled in pure white (`#ffffff`). White text on a light gray fill resulted in a contrast ratio below $1.8:1$, rendering both the bar and label virtually invisible to the user.
- **The Solution**:
  - Implement high-contrast solid backgrounds with verified $\ge 3:1$ contrast against the track.
  - Position regret value labels with dark text (`#0f172a`, contrast $> 7:1$) on the header line.
  - Add a "Zoom (drop Random Choice)" toggle that expands the scale across the competitive $0 - 3\text{ dB}$ domain.
