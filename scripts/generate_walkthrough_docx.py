"""
Script to Generate Comprehensive Step-by-Step Walkthrough Word Document (.docx)
Project Triathlon: Fidelity-Driven Per-Image Enhancement Selection
Authors: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (GSV Vadodara)
Supervisor: Prof. Kishor P. Upla (SVNIT Surat)
"""

import os
import sys

# Add Library site-packages for python-docx
sys.path.insert(0, "/Users/home/Library/Python/3.14/lib/python/site-packages")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUTPUT_DOCX = "report/Project_Triathlon_Step_by_Step_Walkthrough.docx"


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tc_pr.append(tc_mar)


def create_callout_box(doc, title, text, bg_hex="EFF6FF", border_hex="3B82F6"):
    """Creates a styled callout box in Word."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Left border only
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"{title}\n")
    r_title.bold = True
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    r_text = p.add_run(text)
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(6)


def build_document():
    doc = docx.Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Colors
    c_primary = RGBColor(0x1E, 0x3A, 0x8A)      # Deep Navy
    c_secondary = RGBColor(0x25, 0x63, 0xEB)    # Royal Blue
    c_dark = RGBColor(0x0F, 0x17, 0x2A)         # Slate Dark
    c_gray = RGBColor(0x47, 0x55, 0x69)         # Text Muted

    # -------------------------------------------------------------
    # Document Header / Title
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Project Triathlon: End-to-End Walkthrough")
    r_title.bold = True
    r_title.font.size = Pt(24)
    r_title.font.color.rgb = c_primary

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Fidelity-Driven Algorithm Selection for Low-Light Image Restoration\nFrom Input Image Ingestion to the Optimal Restoration Solution")
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = c_secondary

    # Author Metadata Table
    tbl_meta = doc.add_table(rows=2, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r in tbl_meta.rows:
        for c in r.cells:
            set_cell_background(c, "F8FAFC")
            set_cell_margins(c, top=80, bottom=80, left=140, right=140)

    tbl_meta.cell(0, 0).paragraphs[0].add_run("Authors & Student Team:").bold = True
    tbl_meta.cell(0, 0).paragraphs[0].runs[0].font.size = Pt(10)
    p_team = tbl_meta.cell(0, 1).paragraphs[0]
    p_team.add_run("Ananya Kasaudhan, Suyash Srivastav, Logi Vasan\nDept. of Electronics & Communication Engineering (ECE)\nGati Shakti Vishwavidyalaya, Vadodara, Gujarat")
    p_team.runs[0].font.size = Pt(9.5)

    tbl_meta.cell(1, 0).paragraphs[0].add_run("Supervisor:").bold = True
    tbl_meta.cell(1, 0).paragraphs[0].runs[0].font.size = Pt(10)
    p_sup = tbl_meta.cell(1, 1).paragraphs[0]
    p_sup.add_run("Prof. Kishor P. Upla, Senior Member, IEEE\nDept. of Electronics Engineering\nSardar Vallabhbhai National Institute of Technology (SVNIT), Surat, Gujarat")
    p_sup.runs[0].font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------
    # Executive Overview
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Overview & The Problem We Solve", level=1)
    h1.runs[0].font.color.rgb = c_primary

    p = doc.add_paragraph(
        "Digital imaging under adverse environmental conditions—such as nighttime surveillance, autonomous driving, "
        "and consumer mobile photography—suffers from severe visibility loss, low contrast, and sensor photon shot noise. "
        "While dozens of enhancement operators exist (e.g., Gamma remapping, CLAHE, Histogram Equalization, Retinex, and edge-preserving filters), "
        "no single enhancement algorithm dominates across all real-world scenes."
    )
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Conventional vision systems deploy a single fixed filter on all incoming frames, causing catastrophic quality failures "
        "on out-of-distribution frames (e.g., Retinex blowing out contrast or amplifying grain into heavy noise). "
        "Furthermore, recent research (BlindPSNR, IEEE SPL 2026) proved that standard blind No-Reference Quality Assessment (NR-IQA) "
        "metrics (such as BRISQUE and NIQE) fail catastrophically at ranking enhancement algorithms, achieving exactly 0% top-1 accuracy."
    )
    p.paragraph_format.space_after = Pt(6)

    create_callout_box(
        doc,
        "Core Project Innovation: Per-Image Algorithm Selection",
        "Instead of running heavy deep neural networks or trial-and-error filters at runtime, Project Triathlon extracts 7 computationally "
        "cheap physical degradation descriptors directly from the degraded input in milliseconds (< 20 ms). A lightweight machine learning "
        "selector (Random Forest) predicts the optimal filter from a six-method bank, recovering 57.4% of the theoretical Oracle headroom "
        "and slashing decision regret by 1.574 dB (p < 10^-10)."
    )

    # -------------------------------------------------------------
    # Step-by-Step Visual Walkthrough
    # -------------------------------------------------------------
    h1 = doc.add_heading("2. Step-by-Step Pipeline: From Input to Optimal Output", level=1)
    h1.runs[0].font.color.rgb = c_primary

    # STEP 1
    doc.add_heading("Step 1: Input Image Selection & Ingestion", level=2)
    p = doc.add_paragraph(
        "The pipeline begins by ingesting a degraded low-light image x. In this walkthrough, we trace benchmark sample image "
        "1.png from the official LOL evaluation split (eval15). The image is captured in adverse illumination conditions where scene objects "
        "are nearly invisible to human observers and automated feature detectors."
    )
    tbl_img = doc.add_table(rows=4, cols=2)
    tbl_img.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r in tbl_img.rows:
        for c in r.cells:
            set_cell_background(c, "FAFAFA")
            set_cell_margins(c, top=60, bottom=60, left=120, right=120)

    specs = [
        ("Source Dataset", "LOL Benchmark (eval15 split / our485 training set)"),
        ("Original Dimensions", "600 x 400 pixels (3-channel BGR)"),
        ("Dynamic Compression", "Severe underexposure; > 85% of pixels crushed below level 35"),
        ("Ground Truth Status", "Paired clean ground-truth reference available for offline validation"),
    ]
    for idx, (prop, val) in enumerate(specs):
        tbl_img.cell(idx, 0).paragraphs[0].add_run(prop).bold = True
        tbl_img.cell(idx, 1).paragraphs[0].add_run(val)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # STEP 2
    doc.add_heading("Step 2: Standardization & Preprocessing Contract", level=2)
    p = doc.add_paragraph(
        "To prevent resolution bias and computational bottlenecking, the input is standardized according to Chapter 23 of the Handbook:\n"
        "1. Long-Side Rescaling: The image longest dimension is resized to exactly 512 pixels using cv2.INTER_AREA area-averaging interpolation.\n"
        "2. Exact Aspect Ratio: The aspect ratio is strictly preserved (600x400 -> 512x341, scale factor = 0.853).\n"
        "3. Pixel Precision: Loaded into 8-bit uint8 BGR representation [0, 255]."
    )

    # STEP 3
    doc.add_heading("Step 3: Handcrafted Physical Degradation Extraction (< 20 ms)", level=2)
    p = doc.add_paragraph(
        "CRITICAL SCIENTIFIC RULE: All features phi(x) are computed strictly on the degraded input image without any access to "
        "ground truth reference images or candidate enhanced outputs. Seven physically grounded statistics are calculated:"
    )

    tbl_feats = doc.add_table(rows=9, cols=4)
    tbl_feats.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Feature", "Mathematical Formula", "Physical Meaning", "Value (Sample 1.png)"]
    for j, h in enumerate(headers):
        c = tbl_feats.cell(0, j)
        set_cell_background(c, "1E3A8A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    feat_data = [
        ("Brightness (mu)", "Mean of grayscale intensity", "Exposure deficit and scene gloom", "24.64 / 255"),
        ("Contrast (sigma)", "Standard deviation of grayscale", "Tonal dynamic range compression", "23.20"),
        ("Shannon Entropy (H)", "-sum(p * log2(p)) over 256 bins", "Information dispersion / richness", "5.44 bits"),
        ("Sensor Noise (sigma_n)", "1.4826 * median(|Laplacian|) flattest decile", "Photon shot noise in smooth areas", "5.93"),
        ("Dark Channel (D)", "Mean of local 15x15 eroded min_c I_c", "Dark shadow depth & haze estimate", "9.93"),
        ("Edge Mean", "Mean of Sobel gradient magnitude", "Overall scene boundary strength", "13.92"),
        ("Edge p90", "90th percentile of Sobel gradient", "Salient boundary definition", "34.06"),
        ("Colourfulness (C)", "Hasler-Suesstrunk chromatic metric", "Degree of color saturation & wash-out", "18.40"),
    ]

    for i, row in enumerate(feat_data):
        for j, val in enumerate(row):
            c = tbl_feats.cell(i + 1, j)
            set_cell_background(c, "F8FAFC" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # STEP 4
    doc.add_heading("Step 4: Machine Learning Selection (Random Forest Classifier)", level=2)
    p = doc.add_paragraph(
        "The 8-dimensional feature vector is passed to the trained Random Forest Selector (300 decision trees, GroupKFold trained). "
        "The classifier outputs both the predicted winner and posterior class probabilities:"
    )

    tbl_prob = doc.add_table(rows=6, cols=3)
    tbl_prob.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(["Candidate Method", "Model Confidence (%)", "Algorithm Decision"]):
        c = tbl_prob.cell(0, j)
        set_cell_background(c, "1E3A8A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    class_probs = [
        ("Histogram Equalization (hist_eq)", "76.67%", "★ WINNER SELECTED ★"),
        ("Gamma Correction (gamma)", "15.00%", "Rejected (sub-optimal contrast)"),
        ("Single-Scale Retinex (retinex)", "8.33%", "Rejected (risk of color distortion)"),
        ("CLAHE (clahe)", "0.00%", "Rejected"),
        ("Identity (identity)", "0.00%", "Rejected"),
    ]

    for i, row in enumerate(class_probs):
        is_win = (i == 0)
        for j, val in enumerate(row):
            c = tbl_prob.cell(i + 1, j)
            set_cell_background(c, "DCFCE7" if is_win else ("F8FAFC" if i % 2 == 0 else "FFFFFF"))
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if is_win:
                r.bold = True
                r.font.color.rgb = RGBColor(0x15, 0x80, 0x3D)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # STEP 5
    doc.add_heading("Step 5: Execution of the Optimal Enhancement Operator", level=2)
    p = doc.add_paragraph(
        "The pipeline executes the winning operator (Histogram Equalization on LAB Luminance Channel). "
        "The operation spreads the narrow histogram across the full [0, 255] tonal spectrum while preserving color chromaticity "
        "channels (A, B) to prevent hue shifting. Execution time: 122.97 ms."
    )

    # STEP 6
    doc.add_heading("Step 6: Offline Validation & Decision Regret Computation", level=2)
    p = doc.add_paragraph(
        "When ground truth is available (during training and benchmark verification), all 6 candidate methods are scored "
        "against the clean reference image using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM):"
    )

    tbl_scores = doc.add_table(rows=7, cols=5)
    tbl_scores.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(["Enhancer Method", "PSNR (dB)", "SSIM", "Decision Regret (dB)", "Evaluation Result"]):
        c = tbl_scores.cell(0, j)
        set_cell_background(c, "1E3A8A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    method_scores = [
        ("Histogram Equalization", "18.20 dB", "0.736", "0.000 dB", "★ THEORETICAL ORACLE WINNER ★"),
        ("Single-Scale Retinex", "16.89 dB", "0.684", "1.310 dB", "Inferior (loss of edge sharpness)"),
        ("Gamma Correction", "14.12 dB", "0.592", "4.080 dB", "Inferior (remains underexposed)"),
        ("CLAHE", "12.45 dB", "0.521", "5.750 dB", "Inferior (excessive noise gain)"),
        ("Bilateral + Unsharp", "10.22 dB", "0.418", "7.980 dB", "Inferior (insufficient brightening)"),
        ("Identity (Unprocessed)", "7.84 dB", "0.215", "10.360 dB", "Worst (completely dark)"),
    ]

    for i, row in enumerate(method_scores):
        is_win = (i == 0)
        for j, val in enumerate(row):
            c = tbl_scores.cell(i + 1, j)
            set_cell_background(c, "DCFCE7" if is_win else ("F8FAFC" if i % 2 == 0 else "FFFFFF"))
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if is_win:
                r.bold = True
                r.font.color.rgb = RGBColor(0x15, 0x80, 0x3D)

    create_callout_box(
        doc,
        "Result: TOP-1 EXACT MATCH (Zero Decision Regret)",
        "On test sample 1.png, our selector predicted Histogram Equalization as the winning method. "
        "Full-reference oracle evaluation confirms Histogram Equalization is indeed the highest-fidelity enhancer (PSNR: 18.20 dB), "
        "yielding exactly 0.000 dB Decision Regret! A static Retinex baseline would have suffered a 1.310 dB fidelity penalty."
    )

    # -------------------------------------------------------------
    # Visual Demonstration
    # -------------------------------------------------------------
    h1 = doc.add_heading("3. Embedded Visual Demonstrations & Benchmark Figures", level=1)
    h1.runs[0].font.color.rgb = c_primary

    # Embed Demo Comparison Grid
    demo_grid_p = "results/demo_output/enhancement_comparison_grid.png"
    if os.path.exists(demo_grid_p):
        doc.add_heading("Figure A: End-to-End Single-Image Selection & Candidate Comparison", level=2)
        doc.add_picture(demo_grid_p, width=Inches(6.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure A: Visual comparison grid generated on eval15 test sample 1.png. Contrasting degraded input, ground truth reference, the selected enhancer (Histogram Equalization), and all candidate portfolio filters.")
        r_cap.font.size = Pt(9)
        r_cap.font.italic = True
        doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Embed Qualitative Strip
    strip_p = "results/figures/fig5_qualitative_strip.png"
    if os.path.exists(strip_p):
        doc.add_heading("Figure B: Multi-Scene Qualitative Restoration Strip across Diverse Cases", level=2)
        doc.add_picture(strip_p, width=Inches(6.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure B (Paper Fig 5): Qualitative comparison strip across three diverse benchmark scenes. Shows where the static Retinex baseline washes out texture or amplifies noise, while our selector identifies Gamma, Histogram Equalization, or Retinex dynamically.")
        r_cap.font.size = Pt(9)
        r_cap.font.italic = True
        doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Embed Downstream Utility
    downstream_p = "results/figures/fig6_downstream_utility.png"
    if os.path.exists(downstream_p):
        doc.add_heading("Figure C: Downstream Computer Vision Utility (Keypoint Repeatability)", level=2)
        doc.add_picture(downstream_p, width=Inches(6.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure C (Paper Fig 6): Downstream keypoint matching repeatability. Raw low-light images retain only 6.10% of true ground-truth landmarks, whereas our selector restores structural keypoint repeatability to 49.94%.")
        r_cap.font.size = Pt(9)
        r_cap.font.italic = True
        doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # Statistical Benchmark Summary
    # -------------------------------------------------------------
    h1 = doc.add_heading("4. Comprehensive Benchmark Results & Statistical Validation", level=1)
    h1.runs[0].font.color.rgb = c_primary

    p = doc.add_paragraph(
        "The model was rigorously validated across 15 cross-validation folds (5-Fold GroupKFold by physical scene x 3 random seeds) "
        "on the LOL benchmark (485 training pairs):"
    )

    tbl_bench = doc.add_table(rows=7, cols=5)
    tbl_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(["Model / Selection Strategy", "Accuracy Mean (%)", "Mean Regret (dB)", "p-value vs Retinex", "Max Regret (dB)"]):
        c = tbl_bench.cell(0, j)
        set_cell_background(c, "1E3A8A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    bench_data = [
        ("Random Choice (Chance)", "17.66% ± 4.77%", "6.916 ± 0.511 dB", "p = 8.68e-17", "18.085 dB"),
        ("Best Fixed Method (Retinex)", "45.57% ± 3.41%", "2.734 ± 0.118 dB", "Baseline", "17.815 dB"),
        ("Hand-Written Heuristic Rule", "12.16% ± 3.29%", "6.926 ± 0.360 dB", "p = 1.52e-17", "17.064 dB"),
        ("Multinomial Logistic Regression", "61.86% ± 2.53%", "1.421 ± 0.213 dB", "p = 2.26e-06", "15.032 dB"),
        ("Random Forest Selector (Proposed)", "68.18% ± 3.49%", "1.160 ± 0.236 dB", "p = 1.90e-11", "15.181 dB"),
        ("Theoretical Oracle Upper Bound", "100.00% ± 0.00%", "0.000 ± 0.000 dB", "Upper Bound", "0.000 dB"),
    ]

    for i, row in enumerate(bench_data):
        is_prop = (i == 4)
        for j, val in enumerate(row):
            c = tbl_bench.cell(i + 1, j)
            set_cell_background(c, "DCFCE7" if is_prop else ("F8FAFC" if i % 2 == 0 else "FFFFFF"))
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if is_prop:
                r.bold = True
                if j in [1, 2]:
                    r.font.color.rgb = RGBColor(0x15, 0x80, 0x3D)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------
    # How to Run Everything (Quickstart Guide)
    # -------------------------------------------------------------
    h1 = doc.add_heading("5. Reproducibility & CLI Execution Guide", level=1)
    h1.runs[0].font.color.rgb = c_primary

    p = doc.add_paragraph("All steps in this project can be reproduced with single-line commands via the Makefile:")
    
    commands = [
        ("make app", "Launch the live interactive web dashboard (http://localhost:8501) with real-time feature gauges and before/after slider."),
        ("make demo", "Run end-to-end single image inference on any sample image and generate the 6-panel comparison grid."),
        ("make benchmark", "Execute the full 15-run GroupKFold cross-validation pipeline across 3 seeds on /Users/home/Desktop/LOLdataset."),
        ("make downstream", "Run the Phase 1 downstream keypoint repeatability benchmark comparing feature matching against ground truth."),
        ("make test", "Run the automated unit test suite (tests/test_pipeline.py) verifying 6/6 tests."),
        ("make package", "Bundle the camera-ready Overleaf LaTeX zip archive (report/IEEE_Paper_Overleaf_Package.zip)."),
    ]

    tbl_cmd = doc.add_table(rows=len(commands) + 1, cols=2)
    tbl_cmd.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(["Command", "What It Executes"]):
        c = tbl_cmd.cell(0, j)
        set_cell_background(c, "1E3A8A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, (cmd, desc) in enumerate(commands):
        c0 = tbl_cmd.cell(i + 1, 0)
        c1 = tbl_cmd.cell(i + 1, 1)
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF" if i % 2 == 0 else "F8FAFC")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        r0 = c0.paragraphs[0].add_run(cmd)
        r0.bold = True
        r0.font.name = "Courier New"
        r0.font.size = Pt(9.5)
        r1 = c1.paragraphs[0].add_run(desc)
        r1.font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------
    # Footer / Sign-off
    # -------------------------------------------------------------
    p_end = doc.add_paragraph()
    p_end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_end = p_end.add_run(
        "Project Triathlon · Capstone Phase 0 Complete\n"
        "Students: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (ECE, Gati Shakti Vishwavidyalaya, Vadodara)\n"
        "Supervisor: Prof. Kishor P. Upla (ECE, SVNIT Surat)\n"
        "Date: November 2026"
    )
    r_end.font.size = Pt(9.5)
    r_end.font.italic = True
    r_end.font.color.rgb = c_gray

    # Save
    os.makedirs(os.path.dirname(OUTPUT_DOCX), exist_ok=True)
    doc.save(OUTPUT_DOCX)
    print(f"Successfully generated Word Walkthrough Document at: {OUTPUT_DOCX} ({os.path.getsize(OUTPUT_DOCX) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_document()
