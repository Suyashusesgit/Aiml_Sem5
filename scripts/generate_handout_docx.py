#!/usr/bin/env python3
"""
Generates the 1-Page Executive Summary Handout for Prof. Kishor P. Upla.
Outputs to report/Executive_Summary_Prof_Upla.docx
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

COLOR_PRIMARY = "0F172A"    # Deep Navy
COLOR_SECONDARY = "2563EB"  # Royal Blue
COLOR_ACCENT = "0D9488"     # Deep Teal
COLOR_MUTED = "64748B"      # Slate Gray
COLOR_LIGHT_BG = "F8FAFC"   # Light background
COLOR_BORDER = "CBD5E1"     # Light Border


def set_cell_background(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def build_handout():
    doc = Document()

    # Narrow margins to comfortably fit 1 page
    for section in doc.sections:
        section.top_margin = Inches(0.4)
        section.bottom_margin = Inches(0.4)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    # Header Bar
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_cat = title_p.add_run("EXECUTIVE BRIEFING • CAPSTONE RESEARCH PROJECT • PHASES 0, 1 & 2\n")
    run_cat.font.name = "Arial"
    run_cat.font.size = Pt(8.5)
    run_cat.font.bold = True
    run_cat.font.color.rgb = RGBColor(37, 99, 235)

    run_title = title_p.add_run("Fidelity-Driven Algorithm Selection for Low-Light Image Enhancement")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(15)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    # Metadata Subtitle Table
    meta_table = doc.add_table(rows=1, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    meta_table.columns[0].width = Inches(4.3)
    meta_table.columns[1].width = Inches(3.2)

    c0 = meta_table.cell(0, 0)
    set_cell_background(c0, "F1F5F9")
    set_cell_margins(c0, 60, 60, 100, 100)
    p0 = c0.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(0)
    r = p0.add_run("Student Investigators: ")
    r.font.bold = True
    r.font.size = Pt(8.5)
    p0.add_run("Ananya Kasaudhan, Suyash Srivastav, Logi Vasan\n").font.size = Pt(8.5)
    r2 = p0.add_run("Affiliation: ")
    r2.font.bold = True
    r2.font.size = Pt(8.5)
    p0.add_run("Dept. of ECE, Gati Shakti Vishwavidyalaya (GSV), Vadodara").font.size = Pt(8.5)

    c1 = meta_table.cell(0, 1)
    set_cell_background(c1, "F1F5F9")
    set_cell_margins(c1, 60, 60, 100, 100)
    p1 = c1.paragraphs[0]
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(0)
    r3 = p1.add_run("Supervisor: ")
    r3.font.bold = True
    r3.font.size = Pt(8.5)
    p1.add_run("Prof. Kishor P. Upla, Senior Member IEEE\n").font.size = Pt(8.5)
    r4 = p1.add_run("Affiliation: ")
    r4.font.bold = True
    r4.font.size = Pt(8.5)
    p1.add_run("Dept. of Electronics Eng., SVNIT Surat").font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # Core Problem & Breakthrough Callout
    callout_tbl = doc.add_table(rows=1, cols=1)
    callout_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    callout_tbl.columns[0].width = Inches(7.5)
    c_call = callout_tbl.cell(0, 0)
    set_cell_background(c_call, "EFF6FF")
    set_cell_margins(c_call, 80, 80, 120, 120)
    p_c = c_call.paragraphs[0]
    p_c.paragraph_format.space_before = Pt(0)
    p_c.paragraph_format.space_after = Pt(0)
    r_h = p_c.add_run("The Core Scientific Discovery & Multi-Phase Completion:\n")
    r_h.font.bold = True
    r_h.font.size = Pt(9.5)
    r_h.font.color.rgb = RGBColor(37, 99, 235)
    p_c.add_run(
        "Standard blind quality assessment metrics (NR-IQA such as BRISQUE/NIQE) achieve 0% top-1 accuracy in identifying the "
        "PSNR-optimal enhancer (BlindPSNR, IEEE SPL 2026). Our project solves this by formulating enhancement selection through "
        "decision theory: we extract an 8D physical feature space, deploy an ultra-lightweight Random Forest selector, validate "
        "downstream YOLOv8 object detection (+18.6% recall gain), profile CPU edge deployment via ONNX (84.7 FPS, 54 MB RAM), "
        "and generalize across blur, haze, and noise via a 10D hierarchical cascade (+0.915 dB net regret gain)."
    ).font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # Primary Benchmark Table
    sec_p = doc.add_paragraph()
    sec_p.paragraph_format.space_before = Pt(2)
    sec_p.paragraph_format.space_after = Pt(2)
    r_sec = sec_p.add_run("Primary Benchmark Results (5-Fold GroupKFold CV across 3 Seeds, 15 Total Folds):")
    r_sec.font.bold = True
    r_sec.font.size = Pt(9.5)
    r_sec.font.color.rgb = RGBColor(15, 23, 42)

    res_table = doc.add_table(rows=6, cols=5)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_table.columns[0].width = Inches(2.7)
    res_table.columns[1].width = Inches(1.2)
    res_table.columns[2].width = Inches(1.2)
    res_table.columns[3].width = Inches(1.2)
    res_table.columns[4].width = Inches(1.2)

    headers = ["Model / Selection Strategy", "Accuracy (%)", "Mean Regret (dB)", "Max Regret (dB)", "p-value vs. RF"]
    for c_idx, h in enumerate(headers):
        cell = res_table.cell(0, c_idx)
        cell.text = h
        set_cell_background(cell, COLOR_PRIMARY)
        set_cell_margins(cell, 50, 50, 80, 80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        ("Random Choice (Chance Baseline)", "17.66 ± 4.77%", "6.916 ± 0.511 dB", "18.085 dB", "p = 8.68e-17"),
        ("Best Fixed Method (Static Retinex)", "45.57 ± 3.41%", "2.734 ± 0.118 dB", "17.815 dB", "p = 1.90e-11"),
        ("Hand-Written Domain Rule", "12.16 ± 3.29%", "6.926 ± 0.360 dB", "17.064 dB", "p = 1.52e-17"),
        ("Multinomial Logistic Regression", "61.86 ± 2.53%", "1.421 ± 0.213 dB", "15.032 dB", "p = 2.26e-06"),
        ("Random Forest Selector (Proposed)", "68.18 ± 3.49%", "1.160 ± 0.236 dB", "15.181 dB", "PROPOSED"),
    ]

    for r_idx, row in enumerate(rows_data, start=1):
        is_rf = "Proposed" in row[0]
        bg_col = "DBEAFE" if is_rf else ("FFFFFF" if r_idx % 2 == 1 else "F8FAFC")
        for c_idx, val in enumerate(row):
            cell = res_table.cell(r_idx, c_idx)
            cell.text = val
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 40, 40, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.size = Pt(8.5)
                if is_rf:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(37, 99, 235)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # 4 Key Quantitative Takeaways
    summary_tbl = doc.add_table(rows=1, cols=4)
    summary_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i in range(4):
        summary_tbl.columns[i].width = Inches(1.875)

    stats = [
        ("+22.61% Acc", "Accuracy & Regret", "Beats Retinex by 22.6 pts; cuts regret from 2.73 to 1.16 dB (p=1.9e-11)."),
        ("67.15% Recall", "YOLOv8 Detection", "+18.62% recall gain over Retinex (37 vs 30 boxes detected)."),
        ("84.7 FPS", "ONNX Edge Speed", "11.81 ms single-core CPU latency, 54.4 MB RAM, zero GPU dependency."),
        ("+0.915 dB", "Multi-Degradation", "10D cascade cuts blur regret by 4.70 dB and haze regret by 1.17 dB."),
    ]

    for i, (val, title, desc) in enumerate(stats):
        cell = summary_tbl.cell(0, i)
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, 60, 60, 80, 80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r_val = p.add_run(val + "\n")
        r_val.font.bold = True
        r_val.font.size = Pt(11)
        r_val.font.color.rgb = RGBColor(37, 99, 235)
        r_t = p.add_run(title + "\n")
        r_t.font.bold = True
        r_t.font.size = Pt(8)
        r_t.font.color.rgb = RGBColor(15, 23, 42)
        r_d = p.add_run(desc)
        r_d.font.size = Pt(7.5)
        r_d.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # Deliverables & Next Steps Table
    deliv_p = doc.add_paragraph()
    deliv_p.paragraph_format.space_before = Pt(2)
    deliv_p.paragraph_format.space_after = Pt(2)
    r_del = deliv_p.add_run("Completed Multi-Phase Deliverables Ready for Review:")
    r_del.font.bold = True
    r_del.font.size = Pt(9.5)
    r_del.font.color.rgb = RGBColor(15, 23, 42)

    d_table = doc.add_table(rows=2, cols=2)
    d_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    d_table.columns[0].width = Inches(3.75)
    d_table.columns[1].width = Inches(3.75)

    items = [
        ("IEEE Paper Manuscript & Package", "Overleaf-ready LaTeX package with IEEEtran template, 11 figures, 9 tables, mathematical formalization (report/IEEE_Paper_Overleaf_Package.zip)."),
        ("15-Slide Master Defense Presentation", "16:9 widescreen slide deck with embedded qualitative comparisons, YOLOv8 benchmarks, and ONNX edge profiling (report/Project_Triathlon_Executive_Presentation.pptx)."),
        ("Live Interactive Web App (app.py)", "Streamlit web dashboard with Side-by-Side, Interactive Slider, Keypoints Overlay, Edge Delta Heatmap, and real-time inference."),
        ("Edge Engine & Unit Tested Codebase", "100% passing test suite (9/9 tests), models/rf_selector.onnx runtime engine (11.8 ms), and multi-hazard benchmark."),
    ]

    for idx, (title, desc) in enumerate(items):
        r = idx // 2
        c = idx % 2
        cell = d_table.cell(r, c)
        set_cell_background(cell, "FFFFFF")
        set_cell_margins(cell, 40, 40, 60, 60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.add_run("• ").font.bold = True
        r1 = p.add_run(title + ": ")
        r1.font.bold = True
        r1.font.size = Pt(8.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run(desc)
        r2.font.size = Pt(8)
        r2.font.color.rgb = RGBColor(71, 85, 105)

    out_docx = "report/Executive_Summary_Prof_Upla.docx"
    doc.save(out_docx)
    print(f"Successfully generated 1-Page Executive Summary at: {out_docx} ({os.path.getsize(out_docx) / 1024:.1f} KB)")


if __name__ == "__main__":
    build_handout()

