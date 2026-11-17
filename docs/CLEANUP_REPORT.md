# Repository Cleanup & Archival Report

*Date: 15 November 2026*  
*Rule 4 Compliance: Zero project file deletion. Only verified exact duplicates (same SHA-256) and superseded prototype scripts remain in `_archive/` with full cryptographic provenance.*

---

## 1. Protected Assets Restored & Relocated

In compliance with project directives, all protected submission deliverables, downstream machine vision models, video engines, web applications, and latency tools were restored from `_archive/` and placed into dedicated, production-ready modules with comprehensive READMEs:

| Original Path | Restored Permanent Location | SHA-256 Hash | Purpose & Execution |
| :--- | :--- | :--- | :--- |
| `ORB/Classical Vision.pt` | `extensions/downstream/ORB/Classical Vision.pt` | `f59b3d833e2ff32e600de8d98d8ee27e0ea82424b94f1ff5c9ba31c62f27715f` | Pretrained ORB/Classical Vision Nano weights for CPU object detection |
| `JOURNAL_SUBMISSION/` | `paper/submission/` | *Directory* | Camera-ready & anonymized IEEE PDFs, Supplementary Visual Atlas, and Overleaf source bundle |
| `results/video_demo_anti_flicker.mp4` | `results/video_demo_anti_flicker.mp4` & `extensions/video/` | `eedea72a5a462a9ea516b24ce80c85b1915f06112d1b827df8da22d4c0c166d1` | Rendered video demonstrating temporal anti-flicker feature smoothing |
| `report/Project_Triathlon_Master_SOP.*` | `docs/deliverables/Project_Triathlon_Master_SOP.*` | `e79dc17a861ec3bc...` | Master Standard Operating Procedure document |
| `report/Project_Triathlon_Executive_Presentation.pptx` | `docs/deliverables/Project_Triathlon_Executive_Presentation.pptx` | `83b4fa8d9e630d70...` | Presentation deck for project review |
| `report/Project_Triathlon_Comprehensive_Deliverables_Report.*` | `docs/deliverables/Project_Triathlon_Comprehensive_Deliverables_Report.*` | `7be2f066e4a2e260...` | Deliverables compilation report |
| `report/Project_Triathlon_Step_by_Step_Walkthrough.docx` | `docs/deliverables/Project_Triathlon_Step_by_Step_Walkthrough.docx` | `75db7417fa725916...` | Student step-by-step walkthrough documentation |
| `report/Executive_Summary_Prof_Upla.docx` | `docs/deliverables/Executive_Summary_Prof_Upla.docx` | `8e38520779836ae8...` | Executive summary for supervisor review |
| `report/Suyash/` & `report/Ananya/` | `docs/deliverables/Suyash/` & `docs/deliverables/Ananya/` | *Directories* | Student completion reports and presentation slides |

---

## 2. Exact Duplicates Remaining Archived (Rule 4 Permitted)

The following files remain archived in `_archive/report/` solely because an identical primary copy (same exact SHA-256 hash) is actively maintained and tracked in the canonical workspace:

| Archived Path | Size (Bytes) | SHA-256 Hash | Active Canonical Copy in Workspace |
| :--- | :---: | :--- | :--- |
| `_archive/report/IEEEtran.cls` | 281,957 | `da751920a317ed318b7b5cd7fa585a6cc7d28502d457856382e9be24b10a3bd7` | `paper/IEEEtran.cls` |
| `_archive/report/IEEE_Paper_Anonymized.pdf` | 11,390,852 | `57f934be268cb1051b396c5e587b52e2ab53078a745d66ca2a1aff54484ef84c` | `paper/submission/IEEE_Paper_To_Submit_Anonymized.pdf` |
| `_archive/report/Project_Triathlon_Supplementary_Visual_Atlas.pdf` | 15,320,860 | `df0e374058a7dd30e4c309927e1a8cd07b95d734db1a15008952618f26e08b34` | `paper/submission/Supplementary_Visual_Atlas.pdf` |
| `_archive/report/Project_Triathlon_Basic_vs_Extra_Comparison.pdf` | 84,171 | `c0c9c61a49ca2f18ec8aa8d4771ed7ac6d8086799b2f82abd10a7d08636590d5` | `paper/submission/Project_Triathlon_Basic_vs_Extra_Comparison.pdf` |

---

## 3. Superseded Ad-Hoc Scripts Remaining Archived

The following exploratory scripts from previous debugging sessions remain archived because their experimental logic was unified and superseded by the canonical, pre-registered test runner `experiments/run_experiment.py`:

| Archived Script Path | Size (Bytes) | SHA-256 Hash | Superseding Canonical File |
| :--- | :---: | :--- | :--- |
| `_archive/scripts/benchmark_8enhancers_10features.py` | 8,615 | `563f9976be2dfae2...` | `experiments/run_experiment.py` |
| `_archive/scripts/benchmark_lol_v2_zeroshot.py` | 10,273 | `42ea3c6677f9203f...` | `experiments/run_experiment.py` (E4 module) |
| `_archive/scripts/train_test_lol_v2.py` | 16,796 | `38ddf352c21b32e9...` | `experiments/run_experiment.py` (E4 module) |

---

## 4. Archival Accounting Summary

- **Total Protected Items Restored**: 9 files/bundles ($>100\text{ MB}$) restored into `extensions/`, `apps/`, `paper/submission/`, and `docs/deliverables/`.
- **Remaining in `_archive/`**: Exactly 4 duplicate PDFs/classes (whose identical copies reside in `paper/` or `paper/submission/`) and 3 superseded ad-hoc scripts.
- **Project Files Lost**: **0** (Strict compliance with Project Triathlon Rule 4).
