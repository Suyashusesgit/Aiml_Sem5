# Repository Cleanup & Archival Report

*Date: 15 November 2026*
*Rule 4 Compliance: Zero project file deletion. All non-regenerable artifacts moved to `_archive/` with full provenance.*

| Original Path | Size (Bytes) | SHA-256 Hash | Reason for Archiving | Canonical Replacement |
| :--- | :---: | :--- | :--- | :--- |
| `yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e...` | Pretrained deep learning weight file outside CPU-only classical scope | None (downstream YOLO evaluation is preserved under extensions/) |
| `JOURNAL_SUBMISSION` | 50,815,799 | `2ecfd4f208ff38f3...` | Exact duplicate bundle of report/ paper archives and PDFs | report/ (canonical paper source and assets) |
| `results/video_demo_anti_flicker.mp4` | 1,828,012 | `eedea72a5a462a9e...` | Large binary generated video artifact | Regenerable via extensions/generate_video_demo.py |
| `scripts/benchmark_8enhancers_10features.py` | 8,615 | `563f9976be2dfae2...` | Ad-hoc exploratory script from previous audit session | experiments/ (canonical structured decision experiments) |
| `scripts/benchmark_lol_v2_zeroshot.py` | 10,273 | `42ea3c6677f9203f...` | Ad-hoc exploratory script from previous audit session | experiments/ (canonical structured decision experiments) |
| `scripts/train_test_lol_v2.py` | 16,796 | `38ddf352c21b32e9...` | Ad-hoc exploratory script from previous audit session | experiments/ (canonical structured decision experiments) |

## Summary
- Total items archived: 6
- Total archived size: 59,229,291 bytes
|  | 92,298,574 |  | Legacy paper directory replaced by canonical paper/ | paper/ |
