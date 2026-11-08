.PHONY: help install benchmark demo app downstream downstream_semantic robustness presentation docx test package clean

help:
	@echo "Project Triathlon: Image Enhancement & Per-Image Selection"
	@echo "Available commands:"
	@echo "  make install    - Install required dependencies"
	@echo "  make benchmark  - Run end-to-end GroupKFold benchmark pipeline (main.py)"
	@echo "  make demo       - Run inference demo on a sample image (src/demo.py)"
	@echo "  make app        - Launch interactive web application dashboard (app.py)"
	@echo "  make downstream - Run Phase 1 downstream ORB keypoint benchmark"
	@echo "  make downstream_semantic - Run SIFT RANSAC & Canny boundary benchmark"
	@echo "  make robustness - Run sensor noise sensitivity sweep (src/robustness_sweep.py)"
	@echo "  make presentation - Generate 12-slide PowerPoint deck and 1-page handout"
	@echo "  make docx       - Generate comprehensive step-by-step Word (.docx) walkthrough"
	@echo "  make package    - Build Overleaf camera-ready LaTeX zip archive"
	@echo "  make test       - Run unit tests (tests/test_pipeline.py)"
	@echo "  make clean      - Remove bytecode, cache, and temporary demo outputs"

install:
	pip install -r requirements.txt

benchmark:
	python3 main.py

demo:
	python3 src/demo.py --input data/lol_dataset/eval15/low/1.png --reference data/lol_dataset/eval15/high/1.png

app:
	python3 app.py --port 8501

downstream:
	python3 src/downstream_detection.py

downstream_semantic:
	python3 src/downstream_semantic_eval.py

robustness:
	python3 src/robustness_sweep.py

presentation:
	python3 scripts/generate_presentation.py
	python3 scripts/generate_handout_docx.py

docx:
	python3 scripts/generate_walkthrough_docx.py

package:
	python3 -c "import os, zipfile; z = zipfile.ZipFile('report/IEEE_Paper_Overleaf_Package.zip', 'w', zipfile.ZIP_DEFLATED); z.write('report/IEEE_Paper_Image_Enhancement_Selection.tex', 'main.tex'); z.write('report/IEEEtran.cls', 'IEEEtran.cls'); [z.write(os.path.join('results/figures', f), f) for f in os.listdir('results/figures') if f.endswith('.png')]; [z.write(os.path.join('results/figures', f), os.path.join('figures', f)) for f in os.listdir('results/figures') if f.endswith('.png')]; z.close(); print('Overleaf Camera-Ready Package ready at report/IEEE_Paper_Overleaf_Package.zip')"

test:
	python3 -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf results/demo_output

