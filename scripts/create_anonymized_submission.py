"""
Creates an Anonymized IEEE Transactions Manuscript for Double-Blind Peer Review.
Generates:
    - report/IEEE_Paper_Anonymized.tex
    - report/IEEE_Paper_Anonymized.pdf
    - report/IEEE_Transactions_Anonymized_Submission_Package.zip
"""

import os
import re
import subprocess
import zipfile

def create_anonymized_paper():
    tex_path = "report/IEEE_Paper_Image_Enhancement_Selection.tex"
    with open(tex_path, "r") as f:
        content = f.read()

    # 1. Anonymize Author Block
    old_author_pattern = r"\\author\{.*?\n\\thanks\{.*?\n\\thanks\{.*?\n\\thanks\{.*?\n\}\}"
    new_author = r"""\author{Anonymous Authors for Double-Blind Peer Review%
\thanks{Manuscript submitted to IEEE Transactions on Image Processing (TIP) for peer review. Source code, models, and supplementary materials are available in an anonymized repository.}%
}"""

    anonymized_content = re.sub(
        r"\\author\{.*?\\thanks\{.*?\\thanks\{.*?\\thanks\{.*?\}\}",
        new_author,
        content,
        flags=re.DOTALL
    )

    # 2. Anonymize Institutional References
    anonymized_content = anonymized_content.replace("Gati Shakti Vishwavidyalaya", "Anonymous University")
    anonymized_content = anonymized_content.replace("SVNIT Surat", "Anonymous Research Institution")
    anonymized_content = anonymized_content.replace("Department of Electronics and Communication Engineering", "Department of Computer Science and Engineering")
    anonymized_content = anonymized_content.replace("Ananya Kasaudhan", "Author 1")
    anonymized_content = anonymized_content.replace("Suyash Srivastav", "Author 2")
    anonymized_content = anonymized_content.replace("Logi Vasan", "Author 3")
    anonymized_content = anonymized_content.replace("Prof. Kishor P. Upla", "Senior Faculty Supervisor")

    # 3. Anonymize GitHub URL
    anonymized_content = anonymized_content.replace(
        "https://github.com/Suyashusesgit/Aiml_Sem5",
        "https://anonymous.4open.science/r/Project-Triathlon-Blind"
    )

    # 4. Remove Acknowledgments Section if present
    anonymized_content = re.sub(
        r"\\section\*\{Acknowledgment\}.*?\\end\{document\}",
        r"\\section*{Acknowledgment}\nThe authors acknowledge the compute facilities provided by the anonymous institution.\n\n\\end{document}",
        anonymized_content,
        flags=re.DOTALL
    )

    out_tex = "report/IEEE_Paper_Anonymized.tex"
    with open(out_tex, "w") as f:
        f.write(anonymized_content)
    print(f"Wrote anonymized LaTeX source to {out_tex}")

    # Compile with Tectonic
    print("Compiling anonymized paper with Tectonic...")
    cmd = ["tectonic", out_tex]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully compiled report/IEEE_Paper_Anonymized.pdf ({os.path.getsize('report/IEEE_Paper_Anonymized.pdf') / (1024*1024):.2f} MB)")
    else:
        print("Tectonic compilation error:", res.stderr[:500])

    # Package into clean Overleaf/IEEE Submission ZIP
    zip_path = "report/IEEE_Transactions_Anonymized_Submission_Package.zip"
    figure_names = [
        "fig1_oracle_distribution.png",
        "fig2_confusion_matrix.png",
        "fig3_regret_cdf.png",
        "fig4_feature_importances.png",
        "fig5_qualitative_strip.png",
        "fig6_downstream_utility.png",
        "fig7_downstream_geometric_eval.png",
        "fig8_robustness_noise_sweep.png",
        "fig9_edge_profiling.png",
        "fig10_multi_degradation_cascade.png",
        "fig11_downstream_yolo_detection.png",
        "IEEEtran.cls",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_tex, arcname="IEEE_Paper_Anonymized.tex")
        for fname in figure_names:
            found = False
            for parent in ["report", "results/figures", "."]:
                fpath = os.path.join(parent, fname)
                if os.path.exists(fpath):
                    z.write(fpath, arcname=fname)
                    found = True
                    break
            if not found:
                print(f"Warning: could not find {fname}")
    print(f"Created submission bundle: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_anonymized_paper()
