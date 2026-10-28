"""Prepare a copy of the paper's .tex files for pandoc -> docx.

Usage: python docx_prep.py <tex dir> <out dir>
Resolves \\ref/\\eqref from main.aux, numbers display equations with \\tag,
and uses the PNG of the model diagram (Word cannot embed PDF).
"""
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}",
                         (src / "main.aux").read_text()))

FILES = ["intro", "related", "modeling", "method", "dataset", "evaluation",
         "conclusion"]
n_eq = 0


def tag_equations(text: str) -> str:
    # One numbered equation per display line (a gather becomes several).
    def sub(m):
        global n_eq
        body = re.sub(r"\\label\{[^}]*\}", "", m.group(2))
        parts = []
        for line in body.split(r"\\"):
            n_eq += 1
            # \tag is dropped by the docx writer, so write the number in.
            parts.append(f"\\begin{{equation}}{line.rstrip()} \\qquad ({n_eq})\n\\end{{equation}}")
        return "\n".join(parts)

    pat = re.compile(r"\\begin\{(equation|gather)\}(.*?)\\end\{\1\}", re.S)
    return pat.sub(sub, text)


def number_captions(text: str) -> str:
    # The docx writer does not number captions: prefix "Figure n:"/"Table n:".
    def sub(m):
        env, body = m.group(1), m.group(2)
        label = re.search(r"\\label\{([^}]*)\}", body)
        if label:
            kind = "Figure" if env == "figure" else "Table"
            body = body.replace(r"\caption{",
                                rf"\caption{{{kind} {labels[label.group(1)]}: ", 1)
        return rf"\begin{{{env}}}{body}\end{{{env}}}"

    return re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", sub, text,
                  flags=re.S)


for name in FILES:
    t = (src / f"{name}.tex").read_text()
    t = t.replace("plot/model.pdf", "plot/model.png")
    # Tables inside \resizebox are dropped by pandoc.
    t = t.replace("\\resizebox{\\linewidth}{!}{%", "")
    t = t.replace("\\end{tabular}}", "\\end{tabular}")
    t = re.sub(r"\\cmidrule\([a-z]*\)\{[^}]*\}", "", t)
    t = t.replace("$^{*}$", "*")
    # Plain-text ± and ° fit table cells better than inline maths.
    t = re.sub(r"\$\\mathbf\{([\d.]+)\\pm([\d.]+)\}\$", r"\\textbf{\1±\2}", t)
    t = re.sub(r"\$([\d.]+)\\pm([\d.]+)\$", r"\1±\2", t)
    t = t.replace("($^\\circ$C)", "(°C)")
    t = t.replace("height=0.42\\textheight", "width=0.4\\linewidth")
    t = number_captions(t)
    t = tag_equations(t)
    t = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels[m.group(1)]})", t)
    t = re.sub(r"~?\\ref\{([^}]*)\}", lambda m: f"~{labels[m.group(1)]}", t)
    (out / f"{name}.tex").write_text(t)

print("equations:", n_eq)
