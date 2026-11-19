"""Convert AUDIT_REPORT.md to LaTeX and compile to PDF with tectonic.

A small, dependency-free Markdown->LaTeX converter covering the constructs used
in the report: headings, bold/italic, inline code, bullet/numbered lists, block
quotes, pipe tables, and a full-page flowchart figure. Compiles with the local
tectonic binary. Output: AUDIT_REPORT.pdf.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TECTONIC = r"D:\Documents\qwen-agent\i15g3v3rg3\论文C21\_tools\tectonic.exe"
MD = os.path.join(HERE, "AUDIT_REPORT.md")
TEX = os.path.join(HERE, "AUDIT_REPORT.tex")
PDF = os.path.join(HERE, "AUDIT_REPORT.pdf")


UNI = {
    "₀": r"$_0$", "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$", "₄": r"$_4$",
    "₅": r"$_5$", "₆": r"$_6$", "₇": r"$_7$", "₈": r"$_8$", "₉": r"$_9$",
    "⁻": r"$^-$", "¹": r"$^1$", "²": r"$^2$", "³": r"$^3$", "⁴": r"$^4$",
    "⁵": r"$^5$", "⁶": r"$^6$", "⁷": r"$^7$", "⁸": r"$^8$", "⁹": r"$^9$",
    "∂": r"$\partial$",
    "ν": r"$\nu$", "τ": r"$\tau$", "ρ": r"$\rho$", "φ": r"$\varphi$",
    "λ": r"$\lambda$", "ε": r"$\varepsilon$", "π": r"$\pi$", "∞": r"$\infty$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "≠": r"$\neq$",
    "→": r"$\rightarrow$", "↦": r"$\mapsto$", "⇒": r"$\Rightarrow$",
    "∈": r"$\in$", "∉": r"$\notin$", "⊂": r"$\subset$", "⊃": r"$\supset$",
    "×": r"$\times$", "·": r"$\cdot$", "–": "--", "—": "---",
    "‖": r"$\|$", "†": r"$\dagger$", "ᵀ": r"$^{\mathsf T}$",
    "ℚ": r"$\mathbb{Q}$", "ₘ": r"$_m$", "ₐ": r"$_a$", "ₑ": r"$_e$",
    "ₗ": r"$_l$", "ₙ": r"$_n$", "ₚ": r"$_p$", "ₛ": r"$_s$", "ₜ": r"$_t$",
    "ᵢ": r"$_i$", "ⱼ": r"$_j$", "ₖ": r"$_k$", "ₗ": r"$_l$",
    "•": r"$\bullet$", "✓": r"\checkmark", "▶": r"$\blacktriangleright$",
}


def esc(s):
    # 1) map unicode math/subscripts to LaTeX, protecting each fragment
    frags = []
    for u, rep in UNI.items():
        if u in s:
            s = s.replace(u, "\x01%d\x01" % len(frags))
            frags.append(rep)
    # 2) escape the remaining plain text
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    # 3) restore protected LaTeX fragments
    for k, rep in enumerate(frags):
        s = s.replace("\x01%d\x01" % k, rep)
    return s


def inline(s):
    # protect inline code first
    codes = []
    def stash(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)
    s = re.sub(r"`([^`]+)`", stash, s)
    s = esc(s)
    # bold then italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    # restore code
    def unstash(m):
        return r"\texttt{" + esc(codes[int(m.group(1))]) + "}"
    s = re.sub(r"\x00(\d+)\x00", unstash, s)
    return s


def convert(lines):
    out = []
    i = 0
    in_list = False
    list_type = None
    def close_list():
        nonlocal in_list, list_type
        if in_list:
            out.append("\\end{%s}" % list_type)
            in_list = False
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        # table detection
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            close_list()
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = len(header)
            # wide tables (>=3 cols or long cells) use wrapping p-columns
            maxlen = max([len(c) for c in header] + [len(c) for r in rows for c in r])
            if ncol >= 3 or maxlen > 40:
                w = "%.2f\\textwidth" % (0.94 / ncol)
                colspec = "@{}" + ("p{%s}" % w) * ncol + "@{}"
            else:
                colspec = "@{}" + "l" * ncol + "@{}"
            out.append("\\begin{center}\\footnotesize")
            out.append("\\begin{tabular}{%s}" % colspec)
            out.append("\\hline")
            out.append(" & ".join(inline(c) for c in header) + " \\\\")
            out.append("\\hline")
            for r in rows:
                r = (r + [""] * ncol)[:ncol]
                out.append(" & ".join(inline(c) for c in r) + " \\\\")
            out.append("\\hline")
            out.append("\\end{tabular}\\end{center}")
            continue
        if ln.startswith("# "):
            close_list(); out.append("\\section*{%s}" % inline(ln[2:]))
        elif ln.startswith("## "):
            close_list(); out.append("\\subsection*{%s}" % inline(ln[3:]))
        elif ln.startswith("### "):
            close_list(); out.append("\\subsubsection*{%s}" % inline(ln[4:]))
        elif ln.startswith("> "):
            close_list(); out.append("\\begin{quote}\\emph{%s}\\end{quote}" % inline(ln[2:]))
        elif re.match(r"^\s*[-*] ", ln):
            if not in_list or list_type != "itemize":
                close_list(); out.append("\\begin{itemize}"); in_list = True; list_type = "itemize"
            out.append("\\item %s" % inline(re.sub(r"^\s*[-*] ", "", ln)))
        elif re.match(r"^\s*\d+\. ", ln):
            if not in_list or list_type != "enumerate":
                close_list(); out.append("\\begin{enumerate}"); in_list = True; list_type = "enumerate"
            out.append("\\item %s" % inline(re.sub(r"^\s*\d+\. ", "", ln)))
        elif ln.strip() == "---":
            close_list(); out.append("\\vspace{4pt}\\hrule\\vspace{4pt}")
        elif ln.strip() == "":
            close_list(); out.append("")
        else:
            close_list(); out.append(inline(ln))
        i += 1
    close_list()
    return "\n".join(out)


def main():
    lines = open(MD, encoding="utf-8").read().split("\n")
    body = convert(lines)
    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{amssymb}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage{titlesec}
\sloppy
\titleformat{\section}{\Large\bfseries\color{blue!40!black}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{blue!30!black}}{}{0pt}{}
\setlist{nosep,leftmargin=1.4em}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
{\LARGE\bfseries The C21 Paper \& Code Audit Report}\\[6pt]
{\large An honest account of 13 external audits, 6 self-audits, and 8 mining directions}\\[4pt]
{\itshape Project started February 2025 \quad\textbullet\quad Report November 2026 \quad\textbullet\quad Package v24}
\end{center}
\vspace{6pt}
\begin{center}
\includegraphics[width=0.92\textwidth]{audit_journey_flowchart.png}
\end{center}
\newpage
"""
    # strip the first H1 title (already in preamble) to avoid duplication
    body = re.sub(r"^\\section\*\{The C21 Paper.*?\}\n", "", body, count=1)
    tex = preamble + body + "\n\\end{document}\n"
    open(TEX, "w", encoding="utf-8").write(tex)
    r = subprocess.run([TECTONIC, "--keep-logs", TEX], cwd=HERE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("tectonic exit", r.returncode)
    if os.path.exists(PDF):
        import fitz
        d = fitz.open(PDF)
        print("AUDIT_REPORT.pdf pages:", d.pageCount)
    else:
        print(r.stdout.decode("utf-8", "ignore")[-2000:])


if __name__ == "__main__":
    main()
