"""
build_codedoc.py -- generate the code documentation LaTeX from the verification source.

The documentation is generated, not hand-written, so that every listing is byte-identical
to the program that produced the numbers.  Each module contributes:
  * a prose paragraph explaining what it does and why,
  * its figure,
  * a listing in which every source line carries its own right-hand comment.

Run:  python build_codedoc.py
Out:  ../03_code_documentation/codedoc.tex
"""
import os                                                          # Paths.
import re                                                          # Splitting source lines into code and comment.
import json                                                        # Reading the measured results for the prose.

HERE = os.path.dirname(os.path.abspath(__file__))                  # This directory.
SRC = os.path.join(HERE, "verify_prd.py")                          # The verification program.
OUT = os.path.abspath(os.path.join(HERE, "..", "03_code_documentation", "codedoc.tex"))  # Target.
RES = os.path.abspath(os.path.join(HERE, "..", "06_results", "results.json"))  # Measured numbers.

# Prose for each module: what it does, and why it is built that way.
MODULES = [
    ("0", "Infrastructure", "fig0_pipeline",
     "Imports, output paths, the fixed seed and the publication palette. This module also draws the "
     "pipeline diagram: one script in, nine modules through, three documents out. The seed is fixed "
     "so that the census, the exact ranks and the assertion outcomes reproduce bit for bit; the "
     "palette assigns one colour to each semantic role and reuses it in every figure, so the twelve "
     "figures (nine wide plus single-column renditions) read as one set."),
    ("1", "Group theory and the exact census", "fig1_census",
     "The only place where physics enters. A table of Clebsch--Gordan coefficients, the coefficient "
     "matrix with one column per electroweak bidoublet, its rank in exact rational arithmetic, and "
     "the enumeration of all eighteen Higgs contents. Four exclusion criteria are applied: unfed "
     "sectors, fixed proportionality, transpose degeneracy, and masslessness when every Yukawa matrix "
     "is antisymmetric. No floating tolerance enters the classification."),
    ("2", "The bidoublet parameterisation", "fig2_bidoublet",
     "Maps a real parameter vector to the three mass matrices. The crucial routine is the SO(10) "
     "builder: the 120 carries two bidoublets sharing one antisymmetric Yukawa matrix, and their vevs "
     "enter the down and lepton sectors as v1+v15 and v1-3v15 respectively. A switch selects the "
     "correct parameterisation or the collapsed one used in an earlier draft, so the two can be "
     "compared on identical random draws."),
    ("3", "Observables and Jacobian ranks", "fig3_ranks",
     "The observable vector is nine singular values, three CKM moduli and the Jarlskog invariant. "
     "Singular values are used rather than moduli of eigenvalues, which differ for non-Hermitian "
     "matrices. The Jacobian is a central difference; its rank is read from the singular spectrum at "
     "a relative threshold. Points with degenerate spectra are rejected because the sorted-mass map "
     "is not differentiable there."),
    ("4", "The withdrawn identity and bound", "fig4_withdrawn",
     "Compares the correct and collapsed parameterisations of 120+126 on four thousand random points "
     "each. Three quantities are recorded: the failure of the identity Me=-2Md-Md^T, the accuracy of "
     "the surviving relation sym(Me)=-3 sym(Md), and the range of the ratio that was claimed bounded. "
     "The identity holds to machine precision only in the collapsed case."),
    ("5", "Seesaw surjectivity and type-II", "fig5_seesaw",
     "Verifies the neutrino theorem numerically. For an arbitrary invertible symmetric target S, the "
     "construction MR = -MD S^-1 MD^T is symmetric, invertible, and maps back to S; five hundred "
     "targets are reproduced to better than one part in 10^12. Two type-II diagnostics confirm that "
     "setting Y10 to zero forces the CKM matrix diagonal, and that Me is never proportional to the "
     "type-II neutrino matrix at a generic point."),
    ("6", "The reach landscape", "fig6_landscape",
     "The headline visual. Each content is placed at the rank of its gauge group and its operator "
     "count, at a height equal to its exact reach. Two translucent plates mark the baselines. Stems "
     "for excluded contents rise from the ceiling rather than the floor, so the excess above baseline "
     "is what the eye measures. A companion panel decomposes the viable set into gauge unification, "
     "charge quantisation, and a model-specific part of zero height."),
    ("7", "The counting trap", "fig7_trap",
     "Quantifies what naive operator counting invents. Both affected contents are re-measured under "
     "both countings, and the transpose diagnostic is evaluated on the correct parameterisation. For "
     "10+120 the naive count forces identical down and lepton spectra and wrongly excludes the "
     "content; correctly counted, the spectra separate and the content is viable."),
    ("8", "Assertions and export", "fig8_assertions",
     "Every numerical claim in the three documents is an assertion here. The driver runs all modules, "
     "evaluates the assertions, prints one line per claim, and exports a JSON record so that any "
     "table entry can be audited without re-running the program. Single-column renditions of the "
     "wide figures are also produced here, for the two-column journal format."),
]


def esc(s):                                                        # Escape a string for LaTeX verbatim-like display.
    s = s.replace("\\", "\\textbackslash{}")                        # Backslash first, before introducing new ones.
    for a, b in [("{", "\\{"), ("}", "\\}"), ("$", "\\$"), ("&", "\\&"),  # Special characters.
                 ("#", "\\#"), ("_", "\\_"), ("%", "\\%"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)                                        # Replace each in turn.
    return s                                                       # Safe for typesetting.


def split_line(line):                                              # Split a source line into code and trailing comment.
    # A '#' inside a string literal is not a comment.  Track quote state.
    q = None                                                       # Current quote character, if inside a string.
    i = 0                                                          # Scan position.
    while i < len(line):                                           # Walk the characters.
        c = line[i]                                                # Current character.
        if q:                                                      # Inside a string: look for its terminator.
            if c == q:                                             # Closing quote.
                q = None                                           # Leave the string.
        elif c in "\"'":                                           # Opening quote.
            q = c                                                  # Enter the string.
        elif c == "#":                                             # A hash outside any string starts the comment.
            return line[:i].rstrip(), line[i + 1:].strip()          # Code and comment text.
        i += 1                                                     # Advance.
    return line.rstrip(), ""                                       # No comment on this line.


def parse_modules(path):                                           # Slice the source into its nine modules.
    lines = open(path, encoding="utf-8").read().split("\n")         # All source lines.
    marks = []                                                     # (line index, module number) for each banner.
    for i, ln in enumerate(lines):                                 # Find the module banners.
        m = re.match(r"#\s*=====\s*MODULE\s+(\d+)", ln)             # Banner pattern.
        if m:                                                      # A module starts here.
            marks.append((i, m.group(1)))                           # Record it.
    out = {}                                                       # Module number -> list of source lines.
    for k, (start, num) in enumerate(marks):                        # Slice between consecutive banners.
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)  # Next banner, or end of file.
        out[num] = lines[start:end]                                # The module's source.
    return out                                                     # Nine slices.


def main():                                                        # Build the document.
    mods = parse_modules(SRC)                                      # Slice the program.
    res = json.load(open(RES, encoding="utf-8"))                   # Measured numbers for the prose.
    npass = res["n_passed"]; ntot = res["n_assertions"]             # Assertion counts.
    sur = res["seesaw"]; comp = res["parameterisation_comparison"]  # Key results.

    P = []                                                         # Accumulator for LaTeX lines.
    A = P.append                                                   # Shorthand.
    A("\\documentclass[prd,preprint,superscriptaddress,nofootinbib,amsmath,amssymb,floatfix]{revtex4-2}")
    A("\\usepackage{graphicx}")                                    # Figures.
    A("\\usepackage{booktabs}")                                    # Rules.
    A("\\usepackage{fancyvrb}")                                    # Verbatim with per-line markup.
    A("\\usepackage{framed}")                                      # Background shading that tolerates verbatim.
    A("\\usepackage{float}")                                       # [H] placement for the GUI screenshots.
    A("\\usepackage{xcolor}")                                      # Colour.
    A("\\usepackage{longtable}")                                   # Tables that break across pages.
    A("\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}")
    A("\\graphicspath{{../05_figures/}}")                          # Figure directory.
    A("\\definecolor{codebg}{HTML}{F7F9FB}")                       # Listing background.
    A("\\definecolor{codeframe}{HTML}{D6DDE5}")                    # Listing frame.
    A("\\definecolor{shadecolor}{HTML}{F7F9FB}")                   # framed.sty reads this name.
    A("\\definecolor{commentcol}{HTML}{1F6F4A}")                   # Comment colour: green.
    A("\\definecolor{codecol}{HTML}{1B2430}")                      # Code colour: ink.
    A("\\DefineVerbatimEnvironment{codebox}{Verbatim}{fontsize=\\scriptsize,commandchars=\\\\\\{\\},formatcom=\\color{codecol}}")
    A("\\newcommand{\\cm}[1]{\\textcolor{commentcol}{#1}}")        # Comment markup inside listings.
    A("\\begin{document}")
    A("\\title{Code Documentation for\\\\ ``Grand unification builds the stage, not the script:\\\\")
    A("the exact reach of unified flavour models, with neutrinos''}")
    A("\\author{Yuxuan Zhang}")
    A("\\affiliation{College of Communication Engineering, Jilin University, Changchun 130012, China}")
    A("\\author{Weitong Hu}")
    A("\\affiliation{Aviation University of Air Force, Changchun 130022, China}")
    A("\\author{Wei Zhang}")
    A("\\affiliation{College of Computer Science and Technology, Jilin University, Changchun 130012, China}")
    A("\\date{\\today}")
    A("\\begin{abstract}")
    A("This document reproduces the nine modules of \\texttt{verify\\_prd.py}, one per section, with a "
      "comment on every source line and a figure per module. Listings are generated from the program "
      "that produced the numbers, so they cannot drift from the source; the prose and assertion "
      "register are regenerated from \\texttt{results.json} each build. Running the program reproduces "
      "the census, the exact rational ranks and all %d assertions, and writes a JSON record plus twelve "
      "figures (nine wide and single-column renditions). Dependencies: NumPy, Matplotlib, SciPy "
      "(\\texttt{scipy.linalg} for the grade-(a) maxvol QR); seed fixed at %d. Auxiliary certificate and "
      "diagnosis scripts are summarised in Section~\\ref{app:aux} (purpose, method and verdict; their "
      "sources carry block comments, while the nine module listings below are commented line by "
      "line)." % (ntot, res["seed"]))
    A("\\end{abstract}")
    A("\\maketitle")
    # Widen the number boxes in the table of contents.  With revtex's
    # preprint layout the default boxes are too narrow for the two-level
    # numbering used here, which produces Overfull \hbox warnings in the
    # .toc lines.  These widths are set before \tableofcontents is read.
    A("\\makeatletter")
    A("\\renewcommand*\\l@section{\\@dottedtocline{1}{0em}{2.6em}}")
    A("\\renewcommand*\\l@subsection{\\@dottedtocline{2}{2.6em}{3.4em}}")
    A("\\makeatother")
    A("\\tableofcontents")
    A("\\clearpage")

    A("\\section{How to read this document}")
    A("Each of the nine sections below corresponds to one module of \\texttt{verify\\_prd.py}. "
      "A section opens with a paragraph stating what the module computes and why it is built that "
      "way, then shows its figure, then lists its source. In the listings the left column is "
      "executable Python and the right column, in green, is the comment carried by that same line "
      "in the program. Nothing is paraphrased: the listing is the program.")
    A("")
    A("\\subsection{Reproduction}")
    A("\\begin{verbatim}")
    A("python verify_prd.py      # nine modules + certificates")
    A("                          # -> results.json, fig0..8 + col")
    A("python build_codedoc.py   # regenerate this document")
    A("\\end{verbatim}")
    A("")
    A("\\subsection{Result summary}")
    A("The program reports %d of %d assertions passed. Seven contents are spectrum-splitting and "
      "eleven are excluded; every viable content has a flavour Jacobian of full rank thirteen, so "
      "$P_{\\mathrm{add}}=0$ throughout. The seesaw construction reproduces %d arbitrary symmetric "
      "targets with maximum relative error %.1e. \\textbf{Current verdict (strict Takagi):} the "
      "observable-space dimension is 22 for every content, i.e.\\ zero forced low-energy relations; "
      "the exact pencil identity is matrix-level only. \\textbf{Withdrawn / counting-trap contrast:} "
      "the identity $M_e=-2M_d-M_d^{T}$ (median relative error %.1e in the collapsed parameterisation "
      "and %.2f in the correct one) is \\emph{not} claimed by the paper; it is shown only to illustrate "
      "the bidoublet counting trap."
      % (npass, ntot, sur["n_points"], sur["hit_err_max"],
         comp["collapsed"]["identity_err_median"], comp["correct"]["identity_err_median"]))
    A("\\clearpage")

    for num, title, fig, prose in MODULES:                          # One section per module.
        A("\\section{Module %s: %s}" % (num, esc(title)))
        A(esc(prose))                                              # Prose may contain _ and ^: escape it too.
        A("")
        A("\\begin{figure}[htbp]")
        A("\\centering")
        A("\\includegraphics[width=0.95\\textwidth]{%s.pdf}" % fig)
        A("\\caption{Module %s output.}" % num)
        A("\\end{figure}")
        A("")
        A("\\subsection*{Listing %s}" % num)
        body = mods.get(num, [])                                   # This module's source lines.
        # Render in chunks so that a long module breaks across pages cleanly.
        CH = 46                                                    # Lines per listing block.
        for start in range(0, len(body), CH):                       # Walk the module in chunks.
            chunk = body[start:start + CH]                          # The chunk.
            A("\\begin{shaded*}")                                  # shaded* tolerates verbatim content.
            A("\\begin{codebox}")
            for ln in chunk:                                       # One output line per source line.
                code, comment = split_line(ln)                      # Separate code from its comment.
                if not code and not comment:                        # Blank source line.
                    A("")                                          # Preserve the blank.
                    continue                                       # Next line.
                if not code and comment:                            # A pure comment line.
                    A("\\cm{# %s}" % esc(comment))                  # Render it in comment colour.
                    continue                                       # Next line.
                # Code with an optional trailing comment, aligned into a right-hand column.
                # Long lines are wrapped so that nothing runs past the text block.
                CODEW = 62                                         # Code column width: fits the preprint text block.
                COMW = 62                                          # Comment column width.
                csegs = [code[i:i + CODEW] for i in range(0, len(code), CODEW)] or [""]
                msegs = ([comment[i:i + COMW] for i in range(0, len(comment), COMW)]
                         if comment else [])                       # Comment wrapped independently.
                nrows = max(len(csegs), len(msegs))                 # Rows this source line occupies.
                for k in range(nrows):                             # Emit one printed row per segment pair.
                    cpart = csegs[k] if k < len(csegs) else ""      # Code segment, or blank.
                    mpart = msegs[k] if k < len(msegs) else ""      # Comment segment, or blank.
                    pad = max(1, CODEW + 2 - len(cpart))            # Align the comment column.
                    if mpart:                                      # There is comment text on this row.
                        prefix = "# " if k == 0 else "  "           # Hash only on the first row.
                        A("%s%s\\cm{%s%s}" % (esc(cpart), " " * pad, prefix, esc(mpart)))
                    else:                                          # Code only.
                        A(esc(cpart))                              # Render it.
            A("\\end{codebox}")
            A("\\end{shaded*}")
            A("")
        A("\\clearpage")

    A("\\section{Auxiliary certificate and diagnosis scripts}")
    A("\\label{app:aux}")
    A("Beyond the nine modules, the package ships the scripts below. Each runs independently and "
      "is summarised here with its purpose, method and verdict (block-commented sources); all are "
      "invoked by the driver and their outputs enter the JSON record.")
    A("\\begin{itemize}")
    A("\\item \\texttt{strict\\_obs.py} --- single source of truth for observable dimensions: strict, "
      "basis-independent Autonne--Takagi observables (transpose congruence to the charged-lepton mass "
      "basis, column-phase fixing, rephasing-invariant Majorana phases). Generic rank 22 / zero "
      "relations per content; charged 13. Basis-independent to $\\sim10^{-14}$.")
    A("\\item \\texttt{exact\\_core.py} --- exact Gaussian-rational arithmetic and dual-number "
      "derivatives over $\\mathbb{Q}(i)$; the pencil identity, explicit constraints (E1)--(E2), and "
      "exact matrix-map ranks. The orbit-subtraction dimension method is withdrawn (audit 9).")
    A("\\item \\texttt{charged\\_grade\\_a.py} --- grade-(a) charged dominance for all four SO(10) "
      "viable contents: exact rational invariant Jacobian rank 13 (nine $\\mathrm{tr}(H_f^k)$ masses "
      "plus $H_u$--$H_d$ CKM cross words; $H_e$ cross words excluded because the quark and lepton "
      "left-handed bases rotate independently).")
    A("\\item \\texttt{grade\\_a\\_fast.py} --- the grade-(a) neutrino rank-22 certificate: float-interval "
      "implicit-function differentials of the 22 strict observables, maxvol submatrix selection, and the "
      "Neumann criterion $\\rho=\\|A^{-1}\\|_\\infty\\|E\\|_\\infty<1$ at seeds 2/7/11 (tied-$v_L$ "
      "parameterisation identical to \\texttt{strict\\_obs.build}; E inflated by a conservative "
      "Richardson-FD bound with $|A-J_{\\rm FD}|\\le E$ independently verified, and eigenvalue intervals "
      "independently containment-checked). Requires \\texttt{scipy.linalg}.")
    A("\\item \\texttt{rational\\_route\\_exhaustive.py} --- exhaustive screen of 40 one-/two-$M_\\nu$ "
      "trace words (20 fibre-constant survivors, exact rank 16) plus the fraction-field argument "
      "$d(f/g)\\in\\mathrm{span}\\{df,dg\\}$: no rational invariant of the trace-word fraction field "
      "reaches rank 22.")
    A("\\item \\texttt{rg\\_lowenergy\\_match.py} --- measured-boundary one-loop RG matching: the 120+126 "
      "diagonal-lock exclusion survives running (measured spectrum at GUT $m_\\tau/m_b\\simeq1.07$ vs "
      "the 2.76 wall; round-trip $10^{-14}$).")
    A("\\item \\texttt{fiber\\_proof.py} --- tangent-space demonstration that $\\Sigma=\\mathrm{sym}\\,M_d$ "
      "varies along the 12-dimensional observable fibre while observables are stationary to $O(t^2)$.")
    A("\\item \\texttt{seed42\\_diagnosis.py} --- the seed-42 rank-21 point is a Jarlskog-near-zero "
      "coordinate degeneracy ($J\\simeq-4.6\\times10^{-7}$, $d^2J/dt^2$ stable), not a physical relation.")
    A("\\item \\texttt{pencil\\_mining.py} --- pencil identity scope (holds for 120+126, fails for "
      "10+126 / 10+120+126 / double-126), breaking scale ($\\propto\\varepsilon^2$), and a "
      "prior-dependent statistical preference on the neutrino mass hierarchy (KS $p\\sim10^{-270}$ for "
      "$m_3/m_1$). Requires \\texttt{scipy.stats}.")
    A("\\item \\texttt{declaration\\_scoreboard.py} --- declaration-cost scoreboard: generic fit of the "
      "six indicative GUT ratios gives cost 0 for 10+126 and $\\ge1$ for 120+126 (stuck at the "
      "$m_\\tau/m_b$ wall), breaking the census $P_{\\rm add}=0$ tie.")
    A("\\item \\texttt{pencil\\_prior\\_sensitivity.py} --- prior robustness of the statistical "
      "hierarchy preference: direction (single-$S$ $\\Rightarrow$ weaker hierarchy) holds in 6/6 "
      "priors and across breaking strengths $\\varepsilon=0.3,1,3$; magnitude is measure-dependent. "
      "Requires \\texttt{scipy.stats}.")
    A("\\item \\texttt{exclusion\\_taxonomy.py} --- exclusion classification: 9/11 excluded "
      "contents are CKM-locked ($V_{\\rm CKM}=\\mathbb{1}$) by group theory alone, across all four "
      "groups; the minimal-content exclusion survives measured-boundary RG running.")
    A("\\item \\texttt{pencil\\_attribution\\_control.py} --- attribution control: the weaker-hierarchy "
      "preference appears in 10+126 (which has NO pencil identity), proving it is a single-$S$ seesaw "
      "structure effect, not a pencil consequence. Requires \\texttt{scipy.stats}.")
    A("\\item \\texttt{pencil\\_analytic\\_origin.py} --- algebraic derivation of the pencil identity: "
      "$M_D=aA+bS$ with $A$ antisymmetric $\\Rightarrow$ $B=AS^{-1}$ traceless rank-2 $\\Rightarrow$ "
      "$B^2$ has a repeated eigenvalue $\\Rightarrow$ $\\mathrm{disc}(M_\\nu S^{-1})=0$. Every link "
      "verified to $10^{-13}$; the repeated root is forced by $\\det A=0$ (Pfaffian). The chain is "
      "derived algebraically and confirmed numerically (grade b), not certified in exact arithmetic.")
    A("\\item \\texttt{singleS\\_spectral\\_theorem.py} --- a $\\mathbf{120}+\\mathbf{126}$-only spectral "
      "observation: single-$S$ $M_\\nu$ has 2 distinct eigenvalues ($A^2$ eigs $\\{0,\\lambda,\\lambda\\}$), "
      "broken-$S$ has 3. This $3\\to2$ reduction is \\emph{not} the preference mechanism (10+126 has 3 "
      "eigenvalues yet the same preference); the preference is a single-$S$ $M_R$ structure effect.")
    A("\\item \\texttt{seven\\_content\\_scoreboard.py} --- declaration-cost scoreboard for all seven "
      "viable contents (census Clebsch table): 5 fit with cost 0, $\\mathbf{10}+\\mathbf{120}$ and "
      "$\\mathbf{120}+\\mathbf{126}$ need $\\ge1$; plus the Pfaffian generalisation test (SU(5) 45 "
      "fails: general down Yukawa, no seesaw singlet). The latter is a STRUCTURAL criterion computed "
      "from the Clebsch table plus the declared group-level seesaw-singlet fact, not a numerical "
      "measurement.")
    A("\\end{itemize}")
    A("")
    A("\\section{The graphical control panel (\\texttt{runall.bat})}")
    A("\\label{sec:gui}")
    A("For readers who prefer not to use a command line, the package ships a one-click graphical "
      "control panel. Double-clicking \\texttt{runall.bat} (in the package root) opens the window "
      "below. Every verification program is a button on the left; clicking it shows, in plain "
      "language, \\emph{what} the program does, \\emph{why} it matters, and \\emph{what a good result "
      "looks like}, together with an illustrative figure. The green ``Run'' button executes the "
      "selected program and streams its live output into the console at the bottom. The first "
      "button, ``RUN ALL CHECKS'', runs the master verifier (\\texttt{verify\\_prd.py}, 70 assertions). "
      "No physics background is needed to read the panels: each is written for a curious "
      "non-specialist, and the pictures are real result figures.")
    A("")
    A("\\begin{figure}[H]")
    A("\\centering")
    A("\\includegraphics[width=0.98\\textwidth]{gui_screenshot.png}")
    A("\\caption{The control panel with the master verifier selected. Left: the program list. "
      "Right: plain-language purpose, expected result, and the rank-22 certificate figure. "
      "Bottom: the live-output console.}")
    A("\\end{figure}")
    A("")
    A("\\begin{figure}[H]")
    A("\\centering")
    A("\\includegraphics[width=0.98\\textwidth]{gui_screenshot_control.png}")
    A("\\caption{The same panel with the attribution control selected. The figure shows the "
      "weaker-hierarchy preference appearing in \\emph{both} 120+126 (which has the pencil identity) "
      "and 10+126 (which does not)---the control that proves the preference is a single-$S$ structure "
      "effect, not a pencil consequence.}")
    A("\\end{figure}")
    A("")
    A("\\section{Assertion register}")
    A("Every claim below is evaluated by the program and printed at run time. A failure exits "
      "non-zero.")
    A("")
    A("\\begin{longtable}{@{}clp{0.62\\textwidth}@{}}")
    A("\\toprule")
    A("\\# & status & assertion \\\\")
    A("\\midrule")
    A("\\endhead")
    for i, c in enumerate(res["checks"]):                           # One row per assertion.
        A("%d & %s & %s \\\\" % (i + 1, "PASS" if c["passed"] else "FAIL", esc(c["name"])))
    A("\\bottomrule")
    A("\\end{longtable}")
    A("")
    A("\\section{Measured values}")
    A("\\begin{table}[htbp]")
    A("\\centering")
    A("\\caption{Ranks and conditioning, as exported by the program.}")
    A("\\begin{tabular}{@{}lcccc@{}}")
    A("\\toprule")
    A("content & $n_{\\mathrm{par}}$ & rank & $s_{\\min}/s_{\\max}$ & rel.\\ gap \\\\")
    A("\\midrule")
    for r in res["ranks"]:                                         # One row per viable content.
        A("%s & %d & %d & %.1e & %.3f \\\\" %
          (esc(r["model"]), r["n_params"], r["rank"], r["s_ratio"], r["min_gap"]))
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\end{table}")
    A("\\end{document}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)                # Ensure the target directory.
    open(OUT, "w", encoding="utf-8").write("\n".join(P))            # Write the document.
    print("wrote", OUT)                                            # Report.
    print("modules:", len(MODULES), "assertions:", ntot)            # Summary.


if __name__ == "__main__":                                         # Entry point.
    main()                                                         # Build.
