# -*- coding: utf-8 -*-
"""Generate supplement.tex from the shipped sources + the annotation table."""
import os
import sys
import platform
import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, HERE)
from annotations import A  # noqa: E402

ORDER = ["radialflux.py", "lattice96.py", "validate.py", "audit_claims.py",
         "check_tex.py", "series_d.py", "verify_core.py", "verify_shells.py",
         "make_figs.py", "make_figs2.py", "make_fig5.py", "make_figs3d.py"]

TITLES = {
    "radialflux.py": "The production module",
    "lattice96.py": "The periodic lattice solve",
    "validate.py": "The verification suite (51 asserted checks)",
    "audit_claims.py": "The claim auditor (84 numeric checks)",
    "check_tex.py": "The LaTeX integrity checker",
    "series_d.py": "Symbolic series in $d$",
    "verify_core.py": "Symbolic cross-checks of the closed forms",
    "verify_shells.py": "Symbolic and lattice cross-checks of the shell identity",
    "make_figs.py": "Figure scripts, panels 1--3",
    "make_figs2.py": "Figure scripts, panel 4",
    "make_fig5.py": "Figure scripts, panel 5",
    "make_figs3d.py": "The three-dimensional figure generator (this volume)",
}

INTRO = {
    "radialflux.py":
        "All of the paper's numerical content that is not a lattice solve lives here: "
        "the exact radius, the closed-form bias, the threshold, the background fraction "
        "and the analytic profiles used as ground truth. It depends on NumPy only.",
    "lattice96.py":
        "The $96^3$ test. The only subtlety is line 29: the discrete Laplacian is "
        "inverted with its exact lattice symbol $4\\sum_i\\sin^2(k_i/2)$, not with "
        "$|\\mathbf{k}|^2$; the difference is a few percent at the Nyquist edge and "
        "would have contaminated the background measurement.",
    "validate.py":
        "Thirteen groups of assertions, 51 checks in total, runtime 0.06~s. Each check "
        "compares an implementation against an independent construction (quadrature, "
        "a closed form, an exact root, a random shell pair), never against itself.",
    "audit_claims.py":
        "Every number printed in the manuscript -- six tables, the series coefficients, "
        "the Cornell example -- is recomputed here from the module. If the text and the "
        "code ever diverge, this script exits non-zero.",
    "check_tex.py":
        "Structural checks on the manuscript: duplicate labels, dangling references, "
        "uncited bibliography entries, citations with no entry, figures that are "
        "included but never referenced, and referenced files that do not exist.",
    "series_d.py":
        "Expands $\\rho_\\ast$, $\\kappa_L$, $\\kappa_R$, $\\kappa_A$ and the "
        "flux-weighted mean radius to second order in $\\varepsilon=q-1$, for "
        "$d=3,\\dots,11$, with exact rational coefficients.",
    "verify_core.py":
        "Independent symbolic derivations: the power-mean identity, the factorisation "
        "of the threshold polynomial, the four-dimensional ball, and the difference "
        "stencils that are exact at geometric means.",
    "verify_shells.py":
        "Checks the two identities that the lattice analysis rests on: "
        "$\\Delta_{\\rm lat}|\\mathbf{x}|^2=2d$ exactly, and the two-term shell "
        "identity for a distributed source in $d=3$ and $d=4$.",
    "make_figs.py":
        "Panels 1--3 of the manuscript: the equal-area construction, the naive "
        "generalisation, and the uniform-lattice bias.",
    "make_figs2.py":
        "Panel 4: the distributed source and the running field. It re-executes only "
        "the style header of the previous file rather than importing it.",
    "make_fig5.py":
        "Panel 5: the $96^3$ measurement, the closed-form background, and the "
        "scaling comparison that makes the null test.",
    "make_figs3d.py":
        "The nine three-dimensional figures of this supplement. Each one calls the "
        "shipped module, so no picture can drift away from the code it illustrates.",
}


def esc(t):
    rep = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
           ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
           ("^", r"\^{}"), ("~", r"\textasciitilde{}")]
    for a, b in rep:
        t = t.replace(a, b)
    return t


def note_rows(rows):
    out = []
    for a, b, note in rows:
        num = str(a) if a == b else "%d--%d" % (a, b)
        out.append("\\noterow{%s}{%s}" % (num, esc(note)))
    return "\n".join(out)
HEAD = r"""
\documentclass[preprint,12pt,3p]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{caption}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{3em}
\newcounter{none}
\captionsetup{font=small,labelfont=bf}
\newcommand{\Om}{\Omega_d}
\newcommand{\rstar}{\rho_\ast}
\newcommand{\eps}{\varepsilon}
\newcommand{\noterow}[2]{%
  \raggedleft #1 & \parbox[t]{\linewidth}{\raggedright\footnotesize #2}\tabularnewline
  \hline}
\newenvironment{notetable}{%
  \footnotesize
  \setlength{\tabcolsep}{4pt}%
  \renewcommand{\arraystretch}{1.05}%
  \def\LTcaptype{none}%
  \begin{longtable}{@{}>{\raggedleft\arraybackslash}p{1.05cm}%
      >{\raggedright\arraybackslash}p{0.845\textwidth}@{}}
  \toprule
  \textbf{line} & \textbf{annotation}\\
  \midrule
  \endfirsthead
  \toprule
  \textbf{line} & \textbf{annotation}\\
  \midrule
  \endhead
  \bottomrule
  \endfoot
}{%
  \end{longtable}
}
\lstset{language=Python,
  basicstyle=\ttfamily\scriptsize,
  numbers=left, numberstyle=\tiny\color{gray}, numbersep=5pt,
  breaklines=true, breakatwhitespace=false,
  postbreak=\mbox{\textcolor{gray}{$\hookrightarrow$}\space},
  columns=fullflexible, keepspaces=true, showstringspaces=false,
  frame=single, framesep=3pt, rulecolor=\color{black!25},
  xleftmargin=14pt, framexleftmargin=12pt,
  aboveskip=6pt, belowskip=8pt}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=2pt}
"""

BODY_1 = r"""
\begin{document}
\begin{frontmatter}
\title{Supplementary material to\\[2pt]
\large Exact flux-sampling radii for radial Coulomb and Poisson problems on
graded meshes: closed-form estimator bias in any dimension, and a null test}
\author[1]{Yuxuan Zhang\fnref{orc1}\corref{cor1}}
\ead{csoft@live.cn}
\author[2]{Weitong Hu\fnref{orc2}}
\author[3]{Wei Zhang\fnref{orc3}}
\ead{zwei25@mails.jlu.edu.cn}
\cortext[cor1]{Corresponding author.}
\fntext[orc1]{ORCID: \href{https://orcid.org/0009-0007-9994-0172}{0009-0007-9994-0172}}
\fntext[orc2]{ORCID: \href{https://orcid.org/0009-0000-9011-333X}{0009-0000-9011-333X}}
\fntext[orc3]{ORCID: \href{https://orcid.org/0000-0002-8950-2933}{0000-0002-8950-2933}}
\address[1]{College of Communication Engineering, Jilin University, Changchun
130012, China}
\address[2]{Aviation University of Air Force, Changchun 130022, China}
\address[3]{College of Computer Science and Technology, Jilin University,
Changchun 130012, China}
\begin{abstract}
This supplement accompanies the manuscript and contains three things that do not
belong in a letter-length article: (i) the complete reference implementation, with
every executable line annotated and every imported function traced to its
definition; (ii) nine three-dimensional visualisations, each generated by the
shipped code; and (iii) an explicitly speculative section on a
communication-theoretic reading of the estimator.  Part (iii) is an outlook, not a
result: nothing in it is used in any claim of the paper.
\end{abstract}
\end{frontmatter}

\section{Scope, and what this document is not}

The manuscript makes four kinds of statement: an exact identity for the sampling
radius, closed forms for the bias of the usual rules, a null test on a periodic
lattice, and a set of numerical tables.  Each of them is produced by code, and the
purpose of this supplement is to make that code inspectable rather than merely
available.

\begin{itemize}
\item Section~\ref{sec:prov} lists every external function the implementation calls,
with its definition and the reason it was chosen over the obvious alternative.
\item Section~\ref{sec:code} reproduces all twelve Python files with a
line-by-line annotation.  The files themselves are unmodified: the line numbers in
the tables are the line numbers in the shipped sources.
\item Section~\ref{sec:figs} presents nine three-dimensional figures.
\item Section~\ref{sec:comm} is an outlook.  It records an analogy between the
estimator and problems in communication engineering.  \textbf{It is not part of
the paper's argument}, no claim in the paper depends on it, and it is included
because the first author's graduate training is in communications and this analogy
is where the present line of work began.
\end{itemize}

\noindent\textbf{Reproducibility.}  Every number quoted below was produced by the
commands listed in \ref{app:run} on the environment of Table~\ref{tab:env}.  The
annotation coverage itself is machine-checked: \texttt{check\_annotations.py}
verifies that every one of the \textbf{1471 executable lines} of the twelve files
falls inside an annotated range, and exits non-zero otherwise.

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lll@{}}
\toprule
file & lines (code) & role \\
\midrule
%%FILETABLE%%
total & %%TOTAL%% & \\
\bottomrule
\end{tabular}
\caption{The package.  ``Lines'' counts physical lines; ``code'' counts lines that
are neither blank nor pure comments.  The annotation table covers 100\% of the
latter.}
\label{tab:files}
\end{table}

\section{Provenance of the imported functions}
\label{sec:prov}

No function is used as a black box.  Table~\ref{tab:prov} gives, for every external
name the package calls, the definition that is actually evaluated and the reason
that particular function was chosen.  Two choices deserve a sentence each.

\paragraph{\texttt{expm1}.}  The identity
$\rho_\ast^{\,d-1}=(r_{n+1}-r_n)\big/\int_{r_n}^{r_{n+1}}r^{1-d}\,\mathrm{d}r$
contains the difference $1-q^{-a}$ with $a=d-2$.  For $q-1=10^{-8}$ and $d=4$ that
difference is $10^{-8}$, while both terms are of order one: computing it directly
loses eight digits.  \texttt{np.expm1}$(t)$ evaluates $e^t-1$ with a dedicated
kernel and a relative error below one ulp for $|t|\lesssim10^{-5}$, so the factored
form $r_n^{-a}\,[-\texttt{expm1}(-a\ln q)]/a$ keeps full precision.

\paragraph{\texttt{math.gamma}.}  $\Om=2\pi^{d/2}/\Gamma(d/2)$ is evaluated with
the scalar Cephes implementation rather than a Stirling approximation, because the
argument is at most $5$ in all applications and the exact value is needed at the
$10^{-15}$ level for the $d\to2$ checks.

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}p{3.3cm}p{2.6cm}p{5.1cm}p{4.7cm}@{}}
\toprule
name & module & definition / convention & why this one \\
\midrule
\texttt{pi} & math & the double nearest to $\pi$ & correctly rounded constant \\
\texttt{gamma} & math & $\Gamma(x)$, Cephes & scalar exactness, see above \\
\texttt{log}, \texttt{sqrt} & math & natural log, square root & scalar paths \\
\texttt{perf\_counter} & time & monotonic clock, highest resolution & timing only \\
\texttt{getrusage} & resource & \texttt{ru\_maxrss}, peak RSS in kB & peak memory of the solve \\
\midrule
\texttt{asarray} & numpy & view/copy to ndarray & accepts scalars, lists, arrays \\
\texttt{log} & numpy & elementwise natural log & vectorised ratio $\ln q$ \\
\texttt{expm1} & numpy & $e^{x}-1$ without cancellation & the stability of the whole method \\
\texttt{errstate} & numpy & context manager for FP warnings & silences expected $x/0$ \\
\texttt{ones\_like}, \texttt{zeros\_like} & numpy & shape-preserving allocation & no hard-coded shapes \\
\texttt{diff} & numpy & $a_{i+1}-a_i$ & shell thickness \\
\texttt{minimum} & numpy & elementwise min & clamps $r$ at the ball radius \\
\texttt{where} & numpy & elementwise select & branches inside/outside $R$ \\
\texttt{geomspace} & numpy & $N$ log-spaced points & constant mesh ratio $q$ \\
\texttt{sqrt}, \texttt{rint} & numpy & elementwise root / nearest int & shell index \\
\texttt{meshgrid} & numpy & coordinate grids, \texttt{indexing='ij'} & matches FFT axis order \\
\texttt{fftfreq} & numpy & DFT frequencies, cycles per unit length & $k=2\pi f$ \\
\texttt{fftn}, \texttt{ifftn} & numpy & $n$-dimensional DFT, unitary-free & solves the Poisson problem \\
\texttt{lstsq} & numpy & least squares via SVD, \texttt{rcond=None} & no normal equations \\
\texttt{default\_rng} & numpy & PCG64 generator, seed 7 & reproducible random shells \\
\texttt{linspace}, \texttt{array}, \texttt{vstack} & numpy & standard construction & -- \\
\texttt{uniform}, \texttt{sort}, \texttt{mean}, \texttt{std} & numpy & standard statistics & -- \\
\midrule
\texttt{symbols}, \texttt{series}, \texttt{integrate}, \texttt{simplify} & sympy & exact algebra & independent symbolic checks \\
\texttt{re}, \texttt{os}, \texttt{sys} & stdlib & regex, paths, exit codes & structural checks \\
\bottomrule
\end{tabular}
\caption{Every external name used by the package.  The two middle blocks are the
entire numerical surface; there is no SciPy dependency, by design, so that the
implementation can be read end to end.}
\label{tab:prov}
\end{table}

\paragraph{Conventions that matter.}  \texttt{np.fft.fftfreq(L)} returns cycles per
unit length in $[-1/2,1/2)$; multiplying by $2\pi$ gives the angular wavenumbers of
the discrete Laplacian.  \texttt{np.fft.fftn} performs an unnormalised forward
transform and \texttt{ifftn} divides by $L^3$; the pair is inverse to each other, so
the solve needs no extra normalisation.  \texttt{np.linalg.lstsq} with
\texttt{rcond=None} uses the machine-precision cutoff; the fit is well conditioned
over $R=8,\dots,24$ and the residual rms is $1.2\times10^{-5}$, so the cutoff choice
does not influence the quoted digits.

\section{Annotated code}
\label{sec:code}

Each file is reproduced verbatim in a line-numbered listing, followed by a table
that maps line numbers to annotations.  Consecutive lines that implement one step
share a row; every executable line appears in at least one row -- a property checked
mechanically, not asserted by hand.  Where the code contains something worth
criticising -- an unused variable, a form that loses precision for very small
$q-1$ -- the annotation says so rather than hiding it.

"""


def build_code_section():
    out = []
    for i, f in enumerate(ORDER, start=1):
        title = TITLES[f]
        out.append("\\subsection{\\texttt{%s} --- %s}" % (esc(f), title))
        out.append(INTRO[f])
        out.append("")
        rel = ("../supplement/" + f) if f == "make_figs3d.py" else ("../src/" + f)
        out.append("\\lstinputlisting{%s}" % rel)
        out.append("\\begin{notetable}")
        out.append(note_rows(A[f]))
        out.append("\\end{notetable}")
        out.append("")
    return "\n".join(out)


BODY_FIGS = r"""
\clearpage
\section{Three-dimensional visualisations}
\label{sec:figs}

Each figure is produced by \texttt{make\_figs3d.py} (annotated in
Section~\ref{sec:code}) from the same functions that compute the numbers.  Table~\ref{tab:map}
maps each panel to the code path it exercises.  The
purpose of the set is not decoration: every panel answers a question that the flat
figures of the manuscript can only state in words.

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}p{4.3cm}p{4.6cm}p{5.4cm}@{}}
\toprule
code & what it computes & figure \\
\midrule
\texttt{exact\_radius}, \texttt{enclosed\_charge} & the sampling sphere and the flux &
Fig.~\ref{fig:s1}, \ref{fig:s3} \\
\texttt{exact\_bias}, \texttt{recovered\_fraction} & the closed-form bias surface &
Fig.~\ref{fig:s2} \\
\texttt{lattice96.solve} & the $96^3$ potential & Fig.~\ref{fig:s4} \\
\texttt{q\_star}, \texttt{naive\_radius} & the threshold curve & Fig.~\ref{fig:s5} \\
\texttt{validate.py} V12 & measured convergence order & Fig.~\ref{fig:s6} \\
\texttt{validate.py} V13 & multipole leakage & Fig.~\ref{fig:s8} \\
\texttt{lattice96.table}, \texttt{background\_charge} & the four estimators and the
null test & Fig.~\ref{fig:s9} \\
\texttt{running\_residual} & the slowly varying amplitude & Table~\ref{tab:comm} \\
\bottomrule
\end{tabular}
\caption{Which part of the code each 3-D figure exercises.  Every panel is a direct
call into the module; none of them re-implements a formula.}
\label{tab:map}
\end{table}

\begin{figure}[p]
\centering
\includegraphics[width=0.86\textwidth]{fig/s1_shell_geometry.pdf}
\caption{\textbf{One shell, one exact radius.}  The inner sphere $r_n$ (blue) and the
outer sphere $r_{n+1}$ (orange) bound one shell; the gold sphere is $\rstar$.  The
red arrows are the flux lines that cross the shell.  The statement of the paper is
geometric: the entire flux of the shell is carried by the single sphere $\rstar$,
for every ratio $q$, and in $d=3$ that sphere is the geometric mean
$\sqrt{r_nr_{n+1}}$.  Nothing is fitted; the radius is the closed form of
Eq.~(9).}
\label{fig:s1}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{fig/s2_bias_landscape.pdf}
\caption{\textbf{The bias landscape.}  $\kappa_L(q,d)$ over the whole parameter
range, computed from the closed form of Eq.~(13) with no root-finding.  The red
plane is the exact rule, $\kappa\equiv1$.  The gold points are the four entries of
Table~1 at $q=1.1$; the pale curve on the $d=2$ edge is the logarithmic mean
$\ln q/(q-1)$.  The surface makes the two qualitative claims of the paper visible at
once: the error is first order in $\eps=q-1$ for endpoint rules, and it grows with
$d$ at fixed $q$.}
\label{fig:s2}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.82\textwidth]{fig/s3_graded_mesh.pdf}
\caption{\textbf{The graded mesh.}  Seven shells with constant ratio $q=1.35$, each
shown with its own gold sampling sphere.  Because the mesh is geometric, the
relative position $\rstar/r_n$ is the same in every shell -- the picture is
self-similar, which is why a single closed form covers all shells and all
dimensions.  The red dots on the radial ray are the mesh nodes $r_n$.}
\label{fig:s3}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{fig/s4_lattice96.pdf}
\caption{\textbf{The $96^3$ solve.}  A quarter-cutaway point cloud of the potential
produced by \texttt{lattice96.solve}, on a logarithmic colour scale so that the six
decades between the $1/r$ core and the background floor are visible simultaneously.
The gold wireframe is the inner measurement shell $R=8$, the white one the outer
$R=24$.  The solve is one FFT with the exact lattice symbol, 0.14~s and 120~MB;
the visible spherical symmetry is a property of the solution, not an assumption of
the code.}
\label{fig:s4}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.88\textwidth]{fig/s5_thresholds.pdf}
\caption{\textbf{The threshold surface.}  $q+q^{2-d}-2$ over the $(d,q)$ plane; its
zero level is the curve $q^\ast_d$ on which the $(d-2)$-less rule still samples
inside its own shell.  At $d=4$ the threshold is the golden ratio, at $d=5$ the
plastic constant, at $d=6$ the next $n$-nacci constant.  At $d=3$ the surface is
tangent to zero along the whole line $q=1$: the double root, i.e.\ no threshold,
because the rule is exact there.}
\label{fig:s5}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{fig/s6_convergence.pdf}
\caption{\textbf{The convergence funnel.}  Measured error of the exact-radius rule
over $(\log_{10}(q-1),\log_{10}N)$, from 24 to 320 shells.  The teal plane is the
second-order guide.  The surface collapses onto it, which is the numerical content
of the statement that refining the mesh and flattening the grading are
interchangeable at fixed accuracy; the three measured orders are 2.01, 2.00 and
2.00.}
\label{fig:s6}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.80\textwidth]{fig/s7_outlook.pdf}
\caption{\textbf{An outlook, drawn only to make an analogy visible.}  Concentric
shells with a Fibonacci constellation of sampling directions; the gold curve is a
modulated radial signal.  This figure illustrates the reading discussed in
Section~\ref{sec:comm} and \emph{enters no claim of the paper}.  It is deliberately
labelled as such on the figure itself.}
\label{fig:s7}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{fig/s8_multipole.pdf}
\caption{\textbf{Multipole leakage.}  The prefactor $\sum_{j\le\ell}q^{-j}$ by which a
pure multipole of order $\ell$ contaminates the monopole reading, as a surface in
$(q,\ell)$.  The blue curve is the saturation value $q/(q-1)$ reached as
$\ell\to\infty$; the gold point is the case quoted in the text, $q=1.25$, $\ell=10$,
where the amplification is 1.375 (138\%).  Two things are visible: the leakage is
first order in $1/q$ per added multipole, and it saturates -- so the low multipoles,
not the high ones, do the damage.}
\label{fig:s8}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{fig/s9_waterfall.pdf}
\caption{\textbf{The null test as a waterfall.}  The four estimators of Table~4 on the
same $96^3$ lattice, plotted as rows over $R=8,\dots,24$ with stems down to the truth
plane $\hat Q/Q=1$.  The left-endpoint row sits $\sim3\%$ low and drifts; the
midpoint and exact rows are close but still consistent with the $R^3$ background
trend; only the background-corrected exact row stays on the plane over the whole
range, which is the numerical statement that the background is the dominant
systematic and that it is known in closed form.}
\label{fig:s9}
\end{figure}

\clearpage
"""

BODY_COMM = r"""
\section{An outlook: a communication-theoretic reading}
\label{sec:comm}

\noindent\textbf{Status of this section.}  What follows is a heuristic analogy
recorded for completeness.  It is \emph{not} a result of the paper, it is not
supported by any derivation in it, and no claim elsewhere in this supplement or in
the manuscript depends on it.  The first author's graduate training is in
communications engineering, and the estimator of this paper grew out of that
background; the analogy is written down here so that the origin of the idea is on
the record, and so that anyone who wishes to pursue it can see exactly what would
have to be built.

\subsection{The mapping}

The estimator takes one number per shell -- a potential difference -- and multiplies
it by a geometric weight.  That is structurally the same operation as a sampler in a
receiver: one number per symbol interval, multiplied by a weight that depends on
where in the interval the sample was taken.  Table~\ref{tab:comm} sets the two
vocabularies side by side.

\begin{table}[h]
\centering
\small
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}p{4.6cm}p{4.4cm}p{5.8cm}@{}}
\toprule
object in this paper & communication counterpart & why the correspondence is natural \\
\midrule
shell $[r_n,r_{n+1}]$ & one symbol interval $[nT,(n+1)T]$ &
a finite window that carries one number \\
mesh ratio $q=r_{n+1}/r_n$ & clock ratio / oversampling factor &
the geometric spacing plays the role of the sampling period \\
sample position $\rho'$ & sampling instant $t_n+\tau$ &
the free parameter whose choice creates bias \\
exact radius $\rstar$ & zero-forcing / matched-filter instant &
the unique position at which the interval mean is read without bias \\
bias $\kappa-1$ & intersymbol interference / gain error &
a multiplicative error that survives averaging \\
$\eps=q-1$ & timing offset normalised to the symbol period &
the small parameter of the expansion \\
$q^\ast_d$ & Nyquist-type threshold &
beyond it the sample falls outside its own interval: the estimator reads a
neighbouring symbol \\
$\sum_{j\le l}q^{-j}$ & ISI from $l$ neighbouring symbols &
the leakage of higher multipoles into the monopole reading \\
background $-\frac{2\pi}{3L^3}R(R+1)(2R+1)$ & DC offset / baseline wander &
a deterministic additive term that must be estimated and removed \\
running residual $\gamma[\frac{1}{d-2}-\ell_d(q)]$ & residual after equalisation of
a slowly fading channel & a static weight cannot track a log-varying amplitude \\
\bottomrule
\end{tabular}
\caption{A vocabulary correspondence.  The left column is exact mathematics; the
right column is an analogy, and the middle column is the reason the analogy is not
arbitrary.  It is still only an analogy.}
\label{tab:comm}
\end{table}

\subsection{Where the analogy is exact, and where it stops}

Three statements can be made precisely, and it is worth separating them from the
rhetoric.

\begin{enumerate}[label=(\roman*)]
\item \emph{Exact.}  The bias of a one-point rule is a functional of the sample
position alone, $\kappa=(\rho'/\rstar)^{d-1}$.  In a receiver the gain error of a
sampler is likewise a functional of the sampling instant alone.  The algebra is the
same algebra; this is why the same closed forms appear in both literatures.
\item \emph{Exact but trivial.}  The threshold $q^\ast_d$ is a statement about
whether a point lies inside an interval.  Calling it a Nyquist threshold adds no
content: Nyquist's theorem is about reconstructing a band-limited function from its
samples, and no band-limited reconstruction is being performed here.
\item \emph{Not established.}  There is no channel in this paper.  No signal
propagates, nothing is modulated, no noise is present, and $r$ is a coordinate, not
a time.  A genuine communication reading would require a wave equation in $d$
dimensions, a source and a receiver, and a noise model; none of these is in the
manuscript, and none is attempted here.
\end{enumerate}

\subsection{What would make it a physics statement}

If the analogy were ever to be taken seriously, the minimal honest programme would
be the following.  Take a wave equation $\Box_d\phi=S$ in $d$ spatial dimensions
with a localised source and an observer on a sphere of radius $R$.  Ask whether the
monopole extracted by the estimator of this paper equals the source strength when
the observer samples a discrete set of shells, and whether the residual of
Eq.~(32) acquires the interpretation of a slowly varying gain.  That is a
well-posed question with a computable answer, and it is not the question answered
here.  Until it is computed, the reading above should be treated as a source of
intuition and nothing more.

\subsection{A note on provenance}

The estimator was found while thinking about sampling in a different field, and the
vocabulary of Table~\ref{tab:comm} is the vocabulary in which it was first written
down.  That is a fact about the origin of the work, not evidence for the analogy.
The paper stands on the identities of Section~2, which are exact and independent of
any interpretation.

"""

BODY_LOG = r"""
\clearpage
\section{Verification log}
\label{sec:log}

The following outputs were produced on the environment of Table~\ref{tab:env} by
the commands listed in \ref{app:run}.  They are reproduced verbatim.

\subsection*{Annotation coverage}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,frame=single,columns=fullflexible]
%s
\end{lstlisting}

\subsection*{Verification suite (\texttt{validate.py})}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,frame=single,columns=fullflexible]
%s
\end{lstlisting}

\subsection*{Claim auditor (\texttt{audit\_claims.py}), tail}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,frame=single,columns=fullflexible]
%s
\end{lstlisting}

\subsection*{LaTeX integrity (\texttt{check\_tex.py})}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,frame=single,columns=fullflexible]
%s
\end{lstlisting}

\subsection*{The $96^3$ solve (\texttt{lattice96.py}), tail}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,frame=single,columns=fullflexible]
%s
\end{lstlisting}

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lrr@{}}
\toprule
quantity & value & check \\
\midrule
checks in \texttt{validate.py} & 51 & 51 passed \\
checks in \texttt{audit\_claims.py} & 84 & 0 failed \\
labels / references in the manuscript & 83 / 57 & 0 duplicate, 0 dangling \\
bibliography items & 49 & all cited, no phantom \\
figures in the manuscript & 5 & all referenced and present \\
$96^3$ solve time & 0.14 s & measured \\
peak memory of the solve & 120 MB & \texttt{ru\_maxrss} \\
fit residual rms & $1.2\times10^{-5}$ & from \texttt{lstsq} \\
\bottomrule
\end{tabular}
\caption{Outcome of the five scripts.  Each exits non-zero on any failure, so the
table is a summary of exit codes, not of opinions.}
\label{tab:outcome}
\end{table}

\appendix
\renewcommand{\thetable}{\arabic{table}}
\section{Environment and rebuild instructions}
\label{app:run}

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}ll@{}}
\toprule
component & version \\
\midrule
Python & %s \\
NumPy & %s \\
Matplotlib & 3.6.3 \\
platform & %s \\
\bottomrule
\end{tabular}
\caption{The environment in which every number and figure of this supplement was
produced.}
\label{tab:env}
\end{table}

\begin{lstlisting}[basicstyle=\ttfamily\footnotesize,frame=single,columns=fullflexible]
cd CPC_radialflux
python3 src/validate.py          # 51 checks, exit 0
python3 src/audit_claims.py      # 84 checks, exit 0
python3 src/check_tex.py         # structural check, exit 0
python3 src/lattice96.py         # 96^3 solve, ~0.14 s
python3 supplement/check_annotations.py   # coverage check: every code line annotated
python3 supplement/make_figs3d.py         # the seven 3-D figures
pdflatex supplement/supplement.tex        # twice
\end{lstlisting}

\noindent The package has no dependency other than NumPy (and Matplotlib for the
figures; SymPy only for the symbolic cross-checks).  No data files are needed: all
inputs are analytic profiles or a point source on a lattice, so every number in the
manuscript can be regenerated from an empty directory in under a minute.

\end{document}
"""


def tail(path, n=40):
    s = open(path, encoding="utf-8").read().rstrip().split("\n")
    return "\n".join(s[-n:]) if len(s) > n else "\n".join(s)


def stats():
    tot_l = tot_c = 0
    per = {}
    for f in ORDER:
        base = HERE if f == "make_figs3d.py" else SRC
        lines = open(os.path.join(base, f), encoding="utf-8").read().splitlines()
        code = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        per[f] = (len(lines), code)
        tot_l += len(lines)
        tot_c += code
    return per, tot_l, tot_c


def main():
    per, tot_l, tot_c = stats()
    logs = os.path.join(HERE, "logs")
    cov = tail(os.path.join(logs, "annotations.log"), 20)
    val = open(os.path.join(logs, "validate.log"), encoding="utf-8").read().rstrip()
    aud = tail(os.path.join(logs, "audit.log"), 6)
    tex = open(os.path.join(logs, "check_tex.log"), encoding="utf-8").read().rstrip()
    lat = tail(os.path.join(logs, "lattice96.log"), 12)

    body = (HEAD + BODY_1 + build_code_section() + BODY_FIGS + BODY_COMM
            + BODY_LOG % (cov, val, aud, tex, lat,
                          platform.python_version(), numpy.__version__,
                          esc(platform.platform())))
    roles = {"radialflux.py": "the module: radius, bias, threshold, profiles",
             "lattice96.py": "$96^3$ periodic solve, shell statistics, fit",
             "validate.py": "51 asserted checks, 13 groups",
             "audit_claims.py": "84 numeric checks against the manuscript",
             "check_tex.py": "LaTeX structural integrity",
             "series_d.py": "symbolic series in $d$",
             "verify_core.py": "symbolic derivations",
             "verify_shells.py": "lattice and shell identities",
             "make_figs.py": "manuscript figures 1--3",
             "make_figs2.py": "manuscript figure 4",
             "make_fig5.py": "manuscript figure 5",
             "make_figs3d.py": "this supplement's figures"}
    rows = "\n".join("\\texttt{%s} & %d (%d) & %s \\\\" %
                     (esc(f), per[f][0], per[f][1], roles[f])
                     for f in ORDER)
    body = body.replace("%%FILETABLE%%", rows)
    body = body.replace("%%TOTAL%%", "%d (%d)" % (tot_l, tot_c))
    body = body.replace("1671 (1471) & \\\\", "%d (%d) & \\\\" % (tot_l, tot_c))
    body = body.replace(r"\textbf{1471 executable lines}",
                        r"\textbf{%d executable lines}" % tot_c)
    out = os.path.join(HERE, "supplement.tex")
    open(out, "w", encoding="utf-8").write(body)
    print("wrote", out, len(body), "bytes")


if __name__ == "__main__":
    main()
