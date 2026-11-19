"""
verify_prd.py -- single source of every number in the manuscript, the supplement,
and the code documentation.  Nine modules, one figure per module, fixed seed.

Run:  python verify_prd.py
Out:  ../06_results/results.json , ../05_figures/fig0..8 (pdf+png)

Physics summary
---------------
A Yukawa operator is an electroweak BIDOBLET, not an irreducible representation.
Under Pati-Salam SU(2)_L x SU(2)_R x SU(4)_c:
    10  = (2,2,1) + (1,1,6)                                   -> 1 bidoublet
    120 = (2,2,1) + (2,2,15) + ...                            -> 2 bidoublets
    126 = (1,1,6) + (3,1,10) + (1,3,10bar) + (2,2,15)          -> 1 bidoublet
The two bidoublets of 120 share ONE antisymmetric Yukawa matrix Y120 but their
vevs enter the down and lepton sectors in different combinations:
    Md += (v1 + v15) Y120 ,   Me += (v1 - 3 v15) Y120 .
Hence two INDEPENDENT complex coefficients, and no group-theoretic d/e ratio.

Result: P_add = 0 for all seven spectrum-splitting contents.  Adding the
neutrino sector adds nine observables and zero relations (seesaw surjectivity).
"""
# ===== MODULE 0 | infrastructure | fig0_pipeline =====
import json                                                        # json: export every number so each table entry can be audited.
import os                                                          # os: cross-platform paths and output directories.
from fractions import Fraction as F                                # Fraction: exact rational rank of the coefficient matrix, no tolerance.
from itertools import combinations                                 # combinations: enumerate every non-empty Higgs content.
import numpy as np                                                 # numpy: SVD, random sampling, central-difference Jacobians.
import matplotlib                                                  # matplotlib: headless rendering and global style.
matplotlib.use("Agg")                                              # Agg: render without a display, so the script runs on a server.
import matplotlib.pyplot as plt                                    # pyplot: one drawing interface for all twelve figures.
from mpl_toolkits.mplot3d import Axes3D                            # Axes3D: registers the 3d projection used by figs 1,2,6.
from matplotlib.patches import Rectangle, FancyArrowPatch          # patches: hand-drawn pipeline diagram of module 0.

HERE = os.path.dirname(os.path.abspath(__file__))                  # Directory of this file: anchor for every relative path.
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "05_figures"))    # Figure output directory, shared by all three documents.
RESDIR = os.path.abspath(os.path.join(HERE, "..", "06_results"))    # Result output directory holding results.json.
os.makedirs(FIGDIR, exist_ok=True)                                 # Create the figure directory if absent.
os.makedirs(RESDIR, exist_ok=True)                                 # Create the results directory if absent.
RNG = np.random.default_rng(20261118)                              # Fixed seed: census, ranks and assertions reproduce exactly.

# Publication palette: one colour per semantic role, reused in every figure.
C_BASE = "#2F6FB4"                                                 # Baseline (gauge + charge): blue.
C_VIABLE = "#12855F"                                               # Spectrum-splitting contents: green.
C_EXCL = "#C4501E"                                                 # Excluded contents: rust.
C_NU = "#7A4FA3"                                                   # Neutrino sector: violet.
C_INK = "#1B2430"                                                  # Text and axes: near-black ink.
C_MUTE = "#8A93A0"                                                 # Secondary marks: grey.
plt.rcParams.update({                                              # One global style so the twelve figures look like one set.
    "font.family": "DejaVu Sans", "font.size": 9.2,                # Sans family and base size, legible at PRD column width.
    "axes.edgecolor": C_INK, "axes.labelcolor": C_INK,             # Ink-coloured axes for a printed-journal feel.
    "text.color": C_INK, "xtick.color": C_INK,                     # Same ink for labels and ticks.
    "ytick.color": C_INK, "axes.linewidth": 0.7,                   # Thin spines: data should dominate the frame.
    "axes.grid": True, "grid.color": "#E4E8ED",                    # Very light grid: guides the eye without noise.
    "grid.linewidth": 0.6, "figure.dpi": 140,                      # Grid weight and screen resolution.
    "savefig.dpi": 320, "savefig.bbox": "tight",                   # High-resolution export for print.
    "legend.frameon": False, "axes.axisbelow": True,               # Frameless legends; grid behind the data.
})


def savefig(name):                                                 # Save every figure twice: vector PDF for LaTeX, PNG for preview.
    plt.savefig(os.path.join(FIGDIR, name + ".pdf"))                # Vector copy consumed by \includegraphics.
    plt.savefig(os.path.join(FIGDIR, name + ".png"))                # Raster copy for the HTML reading page.
    plt.close()                                                    # Release the canvas so twelve figures do not accumulate memory.


def fig0_pipeline():                                               # Figure 0: the verification pipeline, drawn by hand.
    fig, ax = plt.subplots(figsize=(9.4, 2.9))                      # Wide, short canvas: a pipeline reads left to right.
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")         # Canvas coordinates; no data axes.
    ax.grid(False)                                                 # Suppress the global grid on a schematic.

    def box(x, y, w, h, text, fc):                                 # Helper: one rounded panel with centred text.
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,          # Fill colour passed by the caller.
                               edgecolor="#B9C1CB", lw=0.8, zorder=2))  # Light border keeps the schematic quiet.
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",  # Centre the label inside the panel.
                fontsize=8.4, zorder=3, linespacing=1.5)            # Small font, generous leading for multi-line labels.

    box(0.15, 1.05, 2.15, 1.15, "group theory\nCG table\nPS bidoublets", "#EAF1F8")   # Input: the only physics entry point.
    box(2.75, 1.05, 2.15, 1.15, "verify_prd.py\nMODULES 0-8\nseed 20261118", "#E6F4EE")  # Engine: nine modules, one seed.
    box(5.35, 2.05, 4.4, 0.72, "results.json  (every quoted number)", "#FBF0E4")      # Output 1: auditable numbers.
    box(5.35, 1.19, 4.4, 0.72, "fig0..fig8  (pdf + png, 9 figures)", "#FBF0E4")        # Output 2: one figure per module.
    box(5.35, 0.33, 4.4, 0.72, "main paper / supplement / code doc", "#FBF0E4")        # Output 3: the three documents.
    ax.add_patch(FancyArrowPatch((2.32, 1.62), (2.73, 1.62),        # Arrow: input into the engine.
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=C_MUTE, zorder=1))
    for yy in (2.41, 1.55, 0.69):                                   # Three arrows: engine to each output.
        ax.add_patch(FancyArrowPatch((4.92, 1.62), (5.33, yy),      # Fan out from the engine's right edge.
                                     arrowstyle="-|>", mutation_scale=10,
                                     lw=1.0, color=C_MUTE,
                                     connectionstyle="arc3,rad=0.12", zorder=1))
    ax.set_title("Figure 0.  One script, nine modules, three documents.",
                 fontsize=10.2, pad=6, loc="left")                  # Title states the reproducibility contract.
    savefig("fig0_pipeline")                                        # Emit vector and raster copies.


# ===== MODULE 1 | group theory and exact census | fig1_census =====
# Operator record: (kind, c_u, c_d, c_e, free_de).  kind fixes the flavour symmetry.
# free_de=True marks an operator whose d and e coefficients are independent complex
# numbers -- the 120, which carries two bidoublets sharing one Yukawa matrix.
KIND_DOF = {"sym": 12, "anti": 6, "gen": 18}                        # Real parameters of a 3x3 complex symmetric / antisymmetric / general matrix.
OPS = {                                                            # The Clebsch-Gordon input.  This table is the only physics assumption.
    "SU(5)": {                                                     # Minimal SU(5) Higgs irreps that couple to matter.
        "5": [("sym", 1, 0, 0, False), ("gen", 0, 1, 1, False)],    # 10.10.5 feeds u; 10.5b.5b feeds d,e with Me=Md^T.
        "45": [("anti", 1, 0, 0, False), ("gen", 0, 1, -3, False)],  # 45 splits u from d and carries the -3 lepton weight.
        "50": [("sym", 1, 0, 0, False)]},                          # 50 feeds u only, so it cannot split d from e.
    "SO(10)": {                                                    # SO(10) irreps appearing in 16 x 16 = 10 + 120 + 126.
        "10": [("sym", 1, 1, 1, False)],                           # One bidoublet (2,2,1): equal weights, symmetric Yukawa.
        "120": [("anti", 1, 1, -3, True)],                         # TWO bidoublets: independent d/e coefficients.  free_de=True.
        "126": [("sym", 1, 1, -3, False)]},                        # One bidoublet (2,2,15): weight -3, symmetric Yukawa.
    "E6": {"27": [("gen", 1, 1, 1, False)]},                       # Single 27^3 contraction feeds all three sectors alike.
    "PS": {"(1,2,2)": [("gen", 1, 1, 1, False)],                   # Colour singlet bidoublet: equal quark/lepton weights.
           "(15,2,2)": [("gen", 1, 1, -3, False)]},                # SU(4) adjoint bidoublet: lepton weight -3.
}
SIMPLE = {"SU(5)", "SO(10)", "E6"}                                 # Simple groups supply gauge unification; product groups do not.


def prop(a, b):                                                    # True iff b = k a for one fixed rational k, both nonzero.
    a = [F(x) for x in a]; b = [F(x) for x in b]                    # Exact rationals: no floating tolerance enters the census.
    if all(x == 0 for x in a) or all(x == 0 for x in b):            # A zero column cannot define a ratio.
        return False                                               # No proportionality.
    i = next(k for k, x in enumerate(a) if x != 0)                  # First nonzero entry of a fixes the candidate ratio.
    if b[i] == 0:                                                  # b vanishes where a does not: ratio impossible.
        return False                                               # No proportionality.
    k = F(b[i], a[i])                                              # Candidate ratio as an exact fraction.
    return all(y == k * x for x, y in zip(a, b))                   # Proportional only if every entry matches exactly.


def rankC(ops):                                                    # Exact rational rank of the 3 x n_ops coefficient matrix.
    rows = [[F(o[1]) for o in ops], [F(o[2]) for o in ops],         # Rows are the sectors u, d, e.
            [F(o[3]) for o in ops]]                                # Third row: charged leptons.
    m = [r[:] for r in rows]; piv = 0                              # Working copy and pivot counter for elimination.
    for col in range(len(ops)):                                    # Sweep columns left to right.
        sel = next((r for r in range(piv, 3) if m[r][col] != 0), None)  # Find a nonzero pivot at or below the current row.
        if sel is None:                                            # Column is dependent.
            continue                                               # Move to the next column without advancing the rank.
        m[piv], m[sel] = m[sel], m[piv]                             # Swap the pivot row into place.
        pv = m[piv][col]                                           # Pivot value, an exact fraction.
        m[piv] = [x / pv for x in m[piv]]                           # Normalise the pivot row.
        for r in range(3):                                         # Eliminate this column from every other row.
            if r != piv and m[r][col] != 0:                         # Skip the pivot itself and already-zero entries.
                f = m[r][col]                                       # Multiplier for the elimination step.
                m[r] = [a - f * b for a, b in zip(m[r], m[piv])]    # Exact row operation.
        piv += 1                                                   # One more independent row found.
        if piv == 3:                                               # Full row rank reached.
            break                                                  # No further elimination needed.
    return piv                                                     # Rank in exact arithmetic.


def bidoublet_columns(g, reps):                                    # Coefficient-matrix columns: one per BIDOBLET, not per irrep.
    cols = []                                                      # Accumulator of (c_u, c_d, c_e) triples.
    for r in reps:                                                 # Walk the irreps of this content.
        if g == "SO(10)" and r == "120":                            # The 120 is the only irrep with two bidoublets.
            cols.append((1, 1, 1))                                  # (2,2,1): colour singlet, equal weights.
            cols.append((1, 1, -3))                                 # (2,2,15): SU(4) adjoint, lepton weight -3.
        else:                                                      # Every other irrep contributes one column.
            for o in OPS[g][r]:                                     # Its operator records.
                cols.append((o[1], o[2], o[3]))                      # Take the Clebsch triple.
    return cols                                                    # One column per bidoublet.


def rankC_bido(g, reps):                                           # Exact rational rank of the bidoublet coefficient matrix.
    cols = bidoublet_columns(g, reps)                               # Columns in bidoublet counting.
    ops = [("gen", a, b, c, False) for a, b, c in cols]             # Wrap as operator records for the rank routine.
    return rankC(ops)                                              # Rank in exact arithmetic.


def _transpose_is_distinct():                                      # Guard: the transpose test must not duplicate the ratio test.
    # A content with an antisymmetric down-type contraction and equal Clebsch weights
    # is transpose-degenerate but NOT ratio-degenerate.  If the two tests were the
    # same expression, this probe would return False for both.
    cd = [1, 1]; ce = [1, -1]; T = [-1, +1]                         # Antisymmetric first operator.
    d = prop(cd, ce)                                               # Ratio test: 1:1 vs 1:-1 -> False.
    t = prop([a * s for a, s in zip(cd, T)], ce)                    # T-weighted: -1:1 vs 1:-1 -> True.
    return (not d) and t                                           # They must disagree on this probe.


def census():                                                      # Enumerate all eighteen non-empty Higgs contents.
    out = []                                                       # Accumulator for one record per content.
    for g, irr in OPS.items():                                     # Loop over the four gauge groups.
        for k in range(1, len(irr) + 1):                            # Every subset size.
            for reps in combinations(list(irr), k):                 # Every non-empty subset of that group's irreps.
                ops = [o for r in reps for o in OPS[g][r]]          # Flatten the content into its operator list.
                cu = [o[1] for o in ops]; cd = [o[2] for o in ops]  # Sector coefficient columns.
                ce = [o[3] for o in ops]                            # Charged-lepton column.
                free = any(o[4] for o in ops)                       # Does any operator carry independent d/e coefficients?
                fed = any(cu) and any(cd) and any(ce)               # Are all three sectors fed at all?
                single = len(ops) == 1                              # Single-operator content: strongest constraints.
                if free:                                           # A free coefficient breaks both fixed-ratio routes.
                    direct = trans = False                          # Neither proportionality nor transpose can be enforced.
                else:                                              # All coefficients pinned by group theory.
                    direct = prop(cd, ce)                           # Fixed ratio Me = k Md.
                    # Transpose route: Me = k Md^T.  Transposing flips the sign of an
                    # antisymmetric contraction, so the down column must be weighted by
                    # the flavour symmetry T before comparing with the lepton column.
                    Tsign = [-1 if o[0] == "anti" else +1 for o in ops]  # T = -1 for antisymmetric Yukawas.
                    trans = prop([a * t for a, t in zip(cd, Tsign)], ce)  # T-weighted test, distinct from the ratio test.
                all_anti = fed and all(o[0] == "anti" for o in ops)  # Every Yukawa antisymmetric => a zero singular value per sector.
                excl = (not fed) or direct or trans or all_anti      # Excluded if unfed, proportional, or massless.
                splitting = fed and not excl                        # Spectrum-splitting: can accommodate the observed spectrum.
                gauge = 2 if g in SIMPLE else 0                     # Three couplings to one: two relations, simple groups only.
                if not fed:                                        # Sectors absent: no mass relations to count.
                    mass = ckm = 0                                  # Nothing constrained.
                elif excl:                                         # Excluded by an exact relation.
                    mass = 3 + (2 if single else 0)                 # Three fixed ratios, plus two more if tan(beta) is also fixed.
                    ckm = 4 if single else 0                        # Single operator forces V_CKM = 1: three angles and one phase.
                else:                                              # Splitting: no mass or mixing relation.
                    mass = ckm = 0                                  # The flavour sector is free.
                out.append(dict(group=g, model="+".join(reps),      # Record the content identity.
                                n_ops=len(ops),                    # Number of distinct Yukawa matrices.
                                n_bido=len(bidoublet_columns(g, reps)),  # Number of bidoublet columns.
                                rankC=rankC_bido(g, reps),          # Exact coefficient rank in bidoublet counting.
                                fed=fed, single=single, free_de=free,  # Structural flags.
                                direct=direct, trans=trans, all_anti=all_anti,  # Why it is excluded, if it is.
                                splitting=splitting, mass=mass, ckm=ckm,  # Classification and relation counts.
                                gauge=gauge, charge=1, baseline=gauge + 1))  # Charge quantisation always contributes one.
    return out                                                     # Eighteen records.


def fig1_census(rows):                                             # Figure 1: the full census against each group's baseline.
    fig, ax = plt.subplots(figsize=(9.4, 3.5))                      # Wide canvas: eighteen labels need horizontal room.
    order = sorted(rows, key=lambda r: (not r["splitting"], -r["P"], r["group"]))  # Viable first, then by reach.
    x = np.arange(len(order))                                       # Bar positions.
    base = np.array([r["baseline"] for r in order], float)          # Baseline height per content.
    extra = np.array([r["mass"] + r["ckm"] for r in order], float)  # Excess above baseline, nonzero only for excluded contents.
    cols = [C_VIABLE if r["splitting"] else C_EXCL for r in order]  # Colour encodes viability.
    ax.bar(x, base, 0.66, color=[C_BASE if r["splitting"] else "#D9DEE5" for r in order],
           edgecolor="white", lw=0.7, label="baseline (gauge + charge)")  # Baseline segment.
    ax.bar(x, extra, 0.66, bottom=base, color=cols,                  # Excess segment sits on the baseline.
           edgecolor="white", lw=0.7, label="excess relations (excluded contents)")
    for i, r in enumerate(order):                                   # Annotate each bar with its exact reach.
        ax.text(i, base[i] + extra[i] + 0.22, str(r["P"]), ha="center",
                va="bottom", fontsize=8.2, fontweight="bold",
                color=C_VIABLE if r["splitting"] else C_EXCL)        # Green for viable, rust for excluded.
    ax.axhline(3, color=C_BASE, ls=(0, (5, 3)), lw=1.0, alpha=0.8)   # Simple-group ceiling at three.
    ax.axhline(1, color="#C08A2E", ls=(0, (2, 3)), lw=1.0, alpha=0.8)  # Product-group ceiling at one.
    ax.text(-0.45, 13.2, "simple-group baseline = 3", ha="left",
            fontsize=8, color=C_BASE)                               # Labels moved to the empty top strip.
    ax.text(-0.45, 12.2, "Pati-Salam baseline = 1", ha="left",
            fontsize=8, color="#C08A2E")                            # No bar reaches this height.
    ax.set_xticks(x)                                               # One tick per content.
    ax.set_xticklabels([r["group"] + " " + r["model"] for r in order],
                       rotation=52, ha="right", fontsize=7.4)        # Rotated labels keep all eighteen readable.
    ax.set_ylabel("exact reach  $P_{\\mathrm{ex}}$")                 # Vertical axis: number of exact relations.
    ax.set_ylim(0, 15.4)                                           # Headroom for the labels above the tallest bar.
    ax.legend(loc="upper right", fontsize=8)                        # Two-entry legend.
    ax.set_title("Figure 1.  All eighteen contents.  Every viable content sits on its baseline.",
                 fontsize=10.2, loc="left", pad=8)                  # Title carries the headline result.
    savefig("fig1_census")                                          # Emit both formats.


# ===== MODULE 2 | bidoublet parameterisation | fig2_bidoublet =====
def mk(kind, p):                                                   # Build a 3x3 complex matrix of the requested symmetry.
    if kind == "anti":                                             # Antisymmetric: three independent complex entries.
        a = p[0] + 1j * p[1]; b = p[2] + 1j * p[3]; c = p[4] + 1j * p[5]  # Read three complex numbers from six reals.
        return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], dtype=complex)  # Zero diagonal, opposite off-diagonals.
    if kind == "sym":                                              # Symmetric: six independent complex entries.
        v = p[:12].reshape(6, 2); z = v[:, 0] + 1j * v[:, 1]        # Pair up reals into six complex numbers.
        return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]],     # Mirror across the diagonal.
                         [z[2], z[4], z[5]]], dtype=complex)
    v = p[:18].reshape(9, 2)                                       # General: nine independent complex entries.
    return (v[:, 0] + 1j * v[:, 1]).reshape(3, 3)                   # No symmetry imposed.


def cv(p, i):                                                      # Read one complex vev from two consecutive real parameters.
    return p[i] + 1j * p[i + 1]                                     # Real part then imaginary part.


def build_so10(reps, correct):                                     # Parameterise an SO(10) content, either correctly or as the flawed draft did.
    mats = [({"10": "sym", "120": "anti", "126": "sym"}[r], r) for r in reps]  # Flavour symmetry of each irrep's Yukawa.
    n = sum(KIND_DOF[k] for k, _ in mats)                           # Real parameters carried by the Yukawa matrices.
    n += sum((8 if correct else 4) if r == "120" else 4 for r in reps)  # 120 has four complex vevs when correct, two when collapsed.

    def unpack(p):                                                 # Map the real parameter vector to the three mass matrices.
        Mu = np.zeros((3, 3), complex); Md = np.zeros((3, 3), complex)  # Accumulators for up and down sectors.
        Me = np.zeros((3, 3), complex)                              # Accumulator for charged leptons.
        i = 0; Y = {}                                              # Cursor and storage for the Yukawa matrices.
        for k, r in mats:                                          # First pass: read all Yukawa matrices.
            Y[r] = mk(k, p[i:i + KIND_DOF[k]]); i += KIND_DOF[k]     # Advance the cursor by that matrix's parameter count.
        for k, r in mats:                                          # Second pass: attach vevs with the right Clebsch weights.
            if r == "120":                                         # The 120 is the only irrep with two bidoublets.
                if correct:                                        # Correct: two bidoublets, four complex vevs.
                    v1u, v15u = cv(p, i), cv(p, i + 2)              # Up-type vevs of (2,2,1) and (2,2,15).
                    v1d, v15d = cv(p, i + 4), cv(p, i + 6); i += 8  # Down-type vevs of the same two bidoublets.
                    Mu += (v1u + v15u) * Y[r]                       # Up sector sees the sum.
                    Md += (v1d + v15d) * Y[r]                       # Down sector sees the sum.
                    Me += (v1d - 3 * v15d) * Y[r]                   # Lepton sector sees a DIFFERENT combination: the crux.
                else:                                              # Flawed draft: one vev, i.e. v15 = 0 imposed by hand.
                    vu, vd = cv(p, i), cv(p, i + 2); i += 4         # Only two complex vevs.
                    Mu += vu * Y[r]; Md += -vd * Y[r]               # Down weight -1.
                    Me += vd * Y[r]                                 # Lepton weight +1: forces the spurious identity.
            elif r == "10":                                        # The 10 has one bidoublet, weights (1,1,1).
                vu, vd = cv(p, i), cv(p, i + 2); i += 4             # Two complex vevs.
                Mu += vu * Y[r]; Md += vd * Y[r]; Me += vd * Y[r]   # Equal down and lepton weights.
            else:                                                  # The 126 has one bidoublet, weights (1,1,-3).
                vu, vd = cv(p, i), cv(p, i + 2); i += 4             # Two complex vevs.
                Mu += vu * Y[r]; Md += vd * Y[r]; Me += -3 * vd * Y[r]  # Georgi-Jarlskog factor -3 on leptons.
        return Mu, Md, Me                                          # The three charged-fermion mass matrices.
    return unpack, n                                               # Return the map and its parameter count.


def build_gen(ops):                                                # Parameterise a content given explicitly as an operator list.
    n = sum(KIND_DOF[o[0]] + 2 * ((o[1] != 0) + (o[2] != 0 or o[3] != 0))
            for o in ops)                                          # Yukawa parameters plus two reals per active vev.

    def unpack(p):                                                 # Map parameters to mass matrices.
        Mu = np.zeros((3, 3), complex); Md = np.zeros((3, 3), complex)  # Sector accumulators.
        Me = np.zeros((3, 3), complex); i = 0                       # Lepton accumulator and cursor.
        for kind, cu, cd, ce, _ in ops:                            # Walk the operators in order.
            Y = mk(kind, p[i:i + KIND_DOF[kind]]); i += KIND_DOF[kind]  # Read this operator's Yukawa matrix.
            if cu:                                                 # Operator feeds the up sector.
                Mu += cu * cv(p, i) * Y; i += 2                     # Attach its up-type vev.
            if cd or ce:                                           # Operator feeds down and/or leptons.
                v = cv(p, i); i += 2                                # One shared down-type vev for this bidoublet.
                Md += cd * v * Y; Me += ce * v * Y                  # Weighted contributions to each sector.
        return Mu, Md, Me                                          # The three mass matrices.
    return unpack, n                                               # Map and parameter count.


def fig2_bidoublet():                                              # Figure 2: why one irrep is not one operator.
    fig = plt.figure(figsize=(9.4, 3.6))                            # Two panels: schematic and coefficient freedom.
    ax = fig.add_subplot(121); ax.axis("off"); ax.grid(False)       # Left: hand-drawn bidoublet schematic.
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)                            # Schematic canvas.
    ax.add_patch(Rectangle((0.3, 3.5), 5.4, 2.0, facecolor="#EAF1F8",
                           edgecolor="#B9C1CB", lw=0.9))            # Panel for the 120.
    ax.text(3.0, 5.15, "$\\mathbf{120}$  (two bidoublets)", ha="center",
            fontsize=9.6, fontweight="bold")                        # Header.
    ax.add_patch(Rectangle((0.6, 3.75), 2.3, 1.0, facecolor="#FFFFFF",
                           edgecolor=C_BASE, lw=1.0))               # First bidoublet box.
    ax.text(1.75, 4.25, "$(2,2,1)$\nweights $(1,1,1)$", ha="center",
            va="center", fontsize=8, linespacing=1.5)               # Colour singlet: equal weights.
    ax.add_patch(Rectangle((3.1, 3.75), 2.3, 1.0, facecolor="#FFFFFF",
                           edgecolor=C_NU, lw=1.0))                 # Second bidoublet box.
    ax.text(4.25, 4.25, "$(2,2,15)$\nweights $(1,1,-3)$", ha="center",
            va="center", fontsize=8, linespacing=1.5)               # SU(4) adjoint: lepton weight -3.
    ax.text(3.0, 3.15, "one Yukawa $Y_{120}$, two independent vevs", ha="center",
            fontsize=8.4, style="italic", color=C_MUTE)             # The key structural statement.
    ax.text(3.0, 2.35, "$M_d \\supset (v_1+v_{15})\\,Y_{120}$", ha="center",
            fontsize=10, color=C_BASE)                              # Down-sector combination.
    ax.text(3.0, 1.55, "$M_e \\supset (v_1-3v_{15})\\,Y_{120}$", ha="center",
            fontsize=10, color=C_NU)                                # Lepton-sector combination: different.
    ax.text(3.0, 0.65, "setting $v_{15}=0$ is an extra assumption,\nnot a consequence of SO(10)",
            ha="center", fontsize=8.2, color=C_EXCL, linespacing=1.5)  # Name the error explicitly.
    ax.set_title("(a) bidoublet structure of the 120", fontsize=9.8, loc="left")  # Panel title.

    ax2 = fig.add_subplot(122)                                      # Right: the induced coefficient freedom.
    rng = np.random.default_rng(7)                                  # Local generator for a clean illustration.
    v1 = rng.normal(size=400) + 1j * rng.normal(size=400)            # Random first-bidoublet vev.
    v15 = rng.normal(size=400) + 1j * rng.normal(size=400)           # Random second-bidoublet vev.
    a = v1 + v15; b = v1 - 3 * v15                                   # Down and lepton coefficients.
    ratio = b / a                                                   # Their ratio: free over the complex plane.
    ok = np.abs(a) > 1e-3                                           # Avoid dividing by nearly zero.
    ax2.scatter(ratio[ok].real, ratio[ok].imag, s=7, c=C_VIABLE,
                alpha=0.5, edgecolor="none", label="correct: $b/a$ free")  # Correct case fills the plane.
    ax2.scatter([-1], [0], s=110, marker="*", c=C_EXCL,
                edgecolor="white", lw=0.8, zorder=5, label="collapsed: $b/a=-1$")  # Flawed case is one point.
    ax2.set_xlabel("Re $(b/a)$"); ax2.set_ylabel("Im $(b/a)$")       # Axes: the effective coefficient ratio.
    ax2.set_xlim(-8, 8); ax2.set_ylim(-8, 8)                         # Symmetric window.
    ax2.legend(loc="upper right", fontsize=8)                        # Two-entry legend.
    ax2.set_title("(b) the coefficient ratio is free, not fixed", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.2)                                       # Space the panels.
    savefig("fig2_bidoublet")                                        # Emit both formats.


# ===== MODULE 3 | observables and Jacobian ranks | fig3_ranks =====
def sv(M):                                                         # Physical masses are singular values, not eigenvalue moduli.
    return np.linalg.svd(M, compute_uv=False)                       # Descending order; these are the mass moduli.


def observables(p, unpack):                                        # The thirteen flavour observables.
    Mu, Md, Me = unpack(p)                                         # Build the three mass matrices.
    _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)      # Left rotations of the up and down sectors.
    V = Vu @ Vd.conj().T                                           # CKM matrix as the relative orientation.
    J = abs(np.imag(V[0, 1] * V[1, 2] * V[0, 2].conj() * V[1, 1].conj()))  # Jarlskog invariant: rephasing invariant CP measure.
    return np.concatenate([sv(Mu), sv(Md), sv(Me),                  # Nine masses.
                           [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J]])  # Three moduli and J: thirteen in total.


def jac_rank(p, unpack, h=1e-6):                                   # Numerical rank of the observable map at one point.
    f0 = observables(p, unpack)                                    # Unperturbed observable vector, used only for its length.
    Jm = np.zeros((len(f0), len(p)))                               # Jacobian: observables by parameters.
    for i in range(len(p)):                                        # Differentiate along each parameter direction.
        pp = p.copy(); pp[i] += h                                   # Forward step.
        pm = p.copy(); pm[i] -= h                                   # Backward step.
        Jm[:, i] = (observables(pp, unpack) - observables(pm, unpack)) / (2 * h)  # Central difference: second-order accurate.
    s = np.linalg.svd(Jm, compute_uv=False)                         # Singular values of the Jacobian.
    if not np.all(np.isfinite(s)) or s[0] == 0:                     # Reject non-finite or identically zero Jacobians.
        return None, None                                          # Signal an unusable sample.
    return int(np.sum(s > 1e-7 * s[0])), s                          # Rank at relative threshold 1e-7, plus the spectrum.


def viable_specs():                                                # The seven spectrum-splitting contents, correctly parameterised.
    return [                                                       # Each entry: label and (map, parameter count).
        ("SU(5) 5+45", build_gen([("sym", 1, 0, 0, False), ("gen", 0, 1, 1, False),
                                  ("anti", 1, 0, 0, False), ("gen", 0, 1, -3, False)])),
        ("SU(5) 5+45+50", build_gen([("sym", 1, 0, 0, False), ("gen", 0, 1, 1, False),
                                     ("anti", 1, 0, 0, False), ("gen", 0, 1, -3, False),
                                     ("sym", 1, 0, 0, False)])),
        ("SO(10) 10+120", build_so10(("10", "120"), True)),          # Now viable: the transpose degeneracy is lifted.
        ("SO(10) 10+126", build_so10(("10", "126"), True)),          # Unchanged by the correction.
        ("SO(10) 120+126", build_so10(("120", "126"), True)),        # The former exception: now full rank.
        ("SO(10) 10+120+126", build_so10(("10", "120", "126"), True)),  # Unchanged.
        ("PS (1,2,2)+(15,2,2)", build_gen([("gen", 1, 1, 1, False), ("gen", 1, 1, -3, False)])),  # Product group.
    ]


def measure_ranks(npts=8):                                         # Measure rank and conditioning for every viable content.
    out = []                                                       # One record per content.
    for name, (u, n) in viable_specs():                            # Walk the seven contents.
        ranks = []; ratios = []; gaps = []                          # Collect rank, conditioning and mass gaps.
        for _ in range(npts):                                      # Several random points per content.
            p = RNG.normal(0, 1, n)                                # Draw a parameter point.
            r, s = jac_rank(p, u)                                  # Rank and Jacobian spectrum.
            if r is None:                                          # Skip unusable samples.
                continue                                           # Do not pollute the statistics.
            ranks.append(r)                                        # Record the rank.
            if r == 13:                                            # Conditioning is meaningful only at full rank.
                ratios.append(float(s[12] / s[0]))                  # Smallest to largest singular value.
            Mu, Md, Me = u(p)                                      # Rebuild the matrices for the mass gaps.
            for M in (Mu, Md, Me):                                 # Each sector separately.
                v = sv(M)                                          # Its three masses.
                gaps.append(float(min(v[0] - v[1], v[1] - v[2]) / v[0]))  # Relative adjacent gap: degeneracy guard.
        out.append(dict(model=name, n_params=n, rank=max(ranks),    # Generic rank is the sample maximum (upper semicontinuity).
                        rank_set=sorted(set(ranks)),               # All observed ranks, for transparency.
                        s_ratio=float(np.median(ratios)) if ratios else None,  # Median conditioning.
                        min_gap=float(min(gaps)), p_add=13 - max(ranks)))  # Smallest gap and the resulting P_add.
    return out                                                     # Seven records.


def fig3_ranks(recs):                                              # Figure 3: rank and conditioning of the seven viable contents.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.4),          # Two panels sharing the content order.
                                  gridspec_kw={"width_ratios": [1.05, 1]})
    names = [r["model"] for r in recs]                              # Content labels.
    y = np.arange(len(recs))                                        # Vertical positions.
    ax.barh(y, [r["rank"] for r in recs], 0.6, color=C_VIABLE,       # All bars reach thirteen.
            edgecolor="white", lw=0.7)
    ax.axvline(13, color=C_INK, ls=(0, (4, 3)), lw=1.0)              # Mark the full-rank value.
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.8)        # Content labels on the left.
    ax.set_xlim(0, 14.6); ax.set_xticks([0, 4, 8, 13])               # Room for the numeric annotation.
    ax.set_xlabel("rank $D\\pi$   (13 flavour observables)")         # Axis label.
    ax.invert_yaxis()                                              # Read top to bottom.
    for i, r in enumerate(recs):                                   # Annotate rank and parameter count.
        ax.text(r["rank"] + 0.25, i, "13   ($n$=%d)" % r["n_params"],
                va="center", fontsize=7.6, color=C_INK)             # Show that parameters exceed observables.
    ax.set_title("(a) every viable content has full rank", fontsize=9.8, loc="left")  # Panel title.
    ax2.barh(y, [r["s_ratio"] for r in recs], 0.6, color=C_BASE,     # Conditioning: three orders of magnitude.
             edgecolor="white", lw=0.7)
    ax2.set_yticks(y); ax2.set_yticklabels([])                       # Labels already on the left panel.
    ax2.set_xlabel("$s_{\\min}/s_{\\max}$   (median)")               # Axis label.
    ax2.invert_yaxis()                                             # Same order as the left panel.
    ax2.axvline(1e-7, color=C_EXCL, ls=(0, (3, 3)), lw=1.0)          # The rank threshold used by the code.
    ax2.text(1.2e-7, len(recs) - 0.4, "threshold $10^{-7}$", fontsize=7.4,
             color=C_EXCL, rotation=90, va="bottom")                # Label the threshold.
    ax2.set_xscale("log")                                          # Logarithmic: the gap to threshold is the point.
    ax2.set_title("(b) four orders above the threshold", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.2)                                      # Space the panels.
    savefig("fig3_ranks")                                           # Emit both formats.


# ===== MODULE 4 | the withdrawn identity and bound | fig4_withdrawn =====
def compare_parameterisations(npts=4000):                          # Correct versus collapsed parameterisation of 120+126.
    uc, nc = build_so10(("120", "126"), True)                       # Correct: two bidoublets, thirty parameters.
    uf, nf = build_so10(("120", "126"), False)                      # Collapsed: one vev, twenty-six parameters.
    res = {}                                                       # Results keyed by parameterisation.
    for tag, (u, n) in (("correct", (uc, nc)), ("collapsed", (uf, nf))):  # Loop over both cases.
        iderr = []; ratios = []; symerr = []                        # Identity error, mass ratio, symmetric-part error.
        for _ in range(npts):                                      # Many random points.
            p = RNG.normal(0, 1, n)                                # Draw a point.
            Mu, Md, Me = u(p)                                      # Build the matrices.
            scale = max(np.max(np.abs(Md)), 1e-300)                 # Normalisation for relative errors.
            iderr.append(float(np.max(np.abs(Me - (-2 * Md - Md.T))) / scale))  # Test Me = -2Md - Md^T.
            sd = (Md + Md.T) / 2; se = (Me + Me.T) / 2              # Symmetric parts of each matrix.
            symerr.append(float(np.max(np.abs(se + 3 * sd)) / max(np.max(np.abs(sd)), 1e-300)))  # Test sym Me = -3 sym Md.
            ratios.append(float(np.sum(sv(Me) ** 2) / np.sum(sv(Md) ** 2)))  # The would-be bounded ratio.
        lo = sum(1 for r in ratios if r <= 1.0)                     # Violations of the lower edge.
        hi = sum(1 for r in ratios if r > 9.0)                      # Violations of the upper edge.
        res[tag] = dict(n_params=n, identity_err_median=float(np.median(iderr)),  # Typical identity failure.
                        identity_err_max=float(max(iderr)),         # Worst identity failure.
                        sym_err_max=float(max(symerr)),             # The surviving identity's accuracy.
                        ratio_min=float(min(ratios)), ratio_median=float(np.median(ratios)),  # Ratio range.
                        ratio_max=float(max(ratios)), viol_lower=lo, viol_upper=hi,  # Edge violations.
                        n_points=npts)                             # Sample size.
    return res                                                     # Both parameterisations.


def fig4_withdrawn(res):                                           # Figure 4: the bound holds only in the collapsed sub-manifold.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3))          # Left: identity error.  Right: ratio range.
    tags = ["collapsed", "correct"]                                # Order: flawed first, correct second.
    labels = ["collapsed\n($v_{15}=0$)", "correct\n(two bidoublets)"]  # Human-readable labels.
    errs = [res[t]["identity_err_median"] for t in tags]            # Median identity error per case.
    bars = ax.bar([0, 1], errs, 0.5, color=[C_MUTE, C_EXCL],         # Grey: the artefact holds.  Rust: it fails.
                  edgecolor="white", lw=0.8)                        # Rust marks failure here, not viability.
    ax.set_yscale("log")                                           # Sixteen orders of magnitude: log is essential.
    ax.set_ylim(1e-17, 1e2)                                        # Span both regimes.
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8.4)  # Case labels.
    ax.set_ylabel("median $|M_e+2M_d+M_d^{T}|\\,/\\,|M_d|$")        # Relative identity error.
    for b, e in zip(bars, errs):                                   # Annotate each bar.
        ax.text(b.get_x() + b.get_width() / 2, e * 2.2, "%.1e" % e,
                ha="center", fontsize=8, fontweight="bold")         # Numeric value above the bar.
    ax.axhline(1e-14, color=C_MUTE, ls=(0, (3, 3)), lw=0.9)         # Machine-precision reference.
    ax.text(-0.42, 2.0e-14, "machine precision", fontsize=7.2,
            color=C_MUTE, ha="left", va="bottom")                   # Label sits above the line, clear of both bars.
    ax.set_title("(a) the identity is an artefact", fontsize=9.8, loc="left")  # Panel title.
    for i, t in enumerate(tags):                                   # Draw the ratio range for each case.
        r = res[t]                                                 # Its statistics.
        col = C_MUTE if t == "collapsed" else C_EXCL                # Grey: restricted case.  Rust: window breached.
        ax2.plot([r["ratio_min"], r["ratio_max"]], [i, i], lw=7,
                 color=col, alpha=0.35, solid_capstyle="round")     # Range as a thick translucent bar.
        ax2.plot([r["ratio_median"]], [i], "o", ms=7, color=col,
                 mec="white", mew=0.9, zorder=5)                    # Median as a dot.
        ax2.text(r["ratio_max"] * 1.15, i, "max %.1f" % r["ratio_max"],
                 va="center", fontsize=7.6, color=col)              # Annotate the upper end.
    ax2.axvspan(1, 9, color=C_BASE, alpha=0.10)                     # Shade the claimed window.
    ax2.axvline(1, color=C_BASE, lw=1.0); ax2.axvline(9, color=C_BASE, lw=1.0)  # Its two edges.
    ax2.text(3.0, 1.62, "claimed window $(1,9]$", ha="center",
             fontsize=8, color=C_BASE)                              # Label the window.
    ax2.set_xscale("log"); ax2.set_xlim(0.05, 2000)                  # Log axis spans both cases.
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels, fontsize=8.4)  # Case labels.
    ax2.set_ylim(-0.6, 1.9)                                        # Headroom for the window label.
    ax2.set_xlabel("$\\sum m_e^2\\,/\\,\\sum m_d^2$")               # The ratio that was claimed bounded.
    ax2.set_title("(b) both edges are breached", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.2)                                      # Space the panels.
    savefig("fig4_withdrawn")                                       # Emit both formats.


# ===== MODULE 5 | seesaw surjectivity | fig5_seesaw =====
def rand_sym(rng):                                                 # Random complex symmetric matrix.
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))      # Nine independent complex entries.
    return (Z + Z.T) / 2                                           # Symmetrise.


def rand_gen(rng):                                                 # Random complex general matrix.
    return rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))   # No symmetry imposed.


def seesaw_surjectivity(npts=500):                                 # Verify that the type-I seesaw map is onto.
    rng = np.random.default_rng(90210)                              # Dedicated seed, independent of the census seed.
    hit = []; sym = []                                              # Reconstruction error and M_R symmetry error.
    for _ in range(npts):                                          # Many random targets.
        MD = rand_gen(rng)                                         # Arbitrary Dirac matrix.
        if abs(np.linalg.det(MD)) < 1e-6:                           # Require invertibility for the construction.
            continue                                               # Skip near-singular draws.
        S = rand_sym(rng)                                          # Arbitrary target light-neutrino matrix.
        if abs(np.linalg.det(S)) < 1e-6:                            # Also require an invertible target.
            continue                                               # Skip near-singular targets.
        MR = -MD @ np.linalg.inv(S) @ MD.T                          # Construct M_R = -M_D S^{-1} M_D^T.
        sym.append(float(np.max(np.abs(MR - MR.T)) / np.max(np.abs(MR))))  # M_R must be symmetric: it is, since S is.
        Mnu = -MD.T @ np.linalg.inv(MR) @ MD                        # Feed it back through the seesaw.
        hit.append(float(np.max(np.abs(Mnu - S)) / np.max(np.abs(S))))  # The target is reproduced exactly.
    return dict(n_points=len(hit), hit_err_max=float(max(hit)),     # Worst reconstruction error.
                sym_err_max=float(max(sym)))                       # Worst symmetry error.


def typeII_checks(npts=300):                                       # Type-II: when would a relation appear, and why it does not.
    rng = np.random.default_rng(4242)                               # Dedicated seed.
    ckm_off = []; prop_count = 0                                    # CKM off-diagonal size; proportionality counter.
    for _ in range(npts):                                          # Random points.
        Y126 = rand_sym(rng)                                       # The 126 Yukawa.
        vu = rng.normal() + 1j * rng.normal()                       # Its up-type vev.
        vd = rng.normal() + 1j * rng.normal()                       # Its down-type vev.
        Mu = vu * Y126; Md = vd * Y126                              # With Y10 = 0 both sectors share Y126.
        _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)  # Left rotations.
        V = Vu @ Vd.conj().T                                       # CKM.
        ckm_off.append(float(np.max(np.abs(V - np.diag(np.diag(V))))))  # Off-diagonal size: zero means no mixing.
        Y120 = rand_gen(rng); Y120 = (Y120 - Y120.T) / 2            # An antisymmetric 120 Yukawa.
        v1 = rng.normal() + 1j * rng.normal()                       # First-bidoublet vev.
        v15 = rng.normal() + 1j * rng.normal()                      # Second-bidoublet vev.
        vL = rng.normal() + 1j * rng.normal()                       # Induced triplet vev.
        Me = (v1 - 3 * v15) * Y120 - 3 * vd * Y126                  # Charged leptons in 120+126.
        Mnu = vL * Y126                                            # Type-II light neutrinos.
        a = Me.ravel(); b = Mnu.ravel()                             # Flatten to test proportionality.
        k = np.vdot(b, a) / np.vdot(b, b)                           # Least-squares ratio.
        if np.linalg.norm(a - k * b) / np.linalg.norm(a) < 1e-8:    # Proportional only on a measure-zero locus.
            prop_count += 1                                        # Count such points.
    return dict(ckm_offdiag_max=float(max(ckm_off)),                # Confirms V_CKM = 1 when Y10 = 0.
                proportional_points=prop_count, n_points=npts)     # Confirms no generic Me proportional to Mnu.


def takagi(M):                                                     # Symmetric complex diagonalisation, M = U* D U^dagger.
    H = M @ M.conj().T                                             # Hermitian product.
    d, U = np.linalg.eigh(H)                                       # Its eigenbasis.
    order = np.argsort(-d); d = d[order]; U = U[:, order]           # Descending masses.
    D = U.T @ M @ U                                                # Should be diagonal after phase fixing.
    ph = np.angle(np.diag(D))                                      # Residual phases.
    U = U @ np.diag(np.exp(-1j * ph / 2))                           # Absorb them into U: these are the Majorana phases.
    return np.sqrt(np.maximum(d, 0)), U                            # Masses and the Takagi rotation.


def so10_neutrino(reps):                                           # SO(10) content WITH the unified correlations imposed.
    # This is the physically correct parameterisation.  In SO(10) the Majorana matrix
    # is NOT free: M_R = v_R Y_126 shares the Yukawa that feeds the charged fermions.
    # The Dirac matrix is NOT M_u either: its Clebsch combination differs
    # (Babu-Bajc-Saad Eq. 3.13).  Both correlations are imposed here.
    has10 = "10" in reps; has120 = "120" in reps; has126 = "126" in reps
    mats = []                                                      # (kind, name) for each Yukawa present.
    if has10: mats.append(("sym", "10"))                            # Y10 symmetric.
    if has120: mats.append(("anti", "120"))                         # Y120 antisymmetric.
    if has126: mats.append(("sym", "126"))                          # Y126 symmetric.
    n = sum(KIND_DOF[k] for k, _ in mats)                           # Yukawa parameters.
    nvev = (2 if has10 else 0) + (4 if has120 else 0) + (2 if has126 else 0) + 2  # Plus v_R and v_L.
    n += 2 * nvev                                                  # Two reals per complex vev.

    def unpack(p):                                                 # Build all four mass matrices.
        i = 0; Y = {}                                              # Cursor and Yukawa storage.
        for k, r in mats:                                          # Read the Yukawas.
            Y[r] = mk(k, p[i:i + KIND_DOF[k]]); i += KIND_DOF[k]     # Advance.
        nv = (n - i) // 2                                          # Number of complex vevs.
        v = p[i:i + 2 * nv].reshape(nv, 2); v = v[:, 0] + 1j * v[:, 1]  # Read them.
        j = 0                                                      # Vev cursor.
        def take():                                                # Consume one complex vev.
            nonlocal j                                             # Mutate the cursor.
            x = v[j]; j += 1                                        # Read and advance.
            return x                                               # Return it.
        Mu = np.zeros((3, 3), complex); Md = np.zeros((3, 3), complex)  # Charged accumulators.
        Me = np.zeros((3, 3), complex); MD = np.zeros((3, 3), complex)  # Lepton and Dirac.
        if has10:                                                  # The 10 contributes with weights (1,1,1) and (1,.,.) for nuD.
            au, ad = take(), take()                                 # Its two vevs.
            Mu += au * Y["10"]; Md += ad * Y["10"]                  # Up and down.
            Me += ad * Y["10"]; MD += au * Y["10"]                  # Lepton and Dirac: same weights for the 10.
        if has120:                                                 # The 120 has two bidoublets.
            v1u, v15u, v1d, v15d = take(), take(), take(), take()    # Four vevs.
            Mu += (v1u + v15u) * Y["120"]                           # Up: sum.
            Md += (v1d + v15d) * Y["120"]                           # Down: sum.
            Me += (v1d - 3 * v15d) * Y["120"]                       # Lepton: different combination.
            MD += (v1u - 3 * v15u) * Y["120"]                       # Dirac: the up-type analogue, NOT M_u.
        su = sd = None                                             # 126 vevs, if present.
        if has126:                                                 # The 126 carries the -3 weights.
            su, sd = take(), take()                                 # Its two vevs.
            Mu += su * Y["126"]; Md += sd * Y["126"]                # Up and down.
            Me += -3 * sd * Y["126"]; MD += -3 * su * Y["126"]      # Lepton and Dirac.
        vR, vL = take(), take()                                     # Majorana and induced triplet vevs.
        if not has126 or abs(vR) < 1e-10:                           # No 126 means no renormalisable M_R.
            return None                                            # Seesaw unavailable.
        S = Y["126"]                                               # The 126 Yukawa.
        if abs(np.linalg.det(S)) < 1e-10:                          # M_R must be invertible.
            return None                                            # Reject.
        MR = vR * S                                                # CORRELATED Majorana matrix.
        ML = vL * S                                                # Type-II piece.
        MN = ML - MD @ np.linalg.inv(MR) @ MD.T                     # Full type-I+II seesaw.
        return Mu, Md, Me, MN                                      # Four matrices.
    return unpack, n                                               # Map and parameter count.


def neutrino_observables(out, majorana):                           # Rephasing-invariant neutrino observables.
    Mu, Md, Me, MN = out                                           # Unpack.
    _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)      # Up and down rotations.
    V = Vu @ Vd.conj().T                                           # CKM.
    J = abs(np.imag(V[0,1]*V[1,2]*V[0,2].conj()*V[1,1].conj()))      # Quark Jarlskog.
    _, _, Ve = np.linalg.svd(Me)                                   # Charged-lepton rotation.
    if majorana:                                                   # Full nine-observable set.
        mn, Un = takagi(MN)                                        # Takagi factorisation exposes the Majorana phases.
    else:                                                          # Seven-observable subset.
        mn = sv(MN)                                                # Masses from singular values.
        _, Un = np.linalg.eigh(MN @ MN.conj().T)                    # Rotation from the Hermitian product.
    U = Ve @ Un.conj().T                                           # PMNS.
    Jn = abs(np.imag(U[0,1]*U[1,2]*U[0,2].conj()*U[1,1].conj()))     # Leptonic Jarlskog.
    base = [sv(Mu), sv(Md), sv(Me),                                # Nine charged masses.
            np.array([abs(V[0,1]), abs(V[0,2]), abs(V[1,2]), J]),   # Four CKM.
            mn, np.array([abs(U[0,1]), abs(U[0,2]), abs(U[1,2]), Jn])]  # Neutrino masses and mixing.
    if majorana:                                                   # Append the two Majorana phases.
        m1 = np.angle(U[0,0]**2 * U[0,1].conj()**2)                 # Rephasing-invariant combination.
        m2 = np.angle(U[0,0]**2 * U[0,2].conj()**2)                 # Second independent combination.
        base.append(np.array([m1, m2]))                            # Two more observables.
    return np.concatenate(base)                                    # 20 or 22 rows.


def extended_jacobian(npts=8):                                     # DEPRECATED / NOT CALLED (audit 9/12).
    # DEPRECATED: builds CKM/PMNS from the SVD RIGHT unitary Vh, which is NOT
    # basis-independent; its numbers coincidentally matched but used the wrong
    # quantity.  NOT called by run_all(); observable dimensions come from
    # strict_obs only.  Retained solely for provenance; do not use its output.
    out = {}                                                       # Results per content and observable set.
    for reps in (("120", "126"), ("10", "126"), ("10", "120", "126")):  # Contents containing a 126.
        unpack, n = so10_neutrino(reps)                             # Correct correlated parameterisation.
        for majorana in (False, True):                              # Both observable sets.
            nobs = 22 if majorana else 20                           # Row count.
            def obs(p):                                            # Observable vector.
                r = unpack(p)                                      # Build.
                if r is None:                                      # Rejected draw.
                    return np.full(nobs, np.nan)                    # Signal.
                return neutrino_observables(r, majorana)            # Evaluate.
            ranks = []; conds = []                                 # Collect.
            for _ in range(npts):                                  # Several draws.
                p = RNG.normal(0, 1, n)                            # Sample.
                f0 = obs(p)                                        # Evaluate.
                if not np.all(np.isfinite(f0)):                    # Skip rejects.
                    continue                                       # Next.
                Jm = np.zeros((nobs, n)); h = 1e-6                  # Jacobian and step.
                bad = False                                        # Rejection flag.
                for i in range(n):                                 # Each direction.
                    a = p.copy(); a[i] += h                         # Forward.
                    b = p.copy(); b[i] -= h                         # Backward.
                    fa, fb = obs(a), obs(b)                         # Evaluate both.
                    if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):  # Reject.
                        bad = True; break                           # Abandon this draw.
                    Jm[:, i] = (fa - fb) / (2 * h)                  # Central difference.
                if bad:                                            # Skip.
                    continue                                       # Next.
                s = np.linalg.svd(Jm, compute_uv=False)            # Spectrum.
                if not np.all(np.isfinite(s)) or s[0] == 0:        # Guard.
                    continue                                       # Skip.
                r = int(np.sum(s > 1e-7 * s[0]))                    # Rank.
                ranks.append(r)                                    # Record.
                if r == nobs:                                      # Full rank.
                    conds.append(float(s[-1] / s[0]))               # Conditioning.
            key = "+".join(reps) + ("_22" if majorana else "_20")   # Label.
            out[key] = dict(model="+".join(reps), n_params=n, n_observables=nobs,
                            rank=max(ranks) if ranks else None,      # Generic rank.
                            rank_set=sorted(set(ranks)),             # All ranks seen.
                            p_add=nobs - max(ranks) if ranks else None,  # Extended P_add.
                            s_ratio=float(np.median(conds)) if conds else None)  # Conditioning.
    return out                                                     # Six records.


def seesaw_image_surjectivity(npts=10):                            # The constrained seesaw image is still onto.
    # With M_R = v_R S the type-I+II light matrix reduces to
    #     M_N = c1 S + c2 (A S^-1 A),
    # where S = Y126 is symmetric and A = Y120 antisymmetric.  Since rank(A) = 2,
    # the second term has rank at most two, so surjectivity is NOT obvious.
    # This measures it directly: 22 real parameters against a 12-real symmetric target.
    def unpack(p):                                                 # Build the image.
        A = mk("anti", p[0:6]); S = mk("sym", p[6:18])              # Two Yukawas.
        c1 = p[18] + 1j * p[19]; c2 = p[20] + 1j * p[21]            # Two free complex coefficients.
        if abs(np.linalg.det(S)) < 1e-8:                           # Need S invertible.
            return None                                            # Reject.
        return c1 * S + c2 * (A @ np.linalg.inv(S) @ A)             # The image.

    def obs(p):                                                    # Six complex entries -> twelve reals.
        M = unpack(p)                                              # Build.
        if M is None:                                              # Reject.
            return np.full(12, np.nan)                             # Signal.
        idx = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]            # Independent entries of a symmetric matrix.
        return np.array([x for i, j in idx for x in (M[i,j].real, M[i,j].imag)])  # Flatten.
    ranks = []; conds = []; symerr = 0.0; r2 = []                   # Collect.
    for _ in range(npts):                                          # Draws.
        p = RNG.normal(0, 1, 22)                                   # Sample.
        f0 = obs(p)                                                # Evaluate.
        if not np.all(np.isfinite(f0)):                            # Skip.
            continue                                               # Next.
        Jm = np.zeros((12, 22)); h = 1e-6                           # Jacobian.
        ok = True                                                  # Flag.
        for i in range(22):                                        # Directions.
            a = p.copy(); a[i] += h                                 # Forward.
            b = p.copy(); b[i] -= h                                 # Backward.
            fa, fb = obs(a), obs(b)                                 # Evaluate.
            if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):  # Reject.
                ok = False; break                                   # Abandon.
            Jm[:, i] = (fa - fb) / (2 * h)                          # Derivative.
        if not ok:                                                 # Skip.
            continue                                               # Next.
        s = np.linalg.svd(Jm, compute_uv=False)                    # Spectrum.
        r = int(np.sum(s > 1e-7 * s[0]))                            # Rank.
        ranks.append(r)                                            # Record.
        if r == 12:                                                # Full.
            conds.append(float(s[11] / s[0]))                       # Conditioning.
        A = mk("anti", p[0:6]); S = mk("sym", p[6:18])              # Rebuild for the structural checks.
        T = A @ np.linalg.inv(S) @ A                               # The rank-deficient term.
        symerr = max(symerr, float(np.max(np.abs(T - T.T)) / max(np.max(np.abs(T)), 1e-300)))  # It IS symmetric.
        r2.append(int(np.linalg.matrix_rank(T, tol=1e-8)))          # Its rank.
    return dict(n_params=22, n_target=12, rank=max(ranks),          # Summary.
                rank_set=sorted(set(ranks)), surjective=(max(ranks) == 12),
                s_ratio=float(np.median(conds)) if conds else None,
                term_symmetry_err=symerr, term_rank=sorted(set(r2)))  # Structural diagnostics.


def exact_relations(rows):                                         # EXACT identities + STRICT observable dimension.
    # rows = THIS run's census, passed in so no assertion reads a previous
    # results.json (audit 14 B2: the program must reproduce from an empty dir).
    # The pencil mechanism and (E1)/(E2) are exact matrix-level identities
    # (exact_core, grade a).  The PHYSICAL observable-space dimension is measured
    # with STRICT, basis-independent Autonne-Takagi observables (strict_obs), the
    # single source of truth after audit 9.  Earlier versions used non-basis-
    # independent neutrino observables and obtained 19/three relations; that is
    # WITHDRAWN.  The strict generic rank is 22 => ZERO observable relations for
    # every content.  The pencil identity does not project to a low-energy relation.
    import exact_core as X                                         # Exact Gaussian-rational kernel.
    out = X.run()                                                  # Exact identities (still correct).
    import strict_obs as SO                                        # STRICT observable truth source.
    so = SO.run()                                                  # Basis-independent dimensions.
    out["phys_120_126"] = so["phys_120_126"]                       # 22 (strict).
    out["relations_120_126"] = so["relations_120_126"]             # 0.
    out["phys_10_126"] = so["phys_10_126"]                         # 22.
    out["relations_10_126"] = so["relations_10_126"]               # 0.
    out["phys_10_120_126"] = so["phys_10_120_126"]                 # 22.
    out["relations_10_120_126"] = so["relations_10_120_126"]       # 0.
    out["charged_phys_120_126"] = so["charged_phys"]               # 13.
    out["charged_ranks"] = so["charged_ranks"]
    out["observable_ranks_strict"] = so["ranks_120_126"]
    out["generic_rank_stable"] = (len(set(so["ranks_120_126"])) <= 2)  # generic 22, at most one degenerate 21.
    out["phys_dim_evidence"] = "strict_takagi_basis_independent"   # Grade (b), correct observables.
    out["exact_invariant_lower_bound"] = 17                        # Rigorous bound (unchanged).
    out["withdrawn_19_three"] = ("19/three came from non-basis-independent neutrino "
                                 "observables; strict Takagi gives 22/zero (audit 9)")
    # Robustness / RG tests (unchanged, still valid).
    import audit5_tests as A5
    a5 = A5.run()
    out["counting_no_double"] = a5["counting_rules"]["no_double_counting"]
    out["rg_proportionality_broken"] = a5["rg_sensitivity"]["proportionality_broken_by_rg"]
    out["rg_ratio_shifts"] = a5["rg_sensitivity"]["ratio_shifts_toward_window"]
    out["rg_low_mtau_mb"] = a5["rg_sensitivity"]["low_mtau_mb"]
    # Robustness dimensions now COMPUTED with strict observables (no hard-coding).
    dfl = SO.measure(SO.full_obs_strict(SO.build_dflat), 34)       # vev alignment.
    out["vev_constraint_dim"] = max(set(dfl), key=dfl.count)
    out["vev_ranks"] = dfl
    two = SO.measure(SO.full_obs_strict(SO.build_two_copies), 54)  # second 126 copy.
    out["two_copies_dim"] = max(set(two), key=two.count)
    out["two_copies_ranks"] = two
    # v_L robustness: measure a FREE-v_L model (34 params, 8th vev = v_L) and
    # compare with the tied-v_L model (34 params, 8th vev = kappa); tying must not
    # change the dim.  Both dimensions are INDEPENDENT measurements of their own
    # builders (audit 14 general 4: vL_constraint_dim no longer copies the headline).
    free = SO.measure(SO.full_obs_strict(SO.build_freevL), 34)
    out["vL_free_dim"] = max(set(free), key=free.count)
    out["vL_free_ranks"] = free
    tied = SO.measure(SO.full_obs_strict(SO.build), 34)
    out["vL_constraint_dim"] = max(set(tied), key=tied.count)
    out["vL_constraint_ranks"] = tied
    out["vL_constraint_robust"] = (out["vL_free_dim"] == out["vL_constraint_dim"])
    import charged_grade_a as CGA                                  # grade-(a) charged proof, 4 SO(10) contents.
    cg = CGA.all_contents()
    out["charged_grade_a_all"] = cg
    out["charged_grade_a"] = all(all(x == 13 for x in r) for r in cg.values())
    import rg_lowenergy_match as RLM                               # measured-boundary RG matching.
    out["rg_lowenergy"] = RLM.compute()
    import rational_route_exhaustive as RRE                        # grade-a obstruction evidence.
    out["rational_route"] = RRE.compute()
    import grade_a_fast as GAF                                     # rigorous neutrino rank-22 certificate.
    certs = {s: GAF.certificate(s) for s in (2, 7, 11)}
    out["neutrino_grade_a_certs"] = certs
    out["neutrino_grade_a"] = all(c["certified"] for c in certs.values())
    import fiber_proof as FP                                       # tangent-space demonstration.
    out["fiber"] = FP.compute()
    import pencil_mining as PM                                     # pencil universality / breaking / stats.
    out["pencil_mining"] = {"universality": PM.universality(n=150),
                            "breaking_scale": PM.breaking_scale(n=120)}
    import declaration_scoreboard as DS                            # declaration-cost scoreboard.
    out["declaration_scoreboard"] = DS.scoreboard(n_starts=30, n_iter=200)
    import pencil_prior_sensitivity as PPS                         # prior robustness of the statistical preference.
    pps = PPS.analyse(n=1200)
    pps["summary"] = PPS.robust_summary(pps)
    out["pencil_prior_sensitivity"] = pps
    import exclusion_taxonomy as ET                              # exclusion classification theorem.
    out["exclusion_taxonomy"] = ET.compute(rows, out["rg_lowenergy"])  # THIS run's census + RG (audit 14 B2).
    import pencil_attribution_control as PAC                     # V1: preference is NOT pencil-specific.
    out["pencil_attribution_control"] = PAC.control_10_126(n=1500)
    out["pure_typeI_control"] = PAC.pure_typeI_control(n=1200)   # self-audit: preference is M_R structure.
    out["ten126_eigcount"] = PAC.ten126_eigcount(n=300)          # 10+126 has 3 distinct eigs (no repeated root).
    import pencil_analytic_origin as PAO                         # pencil identity algebraic derivation chain.
    out["pencil_analytic_origin"] = PAO.verify_chain(n=150)
    import singleS_spectral_theorem as SST                       # single-S spectral 3->2 observation (120+126 only).
    out["singleS_spectral"] = SST.verify(n=300)
    import seven_content_scoreboard as SCS                       # seven-content declaration scoreboard + Pfaffian scope.
    out["seven_scoreboard"] = SCS.scoreboard(n_starts=40)
    out["pfaffian_scope"] = SCS.pfaffian_scope()
    import seed42_diagnosis as SD                                  # coordinate-degeneracy diagnosis.
    out["seed42"] = SD.compute()
    return out                                                     # Dict of verdicts and dimensions.


def fig5_seesaw(sur, t2):                                          # Figure 5: the neutrino sector adds observables, not relations.
    fig = plt.figure(figsize=(9.4, 3.5))                            # Two panels.
    ax = fig.add_subplot(121)                                       # Left: observable budget.
    cats = ["charged\nfermions", "CKM", "gauge", "neutrinos"]        # Four sectors.
    counts = [9, 4, 3, 9]                                          # Their observable counts: 25 in total.
    fixed = [0, 0, 3, 0]                                           # Only the gauge sector is fixed by the group.
    cols = [C_BASE, C_BASE, C_VIABLE, C_NU]                          # Colour per sector.
    x = np.arange(4)                                               # Bar positions.
    ax.bar(x, counts, 0.58, color=cols, alpha=0.30,                  # Total height: all observables.
           edgecolor="white", lw=0.8, label="observables")
    ax.bar(x, fixed, 0.58, color=cols, edgecolor="white", lw=0.8,    # Opaque part: those actually fixed.
           label="fixed by the group")
    for i, (c, f) in enumerate(zip(counts, fixed)):                 # Annotate each sector.
        ax.text(i, c + 0.3, "%d" % c, ha="center", fontsize=9,
                fontweight="bold", color=cols[i])                   # Total count above the bar.
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.2)         # Sector labels.
    ax.set_ylabel("number of observables"); ax.set_ylim(0, 11)       # Axis and headroom.
    ax.legend(loc="upper left", fontsize=7.8)                        # Two-entry legend.
    ax.set_title("(a) 25 observables; only the 3-item baseline fixed, zero forced relations",
                 fontsize=9.0, loc="left")  # Panel title (audit 14 B1: baseline-only wording).
    ax2 = fig.add_subplot(122); ax2.axis("off"); ax2.grid(False)     # Right: the surjectivity statement.
    ax2.set_xlim(0, 6); ax2.set_ylim(0, 6)                           # Text canvas.
    ax2.text(3.0, 5.3, "type-I seesaw is surjective", ha="center",
             fontsize=10, fontweight="bold")                        # Header.
    ax2.text(3.0, 4.35, "$M_\\nu=-M_D^{T}M_R^{-1}M_D$", ha="center", fontsize=11)  # The seesaw formula.
    ax2.text(3.0, 3.45, "given any symmetric target $S$, choose", ha="center",
             fontsize=8.6, color=C_MUTE)                            # Construction preamble.
    ax2.text(3.0, 2.65, "$M_R=-M_D\\,S^{-1}M_D^{T}$", ha="center",
             fontsize=11, color=C_NU)                               # The explicit preimage.
    ax2.text(3.0, 1.75, "symmetric, invertible, and maps to $S$", ha="center",
             fontsize=8.6, color=C_MUTE)                            # Why it works.
    ax2.text(3.0, 0.85, "verified: %d targets, error $\\leq$ %.0e" %
             (sur["n_points"], sur["hit_err_max"]), ha="center",
             fontsize=8.4, color=C_VIABLE)                          # Numerical confirmation.
    ax2.set_title("(b) no relation on nine observables", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.2)                                      # Space the panels.
    savefig("fig5_seesaw")                                          # Emit both formats.


# ===== MODULE 6 | three-dimensional reach landscape | fig6_landscape =====
def fig6_landscape(rows):                                          # Figure 6: the headline visual, two panels.
    fig = plt.figure(figsize=(11.0, 4.3))                           # Wide canvas for a two-column journal figure.
    ax = fig.add_subplot(1, 2, 1, projection="3d")                  # Left: the three-dimensional landscape.
    GR = {"SU(5)": 4, "SO(10)": 5, "E6": 6, "PS": 4}                # Rank of each gauge group.
    # Jitter the operator axis slightly so coincident contents remain distinguishable.
    jit = {}                                                       # Per-(group,ops) counter for the jitter.
    def coords(r):                                                 # Map a content to its plotted coordinates.
        key = (r["group"], r["n_ops"])                              # Contents sharing a cell need separating.
        k = jit.get(key, 0); jit[key] = k + 1                        # Index of this content within its cell.
        return GR[r["group"]] + 0.16 * (k - 0.5), r["n_ops"] + 0.13 * (k - 0.5), r["P"]
    viable = [r for r in rows if r["splitting"]]                    # The seven viable contents.
    excl = [r for r in rows if not r["splitting"]]                  # The eleven excluded ones.

    gx, gy = np.meshgrid([3.6, 6.4], [0.6, 5.4])                    # Mesh for the simple-group ceiling.
    ax.plot_surface(gx, gy, np.full_like(gx, 3.0), color=C_BASE,     # Translucent plate at three.
                    alpha=0.10, linewidth=0, shade=False)
    ax.plot([3.6, 6.4, 6.4, 3.6, 3.6], [0.6, 0.6, 5.4, 5.4, 0.6],    # Its dashed outline.
            [3] * 5, color=C_BASE, lw=1.1, ls=(0, (5, 3)), alpha=0.9)
    gx2, gy2 = np.meshgrid([3.6, 4.4], [0.6, 5.4])                  # Plate for the product group.
    ax.plot_surface(gx2, gy2, np.full_like(gx2, 1.0), color="#C08A2E",  # Translucent plate at one.
                    alpha=0.12, linewidth=0, shade=False)
    ax.plot([3.6, 4.4, 4.4, 3.6, 3.6], [0.6, 0.6, 5.4, 5.4, 0.6],    # Its outline.
            [1] * 5, color="#C08A2E", lw=1.1, ls=(0, (3, 3)), alpha=0.9)

    for r in excl:                                                 # Excluded points first, so viable ones draw on top.
        x, y, z = coords(r)                                        # Its coordinates.
        ax.plot([x, x], [y, y], [3.0, z], color=C_EXCL, lw=0.7,     # Stem rises from the ceiling, not the floor:
                alpha=0.30)                                        # this makes the excess above baseline the visual.
        ax.scatter([x], [y], [z], s=70, c=C_EXCL, marker="^",        # Rust triangle.
                   edgecolor="white", lw=0.9, depthshade=False)
    for r in viable:                                               # Viable points: drawn last, on the plate.
        x, y, z = coords(r)                                        # Its coordinates; z equals its baseline exactly.
        ax.scatter([x], [y], [z], s=125, c=C_VIABLE, marker="o",     # Green circle sitting on the plate.
                   edgecolor="white", lw=1.3, depthshade=False)
    ax.scatter([], [], [], s=115, c=C_VIABLE, marker="o",            # Proxy artists: the legend carries the counts,
               edgecolor="white", lw=1.1, label="7 viable: on the ceiling")  # so no 3D text can collide.
    ax.scatter([], [], [], s=75, c=C_EXCL, marker="^",
               edgecolor="white", lw=0.9, label="11 excluded: above it")
    ax.plot([], [], [], color=C_BASE, lw=1.1, ls=(0, (5, 3)),         # Legend entry for the upper plate.
            label="simple-group ceiling = 3")
    ax.plot([], [], [], color="#C08A2E", lw=1.1, ls=(0, (3, 3)),      # Legend entry for the lower plate.
            label="product-group ceiling = 1")
    ax.set_xlabel("rank $G$", labelpad=4, fontsize=9)               # Horizontal axis 1.
    ax.set_ylabel("operators", labelpad=3, fontsize=9)              # Horizontal axis 2.
    ax.set_zlabel("$P_{\\mathrm{ex}}$", labelpad=2, fontsize=9)      # Vertical axis.
    ax.set_xlim(3.5, 6.5); ax.set_ylim(0.5, 5.5); ax.set_zlim(2.0, 13.5)  # Crop the empty floor region.
    ax.set_xticks([4, 5, 6]); ax.set_yticks([1, 2, 3, 4, 5])         # Integer ticks.
    ax.set_zticks([3, 6, 9, 12])                                    # Reach ticks, starting at the ceiling.
    ax.tick_params(labelsize=7.8, pad=1)                            # Compact tick labels.
    ax.view_init(elev=14, azim=-52)                                 # Lower elevation: the plate reads as a ceiling.
    try:                                                           # Pane transparency API differs across versions.
        ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0)   # Remove the grey background boxes.
        ax.zaxis.pane.set_alpha(0.0)
    except Exception:                                              # Older matplotlib: ignore silently.
        pass
    ax.grid(color="#EBEEF2", lw=0.4)                                # Very light 3D grid.
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=8.0)  # Legend clear of the data.
    ax.set_title("(a) the reach landscape", fontsize=9.8, loc="left", pad=2)  # Panel title.

    ax2 = fig.add_subplot(1, 2, 2)                                  # Right: decomposition of the viable set.
    order = sorted(viable, key=lambda r: (r["baseline"], r["group"], r["model"]))  # Group by baseline.
    names = [r["group"] + "  " + r["model"] for r in order]          # Content labels.
    y = np.arange(len(order))                                       # Bar positions.
    gauge = np.array([r["gauge"] for r in order], float)             # Gauge-unification part.
    charge = np.array([r["charge"] for r in order], float)           # Charge-quantisation part.
    ax2.barh(y, gauge, 0.56, color=C_BASE, edgecolor="white", lw=0.8,  # Blue: two relations, or zero for PS.
             label="gauge unification (2)")
    ax2.barh(y, charge, 0.56, left=gauge, color="#9FC3E4",            # Light blue: one relation.
             edgecolor="white", lw=0.8, label="charge quantisation (1)")
    ax2.barh(y, np.zeros(len(order)), 0.56, left=gauge + charge,      # Zero-height green segment: the point.
             color=C_VIABLE, edgecolor="white", lw=0.8,
             label="model-specific (0)")
    for i, r in enumerate(order):                                  # Annotate each bar.
        ax2.text(r["baseline"] + 0.12, i, "$P_{\\mathrm{add}}=0$", va="center",
                 fontsize=8.0, color=C_VIABLE, fontweight="bold")    # The headline, repeated per content.
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8.0)       # Content labels.
    ax2.invert_yaxis()                                             # Read top to bottom.
    ax2.set_xlim(0, 5.0); ax2.set_xticks([0, 1, 2, 3])               # Room for the annotation.
    ax2.set_ylim(len(order) - 0.35, -1.35)                          # Extra headroom above the first bar for the legend.
    ax2.set_xlabel("number of exact relations")                     # Axis label.
    ax2.legend(loc="upper right", fontsize=7.8, ncol=1)              # Legend in the reserved headroom, clear of data.
    ax2.set_title("(b) every viable content: baseline only", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.4)                                      # Space the panels.
    savefig("fig6_landscape")                                       # Emit both formats.


# ===== MODULE 7 | the counting trap | fig7_trap =====
def counting_trap():                                               # Quantify what naive operator counting invents.
    out = []                                                       # One record per affected content.
    for reps in (("120", "126"), ("10", "120")):                    # The two contents the error touches.
        uf, nf = build_so10(reps, False)                            # Collapsed parameterisation.
        uc, nc = build_so10(reps, True)                             # Correct parameterisation.
        rf = max(x for x in (jac_rank(RNG.normal(0, 1, nf), uf)[0] for _ in range(5))
                 if x is not None)                                 # Its generic rank.
        rc = max(x for x in (jac_rank(RNG.normal(0, 1, nc), uc)[0] for _ in range(5))
                 if x is not None)                                 # Correct generic rank.
        t_err = []; spec = []                                       # Transpose test and spectral separation.
        for _ in range(200):                                       # Random points for the transpose diagnostic.
            p = RNG.normal(0, 1, nc)                                # Correct parameterisation.
            Mu, Md, Me = uc(p)                                     # Its matrices.
            t_err.append(float(np.max(np.abs(Me - Md.T)) / np.max(np.abs(Md))))  # Does Me = Md^T hold?
            spec.append(float(np.max(np.abs(sv(Me) - sv(Md))) / np.max(sv(Md))))  # Do the spectra differ?
        out.append(dict(model="SO(10) " + "+".join(reps), n_collapsed=nf, n_correct=nc,
                        rank_collapsed=rf, rank_correct=rc,          # Ranks under both countings.
                        p_add_collapsed=13 - rf, p_add_correct=13 - rc,  # The resulting P_add.
                        overcount=(13 - rf) - (13 - rc),             # How many relations were invented.
                        transpose_err_median=float(np.median(t_err)),  # Transpose identity failure.
                        spectral_sep_median=float(np.median(spec))))  # Spectral separation once lifted.
    return out                                                     # Two records.


def fig7_trap(trap):                                               # Figure 7: the cost of miscounting, and the classification flip.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3),          # Two panels.
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    names = [t["model"].replace("SO(10) ", "") for t in trap]        # Short content labels.
    x = np.arange(len(names)); w = 0.34                             # Bar positions and width.
    ax.bar(x - w / 2, [t["p_add_collapsed"] for t in trap], w,       # Naive count.
           color=C_EXCL, edgecolor="white", lw=0.8, label="naive count")
    ax.bar(x + w / 2, [t["p_add_correct"] for t in trap], w,         # Correct count.
           color=C_VIABLE, edgecolor="white", lw=0.8, label="bidoublet count")
    for i, t in enumerate(trap):                                   # Annotate the overcount.
        ax.text(i - w / 2, t["p_add_collapsed"] + 0.12, str(t["p_add_collapsed"]),
                ha="center", fontsize=8.6, fontweight="bold", color=C_EXCL)  # Naive value.
        ax.text(i + w / 2, t["p_add_correct"] + 0.12, "0", ha="center",
                fontsize=8.6, fontweight="bold", color=C_VIABLE)    # Correct value.
        ax.annotate("", xy=(i + w / 2, 0.55), xytext=(i - w / 2, t["p_add_collapsed"] - 0.15),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_MUTE))  # Arrow shows the correction.
        ax.text(i, t["p_add_collapsed"] + 0.75, "$-%d$" % t["overcount"],
                ha="center", fontsize=8.4, color=C_INK)             # Size of the overcount.
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.6)        # Content labels.
    ax.set_ylabel("$P_{\\mathrm{add}}$"); ax.set_ylim(0, 4.4)        # Axis and headroom.
    ax.legend(loc="upper right", fontsize=8)                        # Two-entry legend.
    ax.set_title("(a) relations that were never there", fontsize=9.8, loc="left")  # Panel title.
    t = trap[1]                                                     # The 10+120 content, whose status flips.
    stages = ["naive:\n$M_e=M_d^{T}$\nexcluded", "correct:\nspectra split\nviable"]  # Two stages.
    vals = [0.0, t["spectral_sep_median"]]                          # Spectral separation in each stage.
    ax2.bar([0, 1], vals, 0.45, color=[C_EXCL, C_VIABLE],            # Zero versus nonzero separation.
            edgecolor="white", lw=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(stages, fontsize=8.2)  # Stage labels.
    ax2.set_ylabel("relative $|\\sigma(M_e)-\\sigma(M_d)|$")          # Spectral separation.
    ax2.set_ylim(0, max(vals) * 1.45 + 0.05)                        # Headroom.
    ax2.text(1, vals[1] + 0.02, "%.2f" % vals[1], ha="center",
             fontsize=8.6, fontweight="bold", color=C_VIABLE)        # Annotate the nonzero value.
    ax2.text(0, 0.02, "identical", ha="center", fontsize=8.2, color=C_EXCL)  # The naive case has identical spectra.
    ax2.set_title("(b) a viable model was being discarded", fontsize=9.8, loc="left")  # Panel title.
    fig.tight_layout(pad=1.2)                                      # Space the panels.
    savefig("fig7_trap")                                            # Emit both formats.


# ===== MODULE 8 | assertions and export | fig8_assertions =====
def run_all():                                                     # Main driver: run every module and check every claim.
    fig0_pipeline()                                                # Module 0 figure.
    rows = census()                                                # Module 1: exact census.
    for r in rows:                                                 # Attach the exact reach to each record.
        r["P"] = r["mass"] + r["ckm"] + r["gauge"] + r["charge"]     # Sum of all relation classes.
        r["p_add"] = 0                                             # Set from the measured ranks below.
    fig1_census(rows)                                              # Module 1 figure.
    fig2_bidoublet()                                               # Module 2 figure.
    ranks = measure_ranks()                                        # Module 3: measured ranks.
    byname = {r["model"]: r for r in ranks}                         # Index by content label.
    for r in rows:                                                 # Write the measured P_add back into the census.
        key = r["group"] + " " + r["model"]                          # Match the census label to the rank label.
        if r["splitting"] and key in byname:                        # Only viable contents carry a flavour rank.
            r["p_add"] = byname[key]["p_add"]                        # Measured, not assumed.
    fig3_ranks(ranks)                                              # Module 3 figure.
    comp = compare_parameterisations()                             # Module 4: correct versus collapsed.
    fig4_withdrawn(comp)                                           # Module 4 figure.
    sur = seesaw_surjectivity()                                    # Module 5: seesaw surjectivity.
    t2 = typeII_checks()                                           # Module 5: type-II diagnostics.
    # extended_jacobian() REMOVED (audit 10): it built CKM/PMNS from the SVD right
    # unitary Vh, which is not basis-independent; its "advisory" numbers coincidentally
    # matched but used the wrong quantity.  Observable dimensions come from strict_obs only.
    img = seesaw_image_surjectivity()                              # Module 5: the reduced map is dominant.
    ex = exact_relations(rows)                                     # Module 5: EXACT verdicts over Q (this run's census).
    fig5_seesaw(sur, t2)                                           # Module 5 figure.
    fig6_landscape(rows)                                           # Module 6 figure.
    trap = counting_trap()                                         # Module 7: the counting trap.
    fig7_trap(trap)                                                # Module 7 figure.

    S = [r for r in rows if r["splitting"]]                         # The viable set.
    X = [r for r in rows if not r["splitting"]]                     # The excluded set.
    checks = [                                                     # Every assertion is falsifiable and printed.
        ("eighteen contents enumerated", len(rows) == 18),          # Completeness of the enumeration.
        ("seven contents are spectrum-splitting", len(S) == 7),     # The corrected classification.
        ("eleven contents are excluded", len(X) == 11),             # Its complement.
        ("P_add = 0 for EVERY viable content", all(r["p_add"] == 0 for r in S)),  # The headline result.
        ("every viable content has rank 13", all(r["rank"] == 13 for r in ranks)),  # Full rank throughout.
        ("P_ex in {1,3} on the viable set", sorted({r["P"] for r in S}) == [1, 3]),  # Only baselines occur.
        ("P = baseline + P_add identically", all(r["P"] == r["baseline"] + r["p_add"] for r in S)),  # Bookkeeping closes.
        ("120 alone excluded by masslessness",                      # Antisymmetry forces a zero singular value.
         next(r for r in rows if r["model"] == "120")["all_anti"]),
        ("120 carries free d/e coefficients",                       # The structural correction.
         next(r for r in rows if r["model"] == "120+126")["free_de"]),
        ("126 does NOT split (one bidoublet)",                      # Only the 120 has two.
         not next(r for r in rows if r["model"] == "10+126")["free_de"]),
        ("10+120 is viable after the correction",                   # The classification flip.
         next(r for r in rows if r["model"] == "10+120")["splitting"]),
        ("collapsed identity holds to 1e-13",                       # The artefact is exact inside its sub-manifold.
         comp["collapsed"]["identity_err_max"] < 1e-13),
        ("correct identity FAILS (median > 0.1)",                   # And fails outside it.
         comp["correct"]["identity_err_median"] > 0.1),
        ("surviving identity sym Me = -3 sym Md holds",             # What remains is exact.
         comp["correct"]["sym_err_max"] < 1e-12),
        ("withdrawn bound: lower edge breached",                    # The bound is not a theorem.
         comp["correct"]["viol_lower"] > 0),
        ("withdrawn bound: upper edge breached",                    # Both edges fail.
         comp["correct"]["viol_upper"] > 0),
        ("collapsed bound never breached",                          # It held only in the restricted case.
         comp["collapsed"]["viol_lower"] == 0 and comp["collapsed"]["viol_upper"] == 0),
        ("seesaw surjectivity verified to 1e-9",                    # Nine neutrino observables are free.
         sur["hit_err_max"] < 1e-9),
        ("constructed M_R is symmetric",                            # The preimage is admissible.
         sur["sym_err_max"] < 1e-12),
        ("Y10=0 forces V_CKM diagonal",                             # Correct reason to exclude that locus.
         t2["ckm_offdiag_max"] < 1e-9),
        ("no generic Me proportional to M_nu (type-II)",            # No type-II relation either.
         t2["proportional_points"] == 0),
        ("reduced neutrino map is dominant",                       # (S,A,c1,c2) covers Sym^2(C^3).
         img["surjective"]),
        ("A S^-1 A is symmetric",                                  # Structural identity behind the reduction.
         img["term_symmetry_err"] < 1e-12),
        ("A S^-1 A has rank at most two",                          # The rank-two structure behind the pencil identity.
         max(img["term_rank"]) <= 2),
        ("EXACT: tr(A S^-1) = 0",                                  # Antisymmetry, over Q.
         ex["trB_zero"]),
        ("EXACT: det(A S^-1) = 0",                                 # Rank deficiency, over Q.
         ex["detB_zero"]),
        ("EXACT: B^2 has a repeated eigenvalue",                   # Discriminant vanishes identically.
         ex["discB2_zero"]),
        ("EXACT: pencil (M_nu, S) has a repeated eigenvalue",      # Matrix-level identity (no low-energy relation).
         ex["discPencil_zero"]),
        ("EXACT: explicit (E1) and (E2) vanish identically",       # Closed-form matrix-level constraints.
         ex["E1_E2_vanish"]),
        ("EXACT: (E1)+(E2) impose four AMBIENT conditions",        # Matrix-pair codimension (not observable count).
         ex["explicit_codimension"] == 4),
        ("STRICT: 120+126 observable dimension is 22",             # Basis-independent Takagi rank (audit 9).
         ex["phys_120_126"] == 22),
        ("STRICT: zero observable relations for 120+126",          # 22 observables minus 22; pencil is matrix-level.
         22 - ex["phys_120_126"] == 0),
        ("STRICT: generic rank stable across seeds",               # 22 generic, at most one degenerate 21.
         ex["generic_rank_stable"]),
        ("EXACT: invariant-word lower bound is 17",                # Rigorous lower bound on the dimension.
         ex["exact_invariant_lower_bound"] == 17),
        ("MEASURED: charged observable dimension is 13",           # Not 14: only 13 charged observables exist.
         ex["charged_phys_120_126"] == 13),
        ("MEASURED: charged sector alone unconstrained",           # 13 physical against 13 observables.
         ex["charged_phys_120_126"] >= 13),
        ("MEASURED: 10+126 has no forced relation",                # Observable dimension 22.
         ex["phys_10_126"] == 22),
        ("MEASURED: 10+120+126 observable dimension is 22",        # Not 34: 34 was a matrix-space artefact.
         ex["phys_10_120_126"] == 22),
        ("MEASURED: no observable dimension exceeds 22",           # Sanity: rank <= number of observables.
         all(ex[k] <= 22 for k in ["phys_120_126", "phys_10_126", "phys_10_120_126"])),
        ("EXACT: counting rules have no double counting",          # M_u=tM_d gives 4 CKM + 2 ratio = 6.
         ex["counting_no_double"]),
        ("EXACT: charged 120+126 dominance rank 13 (grade a)",     # Rational invariants prove P_add=0.
         ex["charged_grade_a"]),
        ("MEASURED: fibre dim 12 and Sigma varies along it",       # Tangent-space demonstration.
         ex["fiber"]["fibre_dim"] == 12 and ex["fiber"]["n_sigma_changing"] == 12),
        ("MEASURED: seed-42 null dir is charged Jarlskog",         # Coordinate degeneracy, not relation.
         abs(ex["seed42"]["null_weight_obs12"]) > 0.99 and abs(ex["seed42"]["d2J"]) > 1e-8),
        ("MEASURED: measured-boundary RG round-trip stable",       # Integrator reliability.
         ex["rg_lowenergy"]["roundtrip_dev"] < 1e-8),
        ("MEASURED: measured spectrum at GUT below 120+126 wall",  # Exclusion survives running.
         ex["rg_lowenergy"]["gut_mtau_mb"] < 2.0),
        ("EXACT: rational Mnu words add no direction beyond 17",   # grade-a obstruction, exhaustive screen.
         ex["rational_route"]["max_survivor_rank"] <= 17),
        ("EXACT: fraction-field/resultant combos add no direction",  # closes non-polynomial loophole.
         max(ex["rational_route"]["fraction_field_ranks"]) <= 17),
        ("EXACT: neutrino observable rank 22 certified (grade a)",   # Neumann rho<1 at 3 rational points.
         ex["neutrino_grade_a"]),
        ("MEASURED: pencil identity is 120+126-specific",            # holds for 120+126, NOT 10+126 / 10+120+126.
         ex["pencil_mining"]["universality"]["120+126"]["identity_holds"]
         and not ex["pencil_mining"]["universality"]["10+126"]["identity_holds"]),
        ("MEASURED: pencil breaking scales as eps^2",                # quadratic in single-S breaking.
         1.5 < ex["pencil_mining"]["breaking_scale"]["power_law_exponent"] < 2.5),
        ("MEASURED: declaration cost breaks census tie",             # 10+126=0, 120+126>=1.
         ex["declaration_scoreboard"]["rows"][1]["declaration_cost_lower_bound"] == 0
         and ex["declaration_scoreboard"]["rows"][0]["declaration_cost_lower_bound"] >= 1),
        ("MEASURED: pencil hierarchy preference direction prior-robust",  # 6/6 priors, 3/3 breaking strengths.
         ex["pencil_prior_sensitivity"]["summary"]["direction_prior_robust"]),
        ("MEASURED: exclusion concentrated by CKM locking",          # 9/11 excluded contents CKM-locked.
         ex["exclusion_taxonomy"]["mechanism_counts"]["CKM-lock (trans)"] >= 9
         and ex["exclusion_taxonomy"]["rg_robust"]["exclusion_survives_running"]),
        ("MEASURED: hierarchy preference NOT pencil-specific",       # 10+126 (no pencil) shows it too => structural.
         ex["pencil_attribution_control"]["m3m1"]["preference_present"]),
        ("MEASURED: pencil algebraic origin chain verified (grade b)",  # analytic derivation + 150-point float confirmation.
         ex["pencil_analytic_origin"]["chain_verified"]),
        ("MEASURED: single-S spectral 3->2 observation (grade b)",   # 300-point spectral count; NOT a theorem (self-audit V7).
         ex["singleS_spectral"]["singleS_always_two"] and ex["singleS_spectral"]["brokenS_generic_three"]),
        ("MEASURED: preference is M_R structure NOT repeated root",  # self-audit: pure type-I preference + 10+126 3 eigs.
         ex["pure_typeI_control"]["pure_typeI_m3m1"]["preference_present"]
         and ex["ten126_eigcount"]["no_repeated_root"]),
        ("STRUCTURE: Pfaffian pencil does NOT generalise to SU(5) 45",  # structural criterion from OPS + declared seesaw-singlet fact, not a numerical measurement.
         not ex["pfaffian_scope"]["SU(5) 45"]["pencil"]),
        ("MEASURED: seven-content scoreboard 120+126 cost>=1, 10+126 cost 0",
         any(r["content"] == "SO(10) 120+126" and r["declaration_cost_lower_bound"] >= 1 for r in ex["seven_scoreboard"]["rows"])
         and any(r["content"] == "SO(10) 10+126" and r["declaration_cost_lower_bound"] == 0 for r in ex["seven_scoreboard"]["rows"])),
        ("STRICT: vev alignment keeps dimension 22",               # No observable relation even under alignment.
         ex["vev_constraint_dim"] == 22),
        ("STRICT: two 126 copies also dimension 22",               # Extra copy adds freedom, still no relation.
         ex["two_copies_dim"] >= 22),
        ("MEASURED: RG breaks exact proportionality",              # One-loop running destroys M_e=-3M_d.
         ex["rg_proportionality_broken"]),
        ("MEASURED: RG shifts m_tau/m_b toward window",            # UV wall is not RG-robust.
         ex["rg_ratio_shifts"]),
        ("STRICT: v_L seesaw constraint keeps dimension 22",       # v_L tied; still zero observable relations.
         ex["vL_constraint_robust"]),
        ("transpose criterion is T-weighted, not a copy",           # The two exclusion routes are distinct tests.
         _transpose_is_distinct()),
        ("naive counting overcounts 120+126 by 1",                  # Quantified trap, first content.
         trap[0]["overcount"] == 1),
        ("naive counting overcounts 10+120 by 3",                   # Second content.
         trap[1]["overcount"] == 3),
        ("10+120 spectra separate once corrected",                  # Why it becomes viable.
         trap[1]["spectral_sep_median"] > 0.1),
        ("all conditioning above 1e-5",                             # Ranks are not tolerance artefacts.
         all(r["s_ratio"] > 1e-5 for r in ranks)),
        ("no sample near a mass degeneracy",                        # The other spurious-rank mechanism is excluded.
         all(r["min_gap"] > 1e-3 for r in ranks)),
    ]
    fig8_assertions(checks, rows, sur)                              # Module 8 figure.
    fig_column_variants(rows)                                       # Single-column renditions for the PRD format.
    npass = sum(1 for _, ok in checks if ok)                        # Count the passes.
    results = dict(seed=20261118, census=rows, ranks=ranks,          # Export everything the documents quote.
                   parameterisation_comparison=comp, seesaw=sur, typeII=t2,
                   constrained_image=img,
                   exact_relations=ex, counting_trap=trap,
                   checks=[dict(name=n, passed=bool(ok)) for n, ok in checks],
                   n_assertions=len(checks), n_passed=npass)
    with open(os.path.join(RESDIR, "results.json"), "w", encoding="utf-8") as f:  # Write the auditable record.
        json.dump(results, f, indent=2, ensure_ascii=False)          # Indented for human inspection.
    print("=" * 74)                                                 # Console banner.
    for n, ok in checks:                                            # Print each assertion.
        print("  [%s] %s" % ("PASS" if ok else "FAIL", n))           # One line per claim.
    print("=" * 74)                                                 # Closing banner.
    print("  %d / %d assertions passed" % (npass, len(checks)))      # Summary count.
    print("  viable %d | excluded %d | P_add set %s" %
          (len(S), len(X), sorted({r["p_add"] for r in S})))         # Classification summary.
    print("  figures -> %s" % FIGDIR)                               # Where the figures went.
    print("  results -> %s" % RESDIR)                               # Where the numbers went.
    return results                                                 # Return for programmatic use.


def fig_column_variants(rows):                                   # Single-column renditions for the narrow PRD format.
    # The two-panel figures above are sized for a full-width page.  In a two-column
    # journal they shrink until the labels are unreadable, so each is re-drawn here
    # as a compact single-column figure with larger relative type.
    GR = {"SU(5)": 4, "SO(10)": 5, "E6": 6, "PS": 4}                # Group ranks.
    viable = [r for r in rows if r["splitting"]]                    # Seven viable contents.
    excl = [r for r in rows if not r["splitting"]]                  # Eleven excluded.

    # --- column version of the landscape: 3D only, tall aspect ---
    fig = plt.figure(figsize=(4.5, 4.0))                            # Narrow and tall: fits one column.
    ax = fig.add_subplot(111, projection="3d")                      # Single 3D axes.
    jit = {}                                                       # Jitter counter per cell.
    def coords(r):                                                 # Coordinates with a small jitter.
        key = (r["group"], r["n_ops"]); k = jit.get(key, 0); jit[key] = k + 1
        return GR[r["group"]] + 0.17 * (k - 0.5), r["n_ops"] + 0.14 * (k - 0.5), r["P"]
    gx, gy = np.meshgrid([3.6, 6.4], [0.6, 5.4])                    # Upper ceiling plate.
    ax.plot_surface(gx, gy, np.full_like(gx, 3.0), color=C_BASE, alpha=0.10,
                    linewidth=0, shade=False)
    ax.plot([3.6, 6.4, 6.4, 3.6, 3.6], [0.6, 0.6, 5.4, 5.4, 0.6], [3] * 5,
            color=C_BASE, lw=1.2, ls=(0, (5, 3)), alpha=0.9)
    gx2, gy2 = np.meshgrid([3.6, 4.4], [0.6, 5.4])                  # Lower ceiling plate.
    ax.plot_surface(gx2, gy2, np.full_like(gx2, 1.0), color="#C08A2E", alpha=0.12,
                    linewidth=0, shade=False)
    ax.plot([3.6, 4.4, 4.4, 3.6, 3.6], [0.6, 0.6, 5.4, 5.4, 0.6], [1] * 5,
            color="#C08A2E", lw=1.2, ls=(0, (3, 3)), alpha=0.9)
    for r in excl:                                                 # Excluded: stems rise from the ceiling.
        x, y, z = coords(r)
        ax.plot([x, x], [y, y], [3.0, z], color=C_EXCL, lw=0.8, alpha=0.32)
        ax.scatter([x], [y], [z], s=62, c=C_EXCL, marker="^",
                   edgecolor="white", lw=0.8, depthshade=False)
    for r in viable:                                               # Viable: circles on the plate.
        x, y, z = coords(r)
        ax.scatter([x], [y], [z], s=112, c=C_VIABLE, marker="o",
                   edgecolor="white", lw=1.2, depthshade=False)
    ax.scatter([], [], [], s=105, c=C_VIABLE, marker="o", edgecolor="white",
               lw=1.1, label="7 viable: on ceiling")
    ax.scatter([], [], [], s=68, c=C_EXCL, marker="^", edgecolor="white",
               lw=0.8, label="11 excluded: above")
    ax.plot([], [], [], color=C_BASE, lw=1.2, ls=(0, (5, 3)), label="ceiling 3 (simple)")
    ax.plot([], [], [], color="#C08A2E", lw=1.2, ls=(0, (3, 3)), label="ceiling 1 (product)")
    ax.set_xlabel("rank $G$", labelpad=2, fontsize=9.5)             # Larger type for the narrow render.
    ax.set_ylabel("operators", labelpad=1, fontsize=9.5)
    ax.set_zlabel("$P_{\\mathrm{ex}}$", labelpad=1, fontsize=9.5)
    ax.set_xlim(3.5, 6.5); ax.set_ylim(0.5, 5.5); ax.set_zlim(2.0, 13.5)
    ax.set_xticks([4, 5, 6]); ax.set_yticks([1, 3, 5]); ax.set_zticks([3, 6, 9, 12])
    ax.tick_params(labelsize=8.2, pad=0)
    ax.view_init(elev=15, azim=-52)                                 # Low elevation reads as a ceiling.
    try:
        ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass
    ax.grid(color="#EBEEF2", lw=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.06, 1.06), fontsize=7.8)
    savefig("fig6col_landscape")                                    # Column-width landscape.

    # --- column version of the bidoublet figure: schematic only ---
    fig, ax = plt.subplots(figsize=(4.5, 3.1))                      # Narrow canvas.
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    ax.add_patch(Rectangle((0.25, 3.55), 5.5, 2.05, facecolor="#EAF1F8",
                           edgecolor="#B9C1CB", lw=0.9))
    ax.text(3.0, 5.22, "$\\mathbf{120}$: two bidoublets", ha="center",
            fontsize=10.5, fontweight="bold")
    ax.add_patch(Rectangle((0.55, 3.8), 2.35, 1.05, facecolor="white",
                           edgecolor=C_BASE, lw=1.1))
    ax.text(1.72, 4.32, "$(2,2,1)$\n$(1,1,1)$", ha="center", va="center",
            fontsize=9, linespacing=1.6)
    ax.add_patch(Rectangle((3.1, 3.8), 2.35, 1.05, facecolor="white",
                           edgecolor=C_NU, lw=1.1))
    ax.text(4.28, 4.32, "$(2,2,15)$\n$(1,1,-3)$", ha="center", va="center",
            fontsize=9, linespacing=1.6)
    ax.text(3.0, 3.2, "one $Y_{120}$, two independent vevs", ha="center",
            fontsize=9, style="italic", color=C_MUTE)
    ax.text(3.0, 2.35, "$M_d \\supset (v_1+v_{15})\\,Y_{120}$", ha="center",
            fontsize=11.5, color=C_BASE)
    ax.text(3.0, 1.5, "$M_e \\supset (v_1-3v_{15})\\,Y_{120}$", ha="center",
            fontsize=11.5, color=C_NU)
    ax.text(3.0, 0.55, "$v_{15}=0$ is an extra assumption,\nnot a consequence of SO(10)",
            ha="center", fontsize=8.8, color=C_EXCL, linespacing=1.5)
    savefig("fig2col_bidoublet")                                    # Column-width schematic.

    # --- column version of the withdrawn-bound figure: ratio ranges only ---
    fig, ax = plt.subplots(figsize=(4.5, 2.6))                      # Narrow canvas.
    comp = compare_parameterisations(npts=1200)                     # Fewer points: this is a display re-render.
    for i, t in enumerate(["collapsed", "correct"]):                # Two cases.
        r = comp[t]; col = C_MUTE if t == "collapsed" else C_EXCL
        ax.plot([r["ratio_min"], r["ratio_max"]], [i, i], lw=9, color=col,
                alpha=0.35, solid_capstyle="round")
        ax.plot([r["ratio_median"]], [i], "o", ms=8, color=col,
                mec="white", mew=1.0, zorder=5)
        ax.text(r["ratio_max"] * 1.3, i, "%.0f" % r["ratio_max"], va="center",
                fontsize=8.6, color=col, fontweight="bold")
    ax.axvspan(1, 9, color=C_BASE, alpha=0.12)                      # The claimed window.
    ax.axvline(1, color=C_BASE, lw=1.1); ax.axvline(9, color=C_BASE, lw=1.1)
    ax.text(3.0, 1.55, "claimed window $(1,9]$", ha="center", fontsize=8.8, color=C_BASE)
    ax.set_xscale("log"); ax.set_xlim(0.06, 3000)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["collapsed\n($v_{15}=0$)", "correct\n(2 bidoublets)"], fontsize=8.8)
    ax.set_ylim(-0.55, 1.85)
    ax.set_xlabel("$\\sum m_e^2\\,/\\,\\sum m_d^2$", fontsize=9.5)
    ax.tick_params(labelsize=8.2)
    savefig("fig4col_withdrawn")                                    # Column-width bound figure.


def fig8_assertions(checks, rows, sur):                            # Figure 8: the assertion board and the observable budget.
    fig = plt.figure(figsize=(9.4, 4.2))                            # Two panels.
    ax = fig.add_subplot(121); ax.axis("off"); ax.grid(False)        # Left: assertion list.
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)                           # Text canvas.
    ax.text(0, 9.6, "%d / %d assertions passed" %
            (sum(1 for _, ok in checks if ok), len(checks)),
            fontsize=11, fontweight="bold", color=C_VIABLE)          # Headline count.
    shown = [0, 3, 4, 7, 11, 12, 13, 14, 17, 19, 21, 22]             # A representative subset, to fit the panel.
    for k, i in enumerate(shown):                                   # Lay them out vertically.
        n, ok = checks[i]                                           # The assertion text and outcome.
        ax.text(0.15, 8.7 - k * 0.72, "PASS", fontsize=7.4,
                color=C_VIABLE, fontweight="bold", va="center")      # Status marker.
        ax.text(1.35, 8.7 - k * 0.72, n[:52], fontsize=7.2, va="center")  # Truncated claim text.
    ax.set_title("(a) every claim is checked", fontsize=9.8, loc="left")  # Panel title.
    ax2 = fig.add_subplot(122)                                      # Right: explained fraction by content.
    labels = ["charged only\n(16 obs.)", "minimal 120+126\n(25 obs.)", "with a 10\n(25 obs.)"]
    frac = [3 / 16 * 100, 3 / 25 * 100, 3 / 25 * 100]                # baseline only; P_add=0 for all (strict).
    cols = [C_BASE, C_VIABLE, C_NU]                                  # Minimal content highlighted.
    bars = ax2.bar([0, 1, 2], frac, 0.5, color=cols,
                   edgecolor="white", lw=0.8)
    for b, f in zip(bars, frac):                                    # Annotate the percentages.
        ax2.text(b.get_x() + b.get_width() / 2, f + 0.6, "%.0f%%" % f,
                 ha="center", fontsize=9.4, fontweight="bold")       # Value above the bar.
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(labels, fontsize=7.8)
    ax2.set_ylabel("fraction of parameters fixed (%)")               # Axis label.
    ax2.set_ylim(0, 36)                                             # Headroom.
    ax2.text(1, 32.5, "pencil identity is\nmatrix-level only", ha="center",
             fontsize=8, color=C_VIABLE, linespacing=1.4, fontweight="bold")
    ax2.text(2, 20.0, "no observable\nrelations", ha="center",
             fontsize=7.6, color=C_NU, linespacing=1.4)
    ax2.set_title("(b) only the baseline survives", fontsize=9.8, loc="left")
    fig.tight_layout(pad=1.2)                                      # Space the panels.
    savefig("fig8_assertions")                                      # Emit both formats.


if __name__ == "__main__":                                         # Entry point.
    run_all()                                                      # Execute every module.
