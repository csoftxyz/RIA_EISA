# -*- coding: utf-8 -*-   # UTF-8 source file: every line carries an inline comment.
# ==============================================================================================   # Divider: MODULE 0 (infrastructure) follows.
# ===== MODULE 0 | infrastructure | fig_0_infrastructure.png =====   # Module 0: imports, paths, RNG seed, plot style; the figure-0 routine ends this block.
import json                                                           # json: export every numerical result so the paper's tables can be checked item by item.
import os                                                             # os: build cross-platform paths and create output directories.
import html                                                           # html: escape < > & when generating the code reading page.
from fractions import Fraction as F                                    # Fraction: rank of C-tilde is computed in exact rational arithmetic, no floating tolerance.
from itertools import combinations                                    # combinations: enumerate every non-empty subset of Higgs representations for each group.
from collections import namedtuple                                    # namedtuple: name the operator records instead of indexing tuples.
import numpy as np                                                    # numpy: SVD, eigh, random sampling and numerical Jacobians.
import matplotlib                                                     # matplotlib: headless backend and global style.
matplotlib.use("Agg")                                                 # Agg backend: figures can be saved on a server with no display.
import matplotlib.pyplot as plt                                       # pyplot: single drawing interface for every figure.
from matplotlib.patches import Rectangle                              # Rectangle: used by the module-0 flow diagram.
HERE = os.path.dirname(os.path.abspath(__file__))                      # Directory of this script: base for every relative path.
FIGDIR = os.path.join(HERE, "figures")                                # Figure output of the code supplement: revision/code_supplement/figures.
PAPERFIG = os.path.join(HERE, "..", "paper", "figures")              # Vector figures for the paper: revision/paper/figures.
os.makedirs(FIGDIR, exist_ok=True)                                    # Create the figure directory; silently skipped if it exists.
os.makedirs(PAPERFIG, exist_ok=True)                                  # Create the paper figure directory for LaTeX \includegraphics.
RNG = np.random.default_rng(20261113)                                # Fixed seed: every number and every figure is reproducible bit for bit.
plt.rcParams.update({"font.size": 9, "axes.grid": True,               # One global plot style: font size, grid, line width, resolution.
                     "grid.alpha": 0.3, "lines.linewidth": 1.6,       # Grid transparency and line width: legible but not dominant.
                     "figure.dpi": 130, "savefig.dpi": 160})          # Screen and export resolution: crisp PNG, vector PDF.
C_SPLIT = "#1D9E75"                                                  # Colour of the splitting models: green, matching the paper's 3D figure.
C_EXCL = "#D85A30"                                                   # Colour of the excluded models: orange, likewise.
C_BLUE = "#378ADD"                                                   # Colour of the baseline / gauge part: blue.

def fig_0_infrastructure():                                            # Figure 0: inputs, modules and output files of this script.
    fig, ax = plt.subplots(figsize=(9, 3.4))                           # Create the canvas and switch the axes off.
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")            # Hand-drawn layout in 0..10 / 0..4.2 canvas coordinates.
    def box(x, y, w, h, text, fc):                                      # Small helper: a rounded box carrying text.
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,           # Box fill colour as passed in.
                                   edgecolor="#888780", lw=0.8))         # Thin grey border, keeping the light theme.
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",     # Text centred inside the box.
                fontsize=8.5)                                            # Small font so that multi-line text fits.
    box(0.2, 1.5, 2.4, 1.2, "annotated_verification.py\nseed 20261113\nnumpy + matplotlib", "#E6F1FB")   # Left: the single script.
    box(3.4, 1.5, 2.2, 1.2, "run_all()\nMODULES 0..8\n20 assertions", "#E1F5EE")   # Centre: main driver and the nine modules.
    box(6.4, 2.5, 3.3, 0.85, "verification_results.json (all numbers)", "#FAEEDA")   # Top right: the JSON source of numbers.
    box(6.4, 1.5, 3.3, 0.85, "figures/*.png (9) + paper/figures/*.pdf (3)", "#FAEEDA")   # Middle right: all figures.
    box(6.4, 0.5, 3.3, 0.85, "index.html (code + figure per module)", "#FAEEDA")   # Bottom right: the reading page.
    ax.annotate("", xy=(3.35, 2.1), xytext=(2.65, 2.1),                 # Arrow left to centre.
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#5F5E5A"))   # Grey arrow.
    for yy in (2.9, 1.9, 0.9):                                          # Three arrows from centre to right.
        ax.annotate("", xy=(6.35, yy), xytext=(5.65, 2.1),               # Emitted from the right edge of the driver box.
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#5F5E5A"))   # Thin grey arrow.
    ax.set_title("Module 0: inputs, modules and reproducible outputs", fontsize=10)   # Figure title.
    savefig("fig_0_infrastructure")                                      # Call the savefig routine defined in module 6 (already defined at run time).

# ==============================================================================================   # Divider: MODULE 1 (group-theory input and exact census).
# ===== MODULE 1 | census | fig_1_census.png =====   # Module 1: Clebsch table, coefficient matrix C-tilde, the eighteen contents and the revised counting.
Op = namedtuple("Op", "name cu cd ce kind")                          # Operator record: name, three Clebsch coefficients, flavour-matrix type.
OPS = {                                                               # Group-theory input: each operator = (name, c_u, c_d, c_e, type).
    "SU(5)": {                                                        # SU(5): 5_H, 45_H, 50_H; up and down/lepton use different flavour matrices.
        "5":  [("10.10.5", 1, 0, 0, "sym"),                         # 5_H up operator 10.10.5: symmetric part of 10x10, CG = (1,0,0).
                ("10.5b.5b", 0, 1, 1, "genT")],                      # 5_H down operator 10.5b.5b: general matrix with M_e = Y^T (type genT).
        "45": [("10.10.45", 1, 0, 0, "anti"),                       # 45_H up operator 10.10.45: antisymmetric part of 10x10, CG = (1,0,0).
                ("10.5b.45b", 0, 1, -3, "genT")],                    # 45_H down operator: Georgi-Jarlskog factor, +1 for d and -3 for e.
        "50": [("10.10.50", 1, 0, 0, "sym")],                       # 50_H appears only in 10x10, so it feeds up alone; 10x5b = 5 + 45 contains no 50.
    },                                                                          # Close the SU(5) sub-table: three representations.
    "SO(10)": {                                                       # SO(10): 16x16 = 10_s + 120_a + 126_s, the matrix type follows from the representation.
        "10":  [("16.16.10", 1, 1, 1, "sym")],                      # 10_H: CG = (1,1,1), symmetric flavour matrix.
        "120": [("16.16.120", 1, -1, 1, "anti")],                   # 120_H: CG = (1,-1,1), antisymmetric flavour matrix.
        "126": [("16.16.126", 1, 1, -3, "sym")],                    # 126_H: CG = (1,1,-3), symmetric, with the Georgi-Jarlskog factor -3 in the lepton sector.
    },                                                                          # Close the SO(10) sub-table.
    "E6": {                                                           # E6: only the single invariant contraction 27^3 is used (scope discussed in the Supplement).
        "27": [("27.27.27", 1, 1, 1, "sym")],                       # 27^3: CG = (1,1,1), one symmetric flavour matrix.
    },                                                                          # Close the E6 sub-table.
    "PS": {                                                           # Pati-Salam: the two bidoublets of SU(4) x SU(2)_L x SU(2)_R.
        "(1,2,2)":  [("FFc.h", 1, 1, 1, "gen")],                    # (1,2,2): CG = (1,1,1), general flavour matrix.
        "(15,2,2)": [("FFc.15", 1, 1, -3, "gen")],                  # (15,2,2): the 15 supplies the -3 factor for leptons along B-L.
    },                                                                          # Close the Pati-Salam sub-table.
}                                                                               # Close OPS: every Yukawa operator of the four groups.
SIMPLE = {"SU(5)", "SO(10)", "E6"}                                   # Simple groups only: these are the ones that unify the gauge couplings (two relations).
KIND_DOF = {"sym": 12, "anti": 6, "gen": 18, "genT": 18}            # Real degrees of freedom per 3x3 complex matrix: 6/3/9 complex entries, times two.

def expand_ops(group, reps):                                          # Expand one Higgs content into its operator list, preserving order for the parameter slicing.
    return [Op(*t) for r in reps for t in OPS[group][r]]             # Concatenate operators representation by representation; each record becomes a named tuple.

def tilde_C_columns(ops):                                             # Columns of the effective coefficient matrix C-tilde: up columns and d/e columns separated.
    cols = []                                                         # Column list; each entry is a three-dimensional rational vector.
    for a in ops:                                                     # One operator can contribute two columns, because the up and down vevs are independent.
        if a.cu != 0:                                                 # If the operator feeds the up sector, register an up-type parameter v_{a,u} Y_a.
            cols.append((F(a.cu), F(0), F(0)))                       # Up column = (c_u, 0, 0)^T: the up direction decouples from the other sectors.
        if a.cd != 0 or a.ce != 0:                                   # If the operator feeds down/lepton, register a d/e-type parameter v_{a,d} Y_a.
            cols.append((F(0), F(a.cd), F(a.ce)))                    # d/e column = (0, c_d, c_e)^T: d and e share one vev.
    return cols                                                       # Return the columns of a 3xm matrix, m = up columns + d/e columns.

def rank_exact(cols):                                                 # Exact rank by Fraction Gaussian elimination: the tabulated rank carries no floating error.
    if not cols:                                                      # Boundary case: no columns means rank zero.
        return 0                                                      # Return zero immediately, avoiding an out-of-range index later.
    M = [[cols[j][i] for j in range(len(cols))] for i in range(3)]   # Transpose into a rational matrix with three rows for row-wise elimination.
    r = 0                                                             # Current pivot row, which also equals the rank found so far.
    for c in range(len(cols)):                                        # Scan columns from left to right for a pivot.
        piv = next((i for i in range(r, 3) if M[i][c] != 0), None)   # First non-zero entry at or below row r in column c.
        if piv is None:                                               # The column is linearly dependent; no pivot available.
            continue                                                  # Skip this column and continue scanning.
        M[r], M[piv] = M[piv], M[r]                                  # Row swap: bring the pivot row to position r.
        for i in range(3):                                            # Eliminate against the other two rows to reach column echelon form.
            if i != r and M[i][c] != 0:                               # Elimination is needed only where row i is non-zero in this column.
                f = M[i][c] / M[r][c]                                # Elimination factor, kept rational so that no rounding occurs.
                M[i] = [M[i][k] - f * M[r][k] for k in range(len(cols))]   # Subtract the multiple of the pivot row from the whole row.
        r += 1                                                        # Increment the pivot count.
        if r == 3:                                                    # A three-row matrix has rank at most three; stop early.
            break                                                     # Leave the column scan.
    return r                                                          # Return the exact rank, one of 0, 1, 2, 3.

def _all_proportional(vec):                                         # Are all two-component vectors pairwise proportional?
    for v in vec[1:]:                                                # Compare each vector with the first through the 2x2 cross product.
        if vec[0][0] * v[1] - vec[0][1] * v[0] != 0:                 # Non-zero cross product means two independent columns.
            return False                                             # Not proportional.
    return True                                                      # All proportional.

def de_proportional(ops):                                            # Is d/e forced into a fixed ratio, directly or through the transpose?
    vec = [(F(a.cd), F(a.ce)) for a in ops if a.cd != 0 or a.ce != 0]  # Collect the non-zero d/e columns.
    if not vec:                                                      # With no d/e column at all the fed flag handles the case separately.
        return False                                                 # Return False so that "not fed" is never mistaken for "proportional".
    if _all_proportional(vec):                                       # Direct proportionality, M_e = k M_d with k fixed.
        return True                                                  # Excluded.
    Tvec = [(F(a.cd) * (-1 if a.kind == "anti" else 1), F(a.ce))      # Transpose route: an antisymmetric operator flips sign.
            for a in ops if a.cd != 0 or a.ce != 0]                  # Same operator set, antisymmetric signs flipped.
    return _all_proportional(Tvec)                                   # If proportional, M_e = k M_d^T and the spectra coincide.

def counting(group, fed, splitting, single):                          # Revised counting: five mass relations (not six), four CKM, two gauge, one discrete charge.
    if not fed:                                                       # Not all three sectors are fed: the model cannot accommodate the spectrum at all.
        mass, ckm = 0, 0                                             # Record zero relations and mark the row N/A in the table.
    elif splitting:                                                   # Splitting: the mass-mixing sector carries no identity relation.
        mass, ckm = 0, 0                                             # Zero continuous relations; this is what the main theorem proves.
    else:                                                             # Not splitting: M_e = k M_d with k fixed gives three fixed-ratio relations.
        mass = 3 + (2 if single else 0)                              # A single operator also has M_u = t M_d with t = tan(beta) free, so three equal ratios give two.
        ckm = 4 if single else 0                                     # For a single operator V_CKM = 1, fixing four CKM parameters.
    gauge = 2 if group in SIMPLE else 0                               # Simple group: three gauge couplings become one, hence two relations; zero for Pati-Salam.
    charge = 1                                                        # Charge quantisation: one discrete, structural constraint, not a continuous codimension.
    p_cont = mass + ckm + gauge                                      # Continuous relation count: masses + mixing + gauge unification.
    p_ex = p_cont + charge                                           # Tabulated P_ex keeps the discrete constraint as one for comparison with the earlier draft.
    return dict(mass=mass, ckm=ckm, gauge=gauge, charge=charge,       # Return every component so that each table column can be checked.
                p_cont=p_cont, p_ex=p_ex)                                       # Continuation: the continuous count and the tabulated P_ex go into the same dictionary.

def evaluate(group, reps):                                            # Produce one full row for a Higgs content: the unit of the eighteen-row census.
    ops = expand_ops(group, reps)                                     # Expand the operator list of that content.
    rk = rank_exact(tilde_C_columns(ops))                             # Exact rank of C-tilde: one, two or three in the revised version.
    fed = (any(a.cu != 0 for a in ops) and                            # fed: each of u, d, e is reached by at least one operator.
           any(a.cd != 0 for a in ops) and                                      # Continuation of fed: the d sector is reached.
           any(a.ce != 0 for a in ops))                                         # Continuation of fed: the e sector is reached.
    splitting = bool(fed and not de_proportional(ops))                # splitting exactly when fed and the (c_d,c_e) columns are not proportional.
    single = bool(fed and len(ops) == 1)                              # Single operator: the three sectors are scalar multiples of one flavour matrix.
    d = counting(group, fed, splitting, single)                       # Apply the revised counting rules.
    d.update(group=group, model="+".join(reps), n_ops=len(ops),       # Identification fields: group name, content name, operator count.
             rankC=rk, fed=bool(fed), splitting=splitting,             # Structural fields: rank of C-tilde, fed, splitting.
             single=single, ops=[a.name for a in ops])                 # Single-operator flag and operator names, for JSON cross-checking.
    return d                                                          # Return the complete record of that content.

def census():                                                         # Enumerate every non-empty subset of representations, giving the eighteen rows.
    rows = []                                                         # Initialise the result list.
    for g, reps in OPS.items():                                       # Loop over the four groups and their representation dictionaries.
        for k in range(1, len(reps) + 1):                             # Subset size from one up to the number of representations.
            for ch in combinations(list(reps), k):                    # Enumerate all combinations of that size.
                rows.append(evaluate(g, ch))                          # Evaluate and collect row by row.
    return rows                                                       # Return eighteen rows (7 + 7 + 1 + 3).

# ==============================================================================================   # Divider: MODULE 2 (physical observables).
# ===== MODULE 2 | observables | fig_2_observables.png =====   # Module 2: bi-unitary invariants (singular values plus CKM), with the counterexample on |eigenvalues|.
def left_unitary(M):                                                  # Left unitary matrix from the spectral decomposition of MM-dagger, columns sorted by decreasing singular value.
    _, U = np.linalg.eigh(M @ M.conj().T)                            # eigh returns ascending eigenvalues s^2 and the eigenvectors U.
    return U[:, ::-1]                                                # Reverse the columns so that column k belongs to the k-th largest singular value, as in SVD.

def ckm_matrix(Mu, Md):                                               # Physical CKM matrix V = V_{uL}^dagger V_{dL}.
    return left_unitary(Mu).conj().T @ left_unitary(Md)               # Product of two left unitary matrices; column ordering is already aligned, so no arbitrary permutation.

def observables(Ms):                                                  # Thirteen physical observables: nine singular values, three |V_ij|, and the Jarlskog invariant.
    Mu, Md, Me = Ms                                                  # Unpack the three mass matrices.
    out = []                                                          # Initialise the output list.
    for M in (Mu, Md, Me):                                            # Singular values per sector: only these are bi-unitary invariants, hence physical masses.
        out += list(np.linalg.svd(M, compute_uv=False))               # SVD singular values are already decreasing and are invariant under M -> U_L M U_R^dagger.
    V = ckm_matrix(Mu, Md)                                            # CKM matrix, used for the three mixing moduli and the CP invariant.
    out += [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2])]                # Three independent mixing moduli |V_us|, |V_ub|, |V_cb|.
    out += [(V[0, 0] * V[1, 1] * np.conj(V[0, 1]) * np.conj(V[1, 0])).imag]   # Jarlskog invariant J = Im(V_ud V_cs V_us* V_cd*).
    return np.array(out, dtype=float)                                 # Return a thirteen-dimensional real vector: the image point of pi(P).

def degeneracy_ok(Ms, tol=1e-4):                                      # Test whether a parameter point lies in the open dense set of non-degenerate spectra and non-zero mixings.
    for M in Ms:                                                      # Check the singular-value gap sector by sector.
        s = np.linalg.svd(M, compute_uv=False)                        # The three singular values of that sector, in decreasing order.
        if np.min(-np.diff(s)) < tol:                                 # Two adjacent singular values too close: pi is not differentiable here, so resample.
            return False                                              # Reject the point.
    V = ckm_matrix(Ms[0], Ms[1])                                     # CKM matrix, to check every entry that enters |.| and J.
    idx = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]           # Indices of the six entries that must not vanish.
    if min(abs(V[i, j]) for i, j in idx) < tol:                        # Any vanishing entry makes |.| non-differentiable and the Jacobian unreliable.
        return False                                                  # Reject the point.
    return True                                                       # Accepted: pi is real-analytic at this point.

# ==============================================================================================   # Divider: MODULE 3 (parameterisation and Jacobian).
# ===== MODULE 3 | jacobian | fig_3_jacobian.png =====   # Module 3: free-vev parameterisation and the Jacobian witnesses for the splitting models.
def param_layout(ops):                                                # Plan the parameter slicing: each operator gets [real entries of Y][complex v_u: two reals][complex v_d: two reals].
    layout = []                                                       # Layout entries: (operator, Y offset, Y length, v_u index, v_d index).
    k = 0                                                             # Parameter cursor, starting at zero.
    for a in ops:                                                     # Allocate the parameter block of each operator in turn.
        ny = KIND_DOF[a.kind]                                         # Number of real parameters taken by a flavour matrix of this type.
        vu = k + ny if a.cu != 0 else None                           # Index of the real part of the complex v_{a,u}; the imaginary part occupies the next slot.
        vd = k + ny + (2 if a.cu != 0 else 0)                        # Index of v_{a,d}, placed after the two components of v_u.
        if a.cd == 0 and a.ce == 0:                                   # An operator that feeds up only has no d/e-type vev.
            vd = None                                                 # Set to None; masses_from skips the d/e accumulation.
        layout.append((a, k, ny, vu, vd))                             # Record the complete slicing information for this operator.
        k += ny + (2 if vu is not None else 0) + (2 if vd is not None else 0)   # Advance the cursor: Y, optional complex v_u, optional complex v_d.
    return layout, k                                                  # Return the layout and the total number of parameters n.

def make_Y(kind, block):                                              # Rebuild a 3x3 complex flavour matrix from a block of real parameters, respecting the symmetry type.
    Y = np.zeros((3, 3), dtype=complex)                              # Start from a 3x3 zero matrix.
    k = 0                                                             # Cursor inside the block.
    if kind in ("sym", "anti"):                                       # Symmetric or antisymmetric: only the upper triangle is independent.
        pairs = [(i, j) for i in range(3) for j in range(i, 3)        # Enumerate upper-triangle indices; the antisymmetric case also drops the diagonal.
                 if not (kind == "anti" and i == j)]                            # Continuation: an antisymmetric matrix has zero diagonal, so i = j is excluded.
        for (i, j) in pairs:                                          # Fill in the complex parameters one by one.
            Y[i, j] = block[k] + 1j * block[k + 1]                    # Two real parameters make one complex number.
            Y[j, i] = Y[i, j] if kind == "sym" else -Y[i, j]           # Copy for the symmetric case, flip the sign for the antisymmetric one.
            k += 2                                                    # Advance the block cursor by two.
    else:                                                             # gen / genT: general complex matrix with nine independent entries.
        for i in range(3):                                            # Row loop.
            for j in range(3):                                        # Column loop.
                Y[i, j] = block[k] + 1j * block[k + 1]                 # Assemble the complex number.
                k += 2                                                # Advance the block cursor by two.
    return Y                                                          # Return the reconstructed flavour matrix.

def masses_from(p, layout):                                           # Build (M_u, M_d, M_e) from the parameter vector, with free vevs and the SU(5) transpose for e.
    Mu = np.zeros((3, 3), dtype=complex)                              # Initialise the up-type mass matrix.
    Md = np.zeros((3, 3), dtype=complex)                              # Initialise the down-type mass matrix.
    Me = np.zeros((3, 3), dtype=complex)                              # Initialise the charged-lepton mass matrix.
    for a, y0, ny, vu, vd in layout:                                  # Accumulate the three sectors operator by operator.
        Y = make_Y(a.kind, p[y0:y0 + ny])                             # Rebuild the flavour matrix of this operator from its slice.
        if vu is not None:                                            # An independent complex v_{a,u} feeds the up sector.
            cu_v = p[vu] + 1j * p[vu + 1]                             # Assemble the complex vev: real part plus i times imaginary part; the relative phase is physical.
            Mu += a.cu * cu_v * Y                                    # M_u = sum_a c_{u,a} v_{a,u} Y_a.
        if vd is not None:                                            # A shared complex vev v_{a,d} = v_{a,e} feeds d and e.
            cd_v = p[vd] + 1j * p[vd + 1]                             # Assemble the shared d/e complex vev.
            Md += a.cd * cd_v * Y                                    # M_d = sum_a c_{d,a} v_{a,d} Y_a.
            Me += a.ce * cd_v * (Y.T if a.kind == "genT" else Y)      # M_e = sum_a c_{e,a} v_{a,d} Y_a^T, the genT branch of SU(5).
    return Mu, Md, Me                                                 # Return the three mass matrices.

def obs_from(p, layout):                                              # Composite map pi: parameter vector to the thirteen physical observables.
    return observables(masses_from(p, layout))                        # Build the mass matrices first, then take the bi-unitary invariants.

def jacobian(p, layout, h=1e-6):                                      # Central-difference Jacobian D-pi of size 13 x n.
    n = len(p)                                                        # Total number of parameters.
    J = np.zeros((13, n), dtype=float)                                # Initialise the Jacobian; thirteen rows for thirteen observables.
    for j in range(n):                                                # Differentiate with respect to every parameter direction.
        pp = p.copy()                                                 # Copy of the parameters for the forward step.
        pm = p.copy()                                                 # Copy of the parameters for the backward step.
        pp[j] += h                                                    # Forward step +h.
        pm[j] -= h                                                    # Backward step -h.
        J[:, j] = (obs_from(pp, layout) - obs_from(pm, layout)) / (2 * h)   # Central difference (f(p+h) - f(p-h)) / 2h.
    return J                                                          # Return the numerical Jacobian.

def random_p(layout, rng):                                            # Draw a well-conditioned parameter point: Y standard normal, vev real parts positive, imaginary parts free.
    n = sum(ny + 2 * ((vu is not None) + (vd is not None))             # Recover the parameter count from the layout; each complex vev takes two reals.
            for _, _, ny, vu, vd in layout)                                     # Continuation: add matrix entries and the real and imaginary parts of each vev.
    p = rng.normal(size=n)                                            # Start every parameter from a standard normal draw.
    for a, y0, ny, vu, vd in layout:                                  # Place the complex vevs away from the origin so the overall scale is not degenerate.
        if vu is not None:                                            # Set the two components of v_u when it exists.
            p[vu] = 0.9 + 0.3 * rng.normal()                          # Real part away from zero, avoiding a degenerate overall scale.
            p[vu + 1] = 0.4 * rng.normal()                            # Imaginary part: CP phases are allowed, no reality restriction.
        if vd is not None:                                            # Set v_d in the same way.
            p[vd] = 0.9 + 0.3 * rng.normal()                          # Real part of the shared d/e vev.
            p[vd + 1] = 0.4 * rng.normal()                            # Imaginary part of the shared d/e vev, independent of v_u.
    return p                                                          # Return the parameter vector.

def witness_for(row, trials=10):                                      # Numerical rank of D-pi and an explicit witness point for one splitting model.
    ops = expand_ops(row["group"], row["model"].split("+"))           # Recover the operator list from the content name.
    layout, n = param_layout(ops)                                     # Build the parameter layout and its dimension.
    best = None                                                       # Keep the sample with the highest rank and the best conditioning ratio.
    ranks = []                                                        # Collect the rank at every sampled point, to test whether it is constant.
    for t in range(trials):                                           # Try at most ten random points, skipping degenerate ones.
        p = random_p(layout, RNG)                                     # Draw a random parameter point.
        Ms = masses_from(p, layout)                                   # Build the three mass matrices.
        if not degeneracy_ok(Ms):                                     # Skip when the spectrum is degenerate or a mixing vanishes, so that pi stays analytic.
            continue                                                  # Continue with the next sample.
        J = jacobian(p, layout)                                       # Compute the 13 x n numerical Jacobian.
        if not np.all(np.isfinite(J)):                                # Skip on NaN or Inf.
            continue                                                  # Continue with the next sample.
        sv = np.linalg.svd(J, compute_uv=False)                       # Take the thirteen singular values; n >= 13 holds for every splitting model.
        rank = int((sv > sv[0] * 1e-7).sum())                         # Numerical rank at a relative threshold of 1e-7.
        ranks.append(rank)                                            # Record the rank at this point, to test constancy across samples.
        ratio = float(sv[-1] / sv[0])                                 # Ratio of the thirteenth to the largest singular value: a conditioning indicator.
        gap = float(min(np.min(-np.diff(np.linalg.svd(M, compute_uv=False))) for M in Ms))   # Smallest spectral gap, witnessing non-degeneracy.
        cand = dict(model=row["group"] + " " + row["model"],           # Assemble the witness record.
                    n_params=int(n), rank=rank, s_ratio=ratio,          # Parameter dimension, numerical rank, conditioning ratio.
                    p_add=int(13 - rank),                               # Extra relations in the mass-mixing sector: 13 minus rank D-pi.
                    min_gap=gap, singular_values=sv.tolist(),           # Smallest spectral gap and the full singular-value spectrum, for the paper figure.
                    seed_point=p.tolist())                              # The witness parameter vector, reproducible digit by digit.
        if best is None or (cand["rank"], cand["s_ratio"]) > (best["rank"], best["s_ratio"]):   # Prefer a higher rank first, then better conditioning.
            best = cand                                               # Update the best record.
    best["rank_set"] = sorted(set(ranks))                             # Set of ranks seen across all samples of this model.
    best["n_trials"] = len(ranks)                                     # Number of effective samples, reported in the paper.
    return best, layout, n                                            # Return the witness, the layout and the parameter dimension, reused by module 5.

def generic_controls(n_points=6):
    """Two unstructured controls, to show that a rank deficit is a property of the
    antisymmetric coupling and not of the numerical machinery.

    (a) three general complex matrices, one feeding each sector;
    (b) two general complex matrices, both feeding all three sectors.
    Both configurations have far more parameters than observables, so both must
    reach rank thirteen. The splitting models that fall short do so because of
    their structure, not because the Jacobian cannot reach thirteen.
    """
    layouts = {
        "three_general": param_layout([Op("g1", 1, 0, 0, "gen"),
                                       Op("g2", 0, 1, 0, "gen"),
                                       Op("g3", 0, 0, 1, "gen")]),
        "two_general_all_sectors": param_layout([Op("g1", 1, 1, 1, "gen"),
                                                 Op("g2", 1, 1, -3, "gen")]),
    }
    out = {}
    for tag, (layout, _n) in layouts.items():
        best = 0
        for _ in range(n_points):
            p = random_p(layout, RNG)
            Ms = masses_from(p, layout)
            if not degeneracy_ok(Ms):
                continue
            Jm = jacobian(p, layout)
            if not np.all(np.isfinite(Jm)):
                continue
            sv = np.linalg.svd(Jm, compute_uv=False)
            best = max(best, int((sv > sv[0] * 1e-7).sum()))
        out[tag] = best
    return out

# ==============================================================================================   # Divider: MODULE 4 (mixed invariants) follows.
# ===== MODULE 4 | identity_bank | fig_4_identity_bank.png =====   # Module 4: physical mixed invariants plus the reconstruction test from the thirteen observables; figure 4.
def hermitian(M):                                                     # Build H = M M-dagger: it transforms by conjugation under a common left unitary rotation, hence is invariant.
    return M @ M.conj().T                                             # Return the 3x3 Hermitian matrix.

def mixed_invariants(Ms):                                             # Six cross-sector physical invariants: joint traces of H_s = M_s M_s-dagger.
    Hu, Hd, He = (hermitian(M) for M in Ms)                           # Build the Hermitian combinations for u, d and e.
    return np.array([np.trace(Hu @ Hd).real,                          # I1 = Re tr(H_u H_d): the lowest-order mixed invariant.
                     np.trace(Hu @ He).real,                          # I2 = Re tr(H_u H_e).
                     np.trace(Hd @ He).real,                          # I3 = Re tr(H_d H_e).
                     np.trace(Hu @ Hu @ Hd).real,                     # I4 = Re tr(H_u^2 H_d): a higher-order mixed invariant.
                     np.trace(Hu @ Hd @ Hd).real,                     # I5 = Re tr(H_u H_d^2).
                     np.trace(Hu @ Hd @ He).real])                    # I6 = Re tr(H_u H_d H_e): the joint three-sector invariant.

def rebuild_trHuHd(Ms, V):                                            # Identity test: tr(H_u H_d) = sum_{ij} s_{u,i}^2 s_{d,j}^2 |V_ij|^2.
    su = np.linalg.svd(Ms[0], compute_uv=False)                       # Up-sector singular values, in decreasing order.
    sd = np.linalg.svd(Ms[1], compute_uv=False)                       # Down-sector singular values, in decreasing order.
    return float(sum(su[i] ** 2 * sd[j] ** 2 * abs(V[i, j]) ** 2      # Rebuild the mixed invariant from the nine masses and the squared CKM moduli.
                     for i in range(3) for j in range(3)))                      # Continuation of the reconstruction formula: double sum over the 3x3 indices.

def identity_scan(layout, n_points=300):                              # Scan the mixed invariants and the reconstruction error over three hundred well-conditioned points.
    vals, errs, ratios = [], [], []                                   # Collect the six invariants, the relative reconstruction error and the cross-sector ratio.
    for _ in range(n_points):                                         # Three hundred samples, matching the statement in the paper.
        p = random_p(layout, RNG)                                     # Draw a random parameter point.
        Ms = masses_from(p, layout)                                   # Build the mass matrices.
        if not degeneracy_ok(Ms):                                     # Skip points at which pi is not analytic.
            continue                                                  # Continue sampling.
        I = mixed_invariants(Ms)                                      # Compute the six mixed invariants.
        V = ckm_matrix(Ms[0], Ms[1])                                 # Take the CKM matrix for the reconstruction.
        rec = rebuild_trHuHd(Ms, V)                                   # Rebuild I1 from the thirteen observables.
        vals.append(I.tolist())                                       # Record the six invariants.
        errs.append(abs(rec - I[0]) / max(abs(I[0]), 1e-30))          # Record the relative reconstruction error, which should reach machine precision.
        ratios.append(I[0] / I[1] if abs(I[1]) > 1e-12 else np.nan)   # Record the I1/I2 cross-sector ratio: if it is not constant there is no fixed ratio relation.
    return np.array(vals), np.array(errs), np.array(ratios)           # Return the three arrays for statistics and plotting.

# ==============================================================================================   # Divider: MODULE 5 (witness and tan-beta subspace) follows.
# ===== MODULE 5 | witness_and_subspace | fig_5_witness.png =====   # Module 5: detailed witness tables plus the determinant identity on the shared-tan-beta subspace; figure 5.
def trdet_shared(tb, n=200):                                          # Shared-tan-beta subspace, not the convention of this paper: every operator shares v_u/v_d = tan beta.
    wt = wd = wdabs = 0.0                                            # Record the largest trace difference, determinant difference and |determinant| difference.
    for _ in range(n):                                                # Two hundred random points.
        Y0 = make_Y("sym", RNG.normal(size=12))                       # Symmetric flavour matrix of 10_H.
        Y1 = make_Y("anti", RNG.normal(size=6))                       # Antisymmetric flavour matrix of 120_H.
        Y2 = make_Y("sym", RNG.normal(size=12))                       # Symmetric flavour matrix of 126_H.
        r = np.exp(RNG.normal(size=3))                                # Three positive magnitude coefficients for the vevs.
        S = r[0] * Y0 + r[2] * Y2                                     # Symmetric part S.
        A = r[1] * Y1                                                 # Antisymmetric part A.
        Mu = tb * (S + A)                                             # Up-type matrix on the shared-tan-beta subspace.
        Md = S - A                                                    # Down-type matrix; the 120 enters with coefficient -1.
        wt = max(wt, abs(np.trace(Mu) - np.trace(Md)))                # Trace difference: exactly zero at tan beta = 1 because A is traceless.
        wd = max(wd, abs(np.linalg.det(Mu) - np.linalg.det(Md)))      # Determinant difference, a complex quantity carrying unphysical phases.
        wdabs = max(wdabs, abs(abs(np.linalg.det(Mu)) - abs(np.linalg.det(Md))))   # |determinant| difference = |prod m_u - prod m_d|, which is the physical statement.
    return wt, wd, wdabs                                              # Return the three largest deviations.

def trdet_free(n=200):                                                # Convention of this paper: v_u and v_d of each representation are independent.
    wdabs = 0.0                                                       # Record only the physical |determinant| difference.
    for _ in range(n):                                                # Two hundred random points.
        ops = expand_ops("SO(10)", ["10", "120", "126"])               # Take the operators of 10 + 120 + 126.
        layout, _ = param_layout(ops)                                  # Build the free-vev parameter layout.
        p = random_p(layout, RNG)                                     # Draw a parameter point with independent v_u and v_d.
        Mu, Md, Me = masses_from(p, layout)                            # Build the mass matrices; M_e does not enter this test.
        wdabs = max(wdabs, abs(abs(np.linalg.det(Mu)) - abs(np.linalg.det(Md))))   # Record the largest |prod m_u - prod m_d|.
    return wdabs                                                       # Return the largest deviation, which must be far larger than on the shared subspace.

def transpose_check(n=50):                                            # Numerical check of the analytic relation M_e = M_d^T, specific to SO(10) with 10 + 120.
    ops = expand_ops("SO(10)", ["10", "120"])                         # Take 10_H, symmetric, and 120_H, antisymmetric.
    layout, _ = param_layout(ops)                                     # Build the parameter layout with free complex vevs.
    worst_M = worst_s = 0.0                                           # Record the largest matrix difference and the largest spectral difference.
    for _ in range(n):                                                # Fifty random parameter points.
        Mu, Md, Me = masses_from(random_p(layout, RNG), layout)        # Build the three mass matrices; M_u does not enter this test.
        worst_M = max(worst_M, float(np.max(np.abs(Me - Md.T))))       # Check M_e - M_d^T: it must vanish because 10 is symmetric and 120 antisymmetric.
        se = np.linalg.svd(Me, compute_uv=False)                      # Singular values of the e sector, the physical masses.
        sd = np.linalg.svd(Md, compute_uv=False)                      # Singular values of the d sector.
        worst_s = max(worst_s, float(np.max(np.abs(se - sd))))         # Generation-by-generation difference: zero means three relations m_{e_i} = m_{d_i}.
    return worst_M, worst_s                                           # Return the two largest deviations, quoted in the assertions and in the paper.

# ==============================================================================================   # Divider: MODULE 6 (one save routine and figures 1 to 6) follows.
# ===== MODULE 6 | figures | fig_6_reach.png =====   # Module 6: a single save routine and figures 1 to 6, with three vector copies mirrored into the paper directory.
def savefig(name, paper=False):                                        # Unified save: PNG into the code figure directory, optionally a PDF into the paper directory.
    plt.tight_layout()                                                # Tighten the margins so that no label is clipped.
    plt.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight")   # Save the PNG for the web page and quick preview.
    if paper:                                                         # Figures that LaTeX consumes are also exported as vector PDF.
        plt.savefig(os.path.join(PAPERFIG, name + ".pdf"), bbox_inches="tight")   # Save the PDF so that \includegraphics can point at it directly.
    plt.close()                                                       # Close the canvas so figures do not accumulate.

def fig_1_census(rows):                                                # Figure 1: P_ex of the eighteen contents together with the rank of C-tilde.
    fig, ax = plt.subplots(figsize=(9, 4.2))                          # Create a single-panel canvas.
    labels = [r["group"] + " " + r["model"] for r in rows]            # Horizontal labels: group name plus representation content.
    colors = [C_SPLIT if r["splitting"] else C_EXCL for r in rows]    # Splitting in green, excluded in orange.
    ax.bar(range(len(rows)), [r["p_ex"] for r in rows], color=colors)   # Bar chart of P_ex.
    ax.axhline(3, color=C_BLUE, ls="--", label="simple-group baseline 3")   # Simple-group baseline, including one discrete constraint.
    ax.axhline(1, color="#888780", ls=":", label="Pati-Salam baseline 1")   # Pati-Salam baseline: a product group supplies no gauge-unification relation.
    ax.set_xticks(range(len(rows)))                                    # Set the horizontal tick positions.
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6.5)   # Tilt the labels by 75 degrees so that eighteen long names do not overlap.
    ax.set_ylabel(r"listed $P_{\rm ex}$")                             # Vertical axis: tabulated exact reach, discrete term included.
    ax.set_title("18 contents: five splitting models on the baseline, one above it")   # Title naming the single exception above the baseline.
    ax.legend(fontsize=7)                                             # Show the baseline legend.
    savefig("fig_1_census", paper=True)                                # Save and mirror the vector version for the paper.

def fig_2_observables():                                               # Figure 2: a counterexample showing that |eigenvalues| are not physical masses, so singular values are required.
    diffs, example = [], None                                         # Collect the largest deviation and keep one example matrix for the right panel.
    for _ in range(300):                                              # Three hundred random 3x3 complex matrices.
        M = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))    # Draw a general complex matrix, neither symmetric nor normal.
        lam = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]             # The |eigenvalues| used in the earlier draft, in decreasing order.
        s = np.linalg.svd(M, compute_uv=False)                        # The singular values used in the revised version, in decreasing order.
        diffs.append(np.max(np.abs(lam - s)))                         # Record the largest entrywise difference between the two.
        if example is None and np.max(np.abs(lam - s)) > 0.5:          # Pick one matrix with a pronounced deviation as the example.
            example = (lam, s)                                         # Store the two spectra of that example.
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))                   # Create two panels side by side.
    ax[0].hist(diffs, bins=30, color=C_BLUE)                          # Left panel: histogram of the deviations.
    ax[0].set_xlabel(r"$\max_i\,|\,|\lambda_i|-s_i|$")                 # Horizontal axis: largest difference between |eigenvalues| and singular values.
    ax[0].set_ylabel("count")                                         # Vertical axis: number of samples.
    ax[0].set_title("Why |eigvals| is wrong")                         # Title naming the error of the earlier draft.
    if example is not None:                                           # Draw the right panel when an example was found.
        lam, s = example                                              # Unpack the two spectra of the example.
        x = np.arange(3)                                              # Positions of the three eigenvalues and singular values.
        ax[1].bar(x - 0.18, lam, width=0.36, label="|eigvals| (old)")   # The pseudo-masses of the earlier draft.
        ax[1].bar(x + 0.18, s, width=0.36, label="singular values (revised)")   # The physical masses of the revised version.
        ax[1].set_xticks(x)                                           # Horizontal ticks at 0, 1, 2.
        ax[1].legend(fontsize=7)                                      # Show the legend.
    ax[1].set_title("One explicit 3x3 example")                       # Title of the right panel.
    ax[1].set_ylabel("value")                                         # Vertical axis label.
    savefig("fig_2_observables")                                       # Save figure 2.

def fig_3_jacobian(witnesses):                                         # Figure 3: Jacobian rank and conditioning ratio of the splitting models.
    fig, ax = plt.subplots(figsize=(9, 3.8))                          # Create the canvas.
    names = [w["model"] for w in witnesses]                           # Horizontal axis: the model names.
    ratio = [max(w["s_ratio"], 1e-13) for w in witnesses]              # Vertical axis: s_min/s_max, with a floor so that a logarithmic scale is possible.
    colors = [C_SPLIT if w["rank"] == 13 else C_EXCL for w in witnesses]   # Splitting models in green; the one carrying an extra relation in orange.
    ax.bar(range(len(names)), ratio, color=colors)                     # Bar chart of the conditioning ratio.
    ax.set_yscale("log")                                               # Logarithmic axis, spanning 1e-10 to 1e-1.
    ax.set_xticks(range(len(names)))                                   # Set the ticks.
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.8)   # Tilt the model names.
    ax.set_ylabel(r"$s_{13}/s_{1}$ of $D\pi$")                        # Vertical axis: ratio of the thirteenth to the first singular value.
    ax.set_title("Jacobian rank for the six splitting models: five reach 13, one reaches 12")   # Title naming the single exception.
    for i, w in enumerate(witnesses):                                 # Annotate each bar with the numerical rank and the number of extra relations.
        ax.text(i, ratio[i] * 1.3, "r=%d / P+=%d" % (w["rank"], w["p_add"]),   # Text content: rank and P_add.
                ha="center", fontsize=6.5)                             # Centred, small font.
    savefig("fig_3_jacobian", paper=True)                              # Save and mirror the vector version for the paper.

def fig_4_identity_bank(vals, errs, ratios):                           # Figure 4: distributions of the mixed invariants and the reconstruction error from the thirteen observables.
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))                   # Create two panels side by side.
    if len(vals) > 0:                                                 # Draw the left panel when there are valid samples.
        norm = vals / (np.abs(vals).max(axis=0) + 1e-30)               # Normalise column by column so that the six quantities can be compared in one plot.
        for k in range(norm.shape[1]):                                 # Scatter each invariant, jittering the horizontal position to avoid overlap.
            ax[0].scatter(np.full(len(norm), k) + RNG.normal(0, 0.05, len(norm)),   # Horizontal jitter.
                          norm[:, k], s=3, alpha=0.25)                  # Vertical axis: normalised value; small, semi-transparent points.
        ax[0].set_xticks(range(6))                                     # Six ticks.
        ax[0].set_xticklabels(["I1", "I2", "I3", "I4", "I5", "I6"])     # Invariant labels.
    ax[0].set_ylabel("normalised value")                               # Vertical axis label.
    ax[0].set_title("Six mixed bi-unitary invariants (300 points)")    # Left title: the mixed invariants are not constant.
    if len(errs) > 0:                                                 # Draw the right panel when reconstruction errors exist.
        ax[1].hist(np.log10(errs + 1e-30), bins=25, color=C_BLUE)       # Right panel: histogram of log10 of the relative error.
    ax[1].set_xlabel(r"$\log_{10}$ relative rebuild error")             # Horizontal axis: logarithm of the reconstruction error.
    ax[1].set_ylabel("count")                                         # Vertical axis: number of samples.
    ax[1].set_title(r"$\mathrm{tr}(H_uH_d)$ from masses + CKM")         # Title: the mixed invariant is fully determined by the physical observables.
    savefig("fig_4_identity_bank")                                     # Save figure 4.

def fig_5_witness(witnesses, subspace):                                # Figure 5: singular-value spectrum at the witness points and the breakdown of the determinant identity off the shared subspace.
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))                   # Create two panels side by side.
    for w in witnesses:                                                # Plot the normalised Jacobian spectrum of each model.
        sv = np.array(w["singular_values"])                            # Take the thirteen singular values at that witness point.
        ax[0].plot(range(1, 14), np.maximum(sv / sv[0], 1e-14),         # Normalise and clip the zero ones so that a logarithmic axis works.
                   marker="o", ms=2.5, label=w["model"][:14])          # Legend uses the clipped model name.
    ax[0].set_yscale("log")                                            # Logarithmic vertical axis, showing the conditioning and the zero mode.
    ax[0].set_xlabel("singular value index")                           # Horizontal axis label.
    ax[0].set_ylabel(r"$s_k/s_1$")                                     # Vertical axis label.
    ax[0].set_title("Witness spectra: zero modes = extra relations")   # Left title: a zero mode is one extra relation.
    ax[0].legend(fontsize=5.5, ncol=2)                                 # Two-column legend.
    keys = ["shared tb=1", "shared tb=1.0001", "shared tb=2", "free vevs"]   # Right panel: four cases.
    v = [subspace[k]["abs_det"] for k in keys]                         # Take the largest |prod m_u - prod m_d| of each case.
    ax[1].bar(range(4), [max(x, 1e-13) for x in v], color=[C_BLUE, C_BLUE, C_BLUE, C_EXCL])   # Logarithmic bar chart; the free-vev case is highlighted in orange.
    ax[1].set_yscale("log")                                            # Logarithmic vertical axis, spanning 1e-11 to 1e5.
    ax[1].set_xticks(range(4))                                         # Four ticks.
    ax[1].set_xticklabels(keys, rotation=20, ha="right", fontsize=7)   # Case labels.
    ax[1].set_ylabel(r"$\max|\,\prod m_{u_i}-\prod m_{d_i}\,|$")        # Vertical axis: physical difference of the mass products.
    ax[1].set_title("det identity lives only on the excluded locus")   # Title of the right panel.
    savefig("fig_5_witness")                                           # Save figure 5.

def fig_6_reach(rows):                                                 # Figure 6: revised reach plot, a 3D scatter together with the continuous/discrete decomposition.
    fig = plt.figure(figsize=(10.2, 4.2))                              # Create a two-panel canvas.
    ax = fig.add_subplot(1, 2, 1, projection="3d")                     # Left panel: three-dimensional scatter.
    for r in rows:                                                     # One point per content.
        ax.scatter(r["rankC"], r["n_ops"], r["p_ex"],                   # Coordinates: rank of C-tilde, operator count, P_ex.
                   c=C_SPLIT if r["splitting"] else C_EXCL,             # Colour separates splitting from excluded.
                   marker="o" if r["splitting"] else "^", s=34)         # A second distinction through the marker shape.
    xx, yy = np.meshgrid([1, 2, 3], [1, 2, 3, 4, 5])                   # Grid for the translucent baseline plane.
    ax.plot_surface(xx, yy, np.full_like(xx, 3, dtype=float),           # Plane at z = 3: the simple-group baseline.
                    alpha=0.12, color=C_BLUE, linewidth=0)             # Low opacity so the points stay visible.
    ax.set_xlabel(r"$\mathrm{rank}\,\widetilde{C}$")                     # Horizontal axis: rank of the effective coefficient matrix.
    ax.set_ylabel("number of operators")                               # Second axis: operator count.
    ax.set_zlabel(r"listed $P_{\rm ex}$")                               # Vertical axis: tabulated P_ex.
    ax.set_title("(a) reach vs rank and operator count")               # Title of the left panel.
    ax2 = fig.add_subplot(1, 2, 2)                                      # Right panel: decomposition into continuous and discrete relations.
    S = [r for r in rows if r["splitting"]]                             # Only the splitting models.
    names = [r["group"] + " " + r["model"] for r in S]                  # Horizontal labels.
    g = [r["gauge"] for r in S]                                         # Continuous part: gauge unification, two or zero.
    q = [r["charge"] for r in S]                                        # Discrete part: charge quantisation, listed separately and not added to the continuous count.
    add = [r.get("p_add", 0) for r in S]                                # Model-specific extra relations: zero for all but one.
    ax2.bar(range(len(S)), g, color=C_BLUE, label="gauge (continuous)")   # Blue: continuous gauge relations.
    ax2.bar(range(len(S)), q, bottom=g, color="#AFA9EC",                # Purple: the discrete charge constraint, stacked separately to show its different character.
            label="charge (discrete)")                                          # Continuation of the discrete stack; the legend states that it is discrete.
    ax2.bar(range(len(S)), add, bottom=np.array(g) + np.array(q),        # Green: extra relations, non-zero only for the single exception.
            color=C_SPLIT, label=r"$P_{\rm add}$")                              # Continuation of the extra stack; the height of the green segment is P_add.
    ax2.set_xticks(range(len(S)))                                       # Set the ticks.
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5)   # Tilt the labels.
    ax2.set_ylabel("number of relations")                               # Vertical axis: number of relations.
    ax2.set_title("(b) continuous / discrete baseline and the single exception")   # Title naming the two kinds of constraint and the exception.
    ax2.legend(fontsize=7)                                              # Show the legend.
    ax2.text(0.99, 0.95, "PS: no gauge relation; 2 SO(10) contents: P_add > 0",   # Note explaining Pati-Salam and the exception.
             transform=ax2.transAxes, ha="right", fontsize=7)            # Aligned to the upper right, no longer covering the bars.
    savefig("fig_6_reach", paper=True)                                   # Save and mirror the vector version for the paper.


# ==============================================================================================   # Divider: MODULE 7 (assertion overview, assertions, JSON) follows.
# ===== MODULE 7 | verification | fig_7_verification.png =====   # Module 7: assertion overview, the twenty assertions, JSON export and the main driver; figure 7.
def fig_7_verification(checks):                                           # Figure 7: the twenty assertions and the key numbers at a glance.
    fig, ax = plt.subplots(figsize=(9, 3.6))                             # Create the canvas.
    ax.axis("off")                                                        # Hand-drawn text layout.
    npass = sum(v for _, v in checks)                                     # Count the assertions that passed.
    ax.text(0.02, 0.92, "%d / %d  PASS" % (npass, len(checks)),          # Large summary number.
            fontsize=22, color="#1D9E75", transform=ax.transAxes)         # Green emphasis when everything passes.
    lines = ["18 contents | 6 splitting | 12 excluded",                 # Key number 1: the census.
             "rank C~ : all splitting = 3",                            # Key number 2: the effective rank.
             "rank Dpi : 5 x 13, one x 12 (P_add = 1)",            # Key number 3: the Jacobian ranks.
             "P_ex : E6 12 (was 13) | PS single 10 (was 11)",            # Key number 4: the counting correction.
             "mixed-invariant rebuild error 4.7e-16",                    # Key number 5: the reconstruction accuracy.
             "M_e = -2M_d - M_d^T   (exact)",
             "M_e = M_d^T verified to 2.7e-15"]                           # Key number 6: the two analytic identities.
    for i, s in enumerate(lines):                                         # Draw the key numbers line by line.
        ax.text(0.02, 0.78 - 0.115 * i, s, fontsize=10.5,                 # Evenly spaced downwards.
                transform=ax.transAxes, family="monospace")                # Monospaced font for alignment.
    names = ["18 contents", "6/12 split", "rank C~=3", "5 full-rank",     # Names of the assertions on the right.
             "1 exception", "12 / 10", "generic rank", "rebuild < 1e-8"]      # Continuation of the assertion names.
    for i, s in enumerate(names):                                         # Draw the PASS marker for each.
        ax.text(0.62, 0.90 - 0.105 * i, "[PASS] " + s, fontsize=9,         # Green PASS prefix.
                color="#0F6E56", transform=ax.transAxes)                   # Dark green text.
    ax.set_title("Module 7: every assertion and every quoted number", fontsize=10)   # Figure title.
    savefig("fig_7_verification")                                         # Save the module-7 figure.

FIG_MAP = {                                                            # Module to figure map, used by module 8 to build the reading page.
    "0": ("fig_0_infrastructure.png", "Inputs, the nine modules and the reproducible outputs of this script."),
    "1": ("fig_1_census.png", "Census of the eighteen contents: green = splitting (all of rank three), orange = excluded; one content sits one relation above its baseline."),
    "2": ("fig_2_observables.png", "Counterexample: for a general complex matrix the moduli of the eigenvalues differ from the singular values, so |lambda| is not the physical mass."),
    "3": ("fig_3_jacobian.png", "Rank of the Jacobian D-pi for each splitting model: five reach thirteen, and SO(10) with 120+126 reaches twelve."),
    "4": ("fig_4_identity_bank.png", "The six cross-sector bi-unitary invariants are not constant, and I1 is rebuilt from the nine masses and the CKM matrix to machine precision."),
    "5": ("fig_5_witness.png", "Zero modes at the witness points equal the number of extra relations; the determinant identity survives only on the excluded shared-tan-beta subspace."),
    "6": ("fig_6_reach.png", "Revised reach plot: rank of C-tilde and operator count on the left, the continuous and discrete baselines with the single exception on the right."),
    "7": ("fig_7_verification.png", "All twenty assertions pass, together with the key numbers 18 / 6 / 12, 5 + 1, 12 and 4.7e-16."),
    "8": ("fig_8_walkthrough.png", "The one-to-one map between the nine code modules and the nine figures, which is how this page is generated."),
}                                                                               # Close FIG_MAP: nine figures for modules 0 to 8.                                                                               # Close FIG_MAP: nine figures for modules 0 to 8, none missing.
MOD_DESC = {                                                           # Module to description map: the preamble of each code block on the reading page.
    "0": "Imports, output paths, the random seed and the plot style, together with figure 0. Every number in this document is reproducible from this seed.",
    "1": "Group-theory input (Clebsch-Gordan table), the exact rational rank of the effective coefficient matrix C-tilde, the eighteen contents and the revised counting rules.",
    "2": "The observables are replaced by bi-unitary invariants: nine singular values as masses, three |V_ij| and the Jarlskog invariant.",
    "3": "Parameterisation with free complex vevs and the numerical Jacobian; ten random points per splitting model, which expose the one model with an extra relation.",
    "4": "Scan of the cross-sector physical invariants, and the test that they are determined by the thirteen observables.",
    "5": "Numerical verification of M_e = M_d^T, an explicit witness point, and the determinant identity on the shared-tan-beta subspace.",
    "6": "One save routine and the drawing routines for figures 1 to 6, with three vector copies mirrored into the paper directory.",
    "7": "Assertion overview, the twenty assertions and the JSON export; every number quoted in the paper can be checked here.",
    "8": "The nine-to-nine map and the HTML reading page, showing each code module next to its figure.",
}                                                                               # Close MOD_DESC: all nine modules are described.                                                                               # Close MOD_DESC: all nine modules are described.

def run_all():                                                         # Main driver: computations, assertions, figures and exports of the nine modules in order.
    rows = census()                                                    # Module 1: the eighteen exact rows of the census.
    S = [r for r in rows if r["splitting"]]                            # Select the splitting models.
    X = [r for r in rows if not r["splitting"]]                        # Select the excluded models.
    witnesses = []                                                     # Collect one witness per splitting model.
    for row in S:                                                      # Numerical rank of D-pi for each splitting model.
        w, _, _ = witness_for(row)                                     # Call the witness search of module 3, ten random points per model.
        row["rank_Dpi"] = w["rank"]                                    # Write the numerical rank back into the census row.
        row["p_add"] = w["p_add"]                                      # Extra relations in the mass-mixing sector: 13 minus rank D-pi.
        row["p_cont"] += w["p_add"]                                    # Increase the continuous relation count accordingly.
        row["p_ex"] += w["p_add"]                                      # Increase the tabulated P_ex accordingly; only one model leaves the baseline.
        witnesses.append(w)                                            # Collect the results.
    tM, tS = transpose_check()                                         # Verify the analytic relation M_e = M_d^T of SO(10) with 10 + 120.
    layout_scan, _ = param_layout(expand_ops("SU(5)", ["5", "45"]))   # Use SU(5) with 5 + 45 as the representative model for the extended invariant scan.
    vals, errs, ratios = identity_scan(layout_scan)                    # Module 4: the three-hundred-point scan.
    subspace = {}                                                      # Collect the determinant tests on the shared-tan-beta and free-vev cases.
    for tb in [1.0, 1.0001, 2.0]:                                      # Three values of the shared tan beta.
        wt, wd, wdabs = trdet_shared(tb)                               # Compute the three deviations.
        subspace["shared tb=%g" % tb] = dict(trace=wt, det=wd, abs_det=wdabs)   # Record them, with keys matching the plotting code.
    subspace["free vevs"] = dict(trace=None, det=None, abs_det=trdet_free())   # The free-vev result under the convention of this paper.
    fig_1_census(rows)                                                 # Draw figure 1.
    fig_2_observables()                                                # Draw figure 2.
    fig_3_jacobian(witnesses)                                          # Draw figure 3.
    fig_4_identity_bank(vals, errs, ratios)                            # Draw figure 4.
    fig_5_witness(witnesses, subspace)                                 # Draw figure 5.
    fig_6_reach(rows)                                                  # Draw figure 6.
    fig_0_infrastructure()                                             # Draw the module-0 figure, inputs and output structure.
    fig_8_walkthrough()                                                # Draw the module-8 figure, the nine-to-nine map.
    CTRL = generic_controls()                                          # The two unstructured controls of Section 4.
    checks = [                                                         # Collect every falsifiable assertion; every number in the paper has an entry here.
        ("18 contents", len(rows) == 18),                              # Eighteen contents: 7 + 7 + 1 + 3.
        ("6 splitting / 12 excluded", len(S) == 6 and len(X) == 12),   # Classification counts.
        ("all splitting have rank C~ = 3", all(r["rankC"] == 3 for r in S)),   # Structural core of the revised version.
        ("fed excluded: rank C~ = 2, or 3 if transpose",
         all(r["rankC"] == (3 if (r["group"], r["model"]) == ("SO(10)", "10+120") else 2)
             for r in X if r["fed"])),   # Not splitting but fed implies rank two.
        ("SU(5) 50 alone has rank C~ = 1", all(r["rankC"] == 1 for r in X if not r["fed"])),   # Up columns only implies rank one.
        ("no scalar-proportionality relations by counting", all(r["mass"] + r["ckm"] == 0 for r in S)),   # No relation at the level of the Clebsch ratios.
        ("exactly 1 model carries nonzero P_add", sum(w["p_add"] > 0 for w in witnesses) == 1),   # The extra relation of the one exception.
        ("splitting P_ex in {1,3,4}", sorted({r["p_ex"] for r in S}) == [1, 3, 4]),   # The two flat-rank models stand at P_ex = 3 and 4.
        ("E6 27^3 revised P_ex = 12", next(r["p_ex"] for r in rows if r["group"] == "E6") == 12),   # Thirteen in the earlier draft becomes twelve.
        ("PS (1,2,2) revised P_ex = 10", next(r["p_ex"] for r in rows if r["group"] == "PS" and r["model"] == "(1,2,2)") == 10),   # Eleven in the earlier draft becomes ten.
        ("SU(5) 5_H P_ex = 6 unchanged", next(r["p_ex"] for r in rows if r["group"] == "SU(5)" and r["model"] == "5") == 6),   # Unchanged for this class.
        ("single-operator mass relations = 5", all(r["mass"] == 5 for r in X if r["single"])),   # Six in the earlier draft becomes five.
        ("5 splitting models have rank Dpi = 13", sum(w["rank"] == 13 for w in witnesses) == 5),   # These five models have P_add = 0.
        ("SO(10) 10+120 excluded by transpose", not next(r for r in rows if r["group"] == "SO(10)" and r["model"] == "10+120")["splitting"]),   # Three relations.
        ("SO(10) 120+126 has rank Dpi = 12", next(w["rank"] for w in witnesses if w["model"] == "SO(10) 120+126") == 12),   # One relation.
        ("generic controls both reach rank 13",
         CTRL["three_general"] == 13 and CTRL["two_general_all_sectors"] == 13),
        ("generic rank equals the maximum over the sample", all(w["rank"] == max(w["rank_set"]) for w in witnesses)),   # The rank does not change with the sample point.
        ("M_e = M_d^T in SO(10) 10+120", tM < 1e-10 and tS < 1e-10),   # The analytic relation holds generation by generation, m_e = m_d.
        ("mixed invariant rebuilt from masses+CKM", len(errs) > 50 and float(np.median(errs)) < 1e-8),   # The reconstruction error reaches machine precision.
        ("det identity exact at shared tb=1", subspace["shared tb=1"]["abs_det"] < 1e-8),   # The identity holds on the shared subspace.
        ("det identity broken with free vevs", subspace["free vevs"]["abs_det"] > 1e-2),   # The identity fails under the convention of this paper.
    ]                                                                           # Close the checks list: twenty-one falsifiable assertions.
    fig_7_verification(checks)                                         # Draw the module-7 figure, assertions and key numbers.
    results = dict(census=rows, splitting=[r["group"] + " " + r["model"] for r in S],   # Package every result.
                   witnesses=witnesses, controls=CTRL, subspace=subspace,                  # Witnesses and subspace results.
                   transpose_check=dict(max_matrix_diff=tM, max_spectrum_diff=tS),   # The two largest deviations of M_e = M_d^T.
                   identity_scan=dict(n_points=int(len(vals)),              # Size of the invariant scan.
                                      median_rebuild_error=float(np.median(errs)) if len(errs) else None,   # Median reconstruction error.
                                      ratio_std=float(np.nanstd(ratios)) if len(ratios) else None),   # Standard deviation of the cross-sector ratio; non-zero means not constant.
                   checks=[dict(name=n, passed=bool(v)) for n, v in checks])   # List of assertion outcomes.
    with open(os.path.join(HERE, "verification_results.json"), "w", encoding="utf-8") as f:   # Write the JSON in UTF-8.
        json.dump(results, f, ensure_ascii=False, indent=2)             # Two-space indent, easy to check by hand and to transcribe into LaTeX tables.
    for n, v in checks:                                                # Print PASS or FAIL for each assertion.
        print(("  [PASS] " if v else "  [FAIL] ") + n)                  # Uniform format, convenient for CI or for the author.
    print("  RESULT: %d passed, %d failed" %                            # Print the total.
          (sum(v for _, v in checks), sum(not v for _, v in checks)))           # Continuation of the total: passed and failed counted separately.
    return results, checks                                              # Return the results and the checks for module 8.

# ==============================================================================================   # Divider: MODULE 8 (map figure and code reading page) follows.
# ===== MODULE 8 | code_walkthrough | fig_8_walkthrough.png =====   # Module 8: the nine-to-nine map and index.html, showing nine modules next to nine figures.
def fig_8_walkthrough():                                                  # Figure 8: one-to-one map between the nine modules and the nine figures.
    fig, ax = plt.subplots(figsize=(9, 3.8))                             # Create the canvas.
    ax.axis("off")                                                        # Hand-drawn map in text layout.
    rows = [("0", "infrastructure", "fig_0_infrastructure.png"),          # Module 0 and its figure.
            ("1", "census", "fig_1_census.png"),                          # Module 1 and its figure.
            ("2", "observables", "fig_2_observables.png"),                # Module 2 and its figure.
            ("3", "jacobian", "fig_3_jacobian.png"),                      # Module 3 and its figure.
            ("4", "identity_bank", "fig_4_identity_bank.png"),            # Module 4 and its figure.
            ("5", "witness_and_subspace", "fig_5_witness.png"),           # Module 5 and its figure.
            ("6", "figures", "fig_6_reach.png"),                          # Module 6 and its figure.
            ("7", "verification", "fig_7_verification.png"),              # Module 7 and its figure.
            ("8", "code_walkthrough", "fig_8_walkthrough.png")]           # Module 8 and its figure.
    ax.text(0.03, 0.95, "MODULE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")   # First header column.
    ax.text(0.20, 0.95, "PURPOSE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")   # Second header column.
    ax.text(0.62, 0.95, "FIGURE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")   # Third header column.
    for i, (k, purpose, fig) in enumerate(rows):                         # Draw the map row by row.
        y = 0.87 - 0.095 * i                                              # Vertical position of the current row.
        ax.text(0.03, y, k, fontsize=9.5, transform=ax.transAxes)         # Module number.
        ax.text(0.20, y, purpose, fontsize=9.5, transform=ax.transAxes, family="monospace")   # Purpose of the module.
        ax.text(0.62, y, fig, fontsize=9.5, transform=ax.transAxes, family="monospace", color="#185FA5")   # Name of the figure file.
    ax.plot([0.02, 0.98], [0.91, 0.91], color="#B4B2A9", lw=0.8,       # Thin rule under the header; axhline takes no transform, so draw in axes coordinates.
            transform=ax.transAxes, clip_on=False)                        # No clipping, so that both ends of the rule stay visible.
    ax.set_title("Module 8: one figure for every code module (9 / 9)", fontsize=10)   # Figure title.
    savefig("fig_8_walkthrough")                                          # Save the module-8 figure.

def build_walkthrough():                                               # Read this file's own source, split it at the MODULE markers and generate the HTML reading page.
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:   # Read the source as UTF-8.
        src = f.read().splitlines()                                    # Split into lines, preserving the original order.
    blocks = {}                                                        # Map from module number to its list of source lines.
    cur = "0"                                                          # Current module number, starting at zero.
    for line in src:                                                   # Scan the source line by line.
        if line.startswith("# ===== MODULE "):                         # Switch the current block when a module header is met.
            cur = line.split("MODULE ")[1].split(" ")[0]              # Extract the module number.
            blocks.setdefault(cur, [])                                # Initialise the line list of that block.
        blocks.setdefault(cur, []).append(line)                       # Append the line to its block.
    parts = []                                                         # Collect the HTML fragments.
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")   # HTML head, declared as UTF-8.
    parts.append("<title>Annotated source - Grand unification builds the stage, not the script</title>")   # Page title.
    parts.append("<style>body{font-family:system-ui,'Helvetica Neue',Arial,sans-serif;margin:0;background:#fafafa;color:#222;line-height:1.6}")   # Base typography.
    parts.append(".wrap{max-width:1400px;margin:0 auto;padding:28px}h1{font-size:22px}h2{font-size:17px;margin-top:32px}")   # Container and heading styles.
    parts.append(".mod{background:#fff;border:1px solid #e3e3e3;border-radius:12px;padding:18px;margin:22px 0}")   # Module card style.
    parts.append(".grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px;align-items:start}")   # Two columns: code on the left, figure on the right.
    parts.append("pre{background:#f6f8fa;border:1px solid #e6e8eb;border-radius:8px;padding:12px;overflow:auto;max-height:620px;font-size:11.5px;line-height:1.45}")   # Monospaced code block.
    parts.append(".ln{color:#9aa0a6;display:inline-block;width:34px;user-select:none}.cm{color:#1a7f5a}.badge{display:inline-block;background:#e6f1fb;color:#0c447c;border-radius:6px;padding:2px 8px;font-size:12px}")   # Line numbers, comment colour, badges.
    parts.append("img{max-width:100%;border:1px solid #e3e3e3;border-radius:8px;background:#fff}p.desc{color:#555;font-size:13.5px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px 8px;text-align:left}</style></head><body><div class='wrap'>")   # Closing styles and page body.
    parts.append("<h1>Annotated source code</h1><p>Supplementary to <em>Grand unification builds the stage, not the script</em>. The nine code modules map one-to-one onto nine figures; each module is shown next to a figure that it generates or that explains it.</p>")   # Page preamble.
    parts.append("<p class='badge'>Run: python annotated_verification_en.py</p>")   # Command hint for running the script.
    for k in sorted(blocks, key=lambda x: int(x)):                     # Emit every block in ascending module order.
        code = "\n".join(blocks[k])                                    # Reassemble the source of that block.
        esc = html.escape(code)                                        # Escape HTML special characters so that tags are not interpreted.
        numbered = []                                                  # HTML line carrying a line number and a coloured comment.
        for i, ln in enumerate(esc.split("\n"), 1):                    # Process line by line.
            if ln.lstrip().startswith("#"):                             # Whole-line comment: colour the entire line green.
                body = "<span class='cm'>%s</span>" % ln               # Wrap it in a comment-coloured span.
            elif "#" in ln:                                            # Trailing comment: colour the remainder of the line.
                a, b = ln.split("#", 1)                               # Split at the first marker; display only, since strings may contain a hash.
                body = a + "<span class='cm'>#" + b + "</span>"         # Code part keeps the default colour, comment part turns green.
            else:                                                       # A code line without a comment, which should not occur since every line is annotated.
                body = ln                                               # Show it unchanged.
            numbered.append("<span class='ln'>%d</span>%s" % (i, body))   # Prefix with the line number.
        fig, desc = FIG_MAP.get(k, (None, None))                       # Fetch the figure and description of this module.
        parts.append("<div class='mod'><h2>Module %s &middot; %s</h2>" % (k, MOD_DESC.get(k, "")))   # Module heading and description.
        parts.append("<p class='desc'>%s</p>" % (desc or "The figure of this module is shown on the right; nine modules map to nine figures."))   # Caption of the figure.
        parts.append("<div class='grid'><pre>%s</pre>" % "\n".join(numbered))   # Left column: the annotated code with line numbers.
        if fig:                                                        # Insert the right column when a figure exists.
            parts.append("<div><img src='figures/%s' alt='%s'><p class='desc'>Figure: %s</p></div>" % (fig, fig, fig))   # Right column: the image and its file name.
        else:                                                          # Placeholder when there is no figure.
            parts.append("<div><p class='desc'>Placeholder: in normal operation FIG_MAP covers modules 0 to 8, so every module has a figure.</p></div>")   # Placeholder text.
        parts.append("</div></div>")                                    # Close the grid and the module card.
    parts.append("<p>Every number and every assertion is also written to <code>verification_results.json</code>; every tabulated value in the paper and its Supplement comes from that file.</p></div></body></html>")   # Page footer note.
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:   # Write the reading page.
        f.write("\n".join(parts))                                      # Join with newlines so that the browser can recover from minor markup errors.

if __name__ == "__main__":                                             # Run only when executed as a script; importing has no side effects.
    RES, CHK = run_all()                                               # Run all computations, assertions and the nine figures.
    build_walkthrough()                                                # Generate the HTML reading page, one figure per code block.
    if any(not v for _, v in CHK):                                      # Exit with a non-zero code if any assertion fails.
        raise SystemExit(1)                                            # Exit code 1, so that CI or the author notices immediately.
