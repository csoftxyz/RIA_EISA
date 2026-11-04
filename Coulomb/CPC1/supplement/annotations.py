# -*- coding: utf-8 -*-
"""Line-by-line annotations for every Python file in src/.

Each entry is (first_line, last_line, note).  Line numbers match the files as
shipped in this package; supplement/check_annotations.py verifies that every
non-blank line is covered by at least one note.
"""

A = {}

# =====================================================================
# src/radialflux.py
# =====================================================================
A["radialflux.py"] = [
    (1, 2, "Module docstring: what the package is, and the MIT licence. "
            "Nothing is executed here."),
    (3, 3, "PEP 563: annotations are stored as strings, so the 'float' hints below "
           "work on every Python 3.7+ interpreter without evaluation."),
    (4, 4, "math: scalar libm functions. We use it (not NumPy) for pi and Gamma "
           "so that scalar calls stay exact to the last bit."),
    (5, 5, "NumPy: arrays, vectorised ufuncs, expm1, diff."),
    (7, 9, "Public surface. validate.py and audit_claims.py do 'from radialflux "
           "import *', so this list is the contract those scripts rely on."),
    (11, 11, "omega_d(d): area of the unit (d-1)-sphere, the constant that turns a "
             "flux into a charge."),
    (12, 12, "Docstring: Omega_d = 2 pi^{d/2} / Gamma(d/2). d=3 gives 4 pi, d=4 gives 2 pi^2."),
    (13, 13, "math.gamma is the Cephes double-precision Gamma; relative error < 1e-15 "
             "for the arguments used here (d/2 up to 5)."),
    (15, 15, "exact_radius(r_in, r_out, d) -- the central function of the paper."),
    (16, 19, "Docstring: defines rho_* and states the numerically stable route "
             "(expm1), which is what the implementation actually does."),
    (20, 20, "Accept scalars, lists or arrays; everything downstream is vectorised."),
    (21, 21, "a = d-2 is the exponent gap that appears in every formula; a = 0 is the "
             "d = 2 degenerate case handled separately."),
    (22, 22, "lq = ln(r_out/r_in) = ln q. Working with the ratio avoids overflow when "
             "the radii are large but close."),
    (23, 25, "d = 2: the shell integral of r^{-1} is ln q, so rho is the logarithmic "
             "mean (r_out-r_in)/ln q. At q -> 1 this tends to r_in."),
    (26, 26, "Comment: the antiderivative used on the next line, written with "
             "r_in^{-a} factored out."),
    (27, 27, "Suppress divide-by-zero / invalid warnings: they are expected at r_in = 0 "
             "and are handled by the caller's limits."),
    (28, 28, "-expm1(-a*lq) computes 1 - q^{-a} without cancellation: expm1(t) is "
             "accurate for |t| ~ 1e-16, where 1-exp(t) would return 0."),
    (29, 29, "rho^{d-1} = (r_out-r_in) / integral: the reciprocal mean of r^{1-d}."),
    (30, 30, "(d-1)-th root. d-1 > 0 always, so the power is well defined."),
    (32, 32, "recovered_fraction(rule, q, d): the definition of kappa, used by the "
             "tests to cross-check the closed forms."),
    (33, 33, "Docstring: kappa = Qhat/Q for the four rules."),
    (34, 34, "Vectorise q."),
    (35, 35, "Scale invariance: on the normalised shell [1, q] the exact radius is "
             "rho*/r_in, so one call covers every shell."),
    (36, 36, "Sampling radius of each rule: left = r_n, right = r_{n+1}, "
             "midpoint = arithmetic mean, exact = rho*. The dict is indexed by the "
             "rule name, so an unknown name raises KeyError here."),
    (37, 37, "Flux ratio Qhat/Q = (rho'/rho*)^{d-1}: the entire bias is a power of a "
             "radius ratio."),
    (39, 39, "exact_bias(rule, q, d): the closed form (Eq. 13 of the manuscript; the "
           "docstring says Eq. 12, a leftover from an earlier draft)."),
    (40, 40, "Docstring."),
    (41, 41, "q as array; e = q-1 = epsilon, the small parameter of the series."),
    (42, 42, "a = d-2."),
    (43, 44, "d = 2 limit: kappa_L = ln q/(q-1). The ratio is 0/0 at q = 1; the "
             "caller is expected to stay at q > 1 (the limit is 1)."),
    (45, 46, "General closed form kappa_L = (1-q^{-a})/(a*e). This is algebraically "
             "exact but loses digits for e below about 1e-8 (two nearly equal "
             "numbers are subtracted); exact_radius() is the stable route and is "
             "the one used for computation."),
    (47, 47, "Left-endpoint rule: kappa_L itself."),
    (48, 48, "Right-endpoint rule: multiplying the radius by q multiplies the flux by "
             "q^{d-1}, hence kappa_R = kappa_L q^{d-1}."),
    (49, 49, "Midpoint rule: same power law evaluated at (1+q)/2."),
    (50, 50, "The exact rule is unbiased by construction: kappa = 1 for every q and d."),
    (51, 51, "Fail loudly on a mis-spelled rule name rather than returning a silent "
             "garbage number."),
    (53, 53, "naive_radius: the textbook generalisation that forgets the (d-2) factor."),
    (54, 54, "Docstring: it equals rho_* (d-2)^{-1/(d-1)}."),
    (55, 55, "Single line: the missing factor is a pure constant, so the ratio to "
             "rho_* is independent of q."),
    (57, 57, "enclosed_charge(V, r, d, rule, rho_prime): the estimator that the paper "
             "recommends."),
    (58, 58, "Docstring: Qhat from samples at r[0..N]; exact radius by default."),
    (59, 59, "Coerce to float arrays; the slices below assume 1-D input."),
    (60, 60, "Per-shell radius: rho* by default, or a caller-supplied array (used by "
             "the tests to emulate the left/right/midpoint rules)."),
    (61, 61, "Potential drop across each shell, V_n - V_{n+1} (positive for a "
             "positive charge)."),
    (62, 62, "Discrete Gauss law: Qhat = Omega_d rho^{d-1} dV/dr. np.diff(r) is "
             "r_{n+1}-r_n, the shell thickness."),
    (64, 64, "q_star(d, tol): threshold of the (d-2)-less rule."),
    (65, 71, "Docstring: q* is the smallest ratio at which the naive rule still "
             "samples inside its own shell; d = 3 has no threshold because the rule "
             "is exact there. Note the deliberate decision to return None instead of "
             "raising."),
    (72, 73, "d = 3: (d-2) = 1, the defining equation degenerates to (q-1)^2 = 0, "
             "and the rule is exact for all q. No threshold exists."),
    (74, 74, "The threshold function f(q) = q + q^{2-d} - 2; f(1) = 0 and f is "
             "strictly increasing for d > 3, so the root in (1, inf) is unique."),
    (75, 75, "Bracket [1+1e-9, 2]. The lower end is nudged off the trivial root q = 1."),
    (76, 77, "Expand the upper end until f changes sign -- a safeguard for large d, "
             "where the root approaches 1.01 and 2 would still work, but the loop "
             "makes the routine independent of any guess."),
    (78, 78, "Bisection with a hard cap of 200 iterations: guaranteed termination "
             "even if tol is set to zero."),
    (79, 79, "Midpoint of the current bracket."),
    (80, 81, "If the midpoint is indistinguishable from an endpoint, the bracket is "
             "at floating-point resolution: stop rather than loop forever."),
    (82, 85, "Standard sign test: keep the half that brackets the root."),
    (86, 87, "Relative-width stopping rule; tol defaults to 1e-15, i.e. machine "
             "precision at this scale."),
    (88, 88, "Return the midpoint of the final bracket."),
    (90, 90, "uniform_rule_values(n): the d = 3 uniform-lattice special case."),
    (91, 91, "Docstring: exact kappa for r_n = n a (Eq. 18)."),
    (92, 92, "Vectorise n."),
    (93, 94, "rho* = sqrt(n(n+1)), so kappa_L = n/(n+1), kappa_R = (n+1)/n and "
             "kappa_A = (2n+1)^2/(4n(n+1)) = 1 + 1/(4n(n+1))."),
    (96, 96, "shell_identity: the exact two-term identity (Eq. 24)."),
    (97, 97, "Docstring: valid in every d (the naming is historical)."),
    (98, 98, "a = d-2, used twice."),
    (99, 100, "Flux term plus the optional source term int rho r dr/(d-2). The "
              "backslash continues the expression; the conditional adds exactly 0.0 "
              "when no source integral is supplied."),
    (102, 102, "source_bias: leading bias when the source varies across the shell."),
    (103, 103, "Docstring."),
    (104, 104, "(d-2)/24 (q-1)^2 (s/Q): second order, and proportional to (d-2) -- "
              "which is why it vanishes in two dimensions."),
    (106, 106, "running_residual: residual for a logarithmically running amplitude."),
    (107, 107, "Docstring: exact finite-q residual for A(r) = A0 + gamma ln r."),
    (108, 108, "gamma [1/(d-2) - ln q/(q^{d-2}-1)]. The bracket is 0 at q = 1 and "
              "grows like (q-1)/2 for small steps."),
    (110, 110, "background_charge: the charge read through a neutralising background."),
    (111, 112, "Docstring: fraction 1 - (2pi/3) R(R+1)(2R+1)/L^3 in d = 3. The "
               "R(R+1)(2R+1) is the discrete count of cells inside the shell R."),
    (113, 113, "Vectorise R."),
    (114, 114, "Single expression; math.pi is used so the constant is the correctly "
               "rounded double."),
    (116, 117, "Sphere: the closed-form profiles used as ground truth in Sec. 9."),
    (118, 119, "Constructor stores the profile name, the dimension and the keyword "
               "parameters in a dict (no attribute explosion)."),
    (120, 120, "Q(r): enclosed charge of the profile, vectorised in r."),
    (121, 121, "Local aliases for brevity."),
    (122, 123, "Uniform ball: Q (r/R)^d inside, saturated at Q outside; np.minimum "
               "clamps the radius at R."),
    (124, 126, "Hydrogen 1s: Q(r) = 1 - e^{-2Zr}(1 + 2Zr + 2(Zr)^2), the exact "
               "integral of a radial density proportional to r^2 e^{-2Zr}."),
    (127, 128, "Hernquist: Q = Omega_3 M r^2/(r+a)^2, i.e. V = +GM/(r+a) in the "
               "electric sign convention. Used only as a smooth analytic test "
               "profile, not as a physical claim."),
    (129, 130, "Plummer: Q = Omega_3 M r^3/(r^2+a^2)^{3/2}."),
    (131, 131, "Unknown profile name: raise rather than return a default."),
    (132, 132, "V(r): the potential of the same profile."),
    (133, 134, "Aliases; O = Omega_d."),
    (135, 139, "Uniform ball: the d-dimensional interior solution "
               "Q(dR^2-(d-2)r^2)/(2(d-2)Omega_d R^d) and the exterior "
               "Q/((d-2)Omega_d r^{d-2}); np.where picks the branch elementwise, "
               "so r may straddle R."),
    (140, 142, "Hydrogen 1s: V = (1-(1+Zr)e^{-2Zr})/(4 pi r)."),
    (143, 144, "Hernquist potential."),
    (145, 146, "Plummer potential."),
    (147, 147, "Unknown profile."),
]

# =====================================================================
# src/lattice96.py
# =====================================================================
A["lattice96.py"] = [
    (1, 8, "Docstring: the periodic Coulomb test, the exact lattice symbol, and the "
           "figure/table it reproduces."),
    (9, 12, "NumPy for the FFT, math for pi and sqrt, sys for the exit code, time "
            "for the timings printed in the log."),
    (14, 17, "resource gives the peak RSS. It is POSIX-only, so the import is "
             "guarded and degrades to None on Windows."),
    (19, 19, "Public surface of the module."),
    (21, 21, "OM3 = Omega_3 = 4 pi, written out once to avoid repeated math.pi calls."),
    (24, 25, "solve(L, Q=1): the whole lattice problem in one function."),
    (26, 26, "Angular wavenumbers k = 2 pi fftfreq(L): values in [-pi, pi) with "
             "spacing 2 pi/L. fftfreq returns cycles per unit length, hence the 2 pi."),
    (27, 27, "3-D wavenumber grids; indexing='ij' matches the array layout of "
             "np.fft.fftn (axis 0 = x)."),
    (28, 29, "The exact symbol of the 3-D lattice Laplacian: "
             "4 sum_i sin^2(k_i/2). Equals |k|^2 + O(k^4); using |k|^2 here would "
             "silently change the answer near the Nyquist frequency."),
    (31, 33, "Source: a point charge Q at the centre of the box minus the uniform "
             "background Q/L^3, so that the cell is neutral and the zero mode is "
             "consistent."),
    (35, 35, "Forward transform of the source."),
    (36, 36, "Allocate the transform of V with the same shape and dtype."),
    (37, 38, "Divide by the symbol everywhere except the zero mode, whose value is "
             "set to zero (the mean potential is a free constant)."),
    (39, 39, "Inverse transform; the imaginary part is round-off and is dropped "
             "explicitly rather than left to the printer."),
    (42, 43, "shell_stats: mean potential and mean r^2 on each shell "
             "S_R = {x : round(|x|) = R}."),
    (44, 44, "Box size from the array shape."),
    (45, 46, "Integer coordinates relative to the centre, then the 3-D grids."),
    (47, 47, "Euclidean radius of every site."),
    (48, 48, "Nearest-integer shell index; np.rint returns floats, hence astype(int)."),
    (49, 56, "One pass over the shells: boolean mask, then mean V, mean r^2 and the "
             "site count. Shells that are empty (none here) are skipped."),
    (59, 60, "table(): the four estimators of Table 4, one row per R."),
    (61, 61, "Row accumulator."),
    (62, 64, "Skip a shell if either R or R+1 is missing: the estimator needs the "
             "potential difference across the shell."),
    (65, 65, "dV = <V>_R - <V>_{R+1}, the flux through the shell."),
    (66, 66, "rho* = sqrt(R(R+1)) for a unit lattice. The value is computed for "
             "documentation and is not used by the four columns below (they are "
             "written out explicitly); it is dead but harmless, and is kept so that "
             "the reader can see the identity Omega_3 R(R+1) = Omega_3 rho*^2."),
    (67, 67, "Background term: (r2_R - r2_{R+1})/(2 d L^3) with d = 3 is the exact "
             "lattice Laplacian of V_bg = |x|^2/(2 d L^d) evaluated shell by shell."),
    (68, 73, "The four columns: left endpoint (R^2), midpoint ((R+1/2)^2), exact "
             "radius (R(R+1)), and exact radius with the background removed."),
    (76, 77, "fit_ar_b_cr2: least-squares fit of Vbar(R) = A/R + B + C R^2."),
    (78, 79, "Design vector and data vector, using only the shells present."),
    (80, 80, "Design matrix [1/R, 1, R^2]; np.vstack builds it column-wise, .T makes "
             "it (n, 3)."),
    (81, 81, "Ordinary least squares via SVD; rcond=None keeps NumPy's default "
             "cut-off, which is what the paper's numbers were produced with."),
    (82, 82, "Return A, B, C and the standard deviation of the residuals."),
    (85, 92, "Script entry point: solve and time it. The numbers printed below are "
              "and the fit. The numbers printed here are the ones audited by "
              "audit_claims.py."),
    (93, 94, "Peak resident set size in MB, or 'n/a' where resource is unavailable."),
    (95, 96, "One-line timing summary."),
    (98, 99, "Shells R = 8..24: far enough from the point charge that the continuum "
             "1/R behaviour is clean, close enough to have statistics."),
    (100, 105, "Print the per-shell table and the column means."),
    (107, 112, "Fit and print Q = 4 pi A, the constant B (which must not be dropped), "
               "C against its prediction 1/(6L^3), and the rms residual."),
    (113, 113, "Exit 0 on success; a failure would surface as a traceback and a "
               "non-zero exit code."),
]

# =====================================================================
# src/validate.py
# =====================================================================
A["validate.py"] = [
    (1, 4, "Docstring: every check is asserted; the exit code is 0 only if all "
           "checks pass."),
    (5, 7, "Imports. sys.path.insert makes the script runnable from the package "
           "root (python3 src/validate.py) without installation."),
    (9, 9, "NumPy 2.0 renamed trapz to trapezoid; this picks whichever exists."),
    (10, 11, "Start the clock and initialise the result list: (ok, label) pairs."),
    (14, 17, "chk(): record the result, print PASS/FAIL, return the boolean so that "
             "callers can chain. This is the only way a check is registered."),
    (20, 21, "head(): section banner."),
    (24, 25, "Suite header."),
    (27, 29, "[V1] rho*^(d-1) = 1/<r^(1-d)>: the defining property, checked against "
             "numerical quadrature in six dimensions."),
    (30, 30, "30 geometric shells from 0.1 to 10, so q is constant and close to 1."),
    (31, 31, "The exact radius for every shell, vectorised."),
    (32, 34, "Trapezoidal quadrature with 20001 points per shell: the mean of "
             "r^{1-d}; the list comprehension is over (a, b) pairs of shell edges."),
    (35, 36, "Relative deviation from the identity; must be below 1e-8, i.e. the "
             "quadrature error, not the formula error."),
    (38, 40, "[V2] a point charge is reproduced exactly, in d = 3, 4, 5."),
    (41, 44, "V = 1/((d-2)Omega_d r^{d-2}) is the exact point-charge potential, so "
             "Qhat must equal 1 to machine precision."),
    (45, 45, "Threshold 1e-13: only round-off is allowed."),
    (47, 49, "[V3] the (d-2)-less rule returns exactly Q/(d-2) -- the paper's "
             "sharpest statement about the generalisation."),
    (50, 54, "Same potential, but the radius is replaced by naive_radius; the mean "
             "over all shells must equal 1/(d-2)."),
    (55, 55, "1e-12 tolerance; the printed value shows the exact ratio."),
    (57, 63, "[V4] closed form versus definition: 7 dimensions x 37 ratios x 3 rules."),
    (64, 67, "Track the worst deviation; the closed form and the radius-based "
             "definition must agree to 1e-11."),
    (69, 75, "[V5] uniform lattice, d = 3: the three exact values of Eq. 18, "
             "compared against the closed forms written independently in the test."),
    (77, 79, "[V6] the threshold q*_d. At d = 3 the function must return None."),
    (80, 84, "For d = 4..8: check that the root satisfies the defining polynomial "
             "and lies in (1, 2)."),
    (86, 90, "[V7] five analytic profiles, including a four-dimensional ball."),
    (91, 92, "Instantiate each profile and read its dimension."),
    (93, 94, "400 geometric shells; Qhat from the exact radius."),
    (95, 96, "Compare with the true enclosed charge at the shell centres, masking "
             "the region where the true charge is numerically zero."),
    (97, 98, "Deviation must be below 5e-4: this is O((q-1)^2), not exact, because "
             "the rule is exact for a point source, not for a distributed one."),
    (100, 104, "[V8] the shell identity of Eq. 24 for a distributed source."),
    (105, 106, "2000 random shell pairs, sorted and filtered so that b > a."),
    (107, 107, "Uniform density of a ball of radius 2: rho0 = Q/(Omega_d 2^d/d)."),
    (108, 110, "Left side V(a)-V(b); right side the two-term identity with Q(a), "
               "Q(b) and the explicit source integral."),
    (111, 112, "Normalise by max|dV|; the identity must hold to 1e-12."),
    (114, 120, "[V9] the running-field residual against its two-term expansion in "
               "ln q: agreement to 5e-4 at q <= 1.5."),
    (122, 129, "[V10] stability: the naive difference form against the expm1 form. "
               "The check is marked PASS unconditionally (chk(True, ...)) because "
               "the point is the printed discrepancy: it grows like 1/((d-2)eps), "
               "which is exactly why the implementation uses expm1."),
    (131, 135, "[V11] the periodic-box background fraction at R = 8..32, against "
               "the values quoted in the paper."),
    (137, 139, "[V12] convergence on a graded mesh: 40, 80, 160, 320 shells."),
    (140, 146, "Build the mesh, estimate Qhat, compare with the true charge at shell "
               "centres, and record q for printing."),
    (147, 151, "Successive error ratios give the observed order; print each level."),
    (152, 153, "The three measured orders must lie in (1.9, 2.15), i.e. second order."),
    (155, 166, "[V13] multipole leakage: for a pure multipole of order l the "
               "estimator returns sum_{j<=l} q^{-j}; checked for l = 0..6 and three "
               "ratios."),
    (168, 178, "Summary: elapsed time, list of failures, and an exit code that the "
               "shell and any CI can act on."),
]

# =====================================================================
# src/audit_claims.py
# =====================================================================
A["audit_claims.py"] = [
    (1, 4, "Docstring and imports: this script re-derives every number that appears "
           "in radialflux.tex, including the tables."),
    (6, 8, "ok(): print PASS/FAIL and return the boolean. chk(): append to res. "
           "Splitting them keeps the one-liners on the following lines readable."),
    (10, 10, "Short alias for Omega_d."),
    (11, 13, "Eq. 18 identity: (2n+1)^2/(4n(n+1)) = 1 + 1/(4n(n+1)); checked "
             "algebraically at three n."),
    (15, 17, "Atomic solver claim: the midpoint rule differs from r_n e^{h/2} at "
             "order h^2/8; the ratio to h^2/8 must be 1 within 1e-3."),
    (19, 24, "Table 1: the three biases at q = 1.1 for d = 3..6, to 1e-6."),
    (26, 32, "Table 2: the uniform-lattice percentages, to 0.06 (left/right) and "
             "0.006 (midpoint) percentage points -- the tighter midpoint tolerance "
             "reflects its much smaller bias."),
    (34, 36, "q*_d against the published n-nacci constants, to 1e-6."),
    (38, 42, "Series checks: the slope of rho*/r_n at q = 1 is 1/2 - d eps/24."),
    (43, 48, "Second-order coefficients of kappa_L and kappa_A: the residual after "
             "subtracting the printed series must scale as eps^3, so res/eps^3 is "
             "bounded by 25."),
    (50, 51, "The 6.7 percent bias of the central difference at r = 4a."),
    (53, 55, "Forward difference quoted at the midpoint: kappa - 1 = 1/(4n(n+1))."),
    (57, 59, "Layered shell: (chi-1)^2/(4 chi) for the arithmetic/harmonic mix."),
    (61, 63, "Running residual: 4.69 percent at q = 1.1, and agreement with the "
             "two-term expansion to 5e-7."),
    (65, 66, "Multipole amplification at q = 1.25, l = 10: 1.375, i.e. 138 percent."),
    (68, 70, "The 96^3 background fractions quoted in the text."),
    (72, 73, "The d = 2 logarithmic mean."),
    (75, 82, "Recompute the 96^3 solution and the per-shell means (the same "
             "procedure as lattice96.shell_stats, inlined to keep this script "
             "self-contained)."),
    (83, 90, "Table 4: six shells, four estimators, tolerance 1e-4."),
    (91, 96, "Column means over R = 8..24, tolerance 1e-4."),
    (97, 101, "The A/R + B + C R^2 fit: Q = 4 pi A, C, and B."),
    (103, 114, "Cornell worked example: the exact formula a/(r+a) against the "
               "first-order expansion, at ten (a, r) points, plus the identity that "
               "the relative error equals -[a/(r+a)] alpha_s/alpha and the "
               "hbar*c = 0.1973 GeV fm normalisation."),
    (116, 117, "Print the tally and exit non-zero if anything failed."),
]

# =====================================================================
# src/check_tex.py
# =====================================================================
A["check_tex.py"] = [
    (1, 6, "Docstring: why this script exists -- a duplicate label is only a LaTeX "
           "warning, so it survives '0 errors' and silently redirects references."),
    (7, 7, "Standard library only."),
    (9, 9, "Path to the manuscript, resolved relative to this file so the script "
           "works from any directory."),
    (11, 12, "Read the whole source as text; no LaTeX parser is needed for these "
             "five structural checks."),
    (13, 13, "Every label, in order of appearance."),
    (14, 19, "First pass to find duplicates: a set of seen labels and a list of "
             "repeats."),
    (20, 20, "All ref/eqref/autoref targets."),
    (21, 21, "Every bibitem key."),
    (22, 24, "Every cite key, including comma-separated groups."),
    (25, 25, "Every includegraphics argument, with or without options."),
    (26, 27, "Figure labels and the subset of references that point at figures."),
    (29, 47, "The five findings: duplicates, dangling refs, uncited bibitems, "
             "phantom cites, never-referenced figures, missing figure files."),
    (48, 49, "One-line census, printed whether or not anything is wrong."),
    (50, 54, "Report and exit 1 on any finding."),
    (55, 57, "Success message and exit 0."),
    (59, 60, "Entry point."),
]

# =====================================================================
# symbolic / auxiliary scripts
# =====================================================================
A["series_d.py"] = [
    (1, 1, "SymPy is imported only here and in the two verify_* scripts; the "
           "production module never depends on it."),
    (2, 2, "eps = q-1 as a positive symbol; the shell is normalised to r_n = 1."),
    (3, 3, "Header."),
    (4, 12, "For d = 3..11: build rho* symbolically, then series-expand "
            "rho*, kappa_L, kappa_R, kappa_A and the flux-weighted mean radius to "
            "second order in eps. Every coefficient printed is a rational function "
            "of d."),
    (13, 16, "Print the five expansions per dimension, one block per d."),
    (17, 30, "The closed-form coefficient list that the paper quotes, printed so "
             "that the output can be pasted into the text."),
]

A["verify_core.py"] = [
    (1, 3, "Imports: SymPy for exact algebra, NumPy for a few numeric limits."),
    (4, 6, "Symbols: q, d, r_n, a and eps, all positive."),
    (8, 8, "Omega_d as a SymPy expression."),
    (10, 12, "rho_star(): the definition, used symbolically."),
    (14, 18, "Check 1: rho*^{d-1} equals the reciprocal mean of r^{1-d} exactly, "
             "for d = 3..6."),
    (19, 31, "Check 2: the naive rule, its q -> 1 limit (d-2)^{-1/(d-1)}, and the "
             "factorisation of the threshold polynomial for d = 3..10. The "
             "threshold constants are then computed numerically with q_star()."),
    (32, 45, "Check 3: exact kappa for the three rules and their series in e = q-1, "
             "d = 3..5."),
    (46, 56, "Check 4: uniform lattice, d = 3, symbolic in n. Confirms "
             "rho = sqrt(n(n+1)) and the three kappas, and shows that the report's "
             "expression was the reciprocal."),
    (57, 65, "Check 5: forward, backward and central differences of 1/r are exact "
             "at sqrt(r(r+a)), sqrt(r(r-a)) and sqrt(r^2-a^2)."),
    (66, 76, "Check 6: distributed source -- rho* versus the flux-weighted mean "
             "radius, and the difference (d-2)/24 r_n eps^2."),
    (78, 93, "Check 7: the running field A = ln r: the exact residual and its "
             "series, in d = 3 and 4."),
    (94, 108, "Check 8: the four-dimensional uniform ball, quoting Omega_4 = 2 pi^2."),
]

A["verify_shells.py"] = [
    (1, 2, "Imports: NumPy for the lattice, SymPy for the closed forms."),
    (3, 12, "Check D': the lattice Laplacian of |x|^2 is exactly 2d. The neighbour "
            "values are built explicitly from coordinates instead of np.roll, so "
            "periodic wrap-around cannot contaminate the stencil."),
    (13, 17, "Print the maximum deviation from 2d = 6 and the induced background "
             "density; compare with the continuum shell count."),
    (18, 36, "Check F: the two-term identity, symbolically, for d = 3 and d = 4, "
             "inside a uniform ball; the residual must be zero."),
    (37, 48, "Check G: bias of the point rule when the source varies across the "
             "shell -- rho* minus the weighted mean radius, against (d-2)(q-1)^2/24."),
    (49, 59, "Check H: the running field, exact residual versus the ln-series and "
             "the (q-1)-series, at four ratios."),
    (60, 66, "Check I: the condition number 1/((d-2)eps) of the difference form, "
             "which motivates the expm1 implementation."),
]

A["make_figs.py"] = [
    (1, 3, "Imports and the Agg backend (figures are written to files, not shown)."),
    (4, 9, "House style: serif fonts, DejaVu, 150 dpi for the draft PDFs."),
    (10, 11, "The palette used consistently across all four figures."),
    (12, 14, "save(): write both PDF (for LaTeX) and PNG (for slides) and close."),
    (15, 18, "Omega_d and the two radius formulas, in closed form, so that the "
             "figure scripts do not import the module (they predate it)."),
    (19, 43, "Figure 1: the concept. (a) the equal-area construction of rho*; "
             "(b) the four candidate radii on one shell."),
    (44, 78, "Figure 2: the naive generalisation. (a) rho_naive/r_n versus q for "
             "d = 3..7 with the thresholds marked; (b) the q -> 1 limit 1/(d-2)."),
    (79, 98, "Figure 3: exact bias on a uniform lattice, log-log, with the n^{-1} "
             "and n^{-2} guides."),
]

A["make_figs2.py"] = [
    (1, 1, "Re-use the style block of make_figs.py by executing its header only -- "
           "the split at the FIG 1 banner avoids re-running figures 1-3."),
    (2, 2, "NumPy is needed again after the exec."),
    (4, 47, "Figure 4: (a) bias versus eps for the four rules on a uniform ball -- "
            "the endpoints are first order, the midpoint and the exact radius are "
            "second order; (b) the running-field residual against gamma(q-1)/2."),
]

A["make_fig5.py"] = [
    (1, 9, "Imports, style, palette, and the lattice solver from src/."),
    (10, 30, "Solve the 96^3 problem once and build the per-shell means; this is "
             "the same computation as lattice96 but kept inline so the figure can "
             "be regenerated without parsing a log."),
    (31, 40, "The four estimator columns over R = 8..24."),
    (41, 66, "Panel (a): the four estimates versus R. (b): the closed-form "
             "background fraction. (c): the two scalings -- R^3/L^3 versus "
             "constant or ln R -- that make the null test possible."),
    (67, 70, "Numbers printed for the text: means, ranges, and the A/R+B+CR^2 fit."),
]

A["make_figs3d.py"] = [
    (1, 5, "Docstring: every 3-D figure is computed from the shipped module, so a "
           "picture cannot drift away from the code."),
    (6, 18, "Imports; the package src/ is put on sys.path so radialflux and "
            "lattice96 are imported, not copied."),
    (19, 41, "House style, ink and surface colour maps."),
    (42, 61, "style3d(): one place for the camera, transparent panes and grid."),
    (62, 77, "save()/sphere(): write PDF+PNG; generate a sphere grid, optionally a "
             "lune, for the cut-away views."),
    (78, 162, "S1: the geometry of one shell -- inner and outer sphere, the gold "
              "exact-flux sphere, flux arrows, and the three radii."),
    (163, 232, "S2: the bias landscape kappa_L(q,d) as a surface, with the exact "
               "plane kappa = 1 and the four tabulated points."),
    (233, 287, "S3: a graded mesh of seven shells, each with its own gold sampling "
               "sphere, showing that rho*/r_n is constant."),
    (288, 352, "S4: the 96^3 solution as a quarter-cutaway point cloud on a "
               "logarithmic colour scale."),
    (353, 407, "S5: the threshold surface q + q^{2-d} - 2 and its zero-level curve "
               "q*_d, with the n-nacci constants marked."),
    (408, 462, "S6: the convergence funnel -- measured error over (log(q-1), log N) "
               "with the second-order guide plane."),
    (463, 513, "S7: the outlook figure -- shells as symbol intervals, a Fibonacci "
               "constellation of sampling directions, and a modulated radial curve. "
               "Decorative; nothing in it enters any claim."),
    (514, 557, "S8: multipole leakage, P_l(q) = sum_{j<=l} q^{-j}, as a surface in "
               "(q, l); the blue curve is the saturation value q/(q-1) at l = 12, and "
               "the gold point marks the 138 percent case quoted in the text."),
    (558, 621, "S9: the null test as a waterfall -- the four estimators over R = 8..24 "
               "on one 96^3 lattice, with stems to the truth plane at Qhat/Q = 1."),
    (622, 625, "Entry point: run every figure, or only the ones named on the command "
               "line."),
]
