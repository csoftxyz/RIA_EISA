# Z₃-Graded Vacuum Geometry: High-Energy EFT Validation Guide for ATLAS  
**Version 2.0 — High-Mass Tail Focus**  
**Timestamp: March 9, 2026**

**To:** ATLAS Top Physics Working Group / SMEFT Interpretation Group  
**From:** Yuxuan Zhang  

## 1. Executive Summary
We formally retract all previous phenomenological claims of a possible scalar resonance at ~355 GeV in the tt¯ threshold region. Those estimates relied on a free coupling κ ≈ 0.1 and lacked algebraic rigidity. Current ATLAS-CONF-2025-008 data firmly anchor the peak at ~345 GeV with excellent NRQCD fit (χ²/dof ≈ 1.05).

We now restrict all physical predictions of the Z₃ framework to the **decoupled high-energy EFT regime** (M_tt ≫ 2 m_t). In this limit, the vacuum mode ζ manifests exclusively as a dimension-6 contact interaction. A rigid, zero-parameter algebraic coefficient **C_Z3 = 8/63 ≈ 0.12698** is provided for direct testing in high-mass tail analyses.

## 2. Theoretical Foundation
In the exact 19-dimensional matrix representation of the Z₃-graded Lie superalgebra, the relative strength between the vacuum-mediated dimension-6 operator and standard QCD gluon exchange is uniquely fixed by the ratio of their Super-Killing forms (invariant trace norms):

\[
C_{Z_3} = \frac{\sum_k \operatorname{Tr}(\zeta_k \zeta_k^\dagger)}{\sum_a \operatorname{Tr}(B^a B^{a\dagger})} = \frac{8}{63} \approx 0.12698
\]

This coefficient is a direct computational consequence of the closed algebra (Jacobi residual < 10^{-16}) and contains **zero free parameters**.

The predicted deviation in the high-mass tail is:

\[
\frac{d\sigma_{\rm obs}}{d\sigma_{\rm SM}} \simeq 1 \pm \frac{8}{63} \left( \frac{M_{t\bar{t}}}{\Lambda_{\rm alg}} \right)^2
\]

## 3. Three Projected Validation Scenarios for Run 3 High-Mass Data
We define three clear validation scenarios based on upcoming ATLAS high-mass tt¯ differential measurements (M_tt up to 3–4 TeV).

**Scenario A: No Deviation**  
High-mass tail consistent with SM within uncertainties.  
→ Does not falsify the algebra, but pushes Λ_alg beyond current LHC reach.

**Scenario B: Deviation Exists but Slope Mismatch**  
Non-zero EFT coefficient favored, but extracted slope significantly different from 8/63.  
→ **Falsification** of the specific Z₃ 44-vector lattice geometry.

**Scenario C: Exact Slope Match**  
Clean quadratic growth observed; fit with fixed slope 8/63 yields excellent χ².  
→ Strong evidence for the Z₃ vacuum geometry. The data would confirm not just new physics, but the discrete 19-dimensional algebraic structure of the vacuum.

## 4. Technical Directive for ATLAS Analysts
- Focus exclusively on the high-mass tail (M_tt > 1.5 TeV).  
- Do not attempt resonance fitting in the threshold region.  
- Use the provided UFO model in EFT mode (with M_ζ set artificially high, e.g. 10 TeV) to generate interference templates.  
- Extract the slope coefficient in global SMEFT fits and compare directly to the rigid value **8/63**.

This zero-parameter prediction was committed to the public record on March 9, 2026 — well before any Run 3 final unblinding. We invite the ATLAS collaboration to test this exact rational coefficient in upcoming high-mass tt¯ analyses.

**Full computational verification code and trace-norm extraction** are available in the repository file `z3_algebra_verify_mini_para.py`.

---