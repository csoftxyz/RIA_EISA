# Code–Paper Correspondence (English)

**Paper:** `PAPER_KOIDE_NOTE_v14.pdf` / `.tex`
*"Koide's relation is a pole-mass statement: transcendence of the phase, scheme dependence, and a conditional τ-mass prediction."*

This document maps every number in the paper to the script that produces it, lists the
input provenance, and records the status of each script in the package.

---

## 1. Authoritative scripts

| Script | Corresponds to (paper) | Inputs | Outputs |
|---|---|---|---|
| `z44_final2.py` | §1 (geometry), §2 (460σ), §3 (transcendence theorem), §4 (scheme table, 185σ, 6800×), §5 (CF window), §6 (closed form) | Three mass triplets (pole / MS̄(M_Z) / current PDG) | Q, r, φ (pole & MS̄); R_M0; sensitivity; δQ_exp; minimal polynomial of tan φ*; one-loop QED cross-check |
| `z44_final.py` | §6 closed form `√m_τ = 2(a+b)+√3√(m_e+4√(m_e m_μ)+m_μ)` | pole or MS̄ masses | m_τ(pole)=1776.969 MeV; m_τ(MS̄)=1724.79 MeV |
| `run_and_verify.py` | generates `VERIFY_LOG.md` | the scripts above | full run log + verification table |
| `build_paper_package.py` | builds this package | all scripts | `VERIFY_LOG.md`, `README_CODE.md`, `KOIDE_v14_package.zip` |

---

## 2. Data flow per paper section

- **§1 The parametrization is an identity.** `Q = (1+r²/2)/3 = (3+ρ²)/9`, `cos α = 1/√(3Q)`, `α ∈ [0°, 54.7356°]`, `r < 2`. Purely analytic (no script needed).
- **§2 Empirical question (Q1).** `R_M0` computed directly from `(√2, 2π/3+2/9)` by `z44_final2.py` (section 2) = **206.77032**; compared with the pole measurement 206.7682830(44) ⇒ **≈460σ**.
- **§3 Transcendence theorem.** `tan φ*` is algebraic (verified with `sympy.minimal_polynomial`, **degree 8**); combined with Lindemann–Weierstrass ⇒ `φ*` is a transcendental number of radians. Reproduced in `z44_final2.py` (section 5).
- **§4 Fundamental question (Q2).** Pole vs MS̄ table produced in `z44_final2.py` (section 1). `ΔQ = +1.26×10⁻³` = **185σ_Q** (= 205× the empirical deviation); `Δφ = −1.19×10⁻³ rad` = **6800× |φ*−2/9|**; sensitivity **56.14 rad⁻¹** (numeric derivative, section 3).
- **§5 Not a discrete-symmetry phase (pole scheme).** Window `δφ^{(μ/e)} = 2.1×10⁻⁸ / 56.14 = 3.7×10⁻¹⁰ rad`; continued-fraction σ values.
- **§6 The conditional prediction.** Closed form; pole input ⇒ `m_τ = 1776.969 MeV` (0.85σ); MS̄ input ⇒ `1724.79 MeV` vs actual `1746.56 MeV` (off by 22 MeV).

---

## 3. Input data and provenance

| Quantity | Value | Source |
|---|---|---|
| pole masses `m_e, m_μ, m_τ` | 0.510998918, 105.6583692, 1776.86 MeV | Xing–Zhang [hep-ph/0602134], Eq. (1) (PDG 2004 era) |
| current PDG/CODATA `m_e, m_μ` | 0.51099895000, 105.6583755 MeV | differ in the 8th digit; effect = 0.14σ (immaterial) |
| MS̄ masses at `M_Z` | 0.486755106, 102.740394, 1746.56 MeV | Xing–Zhang, Eq. (8) |
| one-loop QED cross-check | `m̄/m_pole` = 0.9526 / 0.9724 / 0.9830 | independent estimate, agrees to 0.3% |

---

## 4. Script status

| Script | Status |
|---|---|
| `z44_final2.py` | ★ **Authoritative** — all numbers of §1–§6 (mpmath 50-digit, no hard-coding) |
| `z44_final.py` | §6 closed form; contains hard-coded `56.15`/`R_M0` (superseded by `z44_final2.py`) |
| `z44_koide.py` | historical: Koide `√2` = root-shell norm (superseded by the theorem) |
| `z44_koide2.py` | historical: fit of the `2/9` phase (superseded) |
| `z44_derive_2over9.py` | historical: search for a lattice origin of `2/9` (provably impossible, see theorem) |
| `z44_derive_2over9b.py` | historical: dynamical search for `2/9` |
| `z44_derive_2over9c.py` | historical: hexagonal discretisation test (refuted) |
| `z44_eps_scheme_v3.py` | historical: scheme ambiguity ε (premise corrected in review round 6) |
| `z44_rge_phi2.py` | **withdrawn**: §7 scale-scan used a non-physical interpolation (removed since v12) |
| `z44_2loop_v2.py` | historical: two-loop order-of-magnitude estimate (sign undetermined) |
| `run_and_verify.py` | generates `VERIFY_LOG.md` |

---

## 5. How to reproduce

```
cd z44_flavor_probe
python3 z44_final2.py        # all key numbers of the paper
python3 z44_final.py         # closed form
python3 run_and_verify.py    # regenerate VERIFY_LOG.md
```

Environment: Python 3 with `numpy`, `scipy`, `sympy`, `mpmath` (mpmath used at 50-digit precision).
All scripts are pure Python (no external services).

---

## 6. Verification (summary)

`VERIFY_LOG.md` §B confirms, item by item, that every key number in v14 is produced directly
by `z44_final2.py` (and the closed form by `z44_final.py`):

| Paper quantity | Script value | Match |
|---|---|---|
| Q / r / φ (pole) | 0.6666605175 / 1.414200518 / 0.2222296241 | ✓ |
| Q / r / φ (MS̄) | 0.6679239596 / 1.416878173 / 0.2210406444 | ✓ |
| R_M0 | 206.770315973 | ✓ |
| sensitivity | 56.1439 rad⁻¹ | ✓ |
| ΔQ / σ_Q, ΔQ / empirical deviation | 186.4σ, 205.19× | ✓ |
| minimal polynomial of tan φ* | degree 8 | ✓ |
| one-loop QED cross-check | 0.95256 / 0.97238 / 0.98295 | ✓ |
| closed form m_τ | 1776.9689 (pole) / 1724.79 (MS̄) | ✓ |
