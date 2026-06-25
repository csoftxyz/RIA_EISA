# Z₃ Rigidity Proof: Inverted Mass Ordering (IO) is FORCED

**Theorem**: The Z₃-graded Lie superalgebra (19D, 12B+4F+3Z) rigidly predicts Inverted Ordering (IO) for neutrino masses: m₃ < m₁ ≈ m₂. If JUNO/DUNE measure Normal Ordering (NO), the Z₃ algebra is falsified.

---

## Proof Architecture

Four independent pathways converge on the same conclusion. Any permutation of the mass ordering violates at least one constraint.

```
                    ┌──────────────────────┐
                    │   Z₃ Algebra (19D)   │
                    └────────┬─────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   Pathway ①  │  │   Pathway ②  │  │   Pathway ③  │
   │  Algebraic   │  │  Geometric   │  │ Rep Theory   │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  ε_ν₂ < ε_ν₃   │
                    │  1/36  < 1/12  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Pathway ④      │
                    │  Contradiction  │←── Assume NO → algebra breaks
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  IO: m₃ < m₁ ≈ m₂       │
              │  FALSIFIABLE by JUNO/DUNE│
              └──────────────────────────┘
```

---

## Pathway ① — Algebraic (Killing Form)

The democratic coupling ε_ν₂ and root coupling ε_ν₃ are derived directly from the Killing form on the 19D algebra:

```
ε_ν₂ = (‖T_dem‖² / ‖T_su3‖²) / dim(u3) = (1/4) / 9 = 1/36
ε_ν₃ = Δq × dim(Root)/dim(Hyb) = (1/3) × (6/24) = 1/12
```

- `‖T_dem‖² = 1` — U(1) trace generator in u(3) = su(3)⊕u(1)
- `‖T_su3‖² = 4` — 8 SU(3) Gell-Mann generators, each ‖λ_a/2‖² = 1/2
- `dim(u3) = 3² = 9` — dilution factor from projecting 19D adjoint onto 3D flavor space
- `Δq = q(F⁴) − q(F¹²³) = 1/2 − 1/6 = 1/3` — F-sector U(1) charge difference

**Key**: ε_ν₂/ε_ν₃ = 1/3, algebraically locked. Zero free parameters.

---

## Pathway ② — Geometric (Z₃ Lattice Shells)

The 44-vector Z₃ lattice reveals a fundamental asymmetry:

| Shell type | L² values | Vectors per shell | Z₃ character |
|-----------|-----------|-------------------|--------------|
| Democratic | 3, 27, 243 | 2 (± pair ∥ [111]) | χ = 1 |
| Root | 2, 6, 18, 54, 162, 486 | 6 (⊥ [111] plane) | χ = ω, ω² |

The democratic channel couples through a single direction; the root channel couples through a 2D plane with 6-fold multiplicity. This geometric density ratio (1:6) propagates directly to the ε hierarchy.

---

## Pathway ③ — Representation Theory (Z₃ Characters)

Z₃ has three irreducible representations, each coupling to a distinct flavor:

```
ν_τ → χ₀ = 1      (trivial rep → democratic eigenstate |e₀⟩)
ν_e → χ₁ = ω      (complex rep → root eigenstate |e₁⟩)
ν_μ → χ₂ = ω²     (complex rep → root eigenstate |e₂⟩)
```

The democratic eigenstate |e₀⟩ = [1,1,1]/√3 spans a 1D subspace. The root eigenstates |e₁⟩, |e₂⟩ span a 2D subspace. Combined with the shell density (6:1), the root channel has 12× more phase space than the democratic channel.

---

## Pathway ④ — Contradiction Scan (Reverse Verification)

The neutrino mass matrix in the Z₃ eigenstate basis:

```
M_ν = m₀ · [(1 + ε_ν₃)·I + (ε_ν₂ − ε_ν₃)·P_dem]
```

where P_dem = |e₀⟩⟨e₀| is the democratic projector.

**Eigenvalues**:
- Democratic state |e₀⟩: mass ∝ (1 + ε_ν₂) → **lighter** (ε_ν₂ < ε_ν₃, subtractive term)
- Root states |e₁⟩, |e₂⟩: mass ∝ (1 + ε_ν₃) → **heavier**

**Contradiction**: If NO were true, we would need ε_ν₂ > ε_ν₃. This requires:

```
‖T_dem‖² / ‖T_su3‖² > dim(u3) × ε_ν₃ = 9/12 = 0.75
```

But the actual ratio is 1/4 = 0.25 — the trace direction **cannot** exceed the adjoint in SU(3). Hence NO is algebraically impossible.

---

## The Three-Lock Theorem

```
┌─────────────────────────────────────────────────────────┐
│  🔒 LOCK 1: ε_ν₂ < ε_ν₃    ← Killing form (algebraic)  │
│  🔒 LOCK 2: density 1:6     ← Lattice shells (geometric)│
│  🔒 LOCK 3: χ₀ vs χ₁+χ₂     ← Z₃ irreps (rep theory)   │
│                                                         │
│  All three locks independently force the same ordering. │
│  Breaking any one lock breaks the Z₃ algebra.           │
└─────────────────────────────────────────────────────────┘
```

---

## Falsifiability

This proof makes IO a **theorem of the Z₃ algebra**, not a parameter choice. If JUNO or DUNE measure NO, the Z₃ framework is falsified — completely, cleanly, with no escape via parameter adjustment.

That is the strength of a zero-parameter theory.

---

## Run

```bash
python3 z3_io_rigidity_proof.py
```

Generates `z3_io_rigidity_viz.png` — 3D visualization of the 44-lattice, mass ellipsoid, ε hierarchy, and contradiction tree.

---

*Yuxuan Zhang (csoft@live.cn), 2026*
*Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality*
