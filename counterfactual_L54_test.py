"""
counterfactual_L54_test.py
================================================================================
COUNTERFACTUAL TEST: Does the S(k) peak MOVE when L²=54 vectors are removed?

Referee challenge:
"If you remove L²=54 from L44, does S(k) peak shift? If not, the peak
is determined by other shells — L²=54 selection is an independent consequence
of lattice geometry, not circular."

This test directly addresses the circularity accusation.
================================================================================
"""
import numpy as np
import sys
sys.path.insert(0, '/root/.openclaw/workspace')
from z3_structure_factor import generate_44_lattice, project_A2, radial_structure_factor, find_peaks

def main():
    np.set_printoptions(precision=6, suppress=True)
    
    print("=" * 72)
    print("  COUNTERFACTUAL TEST: S(k) PEAK STABILITY UNDER SHELL REMOVAL")
    print("=" * 72)
    
    # Generate full lattice
    lattice = generate_44_lattice()
    L2_vals = np.array([np.sum(v**2) for v in lattice])
    L2_rounded = np.array([round(float(x), 0) for x in L2_vals])
    
    print(f"\n  Full lattice: {len(lattice)} vectors")
    from collections import Counter
    counts = Counter(L2_rounded)
    print(f"  Shell multiplicities: {dict(sorted(counts.items()))}")
    
    # S(k) of full lattice
    v_full = project_A2(lattice)
    k_vals = np.linspace(0.02, 8.0, 800)
    Sk_full = radial_structure_factor(v_full, k_vals)
    k_full, _ = find_peaks(k_vals, Sk_full, min_height=0.003, label="Full L44")
    
    a_g = 0.246
    G_g = 4*np.pi/(np.sqrt(3)*a_g)
    L2_eff_full = G_g / k_full if k_full else None
    print(f"  L²_eff (full) = |G|/k₁ = {G_g:.4f}/{k_full:.4f} = {L2_eff_full:.2f}")
    
    print(f"\n{'='*72}")
    print("  COUNTERFACTUAL: REMOVE EACH SHELL AND RECOMPUTE S(k) PEAK")
    print(f"{'='*72}")
    print(f"\n  {'Shell removed':<16} {'N_remain':<10} {'k_peak':<10} {'L²_eff':<10} {'Δ(L²_eff)':<12} {'Peak shift?'}")
    print(f"  {'-'*68}")
    
    results = {}
    for shell in sorted(counts.keys()):
        mask = L2_rounded != shell
        lattice_reduced = lattice[mask]
        v_reduced = project_A2(lattice_reduced)
        
        if len(v_reduced) < 5:
            continue
        
        Sk_red = radial_structure_factor(v_reduced, k_vals)
        k_red, _ = find_peaks(k_vals, Sk_red, min_height=0.002)
        
        if k_red is None:
            results[shell] = {'k': None, 'L2_eff': None, 'delta': None}
            print(f"  L²={int(shell):<12} {len(lattice_reduced):<10} {'no peak':<10} {'-':<10} {'-':<12} —")
            continue
        
        L2_eff_red = G_g / k_red
        delta = L2_eff_red - L2_eff_full
        shift_pct = abs(delta) / L2_eff_full * 100
        stable = "STABLE" if shift_pct < 5 else "SHIFTED"
        
        results[shell] = {'k': k_red, 'L2_eff': L2_eff_red, 'delta': delta}
        print(f"  L²={int(shell):<12} {len(lattice_reduced):<10} {k_red:<10.4f} {L2_eff_red:<10.2f} {delta:<+12.2f} {stable} ({shift_pct:.1f}%)")
    
    # KEY RESULT: What happens when L²=54 specifically is removed?
    print(f"\n{'='*72}")
    print("  KEY RESULT")
    print(f"{'='*72}")
    
    r54 = results.get(54, {})
    if r54.get('L2_eff') is not None:
        delta54 = r54['delta']
        shift54 = abs(delta54) / L2_eff_full * 100
        print(f"\n  Full lattice S(k) peak: L²_eff = {L2_eff_full:.2f}")
        print(f"  After removing L²=54:   L²_eff = {r54['L2_eff']:.2f}")
        print(f"  Shift: ΔL²_eff = {delta54:+.2f} ({shift54:.1f}%)")
        
        if shift54 < 5:
            print(f"\n  ✓ CONCLUSION: The S(k) peak is STABLE under L²=54 removal.")
            print(f"    The peak at L²_eff ≈ {L2_eff_full:.0f} is determined by the COLLECTIVE")
            print(f"    geometry of the OTHER shells, NOT by L²=54 itself.")
            print(f"    This REFUTES the circularity accusation: L²=54 is selected")
            print(f"    because it is the nearest discrete shell to an INDEPENDENTLY")
            print(f"    determined S(k) peak.")
        else:
            print(f"\n  ✗ WARNING: The S(k) peak SHIFTS when L²=54 is removed.")
            print(f"    This suggests the L²=54 shell contributes to determining")
            print(f"    the peak position. The circularity concern has merit.")
    else:
        print(f"  Could not compute (no peak found after removal)")

if __name__ == "__main__":
    main()
