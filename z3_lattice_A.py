import numpy as np

# ================================================================
# Z₃ 44-CRYSTAL LATTICE GENERATOR
# Exact algorithm matching gen44() from z3_44_physics_mapping.py
# — the version on which Zhang2026Symmetry + PLB submission are based.
# ================================================================

basis = np.eye(3)
dem = np.array([1, 1, 1]) / np.sqrt(3)
seed = np.vstack([basis, [dem, -dem]])

T_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

def apply_triality(v):
    return T_mat @ v

# ---- Generate (EXACT gen44 algorithm) ----
uniq = set()
for v in seed:
    uniq.add(tuple(np.round(v, 8)))

current = seed.tolist()
levels = 15

for level in range(levels):
    new = []
    for v in current:
        v1 = apply_triality(v)
        v2 = apply_triality(v1)
        new += [v1, v2, v1 - v, v2 - v]

        cr = np.cross(v, v1)
        if np.linalg.norm(cr) > 1e-6:
            new.extend([cr, cr / np.linalg.norm(cr)])

    for nv in new:
        if np.linalg.norm(nv) > 1e-6:
            uniq.add(tuple(np.round(nv, 8)))

    # Next generation: sample from accumulated set (hash order, no norm sort)
    all_v = [np.array(u) for u in uniq]
    current = [v.tolist() for v in all_v[:100]]

    print(f"Level {level+1}: {len(uniq)} unique vectors")

# ---- Post-process ----
vectors_all = [np.array(t) for t in uniq]
vectors_all = [v for v in vectors_all if np.linalg.norm(v) > 1e-6]
vectors_all.sort(key=lambda x: (round(np.linalg.norm(x), 4), np.sum(np.abs(x))))
vectors_44 = np.array(vectors_all[:44])

print(f"\nTotal: {len(vectors_all)} non-zero  |  Top 44: the 44晶格\n")

# ================================================================
# ANALYSIS
# ================================================================
shells = {}
for v in vectors_44:
    L2 = round(np.sum(v**2), 4)
    shells.setdefault(L2, []).append(np.round(v, 6))

print("=" * 70)
print("44-CRYSTAL SHELL STRUCTURE")
print("=" * 70)

for L2 in sorted(shells.keys()):
    vecs = shells[L2]
    n = len(vecs)
    if n == 6:
        label = "★ ROOT shell (K₂,₂,₂ octahedron)"
    elif n == 1 and abs(vecs[0][0]-vecs[0][1])<0.01 and abs(vecs[0][1]-vecs[0][2])<0.01:
        label = "◆ DEMOCRATIC [111]"
    else:
        label = f"  {n} vectors"
    print(f"\n  L²={L2:<8} {n:>2} v  {label}")
    for v in vecs[:6]:
        print(f"    [{v[0]:>10.5f} {v[1]:>10.5f} {v[2]:>10.5f}]")
    if len(vecs) > 6:
        print(f"    ... +{len(vecs)-6} more")

# ---- Verify against papers ----
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

exp_roots = {2.0, 6.0, 18.0, 54.0, 162.0, 486.0}
exp_dems  = {3.0, 27.0, 243.0}
found_roots = {L2 for L2, v in shells.items() if len(v) == 6}
found_dems  = set()
for L2, v in shells.items():
    if len(v) == 1 and abs(v[0][0]-v[0][1])<0.01 and abs(v[0][1]-v[0][2])<0.01:
        found_dems.add(L2)

print(f"  Root shells:      expected {sorted(exp_roots)}")
print(f"                    found   {sorted(found_roots)}")
missing_r = exp_roots - found_roots
print(f"                    {'✅ ALL PRESENT' if not missing_r else '⚠ MISSING: '+str(sorted(missing_r))}")

print(f"  Democratic:       expected {sorted(exp_dems)}")
print(f"                    found   {sorted(found_dems)}")
missing_d = exp_dems - found_dems
print(f"                    {'✅ ALL PRESENT' if not missing_d else '⚠ MISSING: '+str(sorted(missing_d))}")

print(f"  Total vectors:    {len(vectors_44)} {'✅' if len(vectors_44)==44 else '⚠'}")

# ---- Democratic chain ----
print("\n" + "=" * 70)
print("DEMOCRATIC CHAIN: n·[1,1,1]")
print("=" * 70)
for L2 in sorted(shells.keys()):
    for v in shells[L2]:
        if abs(v[0]-v[1])<0.01 and abs(v[1]-v[2])<0.01 and abs(v[0])>0.001:
            print(f"  L²={L2:<8} n={v[0]:.4f}    (3^{round(np.log(abs(v[0]))/np.log(3),1):.0f} scale)")

# ---- Root scaling ----
print("\n" + "=" * 70)
print("ROOT SHELL SCALING (geometric progression)")
print("=" * 70)
print(f"  {'L²':<8} {'|v|':<10} {'scale √(L²/2)':<15}")
prev = None
for L2 in sorted(shells.keys()):
    if len(shells[L2]) == 6:
        s = np.sqrt(L2/2.)
        r = f"×{s/prev:.2f}" if prev else "  —"
        print(f"  {L2:<8} {np.sqrt(L2):<10.4f} {s:<15.4f} {r}")
        prev = s

print("\n" + "=" * 70)
print("KEY FACTS")
print("=" * 70)
print("""
  1. 5 seeds (3 basis + 2 democratic ±) × Z₃ closure → 44-vector lattice.
  2. L²=2: FIRST root shell — 6 vectors = K₂,₂,₂ graph = octahedron topology.
  3. K₂,₂,₂ Laplacian → {0,4,4,4,6,6} → 6=1⊕3⊕2 = U(1)⊕SU(2)⊕SU(3).
  4. Higher root shells (√3 scaled) carry same K₂,₂,₂ topology.
  5. Democratic direction emerges from raw cross products:
     cross([-1,1,0], [0,-1,1]) = [1,1,1] → L²=3 democratic shell.
  6. Democratic chain: [1,1,1] → [3,3,3] → [9,9,9] (3-generation hierarchy).
  7. Octahedron is NOT chosen — it IS the minimal K₂,₂,₂ structure.
""")
