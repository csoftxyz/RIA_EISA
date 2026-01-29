import numpy as np

print("=== Z3 Lattice: Higgs/Top Mass Ratio Investigation ===\n")

# Experimental Values (PDG 2024)
m_h = 125.25
m_t = 172.76
ratio_exp = m_h / m_t
print(f"Experimental Ratio m_H / m_t: {ratio_exp:.5f}")

# Geometric Candidates from Lattice
# Basis vectors and combinations
# L=1 (Basis), L=sqrt(2) (Root), L=sqrt(3) (Democratic)

candidates = {
    "1/sqrt(2)": 1/np.sqrt(2),
    "sqrt(1/2)": np.sqrt(0.5),
    "1/sqrt(3)": 1/np.sqrt(3),
    "2/3": 2/3,
    "3/4": 0.75,
    "sqrt(2)/2": np.sqrt(2)/2,
    "sqrt(3)/2": np.sqrt(3)/2,
    "11/15": 11/15, # Random guess
    "8/11": 8/11
}

print("\n--- Testing Geometric Hypotheses ---")
best_cand = None
min_err = 1.0

for name, val in candidates.items():
    err = abs(val - ratio_exp) / ratio_exp
    print(f"{name:<15} | {val:.5f} | Error: {err:.2%}")
    if err < min_err:
        min_err = err
        best_cand = (name, val)

print("-" * 50)
print(f"Best Geometric Match: {best_cand[0]} ({best_cand[1]:.5f})")
print(f"Deviation from Exp:   {min_err:.2%}")

# Advanced: Lattice Vector Ratio
# Check if ratio = |v1| / |v2| for any pair in the 44-lattice
print("\n--- Searching in 44-Vector Lattice Ratios ---")

# (Reuse generation code briefly)
basis = np.eye(3); dem = np.array([1,1,1])/np.sqrt(3)
seed = np.vstack([basis, [dem, -dem]])
T = np.array([[0,0,1],[1,0,0],[0,1,0]])
def apply(v): return T@v
unique = set()
for v in seed: unique.add(tuple(np.round(v, 8)))
current = seed.tolist()
for _ in range(5): # Quick gen
    new = []
    for v in current:
        v=np.array(v); v1=apply(v); v2=apply(v1)
        new+=[v1, v2, v1-v, v2-v, np.cross(v,v1)]
    for n in new:
        if np.linalg.norm(n)>1e-6: unique.add(tuple(np.round(n, 8)))
    current = [np.array(u) for u in unique][-50:]
    if len(unique)>=44: break
    
vecs = [np.array(u) for u in unique]
lengths = sorted(list(set([round(np.linalg.norm(v), 5) for v in vecs if np.linalg.norm(v)>1e-6])))

print(f"Lattice Lengths: {lengths}")

# Check Ratios L_i / L_j
found_ratio = False
for l1 in lengths:
    for l2 in lengths:
        r = l1 / l2
        if abs(r - ratio_exp) < 0.01:
            print(f"Match Found! L1={l1}, L2={l2} -> Ratio={r:.5f} (Error: {abs(r-ratio_exp)/ratio_exp:.2%})")
            found_ratio = True

if not found_ratio:
    print("No direct simple ratio found in low-order lattice.")
    print("Suggestion: Mass is likely determined by Loop Factor (Radiative Correction).")
    print("m_H^2 = m_t^2 * (Geometry) + Loop")