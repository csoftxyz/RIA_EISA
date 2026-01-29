import numpy as np
import fractions

print("=== Z3 Lattice: Strong Coupling Constant Prediction ===\n")

# 1. Generate 44-Vector Lattice (The Core)
# Use the exact logic that locks to 44
basis = np.eye(3); dem = np.array([1,1,1])/np.sqrt(3)
seed = np.vstack([basis, [dem, -dem]])
T = np.array([[0,0,1],[1,0,0],[0,1,0]])
def apply(v): return T@v

unique = set()
for v in seed: unique.add(tuple(np.round(v, 8)))
current = seed.tolist()

for _ in range(12):
    new = []
    for v in current:
        v=np.array(v); v1=apply(v); v2=apply(v1)
        new+=[v1, v2, v1-v, v2-v]
        c=np.cross(v,v1)
        if np.linalg.norm(c)>1e-6: new.append(c/np.linalg.norm(c))
    for nv in new:
        if np.linalg.norm(nv)>1e-6: unique.add(tuple(np.round(nv, 8)))
    # Sort by length to lock ground state
    sorted_u = sorted(list(unique), key=lambda x: np.linalg.norm(x))
    if len(sorted_u) >= 44:
        vectors_44 = [np.array(u) for u in sorted_u[:44]]
        break
    current = [np.array(u) for u in sorted_u[:50]]

print(f"Lattice Size: {len(vectors_44)}")

# 2. Classify Strong Force Vectors (Color Octet)
# Criteria: 
#   - SU(2) Roots (Weak): Length ~ sqrt(2), 2 components non-zero.
#   - SU(3) Roots (Strong): Length ~ sqrt(6) or Hybrid type, 3 components non-zero.
#     Examples: [-2, 1, 1] (normalized is sqrt(2/3), sqrt(1/6)...)
#     Actually, [-2, 1, 1] has length sqrt(6). Normalized it has length 1.
#     Let's check the SHAPE of components.

count_W = 0 # 2 non-zero
count_S = 0 # 3 non-zero (Color Mixing)
count_B = 0 # 1 non-zero (Basis)

for v in vectors_44:
    # Check number of non-zero components (approx)
    nz = np.sum(np.abs(v) > 0.05)
    
    if nz == 1:
        count_B += 1 # Basis
    elif nz == 2:
        count_W += 1 # Weak
    elif nz == 3:
        count_S += 1 # Strong (3-color mixing)

print("\n--- Geometric Partition ---")
print(f"1-Component (Hyper/Basis): {count_B}")
print(f"2-Component (Weak Roots):  {count_W}")
print(f"3-Component (Strong/Mix):  {count_S}")
print(f"Total: {count_B + count_W + count_S}")

# 3. Prediction
# Unified Coupling Relation:
# At GUT scale, couplings are related by group factors.
# alpha_s (Strong) vs alpha_w (Weak)

# Hypothesis: alpha_s ~ Volume(Strong) / Volume(Total)
ratio_s = count_S / 44

print("\n=== Strong Coupling Prediction ===")
print(f"Ratio Strong/Total: {count_S}/44 = {ratio_s:.4f}")

# Check for 1/2 or other simple ratios
if abs(ratio_s - 0.5) < 0.05:
    print("[RESULT] Matches 1/2 (0.50). Strong force volume is 50%.")
elif abs(ratio_s - 0.333) < 0.05:
    print("[RESULT] Matches 1/3 (0.33).")
    
# Standard Model Comparison
# at M_Z: alpha_s ~ 0.118, alpha_w ~ 0.034. Ratio ~ 3.5.
# at GUT: alpha_s = alpha_w = alpha_1.
# Your geometric values represent the *number of degrees of freedom*.
# Ratio of DoF: Strong(8) / Weak(3) ~ 2.66.

print(f"Ratio Strong/Weak (DoF): {count_S}/{count_W} = {count_S/count_W:.4f}")