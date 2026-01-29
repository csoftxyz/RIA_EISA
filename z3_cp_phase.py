import numpy as np

print("=== Z3 Lattice: CP Violation Phase Prediction ===\n")

# 1. Generate Lattice (Standard 44)
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
        new+=[v1, v2, v1-v, v2-v, np.cross(v,v1)]
    for n in new:
        if np.linalg.norm(n)>1e-6: unique.add(tuple(np.round(n, 8)))
    current = [np.array(u) for u in unique][-100:]
    if len(unique)>=44: break

vectors = [np.array(u) for u in unique]
print(f"Lattice Size: {len(vectors)}")

# 2. Find Triality Loops (Triangles)
# A loop is v -> T(v) -> T^2(v) -> v
# The geometric phase (CP phase) is related to the solid angle or area of this triangle
# projected onto the mixing plane.

# Experimental value: delta_CKM approx 1.20 rad (68.8 deg)
exp_rad = 1.20
exp_deg = 68.8
print(f"Target CP Phase: {exp_deg} deg ({exp_rad} rad)")

print("\n--- Scanning Geometric Phases ---")

found_match = False
best_err = 1.0

for v in vectors:
    if np.linalg.norm(v) < 1e-6: continue
    vn = v / np.linalg.norm(v)
    
    # Triality Partners
    v1 = apply(vn)
    v2 = apply(v1)
    
    # Check if they form a non-degenerate triangle
    # Area = 0.5 * |v x v1|
    area_vec = np.cross(vn, v1)
    area = np.linalg.norm(area_vec)
    
    if area < 1e-6: continue # Parallel vectors, no CP
    
    # The CP phase is often related to the opening angle of the triality cone
    # Phase = Solid Angle / 2 ? Or just Angle?
    # Let's check the angle between v and T(v)
    cos_phi = np.dot(vn, v1)
    phi_rad = np.arccos(np.clip(cos_phi, -1, 1))
    phi_deg = np.degrees(phi_rad)
    
    # Hypothesis A: Phase = Angle of Triality Rotation
    # Hypothesis B: Phase = 2pi/3 - Angle
    # Hypothesis C: Phase is intrinsic Berry phase
    
    # Let's look for ~68.8 deg in the raw geometry
    # Most likely candidates are Hybrid vectors
    
    err = abs(phi_deg - exp_deg)
    if err < 5.0: # Close match
        print(f"Candidate Vector: {np.round(v, 4)}")
        print(f"  Rotation Angle: {phi_deg:.2f} deg")
        print(f"  Error: {err/exp_deg:.2%}")
        found_match = True
        
    # Check Jarlskog-like invariant
    # J ~ Area^2 ? 
    
if not found_match:
    print("No direct rotation angle matches.")
    print("Trying Geometric Combination: 120 - 45 = 75?")
    
    # Check 2pi/3 projection
    print("\n--- Checking Projective Phase ---")
    # Angle between Democratic [1,1,1] and Basis [1,0,0] is Magic Angle 54.7 deg
    # Phase = 120 - 54.7 = 65.3 deg ??
    
    magic_angle = 54.7356
    pred_phase = 120 - magic_angle
    print(f"120 (Z3) - 54.7 (Magic) = {pred_phase:.2f} deg")
    print(f"Error from 68.8 deg: {abs(pred_phase - 68.8)/68.8:.2%}")