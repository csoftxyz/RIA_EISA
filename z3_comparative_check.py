import numpy as np

def triality_resonance_check(vec):
    """
    计算矢量的三元性偏差 Delta。
    Check if the specific vectors (e.g., [1,2,7]) are local minima of this metric.
    """
    v = np.array(vec)
    l2 = np.sum(v*v)
    if l2 == 0: return float('inf')
    
    # Triality rotation (cyclic permutation)
    v_tau = np.array([v[2], v[0], v[1]]) 
    
    # Cross product magnitude squared
    cross_prod = np.cross(v, v_tau)
    cross_sq = np.sum(cross_prod * cross_prod)
    
    # Delta metric (dimensionless)
    delta = cross_sq / (l2**2)
    return delta

# Test Candidate Vectors
fermions = {
    "Top": [0,0,1],
    "Bottom": [1,2,7],
    "Charm": [0,9,9],
    "Muon": [0,27,27],
    "Electron": [3,138,579]
}

print("Checking Selection Rules (Triality Stability):")
for name, vec in fermions.items():
    d = triality_resonance_check(vec)
    print(f"{name:10} {str(vec):15} L^2={np.sum(np.array(vec)**2):<10} Stability_Delta = {d:.6f}")

print("\n--- Comparative Check ---")
print("Are these vectors 'special'? Let's check a random neighbor of Bottom [1,2,6]")
print(f"Neighbor   [1,2,6]         L^2={1+4+36:<10} Stability_Delta = {triality_resonance_check([1,2,6]):.6f}")