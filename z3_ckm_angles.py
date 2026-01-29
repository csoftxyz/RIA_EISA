import numpy as np

print("=== Z3 Lattice: Geometric Derivation of CKM Angles (FIXED) ===\n")

# Vacuum Axis (Democratic)
v3 = np.array([1.0, 1.0, 1.0])
v3_norm = np.linalg.norm(v3)

targets = {
    "V_us (Cabibbo)": 0.2245,
    "V_cb (2-3)": 0.0412,
    "V_ub (1-3)": 0.0038
}

def find_best_integer_vector(target, limit=20):
    best_vec = None
    min_error = float('inf')
    
    # Scan integer grid
    for x in range(-limit, limit+1):
        for y in range(-limit, limit+1):
            for z in range(-limit, limit+1):
                if x==0 and y==0 and z==0: continue
                
                u = np.array([x, y, z])
                u_norm = np.linalg.norm(u)
                
                # Cosine with [1,1,1]
                # cos = (x+y+z) / (sqrt(3) * sqrt(x^2+y^2+z^2))
                cos_theta = abs(np.dot(u, v3)) / (u_norm * v3_norm)
                
                # We assume CKM element = sin(theta) (Misalignment)
                if cos_theta > 1.0: cos_theta = 1.0
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                
                # Filter: We want small misalignments, but not zero
                if sin_theta < 1e-9: continue
                
                error = abs(sin_theta - target)
                
                if error < min_error:
                    min_error = error
                    best_vec = u
                    
    return best_vec, min_error

print(f"{'Parameter':<15} | {'Target':<8} | {'Best Vector':<15} | {'Pred':<8} | {'Error'}")
print("-" * 70)

for name, val in targets.items():
    vec, err_val = find_best_integer_vector(val, limit=25) # Increase limit slightly
    
    if vec is not None:
        u_norm = np.linalg.norm(vec)
        cos = abs(np.dot(vec, v3)) / (u_norm * v3_norm)
        pred = np.sqrt(1 - cos**2)
        
        # Check if vector has Z3 structure (e.g. sum of components)
        s = np.sum(vec)
        # Type classification
        if s == 0: vtype = "Root-like (Sum=0)"
        elif abs(vec[0])==abs(vec[1]) or abs(vec[1])==abs(vec[2]): vtype = "Hybrid"
        else: vtype = "General"
        
        print(f"{name:<15} | {val:<8.4f} | {str(vec):<15} | {pred:<8.4f} | {abs(pred-val)/val:<.2%} ({vtype})")

print("-" * 70)
print("[INTERPRETATION]")
print("1. If Error < 5%, the CKM angle is a rational geometric projection.")
print("2. Look for vectors with sum=0 or simple integers. These are 'Resonant' states.")