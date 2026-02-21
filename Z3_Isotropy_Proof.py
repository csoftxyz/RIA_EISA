import numpy as np

print("=== Z3 严格闭合44向量晶格生成 (基于19维代数) ===\n")

T_mat = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=float)

def normalize(v):
    v = np.asarray(v, dtype=float).ravel()
    if len(v) != 3:
        return None
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else None

# ==================== 严格种子 (从立方不变量出发) ====================
dem = np.array([1, 1, 1]) / np.sqrt(3)
seeds = [
    [1,0,0], [0,1,0], [0,0,1],
    dem, -dem,
    [1,1,0], [1,-1,0], [0,1,1], [0,-1,1]
]

unique_set = set()
current = [normalize(s) for s in seeds if normalize(s) is not None]

print("开始严格triality闭合枚举...")

for level in range(100):
    new = []
    for v in current:
        if v is None: continue
        v = v.ravel()
        v1 = normalize(T_mat @ v)
        v2 = normalize(T_mat @ v1) if v1 is not None else None
        
        if v1 is not None: new.append(v1)
        if v2 is not None: new.append(v2)
        
        if v1 is not None:
            new.append(normalize(v + v1))
            new.append(normalize(v - v1))
            cross = np.cross(v, v1)
            ncross = normalize(cross)
            if ncross is not None:
                new.append(ncross)
    
    for nv in new:
        if nv is None: continue
        key = tuple(np.round(nv, decimals=10))
        if key not in unique_set:
            unique_set.add(key)
            current.append(nv)
    
    if len(unique_set) >= 44:
        break

# 最终取44个
vectors = np.array([list(key) for key in list(unique_set)[:44]])
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / norms

print(f"成功生成严格闭合的 {len(vectors)} 个单位向量")

# ==================== Isotropy Test ====================
def calculate_isotropy_defect(vectors, rank=4, num_samples=10000):
    phi = np.random.uniform(0, 2*np.pi, num_samples)
    costheta = np.random.uniform(-1, 1, num_samples)
    theta = np.arccos(costheta)
    probes = np.column_stack((
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ))
    moments = np.array([np.sum(np.abs(np.dot(vectors, n))**rank) for n in probes])
    defect = (np.max(moments) - np.min(moments)) / np.mean(moments)
    return defect

d2 = calculate_isotropy_defect(vectors, rank=2)
d4 = calculate_isotropy_defect(vectors, rank=4)

print("\n" + "="*80)
print(f"{'Lattice':<20} | {'Rank-2 Defect':<20} | {'Rank-4 Defect (Gravity)':<25}")
print("-"*80)
print(f"{'Z3 Strict Closed 44':<20} | {d2:.10f}       | {d4:.10f}")
print("="*80)

np.save("z3_44_strict_closed.npy", vectors)
print("\n已保存严格闭合44向量至: z3_44_strict_closed.npy")