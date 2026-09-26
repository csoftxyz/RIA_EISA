import numpy as np

# Deterministic Z3 triality closure, tracking the generation level of each vector.
T = np.array([[0,0,1],[1,0,0],[0,1,0]], float)

def key(v, d=6):
    return tuple(np.round(v, d))

basis = np.eye(3)
dem = np.array([1,1,1])/np.sqrt(3)
seeds = [basis[0], basis[1], basis[2], dem, -dem]

seen = {}
for s in seeds:
    seen[key(s)] = 0

frontier = [s.copy() for s in seeds]
for level in range(1, 16):
    new = []
    for v in frontier:
        v1 = T @ v
        v2 = T @ v1
        cands = [v1, v2, v1 - v, v2 - v]
        cr = np.cross(v, v1)
        if np.linalg.norm(cr) > 1e-9:
            cands.append(cr)
            cands.append(cr/np.linalg.norm(cr))
        for c in cands:
            if np.linalg.norm(c) < 1e-9:
                continue
            k = key(c)
            if k not in seen:
                seen[k] = level
                new.append(c)
    frontier = new
    print(f"level {level:2d}: +{len(new):3d} new  (total {len(seen)})")

allv = [(np.array(k), lvl) for k, lvl in seen.items()]
allv.sort(key=lambda x: (round(float(np.linalg.norm(x[0])), 4), float(np.sum(np.abs(x[0])))))
top = allv[:44]
print("\nTop 44 by norm-composition:")
from collections import Counter
print("level distribution:", dict(sorted(Counter(l for _, l in top).items())))
shells = {}
for v, l in top:
    L2 = round(float(np.dot(v, v)), 4)
    shells.setdefault(L2, []).append((v, l))
for L2 in sorted(shells):
    vs = shells[L2]
    print(f"L2={L2:<8} n={len(vs):<2} levels={sorted(set(l for _,l in vs))}")
