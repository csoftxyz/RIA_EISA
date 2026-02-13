import numpy as np

# 目标：寻找 sin^2 = 1/45, 1/44.6, 1/44
target_denom = 44.6
tolerance = 1.0  # 允许分母在 43.6 - 45.6 之间

print(f"Searching for vectors yielding 1/N approx 1/{target_denom}...")

# 扩大范围，利用 CPU 算力
limit = 300 # Component up to 300, L^2 up to 90000
for x in range(1, limit):
    for y in range(0, x+1):
        for z in range(0, y+1):
            l2 = x*x + y*y + z*z
            
            # Basis Projection [1,0,0] -> sin^2 = 1 - x^2/L^2
            if l2 > x*x:
                s2 = 1.0 - (x*x)/l2
                denom = 1.0 / s2
                
                # Check for "Magic Integers" like 44, 45, 46
                if abs(denom - 45.0) < 0.0001:
                    print(f"[FOUND 1/45] Vector [{x},{y},{z}], L^2={l2} -> sin^2 = 1/45 = {s2:.5f}")
                elif abs(denom - 44.0) < 0.0001:
                    print(f"[FOUND 1/44] Vector [{x},{y},{z}], L^2={l2} -> sin^2 = 1/44 = {s2:.5f}")
                elif abs(denom - target_denom) < 0.5:
                    print(f"[NEAR MATCH] Vector [{x},{y},{z}], L^2={l2} -> 1/sin^2 = {denom:.2f}")