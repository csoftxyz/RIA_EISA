import numpy as np
import random
from fractions import Fraction

# ==================== 参数设置 ====================
dim = 19
omega = np.exp(2j * np.pi / 3)   # primitive 3rd root of unity
sqrt3 = np.sqrt(3)

def N(g, h):
    return omega ** ((g * h) % 3)

# 生成元矩阵
generators = [np.zeros((dim, dim), dtype=complex) for _ in range(dim)]
grades = [0]*12 + [1]*4 + [2]*3   # B:0-11, F:12-15, ζ:16-18

# ==================== 辅助函数 ====================
def fill(i, j, coeff, target):
    gi = grades[i]
    gj = grades[j]
    generators[i][j, target] += coeff
    generators[j][target, i] -= N(gj, gi) * coeff.conjugate() if np.iscomplexobj(coeff) else -N(gj, gi) * coeff

# ==================== 1. su(3) ⊂ g0 ====================
su3_data = [
    (0,1,2,1), (0,2,1,-1), (1,2,0,1),
    (0,3,6,0.5), (0,6,3,-0.5),
    (1,3,5,0.5), (1,5,3,-0.5),
    (2,3,4,0.5), (2,4,3,-0.5),
    (3,4,2,0.5),
    (3,5,6,0.5), (3,6,5,-0.5),
    (4,5,3,0.5), (4,6,3,-0.5), (5,6,4,0.5),
    (0,4,5,-0.5), (0,5,4,0.5),
    (1,4,6,-0.5), (1,6,4,0.5),
    (2,5,6,-0.5), (2,6,5,0.5),
    (3,7,4, sqrt3/2), (4,7,3, -sqrt3/2),
    (5,7,6, sqrt3/2), (6,7,5, -sqrt3/2),
]
for a,b,c,val in su3_data:
    fill(a,b,val,c)

# ==================== 2. su(2) ⊂ g0 ====================
fill(8,  9,  1.0, 10)
fill(8, 10, -1.0,  9)
fill(9, 10,  1.0,  8)

# ==================== 3. T^a on F ====================
T_data = [
    (0,12,13, 0.5), (0,13,12, 0.5),
    (1,12,13,-0.5j),(1,13,12, 0.5j),
    (2,12,12, 0.5), (2,13,13,-0.5),
    (3,12,14, 0.5), (3,14,12, 0.5),
    (4,12,14,-0.5j),(4,14,12, 0.5j),
    (5,13,14, 0.5), (5,14,13, 0.5),
    (6,13,14,-0.5j),(6,14,13, 0.5j),
    (7,12,12,1/(2*sqrt3)), (7,13,13,1/(2*sqrt3)), (7,14,14,-1/sqrt3),
    (8,14,15, 0.5), (8,15,14, 0.5),
    (9,14,15,-0.5j),(9,15,14, 0.5j),
    (10,14,14, 0.5),(10,15,15,-0.5),
    (11,12,12,1/6), (11,13,13,1/6), (11,14,14,1/6), (11,15,15,0.5),
]
for a,alpha,beta,val in T_data:
    fill(a, alpha, val, beta)

# ==================== 4. S^a on ζ ====================
S_data = [
    (0,16,17,-0.5), (0,17,16,-0.5),
    (1,16,17, 0.5j),(1,17,16, -0.5j),
    (2,16,16,-0.5), (2,17,17, 0.5),
    (3,16,18,-0.5), (3,18,16,-0.5),
    (4,16,18, 0.5j),(4,18,16, -0.5j),
    (5,17,18,-0.5), (5,18,17,-0.5),
    (6,17,18, 0.5j),(6,18,17, -0.5j),
    (7,16,16,-1/(2*sqrt3)), (7,17,17,-1/(2*sqrt3)), (7,18,18, 1/sqrt3),
]
for a,k,l,val in S_data:
    fill(a, k, val, l)

# ==================== 验证 ====================
def bracket(i, j):
    gi, gj = grades[i], grades[j]
    return generators[i] @ generators[j] - N(gi, gj) * generators[j] @ generators[i]

def jacobi_residual(i, j, k):
    gi, gj, gk = grades[i], grades[j], grades[k]
    gYZ = (gj + gk) % 3
    gXY = (gi + gj) % 3
    gXZ = (gi + gk) % 3
    lhs = generators[i] @ bracket(j, k) - N(gi, gYZ) * bracket(j, k) @ generators[i]
    rhs = bracket(i, j) @ generators[k] - N(gXY, gk) * generators[k] @ bracket(i, j) \
        + N(gi, gj) * (generators[j] @ bracket(i, k) - N(gj, gXZ) * bracket(i, k) @ generators[j])
    return np.linalg.norm(lhs - rhs, 'fro')

random.seed(42)
np.random.seed(42)
n_tests = 10_000
max_res = 0.0
for _ in range(n_tests):
    i = random.randint(0, 18)
    j = random.randint(0, 18)
    k = random.randint(0, 18)
    res = jacobi_residual(i, j, k)
    if res > max_res:
        max_res = res

print(f"After {n_tests:,} random tests, maximum Jacobi residual = {max_res:.2e}")

# ==============================================================================
# Z3 EFT Coefficient Extractor — THE RIGID ALGEBRAIC CONSTANT
# ==============================================================================
print("\n=== Extracting the Rigid Z3 EFT Coefficient ===")

# 1. SU(3) 规范扇区强度 (indices 0-7)
trace_B_sq = 0.0
for a in range(8):
    mat_B = generators[a]
    trace_B_sq += np.real(np.trace(mat_B @ np.conjugate(mat_B).T))

# 2. 真空扇区强度 (indices 16-18)
trace_Z_sq = 0.0
for k in range(16, 19):
    mat_Z = generators[k]
    trace_Z_sq += np.real(np.trace(mat_Z @ np.conjugate(mat_Z).T))

C_Z3 = trace_Z_sq / trace_B_sq

print(f"Sum Trace(B^2) = {trace_B_sq:.6f}")
print(f"Sum Trace(ζ^2) = {trace_Z_sq:.6f}")
print(f"THE RIGID ALGEBRAIC CONSTANT C_Z3 = {C_Z3:.6f}")

frac = Fraction(C_Z3).limit_denominator(100)
print(f"Rational form: {frac.numerator}/{frac.denominator}")
print("This number is now locked by the algebra. Use it.")