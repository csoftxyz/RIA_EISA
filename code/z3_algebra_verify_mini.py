import numpy as np
import time
import random

# ==================== 参数设置 ====================
dim = 19
omega = np.exp(2j * np.pi / 3)   # primitive 3rd root of unity
sqrt3 = np.sqrt(3)

def N(g, h):
    """Z3 commutation factor N(g,h) = ω^{g h}"""
    return omega ** ((g * h) % 3)

# 生成元矩阵（adjoint 表示）
generators = [np.zeros((dim, dim), dtype=complex) for _ in range(dim)]
grades = [0]*12 + [1]*4 + [2]*3   # B^{1-12}:0-11, F^{1-4}:12-15, ζ^{1-3}:16-18

# 索引映射（方便阅读）
B = slice(0, 12)      # 0-11
F = slice(12, 16)     # 12-15
Zeta = slice(16, 19)  # 16-18

# ==================== 辅助函数：填充 bracket ====================
def fill(i, j, coeff, target):
    """
    [gen_i, gen_j] = coeff * gen_target
    自动满足 graded skew-symmetry
    """
    gi = grades[i]
    gj = grades[j]
    generators[i][j, target] += coeff
    generators[j][target, i] -= N(gj, gi) * coeff.conjugate() if np.iscomplexobj(coeff) else -N(gj, gi) * coeff

# ==================== 1. su(3) ⊂ g0 (indices 0-7) ====================
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

# ==================== 2. su(2) ⊂ g0 (indices 8,9,10) ====================
fill(8,  9,  1.0, 10)
fill(8, 10, -1.0,  9)
fill(9, 10,  1.0,  8)

# ==================== 3. T^a on F (B^a acting on F^{1-4}, indices 12-15) ====================
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
    fill(a, alpha, val, beta)   # [B^a, F^α] = val F^β

# ==================== 4. S^a on ζ (B^a acting on ζ^{1-3}, indices 16-18) ====================
S_data = [
    (0,16,17,-0.5), (0,17,16,-0.5),
    (1,16,17, 0.5j),(1,17,16,-0.5j),
    (2,16,16,-0.5), (2,17,17, 0.5),
    (3,16,18,-0.5), (3,18,16,-0.5),
    (4,16,18, 0.5j),(4,18,16,-0.5j),
    (5,17,18,-0.5), (5,18,17,-0.5),
    (6,17,18, 0.5j),(6,18,17,-0.5j),
    (7,16,16,-1/(2*sqrt3)), (7,17,17,-1/(2*sqrt3)), (7,18,18, 1/sqrt3),
]
for a,k,l,val in S_data:
    fill(a, k, val, l)   # 直接用真实索引

# ==================== 5. g^{αk}_a : [F^α, ζ^k] = g^{αk}_a B^a ====================
g_data = [
    (12,17,0, 0.5), (13,16,0, 0.5),   # α=1,2  k=2,1  a=1
    (12,17,1,-0.5j),(13,16,1, 0.5j),
    (12,16,2, 0.5), (13,17,2,-0.5),
    (12,18,3, 0.5), (14,16,3, 0.5),
    (12,18,4,-0.5j),(14,16,4, 0.5j),
    (13,18,5, 0.5), (14,17,5, 0.5),
    (13,18,6,-0.5j),(14,17,6, 0.5j),
    (12,16,7,1/(2*sqrt3)), (13,17,7,1/(2*sqrt3)), (14,18,7,-1/sqrt3),
    (12,16,11,1/sqrt3), (13,17,11,1/sqrt3), (14,18,11,1/sqrt3),
    (15,16,8,1.0), (15,17,9,1.0), (15,18,10,1.0),   # F^4 with su(2)
]
for alpha,k,a,val in g_data:
    fill(alpha, k, val, a)   # 注意顺序 [F,ζ] = g B

# ==================== 6. h^{kl}_α : [ζ^k, ζ^l] = h^{kl}_α F^α ====================
h_data = [
    (16,17,12, 1.0),      (17,16,12, -omega),
    (16,18,13, 1.0),      (18,16,13, -omega),
    (17,18,14, 1.0),      (18,17,14, -omega),
]
for k,l,alpha,val in h_data:
    fill(k, l, val, alpha)

# ==================== 验证函数 ====================
def bracket(i, j):
    """graded Lie bracket in matrix representation"""
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

# ==================== 大规模随机测试 ====================
random.seed(42)
np.random.seed(42)

n_tests = 100000
max_res = 0.0

start = time.time()
for _ in range(n_tests):
    i = random.randint(0, 18)
    j = random.randint(0, 18)
    k = random.randint(0, 18)
    res = jacobi_residual(i, j, k)
    if res > max_res:
        max_res = res
        if res > 1e-12:
            print(f"Warning: large residual {res:.2e} at ({i},{j},{k})")

end = time.time()
print(f"After {n_tests:,} random tests, maximum residual = {max_res:.2e}")
print(f"Time taken: {end - start} seconds")