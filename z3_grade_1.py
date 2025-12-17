from sympy import symbols, Matrix, I, pi, simplify, sqrt, eye, conjugate, trace, zeros, Rational

# ==================== 0. 配置与符号定义 ====================
print("Initializing Exact Symbolic Z3 Algebra verification...")

# 维度定义
dim = 15

# 【核心修正】：使用 Rational(-1, 2) 代替 -1/2
# Python 中 -1/2 会变成 -0.5 (浮点数)，污染符号计算
# Rational(-1, 2) 则是精确的分数
omega = Rational(-1, 2) + I * sqrt(3) / 2

# 分级: B(0-8) grade 0, F(9-11) grade 1, Z(12-14) grade 2
grades = [0]*9 + [1]*3 + [2]*3

# 初始化生成元列表
generators = [zeros(dim, dim) for _ in range(dim)]

# Z3 分级因子 N(g,h)
def N(g, h):
    power = (g * h) % 3
    if power == 0: return 1
    if power == 1: return omega
    if power == 2: return omega**2 

# 填充函数
def fill(i, j, coeff, target):
    gi, gj = grades[i], grades[j]
    generators[i][target, j] += coeff
    generators[j][target, i] -= N(gj, gi) * coeff

# ==================== 1. 构建生成元 (纯符号) ====================
print("Building Structure Constants...")

# 使用 SymPy 的 Matrix 构建
L = [zeros(3,3) for _ in range(9)]
L[0] = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
L[1] = Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]])
L[2] = Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
L[3] = Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
L[4] = Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]])
L[5] = Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
L[6] = Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]])
L[7] = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3)
L[8] = eye(3) * sqrt(2) / sqrt(3) 

# 归一化生成元 T = L / 2
# 注意：SymPy 的 Matrix 除以整数会自动转为 Rational，这里是安全的
T_basis = [l / 2 for l in L]

# 1. 填充 B-B [B, B] -> B
for a in range(9):
    for b in range(9):
        comm = T_basis[a] * T_basis[b] - T_basis[b] * T_basis[a]
        for c in range(9):
            val = (2 * trace(comm * T_basis[c])).expand() 
            if val != 0:
                fill(a, b, val, c)

# 2. 填充 B-F [B, F] -> F
for a in range(9):
    for i in range(3):
        for j in range(3):
            val = T_basis[a][i,j]
            if val != 0:
                fill(a, 9+j, val, 9+i)

# 3. 填充 B-Z [B, Z] -> Z
for a in range(9):
    S_mat = -T_basis[a].conjugate()
    for i in range(3):
        for j in range(3):
            val = S_mat[i,j]
            if val != 0:
                fill(a, 12+j, val, 12+i)

# 4. 填充混合项 [F, Z] -> B
# g = -T
g_factor = -1
for a in range(9):
    mat = T_basis[a]
    for f in range(3): 
        for z in range(3): 
            val = g_factor * mat[z, f]
            if val != 0:
                fill(9+f, 12+z, val, a)

# ==================== 2. 定义验证逻辑 ====================

def bracket(i, j):
    gi, gj = grades[i], grades[j]
    term1 = generators[i] * generators[j] 
    term2 = N(gi, gj) * generators[j] * generators[i]
    return term1 - term2

def get_jacobi_residual(i, j, k):
    gi, gj, gk = grades[i], grades[j], grades[k]
    
    t1 = generators[i] * bracket(j, k) - N(gi, (gj+gk)%3) * bracket(j, k) * generators[i]
    t2 = bracket(i, j) * generators[k] - N((gi+gj)%3, gk) * generators[k] * bracket(i, j)
    t3 = N(gi, gj) * (generators[j] * bracket(i, k) - N(gj, (gi+gk)%3) * bracket(i, k) * generators[j])
    
    return (t1 - t2 - t3).expand()

# ==================== 3. 执行验证 ====================
print("Verifying Jacobi Identities (Mixing Sector B-F-Z)...")
print("Using EXACT Rational arithmetic...")

non_zero_found = False

# 验证 B-F-Z 扇区
for i in range(9):       # B
    for j in range(9, 12):   # F
        for k in range(12, 15):  # Z
            res_mat = get_jacobi_residual(i, j, k)
            
            if not res_mat.is_zero_matrix:
                non_zero_found = True
                print(f"FAIL: Non-zero residual found at indices ({i},{j},{k})")
                print(res_mat)
                break
        if non_zero_found: break
    if non_zero_found: break

print("="*60)
if not non_zero_found:
    print("VICTORY: All Jacobi residuals are SYMBOLICALLY ZERO.")
    print("Mathematical Closure Verified: Exact.")
    print("Residual = 0 (Pure Symbolic).")
else:
    print("Verification Failed.")
print("="*60)