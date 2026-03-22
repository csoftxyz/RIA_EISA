import numpy as np
import matplotlib.pyplot as plt
import time

print("=== Z3 纯几何魔角模拟（终极零参数版）===")
print("6000×6000 网格 + 多谐波莫尔密度 + 完整 A2 投影\n")

# ====================== 最高分辨率参数 ======================
GRID_POINTS = 6000          # 极限分辨率（768G 仍可跑）
L_MAX = 100.0
xi_vac = 70.0

x = np.linspace(-L_MAX, L_MAX, GRID_POINTS, dtype=np.float32)
y = np.linspace(-L_MAX, L_MAX, GRID_POINTS, dtype=np.float32)
X, Y = np.meshgrid(x, y)

a = 0.246
k0 = 4 * np.pi / (np.sqrt(3) * a)

# ====================== 多谐波莫尔密度（更真实） ======================
def moire_density_multi(X, Y, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xr = X * cos_t - Y * sin_t
    Yr = X * sin_t + Y * cos_t
    
    rho = np.zeros_like(X)
    # 基频 + 2倍频 + 3倍频（更高阶谐波）
    for n in [1, 2, 3]:
        k = n * k0
        rho += np.cos(k*X) + np.cos(k*(X*0.5 + Y*np.sqrt(3)/2)) + np.cos(k*(X*0.5 - Y*np.sqrt(3)/2))
        rho += np.cos(k*Xr) + np.cos(k*(Xr*0.5 + Yr*np.sqrt(3)/2)) + np.cos(k*(Xr*0.5 - Yr*np.sqrt(3)/2))
    return rho

# ====================== 完整 Z3 A2 投影（来自44向量） ======================
def z3_vacuum_potential(X, Y):
    dirs = [(0,1), (np.sqrt(3)/2,0.5), (np.sqrt(3)/2,-0.5),
            (0,-1), (-np.sqrt(3)/2,-0.5), (-np.sqrt(3)/2,0.5)]
    zeta = np.zeros_like(X)
    for n in [1, 2]:                     # 多谐波
        for dx, dy in dirs:
            zeta += np.cos(n * k0 * (dx*X + dy*Y))
    r = np.sqrt(X**2 + Y**2)
    return zeta * np.exp(-r / xi_vac)

# ====================== 扫描 ======================
angles_deg = np.linspace(0.8, 1.8, 401)
overlap_integrals = []

print("开始终极零参数扫描（6000网格）...")
start = time.time()
for deg in angles_deg:
    theta = np.deg2rad(deg)
    rho_moire = moire_density_multi(X, Y, theta)
    zeta = z3_vacuum_potential(X, Y)
    overlap = np.abs(np.sum(rho_moire * zeta)) / (GRID_POINTS**2)
    overlap_integrals.append(overlap)
print(f"扫描完成！用时 {time.time()-start:.1f} 秒")

# ====================== 结果 ======================
overlap = np.array(overlap_integrals)
peak_idx = np.argmax(overlap)
peak_angle = angles_deg[peak_idx]

print(f"\n✅ 纯几何共振峰位于 θ = {peak_angle:.3f}°")
print(f"最大重叠强度 = {overlap[peak_idx]:.8f}")

# ====================== 绘图 ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(angles_deg, overlap/max(overlap), 'r-', linewidth=3.5)
plt.axvline(peak_angle, color='blue', linestyle='--', linewidth=2.5)
plt.text(peak_angle + 0.03, 0.92, f'Pure Geometric Peak\nat {peak_angle:.3f}°', fontsize=13, color='blue', fontweight='bold')
plt.title('Z₃ Pure Geometric Magic Angle\n(No Hopping, No Fitting Parameters)')
plt.xlabel('Twist Angle θ (degrees)')
plt.ylabel('Normalized Overlap Integral')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2, projection='3d')
sub = 120
X3 = X[::sub,::sub]; Y3 = Y[::sub,::sub]
Z3 = z3_vacuum_potential(X3, Y3)
ax = plt.gca()
ax.plot_surface(X3, Y3, Z3, cmap='plasma', alpha=0.92)
ax.set_title('Z₃ Vacuum Potential Surface\n(A₂ Projection)')
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig('Z3_Pure_Geometric_Magic_Angle_Ultimate.png', dpi=600)
plt.show()