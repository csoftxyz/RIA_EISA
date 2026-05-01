import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ====================== 参数设置 ======================
nk = 60                    # k点网格密度
t1 = 1.0                   # 最近邻跃迁
t2 = 0.3                   # Z3 拓扑强度

# ====================== Kagome 哈密顿量 (Z3 手性注入版) ======================
def kagome_hamiltonian(kx, ky):
    H = np.zeros((3, 3), dtype=complex)
    
    # Z3 相位
    z3_phase = np.exp(1j * 2 * np.pi / 3)
    
    # 1. 标准最近邻跃迁
    f1 = 1 + np.exp(-1j * kx)
    H[0,1] = t1 * f1
    H[1,0] = np.conj(H[0,1])
    
    f2 = 1 + np.exp(-1j * (-0.5*kx + np.sqrt(3)/2*ky))
    H[1,2] = t1 * f2
    H[2,1] = np.conj(H[1,2])
    
    f3 = 1 + np.exp(-1j * (-0.5*kx - np.sqrt(3)/2*ky))
    H[2,0] = t1 * f3
    H[0,2] = np.conj(H[2,0])
    
    # 2. Z3 关键修复：只在 0->1 通道注入相位，形成净 2π/3 磁通
    H[0,1] += t2 * z3_phase
    H[1,0] += np.conj(t2 * z3_phase)
    
    return H

# ====================== Fukui-Hatsugai-Suzuki 算法 ======================
print("正在启动 Fukui-Hatsugai-Suzuki 拓扑张量网络... (nk=60, 约需10秒)")
kx_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
ky_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)

u = np.zeros((nk, nk, 3), dtype=complex)
for i in range(nk):
    for j in range(nk):
        H0 = kagome_hamiltonian(kx_vals[i], ky_vals[j])
        evals, evecs = eigh(H0)
        u[i, j, :] = evecs[:, 0]   # 取最低能带

U_x = np.zeros((nk, nk), dtype=complex)
U_y = np.zeros((nk, nk), dtype=complex)
for i in range(nk):
    for j in range(nk):
        u_current = u[i, j]
        u_next_x = u[(i+1)%nk, j]
        u_next_y = u[i, (j+1)%nk]
        
        link_x = np.vdot(u_current, u_next_x)
        link_y = np.vdot(u_current, u_next_y)
        
        U_x[i, j] = link_x / np.abs(link_x)
        U_y[i, j] = link_y / np.abs(link_y)

F12 = np.zeros((nk, nk), dtype=float)
for i in range(nk):
    for j in range(nk):
        loop = U_x[i, j] * U_y[(i+1)%nk, j] * np.conj(U_x[i, (j+1)%nk]) * np.conj(U_y[i, j])
        F12[i, j] = np.angle(loop)

chern_number = np.sum(F12) / (2 * np.pi)

print(f"\n✅ 量子拓扑解算完成！")
print(f" Chern 数 = {chern_number:.4f}")
if abs(chern_number) > 0.5:
    print(" → [神迹降临] 时间反演对称性被打破，Z3 拓扑绝缘体涌现！")
else:
    print(" → TRS 仍未显著打破（可微调 t2）")

# ====================== 绘图 ======================
Area = (2 * np.pi / nk)**2
berry_curvature_density = F12 / Area
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(berry_curvature_density.T, extent=[-np.pi, np.pi, -np.pi, np.pi],
               origin='lower', cmap='RdBu_r', aspect='auto')
ax.set_title('Z₃ Kagome Model\nBerry Curvature Density\n'
             f'Chern Number = {chern_number:.4f}', fontsize=14, pad=20)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
plt.colorbar(im, label='Berry Curvature')
plt.tight_layout()
plt.savefig('z3_qah_berry_curvature_ultimate.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 最终版图片已保存为 z3_qah_berry_curvature_ultimate.png")