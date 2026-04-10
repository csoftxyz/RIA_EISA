"""
Z3_44_Lattice_Multi_Orbital.py
完全基于 Z3 44 向量晶格 + 拓扑相位壁垒 的多轨道电子云生成器
每种轨道独立生成一张图片
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import os

# ====================== 参数设置 ======================
CLOCK_CYCLES = 8_000_000      # 每轨道模拟步数（可根据需要调整）
BETA = 2.5                    # 拓扑逆温度（控制量子涨落）
LATTICE_STEP = 0.25           # 孤子最大跳跃步长 (a0 为单位)
OUTPUT_DIR = "Z3_Emergent_Orbitals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[*] 启动 Z3 44 晶格多轨道模拟器...")
print(f"[*] 将为每种轨道生成独立图片")

# ====================== Z3 拓扑张力 + 相位壁垒 ======================
@njit(fastmath=True)
def z3_topological_tension(x, y, z, orbital_type):
    """
    44 晶格拓扑张力函数
    radial: 线性张力 r (产生指数衰减)
    angular: 不同轨道的相位壁垒 (模拟角动量节点面)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-3:
        r = 1e-3
    
    # 径向线性张力（所有轨道共用）
    energy = r
    
    # ====================== 轨道特有相位壁垒 ======================
    if orbital_type == "1s":
        pass                                # 纯球对称，无角向惩罚
    
    elif orbital_type == "2s":
        energy += 0.8 * np.abs(r - 2.0)    # 径向节点（模拟 2s 的球壳节点）
    
    elif orbital_type == "2pz":
        # 2p_z：z=0 平面为强相位壁垒 → 哑铃状
        energy += 8.0 * np.exp(-8.0 * z**2)
    
    elif orbital_type == "2px":
        # 2p_x：x=0 平面为壁垒
        energy += 8.0 * np.exp(-8.0 * x**2)
    
    elif orbital_type == "2py":
        # 2p_y：y=0 平面为壁垒
        energy += 8.0 * np.exp(-8.0 * y**2)
    
    elif orbital_type == "3dz2":
        # 3d_z²：z轴方向强，xy 平面弱
        energy += 6.0 * np.exp(-6.0 * z**2) - 2.0 * (x**2 + y**2)
    
    elif orbital_type == "3dxy":
        # 3d_xy：xy 平面四瓣
        energy += 5.0 * np.sin(4 * np.arctan2(y, x))**2
    
    return energy

# ====================== 核心模拟函数 ======================
@njit(fastmath=True)
def simulate_orbital(steps, beta, step_size, orbital_type):
    history_x = np.zeros(steps, dtype=np.float32)
    history_y = np.zeros(steps, dtype=np.float32)
    history_z = np.zeros(steps, dtype=np.float32)
    
    # 初始位置（略微偏离原点）
    x = y = z = 1.0
    current_energy = z3_topological_tension(x, y, z, orbital_type)
    
    for i in range(steps):
        dx = (np.random.rand() * 2 - 1.0) * step_size
        dy = (np.random.rand() * 2 - 1.0) * step_size
        dz = (np.random.rand() * 2 - 1.0) * step_size
        
        xn = x + dx
        yn = y + dy
        zn = z + dz
        
        new_energy = z3_topological_tension(xn, yn, zn, orbital_type)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            x, y, z = xn, yn, zn
            current_energy = new_energy
        
        history_x[i] = x
        history_y[i] = y
        history_z[i] = z
    
    return history_x, history_y, history_z

# ====================== 轨道列表 ======================
orbitals = [
    ("1s",   "1s - Spherical Ground State"),
    ("2s",   "2s - Radial Node"),
    ("2pz",  "2p_z - Dumbbell (z-axis)"),
    ("2px",  "2p_x - Dumbbell (x-axis)"),
    ("2py",  "2p_y - Dumbbell (y-axis)"),
    ("3dz2", "3d_z² - Double Dumbbell"),
    ("3dxy", "3d_xy - Four-lobed")
]

# ====================== 批量生成 ======================
for idx, (orb_type, title) in enumerate(orbitals):
    print(f"\n[{idx+1}/{len(orbitals)}] 正在模拟 {orb_type} 轨道...")
    start = time.time()
    
    x_traj, y_traj, z_traj = simulate_orbital(CLOCK_CYCLES, BETA, LATTICE_STEP, orb_type)
    
    # 取 z ≈ 0 切片（XY 平面投影）
    mask = np.abs(z_traj) < 0.6
    x_slice = x_traj[mask]
    y_slice = y_traj[mask]
    
    # 绘图
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 8))
    
    h = ax.hist2d(x_slice, y_slice, bins=280, cmap='inferno', 
                  range=[[-6, 6], [-6, 6]], density=True, vmax=0.12)
    
    ax.plot(0, 0, marker='+', color='cyan', markersize=14, label='Proton Core (Z₃ Source)')
    
    ax.set_title(f"Z₃ 44 Lattice Emergent Orbital\n{title}", fontsize=15, color='white')
    ax.set_xlabel("Bohr Radius $a_0$", fontsize=12)
    ax.set_ylabel("Bohr Radius $a_0$", fontsize=12)
    ax.legend(loc='upper right')
    
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label(r'Topological Soliton Density $|\psi|^2$', fontsize=11)
    
    filename = f"{OUTPUT_DIR}/Z3_Emergent_{orb_type}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    print(f"   → {orb_type} 完成！耗时 {(time.time()-start):.2f}s → {filename}")

print(f"\n[✔] 全部轨道模拟完成！共生成 {len(orbitals)} 张高清图片")
print(f"   文件夹：{OUTPUT_DIR}")