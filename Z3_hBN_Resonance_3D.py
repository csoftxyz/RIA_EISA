import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 设置黑色背景，显得更有科技感/深空感
plt.style.use('dark_background')

def generate_hex_potential(x, y):
    """
    模拟 Z3 真空晶格 (A2 Root System) 的标量势场
    使用三个平面波的叠加来生成六角形对称势
    """
    k = 4 * np.pi / np.sqrt(3)
    # 三个波矢量，互成 120 度
    v1 = np.cos(k * (0.5 * x + np.sqrt(3)/2 * y))
    v2 = np.cos(k * (0.5 * x - np.sqrt(3)/2 * y))
    v3 = np.cos(k * (-x))
    return (v1 + v2 + v3)

def generate_hBN_lattice(n_cells=3):
    """
    生成 hBN (六方氮化硼) 的原子坐标 (蜂窝状结构)
    """
    B_atoms = []
    N_atoms = []
    
    # hBN 晶格常数匹配真空势场
    a = 1.0 
    
    # 生成晶格点
    for i in range(-n_cells, n_cells+1):
        for j in range(-n_cells, n_cells+1):
            # 基矢量
            x_base = i * np.sqrt(3) * a + j * np.sqrt(3)/2 * a
            y_base = j * 1.5 * a
            
            # 蜂窝结构的两个子晶格
            # Boron
            B_atoms.append([x_base, y_base - 0.5*a])
            # Nitrogen
            N_atoms.append([x_base, y_base + 0.5*a])
            
    return np.array(B_atoms), np.array(N_atoms)

# ==================== 绘图设置 ====================
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# 1. 绘制底层：Z3 真空势场 (Vacuum Potential)
X = np.linspace(-4, 4, 100)
Y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(X, Y)
Z = generate_hex_potential(X, Y)

# 调整 Z 的高度，使其位于底部
Z_surface = Z * 0.3 - 2.0 

# 绘制表面，使用 "viridis" 配色 (蓝绿黄) 代表能量势
surf = ax.plot_surface(X, Y, Z_surface, cmap='magma', alpha=0.6, 
                       edgecolor='none', rstride=1, cstride=1, antialiased=True)

# 2. 绘制上层：hBN 原子层
B_atoms, N_atoms = generate_hBN_lattice(n_cells=2)
Z_atoms = np.ones(len(B_atoms)) * 1.5  # 原子层悬浮在上方

# 筛选视野内的原子
mask_B = (np.abs(B_atoms[:,0]) < 3.5) & (np.abs(B_atoms[:,1]) < 3.5)
mask_N = (np.abs(N_atoms[:,0]) < 3.5) & (np.abs(N_atoms[:,1]) < 3.5)

B_atoms = B_atoms[mask_B]
N_atoms = N_atoms[mask_N]
Z_B = np.ones(len(B_atoms)) * 1.5
Z_N = np.ones(len(N_atoms)) * 1.5

# 绘制化学键 (简单的最近邻连接)
# 这里简化处理，只画原子
ax.scatter(B_atoms[:,0], B_atoms[:,1], Z_B, c='cyan', s=150, edgecolor='white', label='Boron (B)', depthshade=False)
ax.scatter(N_atoms[:,0], N_atoms[:,1], Z_N, c='blue', s=200, edgecolor='white', label='Nitrogen (N)', depthshade=False)

# 3. 绘制“共振光柱” (Resonance Beams)
# 连接原子和其正下方的真空势井
for i in range(len(B_atoms)):
    x, y = B_atoms[i]
    # 计算该位置的势能高度
    z_pot = generate_hex_potential(x, y) * 0.3 - 2.0
    
    # 只有当原子对准了“深势井”时才画强光柱 (模拟共振)
    # 在这个数学模型里，v1+v2+v3 在原子位置会产生极值
    
    ax.plot([x, x], [y, y], [z_pot, 1.5], color='white', alpha=0.3, linewidth=1, linestyle='--')
    
    # 在底部画一个光圈
    ax.scatter(x, y, z_pot, c='yellow', s=50, alpha=0.5, marker='x')

# ==================== 装饰 ====================
# 移除坐标轴刻度，显得更抽象
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.axis('off')

# 设置视角
ax.view_init(elev=25, azim=-60)

# 添加标题和图例
plt.title("Geometric Resonance: hBN Lattice Locking into Z3 Vacuum Potential", 
          fontsize=20, color='white', pad=-20)
plt.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white', fontsize=12)

# 保存
plt.tight_layout()
plt.savefig('Z3_hBN_Resonance_3D.png', dpi=300, facecolor='black')
print("图片已保存为 Z3_hBN_Resonance_3D.png")
plt.show()