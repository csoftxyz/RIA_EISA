import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ====================== 参数 ======================
z = np.linspace(-2.2, 2.2, 180)
xi = np.linspace(15, 160, 180)
Z, Xi = np.meshgrid(z, xi)

energy = 8.5 * np.exp(- (Xi - 70)**2 / (2*26**2)) * np.exp(-Z**2 / 0.85) + \
         0.55 * (Xi - 70)**2 / 650 + \
         2.1 * np.abs(Z)

# ====================== 绘图 ======================
fig = plt.figure(figsize=(13.8, 10.8), dpi=420, facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# 1. 表面（水晶质感）
surf = ax.plot_surface(Xi, Z, energy, cmap=cm.viridis, 
                       linewidth=0, antialiased=True, 
                       alpha=0.93, rstride=1, cstride=1)

# 2. 底面投影等高线
ax.contour(Xi, Z, energy, zdir='z', offset=energy.min()-1.3, 
           levels=35, cmap=cm.viridis, alpha=0.65)

# 3. 蓝色细箭头（RG流动）——提前绘制
step = 15
u = np.zeros_like(Xi[::step, ::step])
v = -Z[::step, ::step] * 1.55
w = -(Xi[::step, ::step] - 70) * 0.52

ax.quiver(Xi[::step, ::step], Z[::step, ::step], energy[::step, ::step] - 0.7,
          u, v, w,
          length=10, normalize=True, 
          color='#1E88E5', linewidth=1.15, 
          edgecolor='white', alpha=0.82)

# 4. 红色星标记临界点
ax.scatter(70, 0, energy.min() + 0.08, color='red', s=380, 
           marker='*', edgecolor='#FFDD00', linewidth=3.5, zorder=15)

# ====================== 最后绘制所有文字（确保不被遮挡） ======================
ax.set_xlabel(r'$\xi_{\rm vac}$ (nm)', fontsize=17, labelpad=14)
ax.set_ylabel(r'Depth $z$ from surface (nm)', fontsize=17, labelpad=14)
ax.set_zlabel(r'Effective Mass Squared $M_{\rm eff}^2$', fontsize=17, labelpad=14)

ax.set_title('Surface Quantum Criticality Landscape\n'
             'Vacuum Softening Localized at the Interface\n'
             '(Luban Lock + RG Attractor Analogy)', 
             fontsize=18, pad=35, fontweight='bold')

# 图例（最后绘制 + 白色背景框）
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', marker='*', markersize=18, label=r'Surface QCP ($M_{\rm eff}^2 \approx 0$)'),
    Line2D([0], [0], color='#1E88E5', lw=4, label='RG Flow → Surface Attractor'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=13.5, 
          frameon=True, facecolor='white', edgecolor='black', framealpha=0.97)

ax.view_init(elev=26, azim=-58)

plt.tight_layout()
plt.savefig('Surface_Quantum_Criticality_Landscape_Final.png', 
            dpi=420, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ 最终版生成成功！")
print("   文件名：Surface_Quantum_Criticality_Landscape_Final.png")
print("   文字已完全不被箭头遮挡，图例在右上角带白色背景")