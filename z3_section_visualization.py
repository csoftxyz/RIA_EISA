import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

# ====================== 数据准备 ======================
c_hel = np.linspace(-1.0, 1.0, 1200)
alpha = 4 / 12.0
omega = np.exp(1j * 2 * np.pi / 3)

dN_SM = 1 - c_hel**2
dN_SM /= np.trapezoid(dN_SM, c_hel)

correction = np.real(omega * (c_hel + 1j * np.sqrt(1 - c_hel**2)))
dN_Z3 = (1 - c_hel**2) + alpha * correction
dN_Z3 /= np.trapezoid(dN_Z3, c_hel)

# ====================== 大图布局 ======================
fig = plt.figure(figsize=(20, 13))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Panel 1: Z3 三体几何示意图
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal')
ax1.axis('off')
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
r = 0.85
x = r * np.cos(angles)
y = r * np.sin(angles)
colors = ['#1f77b4', '#d62728', '#2ca02c']
for i in range(3):
    ax1.plot([0, x[i]], [0, y[i]], color=colors[i], lw=4.5, alpha=0.95)
    ax1.scatter(x[i], y[i], s=800, color=colors[i], edgecolors='white', linewidth=3, zorder=5)
ax1.scatter(0, 0, s=550, color='gold', edgecolors='black', linewidth=4, zorder=6)
ax1.text(0, 0.08, r'$\zeta$', fontsize=26, ha='center', va='center', fontweight='bold')
for i in range(3):
    j = (i + 1) % 3
    ax1.arrow(x[i]*0.55, y[i]*0.55, (x[j]-x[i])*0.35, (y[j]-y[i])*0.35,
              head_width=0.08, color='#ff7f0e', alpha=0.9, linewidth=3.5)
ax1.text(0, 1.08, r'Z$_3$ Cubic Vacuum Triality', fontsize=18, fontweight='bold', ha='center')
ax1.text(0, -1.15, r'$\{F^\alpha, F^\beta, F^\gamma\} = \epsilon^k_{\alpha\beta\gamma} \zeta_k$', fontsize=16, ha='center')

# Panel 2: 光学暗点速度曲线
ax2 = fig.add_subplot(gs[0, 1])
theta = np.linspace(10, 170, 300)
v_conv = 1 / np.sin(np.deg2rad(theta))
v_z3 = v_conv * (1 + 0.5 * np.real(omega * np.exp(1j * 3 * np.deg2rad(theta - 60))))
ax2.plot(theta, v_conv, label='Conventional interference', color='#1f77b4', lw=3)
ax2.plot(theta, v_z3, label=r'Z$_3$ lattice refresh', color='#d62728', lw=3)
ax2.axvline(60, color='gray', ls=':', lw=2)
ax2.axvline(120, color='gray', ls=':', lw=2)
ax2.set_xlabel(r'Crossing angle $\theta$ (deg)', fontsize=14)
ax2.set_ylabel(r'$v_{\rm shadow}/c$', fontsize=14)
ax2.set_title('Superluminal Shadow Velocity', fontsize=16)
ax2.legend(fontsize=13)
ax2.grid(True, ls='--', alpha=0.6)

# Panel 3: c_hel 分布对比
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(c_hel, dN_SM, label='SM / NRQCD', color='#1f77b4', lw=3.5)
ax3.plot(c_hel, dN_Z3, label=r'Z$_3$ correction', color='#d62728', lw=3.5)
ax3.set_xlabel(r'$c_{\rm hel}$', fontsize=14)
ax3.set_ylabel(r'Normalized $dN/dc_{\rm hel}$', fontsize=14)
ax3.set_title(r'$c_{\rm hel}$ Distribution', fontsize=16)
ax3.legend(fontsize=13)
ax3.grid(True, ls='--', alpha=0.6)
ax3.axvline(0.5, color='gray', ls=':', lw=2)
ax3.axvline(-0.5, color='gray', ls=':', lw=2)

# Panel 4: 4×4 Spin Density Matrix
ax4 = fig.add_subplot(gs[1, 1])
matrix = np.array([
    [0.250, 0, 0, 0.0875-0.151j],
    [0, 0.250, 0.0875+0.151j, 0],
    [0, 0.0875-0.151j, 0.250, 0],
    [0.0875+0.151j, 0, 0, 0.250]
])
im = ax4.imshow(np.abs(matrix), cmap='viridis', interpolation='nearest')
ax4.set_title(r'4$\times$4 Spin Density Matrix (Z$_3$)', fontsize=16)
ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# 全局标题
fig.suptitle('Signatures of $\\mathbb{Z}_3$-Graded Vacuum Triality\n'
             'From Superluminal Optical Shadows to Top-Quark Spin Entanglement\n\n'
             'Unified Discrete Lattice Refresh Mechanism', 
             fontsize=22, y=0.96, fontweight='bold')

fig.text(0.95, 0.02, f'Generated on {today}', fontsize=12, color='gray', ha='right')

plt.tight_layout()
plt.savefig('Z3_Section_Visualization.png', dpi=400, bbox_inches='tight')
plt.show()

print("✅ 可视化图片已保存为：Z3_Section_Visualization.png")
print("   请直接打开查看完整 Section 内容！")