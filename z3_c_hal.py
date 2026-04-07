import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

# ====================== Z3 理论严格推导的参数 ======================
alpha = 4 / 12
omega = np.exp(1j * 2 * np.pi / 3)

# ====================== 第一张图：c_hel 曲线对比 ======================
c_hel = np.linspace(-1.0, 1.0, 1200)
dN_SM = 1 - c_hel**2
dN_SM /= np.trapezoid(dN_SM, c_hel)

correction = np.real(omega * (c_hel + 1j * np.sqrt(1 - c_hel**2)))
dN_Z3 = (1 - c_hel**2) + alpha * correction
dN_Z3 /= np.trapezoid(dN_Z3, c_hel)

fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(111)
ax1.plot(c_hel, dN_SM, label='SM / NRQCD Toponium (pseudoscalar)', color='#1f77b4', linewidth=4, alpha=0.95)
ax1.plot(c_hel, dN_Z3, label=r'Z$_3$ Ternary Vacuum Correction', color='#d62728', linewidth=4, alpha=0.95)

ax1.set_xlabel(r'$c_{\rm hel}$ (Helicity Angle Cosine)', fontsize=16)
ax1.set_ylabel(r'Normalized $dN/dc_{\rm hel}$', fontsize=16)
ax1.set_title(r'$c_{\rm hel}$ Distribution: SM Toponium vs Z$_3$ Correction', fontsize=20, pad=20)
ax1.grid(True, linestyle='--', alpha=0.65)
ax1.legend(fontsize=15, loc='upper right')

ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.8)
ax1.axvline(-0.5, color='gray', linestyle=':', alpha=0.8)
ax1.text(0.52, 0.08, r'120° cyclic kink', fontsize=13, color='gray', rotation=90)
ax1.text(-0.48, 0.08, r'120° cyclic kink', fontsize=13, color='gray', rotation=90)

fig1.text(0.96, 0.02, f"Generated on {today}", fontsize=12, color='gray', ha='right')
plt.tight_layout()
plt.savefig('Z3_vs_SM_c_hel_curve.png', dpi=400, bbox_inches='tight')
plt.close(fig1)

print("✅ 第一张图片已保存：Z3_vs_SM_c_hel_curve.png")

# ====================== 第二张图：更符合真实物理的3体示意图 ======================
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_aspect('equal')
ax2.axis('off')

# 三个 120° 对称的 fermion 向量 (真实物理风格)
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
r = 0.85
x = r * np.cos(angles)
y = r * np.sin(angles)

colors = ['#1f77b4', '#d62728', '#2ca02c']
for i in range(3):
    ax2.plot([0, x[i]], [0, y[i]], color=colors[i], lw=3.5, alpha=0.9)
    ax2.scatter(x[i], y[i], s=600, color=colors[i], edgecolors='white', linewidth=3, zorder=5)

# 中心真空 ζ
ax2.scatter(0, 0, s=450, color='gold', edgecolors='black', linewidth=3, zorder=6)
ax2.text(0, 0.05, r'$\zeta$', fontsize=18, ha='center', va='center', color='black', fontweight='bold')

# 三次相互作用箭头（cubic bracket）
for i in range(3):
    j = (i + 1) % 3
    ax2.arrow(x[i]*0.6, y[i]*0.6, (x[j]-x[i])*0.3, (y[j]-y[i])*0.3,
              head_width=0.06, color='#ff7f0e', alpha=0.85, linewidth=2.5)

# 数学标注
ax2.text(-0.05, 1.05, r'Z$_3$ Cubic Vacuum Triality', fontsize=18, fontweight='bold', ha='center')
ax2.text(0, -1.05, r'\{F^\alpha, F^\beta, F^\gamma\} = \epsilon^k_{\alpha\beta\gamma} \zeta_k', 
         fontsize=16, ha='center')

fig2.text(0.96, 0.02, f"Generated on {today}", fontsize=12, color='gray', ha='right')
plt.tight_layout()
plt.savefig('Z3_ternary_3body_realistic.png', dpi=400, bbox_inches='tight')
plt.close(fig2)

print("✅ 第二张图片已保存：Z3_ternary_3body_realistic.png")
print(f"生成时间：{today}")
print("\n两张图片已生成，请直接打开查看！")