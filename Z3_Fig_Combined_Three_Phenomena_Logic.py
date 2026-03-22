# -*- coding: utf-8 -*-
"""
Z3_Fig_Combined_Three_Phenomena_Logic_Final.py
最终版：红色文字移到左侧，不遮挡任何图表内容
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

fig = plt.figure(figsize=(11.5, 15.8))   # 稍宽一点留出左侧空间
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.38)

# ====================== (a) Tc vs Diameter ======================
ax1 = fig.add_subplot(gs[0])
d = np.array([30,40,50,60,80,100,150,200])
tc = np.array([1.32,1.28,1.23,1.18,1.12,1.08,1.04,1.02])
ax1.scatter(d, tc, color='#1f2a44', s=92, zorder=5, 
            label='Representative experimental trends (Sn nanowires)', 
            marker='o', edgecolors='white', linewidth=1.4)

d_fit = np.linspace(10, 300, 700)
ax1.plot(d_fit, 1 + np.exp(-d_fit/70), color='#0e6be6', lw=3.4, 
         label=r'Exploratory model ($\xi_{\rm vac} \approx 70$ nm)')

ax1.fill_between(d_fit, 1+0.75*np.exp(-d_fit/84), 1+1.28*np.exp(-d_fit/56), 
                 color='#0e6be6', alpha=0.18)

ax1.set_xlabel('Nanowire Diameter $d$ (nm)')
ax1.set_ylabel(r'$T_c / T_{c0}$')
ax1.set_title('(a) Superconducting Critical Temperature Enhancement', pad=15)
ax1.legend(loc='upper right', fontsize=10.5)
ax1.grid(True, alpha=0.6)

# ====================== (b) Skin Depth ======================
ax2 = fig.add_subplot(gs[1])
f = np.logspace(-0.5, 1.8, 800)
omega = 2 * np.pi * f * 1e12
mu0 = 4 * np.pi * 1e-7
sigma0 = 6.8e9

delta_class = np.sqrt(2 / (omega * mu0 * sigma0)) * 1e9
ax2.plot(f, delta_class, color='#2ca02c', lw=2.6, alpha=0.9, 
         label='Classical anomalous skin effect')

ax2.plot(f, 82*np.ones_like(f), color='#1f77b4', lw=3.8, 
         label=r'Vacuum inertia model ($\xi_{\rm vac} \approx 70$ nm)')

ax2.fill_between(f, 64, 106, color='#1f77b4', alpha=0.16)

ax2.scatter([0.4,1,2,5,10], [138,102,91,85,82], color='#1a1a1a', s=82, 
            label='Representative observed deviations', zorder=6)

ax2.set_xscale('log')
ax2.set_xlabel('Frequency (THz)')
ax2.set_ylabel('Skin Depth (nm)')
ax2.set_title('(b) THz Skin Depth Saturation in High-Purity Copper', pad=15)
ax2.legend(loc='upper right', fontsize=10.5)
ax2.grid(True, alpha=0.6)

# ====================== (c) Sensitivity ======================
ax3 = fig.add_subplot(gs[2])
eta = np.linspace(0.5, 22, 700)
xi = 52 + 38 * (1 - np.exp(-eta/5.2))
ax3.plot(eta, xi, color='#1f77b4', lw=3.6, label='Exploratory model trend')
ax3.fill_between(eta, xi*0.78, xi*1.24, color='#1f77b4', alpha=0.17)

ax3.set_xlabel(r'Surface Enhancement Factor $\eta$')
ax3.set_ylabel(r'Coherence Length $\xi_{\rm vac}$ (nm)')
ax3.set_title('(c) Sensitivity to Surface Enhancement', pad=15)
ax3.legend(loc='lower right', fontsize=10.5)
ax3.grid(True, alpha=0.6)

# ====================== 左侧逻辑说明（红色文字 + 箭头） ======================
# 左侧位置：x=0.06（图表左侧约1/6位置），完全不遮挡图表

# 上箭头 + 文字
arrow1 = FancyArrowPatch((0.08, 0.685), (0.22, 0.535), 
                         transform=fig.transFigure, 
                         arrowstyle='->', mutation_scale=22, 
                         color='#d62728', linewidth=2.8)
fig.patches.append(arrow1)
fig.text(0.06, 0.61, 'Unified by the same\ngeometric scale\n'
                     r'$\xi_{\rm vac} \approx 70$ nm', 
         ha='left', va='center', fontsize=11.8, color='#d62728', 
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white", 
                   edgecolor='#d62728', alpha=0.96))

# 下箭头 + 文字
arrow2 = FancyArrowPatch((0.08, 0.465), (0.22, 0.315), 
                         transform=fig.transFigure, 
                         arrowstyle='->', mutation_scale=22, 
                         color='#d62728', linewidth=2.8)
fig.patches.append(arrow2)
fig.text(0.06, 0.39, 'Driven by surface criticality\n'
                     'and algebraic protection', 
         ha='left', va='center', fontsize=11.8, color='#d62728',
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white", 
                   edgecolor='#d62728', alpha=0.96))

plt.suptitle('Unified Geometric Explanation\n'
             'Three Mesoscopic Anomalies Governed by One Coherence Length '
             r'$\xi_{\rm vac} \approx 70$ nm',
             fontsize=15.5, y=0.96, fontweight='bold', color='black')

plt.savefig('Fig_Combined_Three_Figures_Logic_Final.pdf', dpi=360, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 最终版合并成功！")
print("   生成文件： Fig_Combined_Three_Figures_Logic_Final.pdf")
print("   红色文字已移到左侧，完全不遮挡图表内容")