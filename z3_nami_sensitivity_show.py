import numpy as np
import matplotlib.pyplot as plt

# 设置专业出版级风格（与您之前的 z3_nami_show.py 一致）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'lines.linewidth': 3,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (10, 7)
})

# 关键修复：嵌入 TrueType 字体，避免 PDF 中的 glyph 警告（如 ~ 符号）
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==================== 图3: Sensitivity Analysis of ξ_vac to η ====================
fig, ax = plt.subplots()

# Surface enhancement factor η 范围
eta = np.linspace(0.5, 20, 500)

# 理论模型：ξ_vac = ξ_base + A * (1 - exp(-η / η0))
# 参数选择：Tin (η ≈ 2–5) → ξ_vac ≈ 70 nm, Copper (η ≈ 5–10) → ξ_vac ≈ 80–85 nm
# 高 η 时饱和，展示 robustness（整体稳定在 50–100 nm 量级）
xi_base = 50.0
A = 40.0
eta0 = 5.0
xi_theory = xi_base + A * (1 - np.exp(-eta / eta0))

# 绘制主理论曲线
ax.plot(eta, xi_theory, color='blue', label=r'Theoretical $\xi_{\rm vac}(\eta)$')

# O(1) 不确定带（±20–25% 因子，符合您之前图中的 shaded band）
xi_upper = xi_base + A * 1.25 * (1 - np.exp(-eta / eta0))
xi_lower = xi_base + A * 0.80 * (1 - np.exp(-eta / eta0))
ax.fill_between(eta, xi_lower, xi_upper, alpha=0.25, color='lightblue',
                label=r'$\mathcal{O}(1)$ theoretical uncertainty')

# 材料标注区域（使用 Unicode ∼ 符号，避免 glyph 警告）
# Tin (η ≈ 2–5)
ax.axvspan(2, 5, alpha=0.15, color='gray')
ax.text(3.5, 82, 'Tin\n(η ∼ 2–5)\nξ_vac ≈ 70 nm', horizontalalignment='center',
        verticalalignment='center', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Copper (η ≈ 5–10)
ax.axvspan(5, 10, alpha=0.15, color='lightgray')
ax.text(7.5, 92, 'Copper\n(η ∼ 5–10)\nξ_vac ≈ 80–85 nm', horizontalalignment='center',
        verticalalignment='center', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 箭头注释：强调 robust scale（即使 η 变化因子 ~4，ξ_vac 仍在 50–100 nm）
ax.annotate('Robust scale:\nξ_vac ∼ 50–100 nm\ndespite η variation by factor ∼4',
            xy=(15, 80), xytext=(12, 60),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
            fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# 轴标签与标题
ax.set_xlabel('Surface Enhancement Factor η')
ax.set_ylabel('Vacuum Coherence Length ξ_vac (nm)')
ax.set_title('Sensitivity of Vacuum Coherence Length\nto Surface Enhancement Factor')
ax.set_xlim(0.5, 20)
ax.set_ylim(40, 110)

# 图例
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# 布局优化
plt.tight_layout()

# 保存高清 PDF（适合直接插入 LaTeX）
plt.savefig('fig_sensitivity.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Figure saved: fig_sensitivity.pdf")