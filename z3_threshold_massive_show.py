import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

# === 你的真实模拟数据 ===
p_vals = np.array([0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600])

pl_l8  = np.array([0.0395, 0.1610, 0.3100, 0.4795, 0.6670, 0.8155])
pl_l12 = np.array([0.0195, 0.1530, 0.4335, 0.7180, 0.8850, 0.9685])
pl_l16 = np.array([0.0190, 0.2060, 0.5780, 0.8640, 0.9760, 0.9985])

# 假设每点TRIALS=2000（你可以改成真实值）
TRIALS = 2000

# 95% Wilson score confidence interval (更准于标准binomial)
def wilson_ci(success, n, z=1.96):
    p = success / n
    denom = n + z**2
    center = (success + z**2/2) / denom
    margin = z * np.sqrt((success*(n-success)/n + z**2/4) / denom)
    return margin

err_l8  = wilson_ci(pl_l8 * TRIALS, TRIALS)
err_l12 = wilson_ci(pl_l12 * TRIALS, TRIALS)
err_l16 = wilson_ci(pl_l16 * TRIALS, TRIALS)

# === Nature Communications 视觉标准 ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'errorbar.capsize': 5,
})

fig, ax = plt.subplots(figsize=(6.5, 5))  # NC常见比例

# Colorblind-friendly palette (Nature推荐)
colors = ['#0072B2', '#D55E00', '#009E73']  # Blue, Orange, Green

ax.errorbar(p_vals, pl_l8, yerr=err_l8, label='$L=8$', fmt='-o', color=colors[0],
            capthick=2, elinewidth=2)
ax.errorbar(p_vals, pl_l12, yerr=err_l12, label='$L=12$', fmt='-s', color=colors[1],
            capthick=2, elinewidth=2)
ax.errorbar(p_vals, pl_l16, yerr=err_l16, label='$L=16$', fmt='^', color=colors[2],
            capthick=2, elinewidth=2)

# No-correction reference line
ax.plot(p_vals, p_vals, '--', color='gray', linewidth=2, label='$P_\\mathrm{L} = p$')

# Shaded threshold region (你的~1.8%)
ax.axvspan(0.015, 0.021, alpha=0.15, color='lightgray', label='Threshold region $\\sim 1.8\\%$')

ax.set_xlabel('Physical error rate $p$')
ax.set_ylabel('Logical error rate $P_\\mathrm{L}$')
ax.set_xlim(0.005, 0.065)
ax.set_ylim(0.0, 1.05)
ax.grid(True, linestyle='--', alpha=0.5, which='both')
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')

plt.tight_layout()

# 保存高分辨率（NC要求）
plt.savefig('figure1_z3_threshold_nc.png', dpi=600, bbox_inches='tight')
plt.savefig('figure1_z3_threshold_nc.pdf', bbox_inches='tight')  # Vector版给LaTeX
plt.show()

print("Figure saved: figure1_z3_threshold_nc.png 和 .pdf（NC标准）")