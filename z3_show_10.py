import numpy as np
import matplotlib.pyplot as plt

# Experimental ratio (PDG 2024 averages)
m_h = 125.25
m_t = 172.76
ratio_exp = m_h / m_t
print(f"Experimental m_H / m_t ≈ {ratio_exp:.6f}\n")

# Candidate geometric ratios
candidates = {
    "1/√2": 1/np.sqrt(2),
    "√(1/2)": np.sqrt(0.5),
    "1/√3": 1/np.sqrt(3),
    "2/3": 2/3,
    "3/4": 0.75,
    "√2/2": np.sqrt(2)/2,
    "√3/2": np.sqrt(3)/2,
    "11/15": 11/15,
    "8/11": 8/11,
    "√3/√7": np.sqrt(3)/np.sqrt(7),
    "√2/√3": np.sqrt(2)/np.sqrt(3)
}

# Compute absolute and relative deviations
names = list(candidates.keys())
values = [candidates[name] for name in names]
abs_dev = np.abs(np.array(values) - ratio_exp)
rel_dev = abs_dev / ratio_exp * 100

# Find best
best_idx = np.argmin(rel_dev)
best_name = names[best_idx]  # 修复了这里的语法错误
best_val = values[best_idx]
best_rel = rel_dev[best_idx]

# Sort by deviation (ascending)
sort_idx = np.argsort(rel_dev)
names = [names[i] for i in sort_idx]
values = [values[i] for i in sort_idx]
rel_dev = [rel_dev[i] for i in sort_idx]

# =============== 可视化 ===============
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('white')

# Horizontal bar chart
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, values, color='lightgray', edgecolor='black', height=0.6)

# Highlight best (top bar after sorting)
bars[0].set_color('gold')
bars[0].set_edgecolor('darkorange')

# Experimental line
ax.axvline(ratio_exp, color='red', linestyle='--', linewidth=2, label=f'Experimental ≈ {ratio_exp:.6f}')

# Annotations
for i, (name, val, dev) in enumerate(zip(names, values, rel_dev)):
    ax.text(val + 0.001, y_pos[i], f'{val:.6f} ({dev:.2f}%)', va='center', fontsize=10, fontweight='bold' if i==0 else 'normal')

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('Ratio Value')
ax.set_title('Z₃ Lattice: Geometric Ratio Candidates vs Higgs/Top Mass Ratio', fontsize=14, pad=20)
ax.invert_yaxis()  # Best at top
ax.grid(True, axis='x', alpha=0.3)

# Legend and notes
ax.legend(loc='lower right')
ax.text(0.02, 0.02, f"Closest: {best_name} = {best_val:.6f} (dev {best_rel:.2f}%)", transform=ax.transAxes,
        fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange'))

plt.tight_layout()
plt.show()