import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Experimental and lattice-derived phases
exp_deg = 68.8          # CKM CP phase
triality_deg = 120.0    # Triality phase
magic_deg = np.degrees(np.arccos(1/np.sqrt(3)))  # ≈54.74°
pred_deg = triality_deg - magic_deg              # ≈65.26°
dev_pct = abs(pred_deg - exp_deg) / exp_deg * 100

print(f"Magic angle: {magic_deg:.2f}°")
print(f"Predicted phase: {pred_deg:.2f}°")
print(f"Deviation from experimental {exp_deg}°: {dev_pct:.2f}%")

# =============== 可视化：角度对比圆盘图 ===============
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

# 设置角度范围（0-180°）
ax.set_theta_offset(np.pi/2)      # 上方为0°
ax.set_theta_direction(-1)        # 顺时针
ax.set_thetalim(0, np.pi)         # 修复：使用 set_thetalim 而非 set_tlim
ax.set_rticks([])                 # 不显示半径刻度
ax.grid(False)

# 背景圆环
theta = np.linspace(0, np.pi, 360)
ax.plot(theta, np.ones_like(theta), color='lightgray', linewidth=8, alpha=0.5)

# Triality 120° 弧线（蓝色）
ax.plot(np.radians([0, triality_deg]), [1, 1], color='blue', linewidth=6, label='Triality 120°')
arc_tri = Arc((0,0), 2, 2, angle=0, theta1=0, theta2=triality_deg, color='blue', linewidth=6)
ax.add_patch(arc_tri)

# Magic angle 54.74° 弧线（绿色，从0°开始）
ax.plot(np.radians([0, magic_deg]), [0.8, 0.8], color='green', linewidth=6, label=f'Magic angle {magic_deg:.2f}°')
arc_magic = Arc((0,0), 1.6, 1.6, angle=0, theta1=0, theta2=magic_deg, color='green', linewidth=6)
ax.add_patch(arc_magic)

# 预测相位差 65.26°（橙色，从magic开始）
start_pred = magic_deg
end_pred = triality_deg
ax.plot(np.radians([start_pred, end_pred]), [0.9, 0.9], color='orange', linewidth=6, label=f'/general Predicted δ ≈ {pred_deg:.2f}°')
arc_pred = Arc((0,0), 1.8, 1.8, angle=0, theta1=start_pred, theta2=end_pred, color='orange', linewidth=6)
ax.add_patch(arc_pred)

# 实验值 68.8°（红色虚线）
ax.plot(np.radians([0, exp_deg]), [0.7, 0.7], color='red', linestyle='--', linewidth=5, label=f'Experimental δ ≈ {exp_deg}°')
ax.text(np.radians(exp_deg + 3), 0.75, f'{exp_deg}°', color='red', fontsize=14, fontweight='bold')

# 标注
ax.text(np.radians(triality_deg/2), 1.2, 'Triality Phase', color='blue', fontsize=14, ha='center')
ax.text(np.radians(magic_deg/2), 1.0, 'Magic Angle', color='green', fontsize=14, ha='center')
ax.text(np.radians((start_pred + end_pred)/2), 1.15, 'Projective Difference', color='orange', fontsize=14, ha='center')

ax.set_title('Z₃ Lattice: Phase Difference vs CKM CP-Violating Phase\n'
             f'120° - {magic_deg:.2f}° = {pred_deg:.2f}° (dev {dev_pct:.2f}% from 68.8°)', 
             fontsize=16, pad=40)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12)

plt.tight_layout()
plt.show()