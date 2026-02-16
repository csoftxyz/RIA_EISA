import numpy as np
import matplotlib.pyplot as plt
import re

# =============== 1. 读取日志文件并提取 1/sin² 值 ===============
file_path = 'Z3_Universe_Solver_output.txt'  # 修改为你的实际路径

# 正则表达式匹配 1/sin^2= 后面的浮点数
pattern = r'1/sin\^2\s*=\s*([0-9\.]+)'

values = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            try:
                val = float(match.group(1))
                values.append(val)
            except ValueError:
                continue

values = np.array(values)
print(f"成功提取 {len(values)} 个 1/sin² 值")
print(f"范围: {values.min():.2f} ~ {values.max():.2f}")
print(f"平均值: {values.mean():.3f}")

# =============== 2. 绘制直方图 ===============
plt.figure(figsize=(12, 8))
plt.hist(values, bins=100, color='skyblue', edgecolor='black', alpha=0.8, density=True)

# 高亮两个尖峰区域
plt.axvspan(43.5, 44.5, color='gold', alpha=0.3, label='Peak ~44 (Lattice Anchor)')
plt.axvspan(44.8, 45.5, color='lime', alpha=0.3, label='Peak ~45 (Vacuum Singlet)')

# 实验值垂直线
exp_val = 44.64
plt.axvline(exp_val, color='red', linestyle='--', linewidth=3, label=f'Experimental ≈{exp_val}')

# 标注
plt.text(exp_val + 0.05, plt.ylim()[1]*0.9, f'Exp: {exp_val}\n(In valley between peaks)', 
         color='red', fontsize=12, fontweight='bold', ha='left')

plt.xlabel('1 / sin²θ₁₃', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of 1/sin²θ₁₃ from Lattice Search\n'
          '(Dual Peaks at ~44 & ~45, Experimental Value in Valley)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 紧凑布局并显示
plt.tight_layout()
plt.show()

# =============== 3. 额外统计（可选打印） ===============
print("\n=== 统计总结 ===")
print(f"44 附近 (<0.5偏差) 计数: {np.sum(np.abs(values - 44) < 0.5)}")
print(f"45 附近 (<0.5偏差) 计数: {np.sum(np.abs(values - 45) < 0.5)}")
print(f"实验值 {exp_val} 附近 (<0.2偏差) 计数: {np.sum(np.abs(values - exp_val) < 0.2)}")