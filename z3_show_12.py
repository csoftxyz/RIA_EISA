import numpy as np
import matplotlib.pyplot as plt

# =============== 1. 生成并锁定44向量晶格 ===============
def generate_lattice_vectors():
    basis = np.eye(3)
    dem = np.array([1, 1, 1]) / np.sqrt(3)
    seed = np.vstack([basis, dem.reshape(1,3), -dem.reshape(1,3)])
    
    T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    
    def apply(v):
        return T @ v
    
    unique_core = set()
    for v in seed:
        unique_core.add(tuple(np.round(v, 8)))
    
    current = seed.tolist()
    
    for level in range(15):
        new = []
        for v in current:
            v = np.array(v)
            v1 = apply(v)
            v2 = apply(v1)
            new += [v1, v2, v1-v, v2-v]
            
            cross = np.cross(v, v1)
            if np.linalg.norm(cross) > 1e-6:
                new.append(cross)
                new.append(cross / np.linalg.norm(cross))
        
        for nv in new:
            nv = np.array(nv)
            if np.linalg.norm(nv) > 1e-6:
                unique_core.add(tuple(np.round(nv, 8)))
        
        all_vecs = [np.array(u) for u in unique_core]
        all_vecs.sort(key=lambda x: (np.round(np.linalg.norm(x), 4), np.sum(np.abs(x))))
        
        if len(all_vecs) >= 44:
            ground_state = all_vecs[:44]
            break
        current = all_vecs[:100]
    
    print(f"生成向量总数 (锁定): {len(ground_state)}")
    return ground_state

vectors = generate_lattice_vectors()

# =============== 2. 按非零分量分类 (阈值 0.05) ===============
threshold = 0.05
count_1 = 0  # 1 non-zero (basis-like)
count_2 = 0  # 2 non-zero
count_3 = 0  # 3 non-zero (full mixing)

for v in vectors:
    nz = np.sum(np.abs(v) > threshold)
    if nz == 1:
        count_1 += 1
    elif nz == 2:
        count_2 += 1
    elif nz == 3:
        count_3 += 1

total = count_1 + count_2 + count_3
ratio_3_total = count_3 / total if total > 0 else 0
ratio_3_over_2 = count_3 / count_2 if count_2 > 0 else 0

print("--- Component Partition ---")
print(f"1-Component (basis-like): {count_1}")
print(f"2-Component: {count_2}")
print(f"3-Component (full mixing): {count_3}")
print(f"Total: {total}")
print(f"\nRatio 3-Component / Total: {count_3}/{total} ≈ {ratio_3_total:.4f}")
print(f"Ratio 3-Component / 2-Component: {count_3}/{count_2} ≈ {ratio_3_over_2:.4f}")

# =============== 3. 可视化：饼图 + 条形图 ===============
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# 饼图：分量分布
ax1 = fig.add_subplot(121)
labels = ['1-Component\n(basis-like)', '2-Component', '3-Component\n(full mixing)']
sizes = [count_1, count_2, count_3]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.05, 0.05, 0.1)  # 突出3-component

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12})
ax1.axis('equal')
ax1.set_title('Vector Distribution by Non-Zero Components\n(Threshold = 0.05)', fontsize=14, pad=20)

# 条形图：比率展示
ax2 = fig.add_subplot(122)
ratios = [ratio_3_total, ratio_3_over_2]
ratio_names = ['3-Comp / Total', '3-Comp / 2-Comp']
bars = ax2.bar(ratio_names, ratios, color=['#99ff99', '#ffcc99'], edgecolor='black', linewidth=1.2)

# 标注数值
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# 参考线：论文提到的~2.67 (gluon/weak dof)
ax2.axhline(2.67, color='red', linestyle='--', linewidth=2, label='~2.67 (gluon/weak dof ref)')
ax2.text(1.02, 2.67, '2.67 (8/3)', color='red', fontsize=12, va='center')

ax2.set_ylim(0, max(3.0, max(ratios) + 0.5))
ax2.set_ylabel('Ratio Value')
ax2.set_title('Numerical Ratios from Component Counts', fontsize=14)
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()