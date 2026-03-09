import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

# ==================== 美观设置 ====================
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['text.usetex'] = False  # 如果你电脑有LaTeX可改为True

# ==================== 参数 ====================
C_Z3 = 8 / 63          # 刚性代数常数 ≈ 0.126984
Lambda_alg = 5000      # 真空能标，单位 GeV（默认5 TeV，可修改）
M_tt_min = 1000        # GeV
M_tt_max = 5000
points = 200

# ==================== 计算数据 ====================
M_tt = np.linspace(M_tt_min, M_tt_max, points)
x = (M_tt / Lambda_alg)**2

# SM 预测（无偏差）
sm_ratio = np.ones_like(M_tt)

# Z3 预测（上下界）
z3_ratio_upper = 1 + C_Z3 * x
z3_ratio_lower = 1 - C_Z3 * x
z3_ratio_central = 1 + 0.5 * C_Z3 * x   # 取中间值作为中心曲线

# ==================== 生成 PDF ====================
pdf_filename = "Z3_HighEnergy_Tail_Prediction.pdf"

with PdfPages(pdf_filename) as pdf:
    
    # ==================== 图1：主预测图 ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(M_tt/1000, sm_ratio, 'k-', lw=2.5, label='Standard Model (SM)')
    ax.plot(M_tt/1000, z3_ratio_central, 'red', lw=3, 
            label=f'Z₃ Prediction (C = 8/63 ≈ {C_Z3:.4f})')
    ax.fill_between(M_tt/1000, z3_ratio_lower, z3_ratio_upper, 
                    color='red', alpha=0.25, label='± Z₃ interference band')
    
    ax.set_xlabel(r'$M_{t\bar{t}}$ (TeV)', fontsize=14)
    ax.set_ylabel(r'$d\sigma_{\rm obs} / d\sigma_{\rm SM}$', fontsize=14)
    ax.set_title('Z₃ Vacuum Geometry High-Energy Tail Prediction\n'
                 'Rigid Algebraic Coefficient C$_{Z3}$ = 8/63', pad=20)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    ax.set_ylim(0.85, 1.35)
    
    # 标注关键区域
    ax.axvline(2, color='gray', ls='--', alpha=0.6)
    ax.text(2.05, 1.25, 'HL-LHC sensitive region', fontsize=11, color='gray')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ==================== 图2：不同能标敏感性 ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    for Lambda in [3000, 5000, 8000]:
        x2 = (M_tt / Lambda)**2
        ratio = 1 + C_Z3 * x2
        label = f'Λ_alg = {Lambda/1000:.0f} TeV'
        ax.plot(M_tt/1000, ratio, lw=2.2, label=label)
    
    ax.set_xlabel(r'$M_{t\bar{t}}$ (TeV)')
    ax.set_ylabel(r'$d\sigma_{\rm obs} / d\sigma_{\rm SM}$')
    ax.set_title('Z₃ Prediction Sensitivity to Vacuum Scale Λ_alg')
    ax.legend(title='Vacuum energy scale')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ==================== 图3：残差图（最直观） ====================
    fig, ax = plt.subplots(figsize=(10, 5))
    deviation = (z3_ratio_central - 1) * 100   # 百分比偏差
    
    ax.plot(M_tt/1000, deviation, 'red', lw=3)
    ax.fill_between(M_tt/1000, 
                    (z3_ratio_lower - 1)*100, 
                    (z3_ratio_upper - 1)*100, 
                    color='red', alpha=0.25)
    
    ax.set_xlabel(r'$M_{t\bar{t}}$ (TeV)')
    ax.set_ylabel('Deviation from SM (%)')
    ax.set_title(f'Z₃ Predicted Deviation (C = 8/63)\nΛ_alg = {Lambda_alg/1000} TeV')
    ax.axhline(0, color='black', lw=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"✅ PDF 文件已生成：{pdf_filename}")
print("包含3张高清图：")
print("1. 主预测对比图（带误差带）")
print("2. 不同真空能标敏感性")
print("3. 百分比残差图（最直观）")
print("\n你可以直接把这个 PDF 发给任何人，或上传到 GitHub 作为视觉证据。")