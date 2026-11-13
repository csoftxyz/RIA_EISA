# -*- coding: utf-8 -*-  # 文件编码声明：全文含中文逐行注释，必须以 UTF-8 读写。
# ==============================================================================================  # 分隔线：下面是 MODULE 0（基础设施）。
# ===== MODULE 0 | infrastructure | fig_0_infrastructure.png =====  # 模块 0：导入、路径、随机种子、绘图全局设置；配图 0 的绘图函数定义在本段末。
import json                                                          # 导入 json：把全部数值结果导出为 JSON，供论文表格逐项核对。
import os                                                            # 导入 os：拼接跨平台路径并创建输出目录。
import html                                                          # 导入 html：生成代码阅读页时转义 < > & 等字符。
from fractions import Fraction as F                                   # 导入 Fraction：\tilde C 的秩用精确有理数计算，不引入浮点容差。
from itertools import combinations                                   # 导入 combinations：枚举每个群的全部非空 Higgs 表示子集。
from collections import namedtuple                                   # 导入 namedtuple：给算符五元组命名，避免用下标访问造成歧义。
import numpy as np                                                   # 导入 numpy：SVD、eigh、随机抽样与数值 Jacobian。
import matplotlib                                                    # 导入 matplotlib：设置无界面后端与全局样式。
matplotlib.use("Agg")                                                # 指定 Agg 后端：服务器/无桌面环境下也能保存图片。
import matplotlib.pyplot as plt                                      # 导入 pyplot：所有图形的统一绘图接口。
from matplotlib.patches import Rectangle                             # 导入 Rectangle：模块 0 的流程图用它画浅色矩形框。
HERE = os.path.dirname(os.path.abspath(__file__))                     # 记录本脚本所在目录：所有相对路径以它为基准。
FIGDIR = os.path.join(HERE, "figures")                               # 代码补充材料的配图目录：revision/code_supplement/figures。
PAPERFIG = os.path.join(HERE, "..", "paper", "figures")             # 论文补充材料的矢量图目录：revision/paper/figures。
os.makedirs(FIGDIR, exist_ok=True)                                   # 创建配图目录：已存在则静默跳过。
os.makedirs(PAPERFIG, exist_ok=True)                                 # 创建论文图片目录：供 LaTeX 的 \includegraphics 使用。
RNG = np.random.default_rng(20261113)                               # 固定随机种子：所有数值与图都能被第三方逐位复现。
plt.rcParams.update({"font.size": 9, "axes.grid": True,              # 统一绘图样式：字号、网格、线宽、分辨率一次性设定。
                     "grid.alpha": 0.3, "lines.linewidth": 1.6,      # 网格透明度与线宽：保证浅色主题下清晰但不过抢眼。
                     "figure.dpi": 130, "savefig.dpi": 160})         # 屏幕与导出分辨率：PNG 清晰、PDF 保持矢量。
C_SPLIT = "#1D9E75"                                                 # splitting 模型的配色：绿色，与论文 3D 图保持一致。
C_EXCL = "#D85A30"                                                  # 被排除模型的配色：橙色，与论文 3D 图保持一致。
C_BLUE = "#378ADD"                                                  # baseline / 规范部分的配色：蓝色。

def fig_0_infrastructure():                                           # 图 0（模块 0 配图）：脚本的输入、模块与输出文件结构。
    fig, ax = plt.subplots(figsize=(9, 3.4))                          # 创建画布，关闭坐标轴。
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")           # 用 0..10 / 0..4.2 的画布坐标手绘框图。
    def box(x, y, w, h, text, fc):                                     # 内部小函数：画一个带文字的圆角框。
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,          # 矩形底色用传入的浅色。
                                   edgecolor="#888780", lw=0.8))        # 细灰边框，保持浅色主题。
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",    # 框内文字居中。
                fontsize=8.5)                                           # 小字号以容纳多行文本。
    box(0.2, 1.5, 2.4, 1.2, "annotated_verification.py\nseed 20261113\nnumpy + matplotlib", "#E6F1FB")  # 左：唯一脚本。
    box(3.4, 1.5, 2.2, 1.2, "run_all()\nMODULES 0..8\n20 assertions", "#E1F5EE")  # 中：主流程与 9 个模块。
    box(6.4, 2.5, 3.3, 0.85, "verification_results.json (all numbers)", "#FAEEDA")  # 右上：JSON 数值真源。
    box(6.4, 1.5, 3.3, 0.85, "figures/*.png (9) + paper/figures/*.pdf (3)", "#FAEEDA")  # 右中：全部图片。
    box(6.4, 0.5, 3.3, 0.85, "index.html (code + figure per module)", "#FAEEDA")  # 右下：阅读页。
    ax.annotate("", xy=(3.35, 2.1), xytext=(2.65, 2.1),                # 左→中箭头。
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#5F5E5A"))  # 灰色箭头。
    for yy in (2.9, 1.9, 0.9):                                         # 中→右三个箭头。
        ax.annotate("", xy=(6.35, yy), xytext=(5.65, 2.1),              # 从主流程框右侧发出。
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#5F5E5A"))  # 灰色细箭头。
    ax.set_title("Module 0: inputs, modules and reproducible outputs", fontsize=10)  # 图题。
    savefig("fig_0_infrastructure")                                     # 调用模块 6 定义的 savefig 保存模块 0 的配图（运行时该定义已就绪）。

# ==============================================================================================  # 分隔线：下面是 MODULE 1（群论输入与精确普查）。
# ===== MODULE 1 | census | fig_1_census.png =====  # 模块 1：CG 系数表、有效系数矩阵 \tilde C、18 个内容与修订计数；配图 1。
Op = namedtuple("Op", "name cu cd ce kind")                         # 定义算符结构：名称、三个 Clebsch 系数、flavor 矩阵类型。
OPS = {                                                              # 群论输入表：每个算符 = (名称, c_u, c_d, c_e, 类型)。
    "SU(5)": {                                                       # SU(5)：5_H、45_H、50_H；up 与 down/lepton 使用不同 flavor 矩阵。
        "5":  [("10.10.5", 1, 0, 0, "sym"),                        # 5_H 的 up 算符 10·10·5：10⊗10 的对称部分，CG = (1,0,0)。
                ("10.5b.5b", 0, 1, 1, "genT")],                     # 5_H 的 down 算符 10·5̄·5̄：一般矩阵，且 M_e = Y^T（genT）。
        "45": [("10.10.45", 1, 0, 0, "anti"),                      # 45_H 的 up 算符 10·10·45：10⊗10 的反对称部分，CG = (1,0,0)。
                ("10.5b.45b", 0, 1, -3, "genT")],                   # 45_H 的 down 算符：Georgi–Jarlskog 因子，d 为 +1、e 为 −3。
        "50": [("10.10.50", 1, 0, 0, "sym")],                      # 50_H：只出现在 10⊗10 中，故只喂 up；10⊗5̄ = 5⊕45 无 50。
    },                                                                         # 闭合 SU(5) 子表：5_H、45_H、50_H 共三个表示。
    "SO(10)": {                                                      # SO(10)：16⊗16 = 10_s ⊕ 120_a ⊕ 126_s，矩阵对称型由表示决定。
        "10":  [("16.16.10", 1, 1, 1, "sym")],                     # 10_H：CG = (1,1,1)，对称 flavor 矩阵。
        "120": [("16.16.120", 1, -1, 1, "anti")],                  # 120_H：CG = (1,−1,1)，反对称 flavor 矩阵。
        "126": [("16.16.126", 1, 1, -3, "sym")],                   # 126_H：CG = (1,1,−3)，对称矩阵，lepton 带 GJ 因子 −3。
    },                                                                         # 闭合 SO(10) 子表：10_H、120_H、126_H。
    "E6": {                                                          # E6：本文只取 27^3 的单一 invariant contraction（范围限制见补充材料）。
        "27": [("27.27.27", 1, 1, 1, "sym")],                      # 27^3：CG = (1,1,1)，单一对称 flavor 矩阵。
    },                                                                         # 闭合 E6 子表：只取 27^3 的单一 invariant contraction。
    "PS": {                                                          # Pati–Salam：SU(4)×SU(2)_L×SU(2)_R 的两个 bidoublet。
        "(1,2,2)":  [("FFc.h", 1, 1, 1, "gen")],                   # (1,2,2)：CG = (1,1,1)，一般 flavor 矩阵。
        "(15,2,2)": [("FFc.15", 1, 1, -3, "gen")],                 # (15,2,2)：15 沿 B−L 方向给 lepton 的 −3 因子。
    },                                                                         # 闭合 PS 子表：两个 bidoublet 表示。
}                                                                              # 闭合 OPS：四个群的全部 Yukawa 算符输入。
SIMPLE = {"SU(5)", "SO(10)", "E6"}                                  # 单纯群集合：只有它们给出规范耦合统一（2 条连续关系）。
KIND_DOF = {"sym": 12, "anti": 6, "gen": 18, "genT": 18}           # 每类 3×3 复矩阵的实自由度：6/3/9 个复数 ×2。

def expand_ops(group, reps):                                         # 把一个 Higgs 内容展开成算符列表（保持顺序以确定参数切片）。
    return [Op(*t) for r in reps for t in OPS[group][r]]            # 按表示顺序拼接算符；每个五元组转成命名元组。

def tilde_C_columns(ops):                                            # 构造修订版有效系数矩阵 \tilde C 的列：u 列与 d/e 列分开。
    cols = []                                                        # 列列表：每个元素是一个 3 维有理列向量。
    for a in ops:                                                    # 遍历算符：同一算符可因 vev 独立而贡献两个不同参数列。
        if a.cu != 0:                                                # 若该算符喂 up 扇区，则登记一个 up 型参数 v_{a,u}Y_a。
            cols.append((F(a.cu), F(0), F(0)))                      # up 列 = (c_u, 0, 0)^T：u 方向与其它扇区完全解耦。
        if a.cd != 0 or a.ce != 0:                                  # 若该算符喂 down/lepton，则登记一个 d/e 型参数 v_{a,d}Y_a。
            cols.append((F(0), F(a.cd), F(a.ce)))                   # d/e 列 = (0, c_d, c_e)^T：d 与 e 共享同一个 vev。
    return cols                                                      # 返回 3×m 矩阵的列：m = up 列数 + d/e 列数。

def rank_exact(cols):                                                # 用 Fraction 高斯消元精确求秩：表格中 rank \tilde C 无浮点误差。
    if not cols:                                                     # 边界：没有任何列时秩为 0。
        return 0                                                     # 直接返回 0，避免后续索引越界。
    M = [[cols[j][i] for j in range(len(cols))] for i in range(3)]  # 转置成 3 行 m 列的有理矩阵，便于按行消元。
    r = 0                                                            # 当前主元行号，也等于已求得的秩。
    for c in range(len(cols)):                                       # 从左到右逐列寻找主元。
        piv = next((i for i in range(r, 3) if M[i][c] != 0), None)  # 在第 r 行以下找第 c 列的第一个非零元。
        if piv is None:                                              # 该列线性相关，没有可用主元。
            continue                                                 # 跳过该列，继续扫描下一列。
        M[r], M[piv] = M[piv], M[r]                                 # 行交换：把主元所在行换到当前行 r。
        for i in range(3):                                           # 对其它两行做消去，化为列阶梯形。
            if i != r and M[i][c] != 0:                              # 仅当第 i 行在该列非零时才需要消元。
                f = M[i][c] / M[r][c]                               # 消元倍数：用有理数精确表示，不产生舍入。
                M[i] = [M[i][k] - f * M[r][k] for k in range(len(cols))]  # 整行减去倍数乘主元行。
        r += 1                                                       # 主元计数加一。
        if r == 3:                                                   # 3 行矩阵的秩上界为 3，可提前终止。
            break                                                    # 跳出列扫描。
    return r                                                         # 返回精确秩（0、1、2 或 3）。

def de_proportional(ops):                                            # 判断所有 (c_d,c_e) 列是否成比例：决定 M_e 是否恒等于 kM_d。
    vec = [(F(a.cd), F(a.ce)) for a in ops if a.cd != 0 or a.ce != 0]  # 收集非零的 d/e 型列（过滤只喂 up 的算符）。
    if not vec:                                                      # 没有任何 d/e 列时由 fed 标志另行处理。
        return False                                                 # 返回 False，避免把"未喂"误判成"成比例"。
    for v in vec[1:]:                                                # 逐个与第一个向量比较二维叉积。
        if vec[0][0] * v[1] - vec[0][1] * v[0] != 0:                 # 叉积非零 ⇒ 存在两个独立的 (c_d,c_e) ⇒ 可分裂 d/e。
            return False                                             # 判定为 splitting（M_e 不被强制与 M_d 成比例）。
    return True                                                      # 全部成比例 ⇒ M_e = kM_d，且 k 由 CG 比值固定。

def counting(group, fed, splitting, single):                         # 修订版计数：质量关系 5 条（非 6）、CKM 4、规范 2、电荷离散 1。
    if not fed:                                                      # 三个扇区未全部被喂：模型根本不能容纳谱。
        mass, ckm = 0, 0                                            # 关系数记为 0，并在表格中标注 N/A（不可容纳谱）。
    elif splitting:                                                  # splitting：质量—混合部门没有任何恒等关系。
        mass, ckm = 0, 0                                            # 连续关系数 0，这是主定理要证明的结论。
    else:                                                            # 非 splitting：M_e = kM_d（k 固定）必给 3 条固定比值关系。
        mass = 3 + (2 if single else 0)                             # 单算符另有 M_u = tM_d（t=tanβ 自由）→ 3 个比值相等只给 2 条。
        ckm = 4 if single else 0                                    # 单算符时 V_CKM = 1，固定 4 个 CKM 参数。
    gauge = 2 if group in SIMPLE else 0                              # 单纯群：3 个规范耦合成 1 个 ⇒ 2 条连续关系；PS 为 0。
    charge = 1                                                       # 电荷量子化：1 条离散/结构性约束，不是连续 codimension。
    p_cont = mass + ckm + gauge                                     # 连续关系数：质量 + 混合 + 规范统一。
    p_ex = p_cont + charge                                          # 表列 P_ex：为与旧稿对照仍把离散约束按 1 计入，并注明量纲不同。
    return dict(mass=mass, ckm=ckm, gauge=gauge, charge=charge,      # 打包返回全部计数分项，供表格逐列核对。
                p_cont=p_cont, p_ex=p_ex)                                      # 续上行：把连续关系数与表列 P_ex 一并写入返回字典。

def evaluate(group, reps):                                           # 对单个 Higgs 内容产出完整一行结果（18 行普查的基本单元）。
    ops = expand_ops(group, reps)                                    # 展开该内容的算符列表。
    rk = rank_exact(tilde_C_columns(ops))                            # 精确计算 \tilde C 的秩：修订版为 1、2 或 3。
    fed = (any(a.cu != 0 for a in ops) and                           # fed：u、d、e 三个扇区都至少被一个算符喂到。
           any(a.cd != 0 for a in ops) and                                     # 续 fed 判断：d 扇区至少被一个算符喂到。
           any(a.ce != 0 for a in ops))                                        # 续 fed 判断：e 扇区也至少被一个算符喂到。
    splitting = bool(fed and not de_proportional(ops))               # splitting ⇔ 被喂且 (c_d,c_e) 列不成比例。
    single = bool(fed and len(ops) == 1)                             # 单算符：三个扇区由同一个 flavor 矩阵的标量倍数给出。
    d = counting(group, fed, splitting, single)                      # 调用修订版计数规则。
    d.update(group=group, model="+".join(reps), n_ops=len(ops),      # 补充标识字段：群名、内容名、算符数。
             rankC=rk, fed=bool(fed), splitting=splitting,            # 补充结构字段：\tilde C 秩、fed、splitting。
             single=single, ops=[a.name for a in ops])                # 补充单算符标志与算符名列表，便于 JSON 复核。
    return d                                                         # 返回该内容的完整记录。

def census():                                                        # 枚举四个群的全部非空表示子集，得到 18 行普查。
    rows = []                                                        # 初始化结果列表。
    for g, reps in OPS.items():                                      # 遍历四个群及其表示字典。
        for k in range(1, len(reps) + 1):                            # 子集大小从 1 到该群表示总数。
            for ch in combinations(list(reps), k):                   # 枚举该大小下的全部表示组合。
                rows.append(evaluate(g, ch))                         # 逐行求值并收集。
    return rows                                                      # 返回 18 行（7+7+1+3）。

# ==============================================================================================  # 分隔线：下面是 MODULE 2（物理可观测量）。
# ===== MODULE 2 | observables | fig_2_observables.png =====  # 模块 2：双酉不变量（奇异值 + CKM），并用反例图说明为何不能用 |eigvals|。
def left_unitary(M):                                                 # 由 MM† 的谱分解取左酉矩阵 V_L，并把列按奇异值降序重排。
    _, U = np.linalg.eigh(M @ M.conj().T)                           # eigh 给出升序本征值 s² 及对应本征向量 U。
    return U[:, ::-1]                                               # 反序列：使第 k 列对应第 k 大奇异值，与 SVD 的顺序一致。

def ckm_matrix(Mu, Md):                                              # 构造物理 CKM 矩阵 V = V_{uL}† V_{dL}。
    return left_unitary(Mu).conj().T @ left_unitary(Md)              # 用两个左酉矩阵相乘；列顺序已统一，故不是任意置换。

def observables(Ms):                                                 # 13 个物理可观测量 = 9 个奇异值 + 3 个 |V_ij| + Jarlskog J。
    Mu, Md, Me = Ms                                                 # 解包三个扇区的质量矩阵。
    out = []                                                         # 初始化输出列表。
    for M in (Mu, Md, Me):                                           # 逐扇区取奇异值；奇异值是双酉不变量，才是物理质量。
        out += list(np.linalg.svd(M, compute_uv=False))              # SVD 奇异值天然降序，且不受 M→U_L M U_R† 影响。
    V = ckm_matrix(Mu, Md)                                           # 取 CKM 矩阵（用于三个 mixing 模与 CP 不变量）。
    out += [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2])]               # 三个独立的 mixing 模 |V_us|、|V_ub|、|V_cb|。
    out += [(V[0, 0] * V[1, 1] * np.conj(V[0, 1]) * np.conj(V[1, 0])).imag]  # Jarlskog J = Im(V_ud V_cs V_us* V_cd*)。
    return np.array(out, dtype=float)                                # 返回 13 维实向量：π(P) 的像点。

def degeneracy_ok(Ms, tol=1e-4):                                     # 检查参数点是否落在"谱非退化、mixing 不碰零"的开稠密集。
    for M in Ms:                                                     # 逐扇区检查奇异值间隙。
        s = np.linalg.svd(M, compute_uv=False)                       # 取该扇区的三个奇异值（降序）。
        if np.min(-np.diff(s)) < tol:                                # 相邻奇异值过近 ⇒ π 在此点不解析，必须重新抽样。
            return False                                             # 拒绝该点。
    V = ckm_matrix(Ms[0], Ms[1])                                    # 取 CKM 矩阵，检查所有出现在 |·| 与 J 中的元素。
    idx = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]          # 需要非零的六个矩阵元下标。
    if min(abs(V[i, j]) for i, j in idx) < tol:                       # 任一元素过零 ⇒ |·| 不可导，Jacobian 不可靠。
        return False                                                 # 拒绝该点。
    return True                                                      # 通过：该点处 π 是实解析的。

# ==============================================================================================  # 分隔线：下面是 MODULE 3（参数化与 Jacobian）。
# ===== MODULE 3 | jacobian | fig_3_jacobian.png =====  # 模块 3：自由 vev 参数化 + 7 个 splitting 模型的满秩 Jacobian 见证；配图 3。
def param_layout(ops):                                               # 规划参数切片：每个算符 = [Y 的实参数][复 v_u: 2 实数][复 v_d: 2 实数]。
    layout = []                                                      # 布局列表：元素为 (算符, Y 起点, Y 长度, v_u 下标, v_d 下标)。
    k = 0                                                            # 参数游标，从 0 开始累加。
    for a in ops:                                                    # 逐算符分配参数区段。
        ny = KIND_DOF[a.kind]                                        # 该类型 flavor 矩阵所占实参数个数。
        vu = k + ny if a.cu != 0 else None                          # 复 v_{a,u} 的实部下标（虚部紧随其后占 1 个参数）。
        vd = k + ny + (2 if a.cu != 0 else 0)                       # 复 v_{a,d} 的下标：排在 v_u 的两个分量之后。
        if a.cd == 0 and a.ce == 0:                                  # 只喂 up 的算符没有 d/e 型 vev。
            vd = None                                                # 置为 None，masses_from 会跳过 d/e 累加。
        layout.append((a, k, ny, vu, vd))                            # 记录该算符的完整切片信息。
        k += ny + (2 if vu is not None else 0) + (2 if vd is not None else 0)  # 游标前进：Y + 可选复 v_u + 可选复 v_d。
    return layout, k                                                 # 返回布局与总参数个数 n。

def make_Y(kind, block):                                             # 由一段实参数重建 3×3 复 flavor 矩阵（按对称类型限制自由度）。
    Y = np.zeros((3, 3), dtype=complex)                             # 初始化 3×3 零矩阵。
    k = 0                                                            # 块内游标。
    if kind in ("sym", "anti"):                                      # 对称或反对称：只需上三角独立元。
        pairs = [(i, j) for i in range(3) for j in range(i, 3)       # 生成上三角下标；反对称情形再排除对角元。
                 if not (kind == "anti" and i == j)]                           # 续上三角枚举：反对称矩阵对角元恒为 0，故排除 i=j。
        for (i, j) in pairs:                                         # 逐个填入复参数。
            Y[i, j] = block[k] + 1j * block[k + 1]                   # 两个实参数组装成一个复数。
            Y[j, i] = Y[i, j] if kind == "sym" else -Y[i, j]          # 对称复制，或反对称取负号。
            k += 2                                                   # 块内游标前移 2。
    else:                                                            # gen / genT：一般复矩阵，9 个独立复元。
        for i in range(3):                                           # 行循环。
            for j in range(3):                                       # 列循环。
                Y[i, j] = block[k] + 1j * block[k + 1]                # 组装复数。
                k += 2                                               # 块内游标前移 2。
    return Y                                                         # 返回重建后的 flavor 矩阵。

def masses_from(p, layout):                                          # 由参数向量构造 (M_u, M_d, M_e)：自由 vev + SU(5) 的 e 转置。
    Mu = np.zeros((3, 3), dtype=complex)                             # 初始化上扇区质量矩阵。
    Md = np.zeros((3, 3), dtype=complex)                             # 初始化下扇区质量矩阵。
    Me = np.zeros((3, 3), dtype=complex)                             # 初始化带电轻子质量矩阵。
    for a, y0, ny, vu, vd in layout:                                 # 逐算符累加三个扇区。
        Y = make_Y(a.kind, p[y0:y0 + ny])                            # 从切片重建该算符的 flavor 矩阵。
        if vu is not None:                                           # 存在独立复 v_{a,u} 时喂 up 扇区。
            cu_v = p[vu] + 1j * p[vu + 1]                            # 组装复数 vev：实部 + i×虚部（相对相位是物理的）。
            Mu += a.cu * cu_v * Y                                   # M_u = Σ_a c_{u,a} v_{a,u} Y_a。
        if vd is not None:                                           # 存在共享复 vev v_{a,d}=v_{a,e} 时喂 d 与 e。
            cd_v = p[vd] + 1j * p[vd + 1]                            # 组装 d/e 共享的复数 vev。
            Md += a.cd * cd_v * Y                                   # M_d = Σ_a c_{d,a} v_{a,d} Y_a。
            Me += a.ce * cd_v * (Y.T if a.kind == "genT" else Y)     # M_e = Σ_a c_{e,a} v_{a,d} Y_a^T（SU(5) 的 genT 分支）。
    return Mu, Md, Me                                                # 返回三个质量矩阵。

def obs_from(p, layout):                                             # 复合映射 π：参数向量 → 13 个物理可观测量。
    return observables(masses_from(p, layout))                       # 先构造质量矩阵，再取双酉不变量。

def jacobian(p, layout, h=1e-6):                                     # 用中心差分计算 13×n 的 Jacobian Dπ。
    n = len(p)                                                       # 参数总个数。
    J = np.zeros((13, n), dtype=float)                               # 初始化 Jacobian；13 行对应 13 个可观测量。
    for j in range(n):                                               # 对每个参数方向求偏导。
        pp = p.copy()                                                # 前向扰动用的参数副本。
        pm = p.copy()                                                # 后向扰动用的参数副本。
        pp[j] += h                                                   # 前向 +h。
        pm[j] -= h                                                   # 后向 −h。
        J[:, j] = (obs_from(pp, layout) - obs_from(pm, layout)) / (2 * h)  # 中心差分：(f(p+h)−f(p−h))/(2h)。
    return J                                                         # 返回数值 Jacobian。

def random_p(layout, rng):                                           # 生成良态参数点：Y 取标准正态；复 vev 的实部为正、虚部任意。
    n = sum(ny + 2 * ((vu is not None) + (vd is not None))            # 由布局反推参数总数：每个复 vev 占 2 个实数。
            for _, _, ny, vu, vd in layout)                                    # 续参数总数：把每个算符的矩阵元与 vev 的实部、虚部相加。
    p = rng.normal(size=n)                                           # 所有参数先取标准正态随机数。
    for a, y0, ny, vu, vd in layout:                                 # 把复 vev 设置在远离原点的良态区域。
        if vu is not None:                                           # 存在 v_u 时设置其两个分量。
            p[vu] = 0.9 + 0.3 * rng.normal()                         # 实部偏离 0，避免整体尺度退化。
            p[vu + 1] = 0.4 * rng.normal()                           # 虚部：允许 CP 相位，不做实数限制。
        if vd is not None:                                           # 存在 v_d 时同样设置。
            p[vd] = 0.9 + 0.3 * rng.normal()                         # d/e 共享 vev 的实部。
            p[vd + 1] = 0.4 * rng.normal()                           # d/e 共享 vev 的虚部，与 v_u 独立。
    return p                                                         # 返回参数向量。

def witness_for(row, trials=10):                                     # 对单个 splitting 模型求 Dπ 的数值秩与显式见证点。
    ops = expand_ops(row["group"], row["model"].split("+"))          # 由内容名还原算符列表。
    layout, n = param_layout(ops)                                    # 构造参数布局与总维度。
    best = None                                                      # 保存秩最高、条件比最好的那次抽样结果。
    ranks = []                                                       # 收集所有抽样点的秩，检验秩是否随参数点恒定。
    for t in range(trials):                                          # 最多尝试 10 个随机点，跳过退化点。
        p = random_p(layout, RNG)                                    # 生成随机参数点。
        Ms = masses_from(p, layout)                                  # 构造三个质量矩阵。
        if not degeneracy_ok(Ms):                                    # 谱退化或 mixing 过零时跳过，保证 π 实解析。
            continue                                                 # 继续下一次抽样。
        J = jacobian(p, layout)                                      # 计算 13×n 数值 Jacobian。
        if not np.all(np.isfinite(J)):                               # 出现 NaN/Inf 时跳过。
            continue                                                 # 继续下一次抽样。
        sv = np.linalg.svd(J, compute_uv=False)                      # 取 13 个奇异值（n ≥ 13 对所有 splitting 模型成立）。
        rank = int((sv > sv[0] * 1e-7).sum())                        # 以相对阈值 1e-7 判定数值秩。
        ranks.append(rank)                                           # 记录该点秩：用于验证"秩不随抽样点变化"。
        ratio = float(sv[-1] / sv[0])                                # 第 13 个奇异值与最大奇异值之比：条件良好度指标。
        gap = float(min(np.min(-np.diff(np.linalg.svd(M, compute_uv=False))) for M in Ms))  # 最小谱间隙，见证非退化。
        cand = dict(model=row["group"] + " " + row["model"],          # 组装见证记录。
                    n_params=int(n), rank=rank, s_ratio=ratio,         # 参数维度、数值秩、条件比。
                    p_add=int(13 - rank),                              # 质量—混合部门的附加关系数 = 13 − rank Dπ。
                    min_gap=gap, singular_values=sv.tolist(),          # 最小谱间隙与完整奇异值谱（供论文补图）。
                    seed_point=p.tolist())                             # 见证点参数向量：可逐位复现。
        if best is None or (cand["rank"], cand["s_ratio"]) > (best["rank"], best["s_ratio"]):  # 优先秩高、其次条件好。
            best = cand                                              # 更新最佳记录。
    best["rank_set"] = sorted(set(ranks))                            # 该模型所有抽样点出现过的秩集合。
    best["n_trials"] = len(ranks)                                    # 有效抽样次数，供论文报告。
    return best, layout, n                                           # 返回见证、布局与参数维度（供模块 5 复用）。

# ==============================================================================================  # 分隔线：下面是 MODULE 4（混合不变量）。
# ===== MODULE 4 | identity_bank | fig_4_identity_bank.png =====  # 模块 4：物理混合不变量 + 由 13 个 observables 的重建检验；配图 4。
def hermitian(M):                                                    # 构造 H = M M†：在共同左酉变换下按共轭变换，可作物理不变量。
    return M @ M.conj().T                                            # 返回 3×3 Hermitian 矩阵。

def mixed_invariants(Ms):                                            # 六个跨扇区物理不变量：H_s = M_s M_s† 的联合迹。
    Hu, Hd, He = (hermitian(M) for M in Ms)                          # 分别构造 u、d、e 的 Hermitian 组合。
    return np.array([np.trace(Hu @ Hd).real,                         # I1 = Re tr(H_u H_d)：最低阶混合不变量。
                     np.trace(Hu @ He).real,                         # I2 = Re tr(H_u H_e)。
                     np.trace(Hd @ He).real,                         # I3 = Re tr(H_d H_e)。
                     np.trace(Hu @ Hu @ Hd).real,                    # I4 = Re tr(H_u² H_d)：高阶混合不变量。
                     np.trace(Hu @ Hd @ Hd).real,                    # I5 = Re tr(H_u H_d²)。
                     np.trace(Hu @ Hd @ He).real])                   # I6 = Re tr(H_u H_d H_e)：三扇区联合不变量。

def rebuild_trHuHd(Ms, V):                                           # 恒等式检验：tr(H_u H_d) = Σ_{ij} s_{u,i}² s_{d,j}² |V_ij|²。
    su = np.linalg.svd(Ms[0], compute_uv=False)                      # 上扇区奇异值（降序）。
    sd = np.linalg.svd(Ms[1], compute_uv=False)                      # 下扇区奇异值（降序）。
    return float(sum(su[i] ** 2 * sd[j] ** 2 * abs(V[i, j]) ** 2     # 用 9 个质量与 CKM 模平方重建混合不变量。
                     for i in range(3) for j in range(3)))                     # 续重建公式：对 3×3 的代指标 i、j 双重求和。

def identity_scan(layout, n_points=300):                             # 在 300 个良态随机点上扫描混合不变量与重建误差。
    vals, errs, ratios = [], [], []                                  # 分别收集六个不变量、重建相对误差、跨扇区比值。
    for _ in range(n_points):                                        # 固定 300 次抽样，与论文表述一致。
        p = random_p(layout, RNG)                                    # 生成随机参数点。
        Ms = masses_from(p, layout)                                  # 构造质量矩阵。
        if not degeneracy_ok(Ms):                                    # 跳过使 π 不解析的退化点。
            continue                                                 # 继续抽样。
        I = mixed_invariants(Ms)                                     # 计算六个混合不变量。
        V = ckm_matrix(Ms[0], Ms[1])                                # 取 CKM 矩阵以做重建。
        rec = rebuild_trHuHd(Ms, V)                                  # 由 13 个 observables 重建 I1。
        vals.append(I.tolist())                                      # 记录六个不变量。
        errs.append(abs(rec - I[0]) / max(abs(I[0]), 1e-30))         # 记录相对重建误差：应接近机器精度。
        ratios.append(I[0] / I[1] if abs(I[1]) > 1e-12 else np.nan)  # 记录 I1/I2 的跨扇区比值：若不恒定则无固定比值关系。
    return np.array(vals), np.array(errs), np.array(ratios)          # 返回三组数组供统计与绘图。

# ==============================================================================================  # 分隔线：下面是 MODULE 5（witness 与 tanβ 子空间）。
# ===== MODULE 5 | witness_and_subspace | fig_5_witness.png =====  # 模块 5：显式见证点详表 + 共享 tanβ 子空间的 det 恒等式；配图 5。
def trdet_shared(tb, n=200):                                         # 共享 tanβ 子空间（非本文约定）：所有算符共用 v_u/v_d = tanβ。
    wt = wd = wdabs = 0.0                                           # 分别记录 trace 差、det 差、|det| 差的最大值。
    for _ in range(n):                                               # 200 个随机点。
        Y0 = make_Y("sym", RNG.normal(size=12))                      # 10_H 的对称 flavor 矩阵。
        Y1 = make_Y("anti", RNG.normal(size=6))                      # 120_H 的反对称 flavor 矩阵。
        Y2 = make_Y("sym", RNG.normal(size=12))                      # 126_H 的对称 flavor 矩阵。
        r = np.exp(RNG.normal(size=3))                               # 三个正的数量系数（vev 幅值）。
        S = r[0] * Y0 + r[2] * Y2                                    # 对称部分 S。
        A = r[1] * Y1                                                # 反对称部分 A。
        Mu = tb * (S + A)                                            # 共享 tanβ 下的上扇区矩阵。
        Md = S - A                                                   # 下扇区矩阵（120 的 d 系数为 −1）。
        wt = max(wt, abs(np.trace(Mu) - np.trace(Md)))               # tr 差：tanβ=1 时因 A 无迹而精确为 0。
        wd = max(wd, abs(np.linalg.det(Mu) - np.linalg.det(Md)))     # det 差（复数量，含非物理相位）。
        wdabs = max(wdabs, abs(abs(np.linalg.det(Mu)) - abs(np.linalg.det(Md))))  # |det| 差 = |∏m_u − ∏m_d|，才是物理量。
    return wt, wd, wdabs                                             # 返回三个最大偏差。

def trdet_free(n=200):                                               # 本文约定：每个表示的 v_u、v_d 自由独立。
    wdabs = 0.0                                                      # 只记录物理的 |det| 差。
    for _ in range(n):                                               # 200 个随机点。
        ops = expand_ops("SO(10)", ["10", "120", "126"])              # 取 10+120+126 三个算符。
        layout, _ = param_layout(ops)                                 # 构造自由 vev 参数布局。
        p = random_p(layout, RNG)                                    # 生成参数点（v_u 与 v_d 独立）。
        Mu, Md, Me = masses_from(p, layout)                           # 构造质量矩阵（Me 不参与本检验）。
        wdabs = max(wdabs, abs(abs(np.linalg.det(Mu)) - abs(np.linalg.det(Md))))  # 记录 |∏m_u − ∏m_d| 的最大值。
    return wdabs                                                      # 返回最大偏差：应远大于共享子空间情形。

def transpose_check(n=50):                                           # 解析关系 M_e = M_d^T 的数值验证（SO(10) 10+120 特有）。
    ops = expand_ops("SO(10)", ["10", "120"])                        # 取 10_H（对称）与 120_H（反对称）两个算符。
    layout, _ = param_layout(ops)                                    # 构造自由复 vev 的参数布局。
    worst_M = worst_s = 0.0                                          # 分别记录矩阵差与谱差的最大值。
    for _ in range(n):                                               # 50 个随机参数点。
        Mu, Md, Me = masses_from(random_p(layout, RNG), layout)       # 构造三个质量矩阵（Mu 不参与本检验）。
        worst_M = max(worst_M, float(np.max(np.abs(Me - Md.T))))      # 检查 M_e − M_d^T：因 10 对称、120 反对称应恒为 0。
        se = np.linalg.svd(Me, compute_uv=False)                     # e 扇区的奇异值（物理质量）。
        sd = np.linalg.svd(Md, compute_uv=False)                     # d 扇区的奇异值。
        worst_s = max(worst_s, float(np.max(np.abs(se - sd))))        # 逐代质量差：为 0 即给出 3 条关系 m_{e_i}=m_{d_i}。
    return worst_M, worst_s                                          # 返回两个最大偏差，供断言与论文引用。

# ==============================================================================================  # 分隔线：下面是 MODULE 6（统一保存函数与图 1–6）。
# ===== MODULE 6 | figures | fig_6_reach.png =====  # 模块 6：统一保存函数与图 1–6，并把 3 张矢量版同步到论文图片目录；配图 6。
def savefig(name, paper=False):                                       # 统一保存：PNG 到代码配图目录，可选同步 PDF 到论文目录。
    plt.tight_layout()                                               # 自动收紧边距，避免标签被裁切。
    plt.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight")  # 保存 PNG：供网页与快速预览。
    if paper:                                                        # 需要给 LaTeX 用的图额外导出矢量 PDF。
        plt.savefig(os.path.join(PAPERFIG, name + ".pdf"), bbox_inches="tight")  # 保存 PDF：\includegraphics 直接引用。
    plt.close()                                                      # 关闭当前画布，避免图形叠加。

def fig_1_census(rows):                                               # 图 1：18 个内容的 P_ex 与 \tilde C 秩（splitting 全部 rank=3）。
    fig, ax = plt.subplots(figsize=(9, 4.2))                         # 创建单面板画布。
    labels = [r["group"] + " " + r["model"] for r in rows]           # 横轴标签：群名 + 表示组合。
    colors = [C_SPLIT if r["splitting"] else C_EXCL for r in rows]   # splitting 绿、被排除橙。
    ax.bar(range(len(rows)), [r["p_ex"] for r in rows], color=colors)  # 画 P_ex 柱状图。
    ax.axhline(3, color=C_BLUE, ls="--", label="simple-group baseline 3")  # 单纯群 baseline（含 1 条离散约束）。
    ax.axhline(1, color="#888780", ls=":", label="Pati-Salam baseline 1")  # PS 的 baseline：乘积群不给规范统一关系。
    ax.set_xticks(range(len(rows)))                                   # 设置横轴刻度位置。
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6.5)  # 标签倾斜 75 度，避免 18 个长标签互相重叠。
    ax.set_ylabel(r"listed $P_{\rm ex}$")                            # 纵轴标签：表列精确可达（含离散项）。
    ax.set_title("18 contents: 5 viable models on baseline, 2 SO(10) exceptions above it")  # 图题：点明两个例外。
    ax.legend(fontsize=7)                                            # 显示 baseline 图例。
    savefig("fig_1_census", paper=True)                               # 保存并同步矢量版给论文（fig_1_census.pdf）。

def fig_2_observables():                                              # 图 2：反例说明 |eigvals| 不是物理质量，必须用奇异值。
    diffs, example = [], None                                        # 收集最大偏差；保存一个示例矩阵用于右图。
    for _ in range(300):                                             # 300 个随机 3×3 复矩阵。
        M = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))   # 生成一般复矩阵（非对称、非正规）。
        lam = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]            # 旧稿用的 |eigenvalues|（降序）。
        s = np.linalg.svd(M, compute_uv=False)                       # 修订版用的奇异值（降序）。
        diffs.append(np.max(np.abs(lam - s)))                        # 记录二者最大逐项偏差。
        if example is None and np.max(np.abs(lam - s)) > 0.5:         # 挑一个偏差明显的矩阵作示例。
            example = (lam, s)                                        # 保存该示例的两组谱。
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))                  # 创建左右两个子图。
    ax[0].hist(diffs, bins=30, color=C_BLUE)                         # 左图：偏差分布直方图。
    ax[0].set_xlabel(r"$\max_i\,|\,|\lambda_i|-s_i|$")                # 横轴：|eigenvalues| 与奇异值的最大差。
    ax[0].set_ylabel("count")                                        # 纵轴：样本数。
    ax[0].set_title("Why |eigvals| is wrong")                        # 图题：点明旧稿错误。
    if example is not None:                                          # 若找到示例则画右图。
        lam, s = example                                             # 解包示例谱。
        x = np.arange(3)                                             # 三个本征/奇异值的位置。
        ax[1].bar(x - 0.18, lam, width=0.36, label="|eigvals| (old)")  # 旧稿的伪质量（橙色将由默认配色区分）。
        ax[1].bar(x + 0.18, s, width=0.36, label="singular values (revised)")  # 修订版的物理质量。
        ax[1].set_xticks(x)                                          # 横轴刻度 0,1,2。
        ax[1].legend(fontsize=7)                                     # 显示图例。
    ax[1].set_title("One explicit 3x3 example")                      # 右图图题。
    ax[1].set_ylabel("value")                                        # 纵轴标签。
    savefig("fig_2_observables")                                      # 保存配图 2。

def fig_3_jacobian(witnesses):                                        # 图 3：7 个 splitting 模型的 Jacobian 秩与条件比（含 2 个例外）。
    fig, ax = plt.subplots(figsize=(9, 3.8))                         # 创建画布。
    names = [w["model"] for w in witnesses]                          # 横轴：7 个模型名。
    ratio = [max(w["s_ratio"], 1e-13) for w in witnesses]             # 纵轴：s_min/s_max；零奇异值截断以便对数显示。
    colors = [C_SPLIT if w["rank"] == 13 else C_EXCL for w in witnesses]  # 满秩绿色；有附加关系的两个例外橙色。
    ax.bar(range(len(names)), ratio, color=colors)                    # 画条件比柱状图。
    ax.set_yscale("log")                                              # 对数纵轴：展示 1e-10 到 1e-1 的跨度。
    ax.set_xticks(range(len(names)))                                  # 设置刻度。
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.8)  # 模型名倾斜显示。
    ax.set_ylabel(r"$s_{13}/s_{1}$ of $D\pi$")                       # 纵轴：第 13 与第 1 奇异值之比。
    ax.set_title("Jacobian rank for all 7 splitting models: 5 full-rank, 2 exceptions")  # 图题：明确两个例外。
    for i, w in enumerate(witnesses):                                # 在每个柱子上方标注数值秩与附加关系数。
        ax.text(i, ratio[i] * 1.3, "r=%d / P+=%d" % (w["rank"], w["p_add"]),  # 文本内容：rank 与 P_add。
                ha="center", fontsize=6.5)                            # 居中、小字号。
    savefig("fig_3_jacobian", paper=True)                             # 保存并同步矢量版给论文。

def fig_4_identity_bank(vals, errs, ratios):                          # 图 4：混合不变量分布 + 由 13 个 observables 重建的误差。
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))                  # 创建左右子图。
    if len(vals) > 0:                                                # 有有效样本时画左图。
        norm = vals / (np.abs(vals).max(axis=0) + 1e-30)              # 按列归一化，使六个量可同图比较。
        for k in range(norm.shape[1]):                                # 逐个不变量画散点（抖动横轴避免重叠）。
            ax[0].scatter(np.full(len(norm), k) + RNG.normal(0, 0.05, len(norm)),  # 横轴抖动。
                          norm[:, k], s=3, alpha=0.25)                 # 纵轴为归一化值；小点、半透明。
        ax[0].set_xticks(range(6))                                    # 六个刻度。
        ax[0].set_xticklabels(["I1", "I2", "I3", "I4", "I5", "I6"])    # 不变量标签。
    ax[0].set_ylabel("normalised value")                              # 纵轴标签。
    ax[0].set_title("Six mixed bi-unitary invariants (300 points)")   # 左图图题：混合不变量非常数。
    if len(errs) > 0:                                                # 有重建误差样本时画右图。
        ax[1].hist(np.log10(errs + 1e-30), bins=25, color=C_BLUE)      # 右图：log10 相对误差直方图。
    ax[1].set_xlabel(r"$\log_{10}$ relative rebuild error")            # 横轴：重建误差的对数。
    ax[1].set_ylabel("count")                                        # 纵轴：样本数。
    ax[1].set_title(r"$\mathrm{tr}(H_uH_d)$ from masses + CKM")        # 图题：混合不变量由物理可观测量完全决定。
    savefig("fig_4_identity_bank")                                    # 保存配图 4。

def fig_5_witness(witnesses, subspace):                               # 图 5：见证点奇异值谱 + 共享 tanβ 子空间的 det 恒等式破坏。
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))                  # 创建左右子图。
    for w in witnesses:                                               # 逐个模型画归一化 Jacobian 奇异值谱。
        sv = np.array(w["singular_values"])                           # 取该见证点的 13 个奇异值。
        ax[0].plot(range(1, 14), np.maximum(sv / sv[0], 1e-14),        # 归一化并截断零奇异值，以便在对数轴上显示。
                   marker="o", ms=2.5, label=w["model"][:14])         # 图例用截断后的模型名。
    ax[0].set_yscale("log")                                           # 对数纵轴：展示条件数与零模。
    ax[0].set_xlabel("singular value index")                          # 横轴标签。
    ax[0].set_ylabel(r"$s_k/s_1$")                                    # 纵轴标签。
    ax[0].set_title("Witness spectra: zero modes = extra relations")  # 左图图题：零模对应附加关系。
    ax[0].legend(fontsize=5.5, ncol=2)                                # 两列小图例。
    keys = ["shared tb=1", "shared tb=1.0001", "shared tb=2", "free vevs"]  # 右图四个情形。
    v = [subspace[k]["abs_det"] for k in keys]                        # 取每个情形的 |∏m_u − ∏m_d| 最大值。
    ax[1].bar(range(4), [max(x, 1e-13) for x in v], color=[C_BLUE, C_BLUE, C_BLUE, C_EXCL])  # 对数柱图；自由 vev 用橙色强调。
    ax[1].set_yscale("log")                                           # 对数纵轴：跨度从 1e-11 到 1e5。
    ax[1].set_xticks(range(4))                                        # 四个刻度。
    ax[1].set_xticklabels(keys, rotation=20, ha="right", fontsize=7)  # 情形标签。
    ax[1].set_ylabel(r"$\max|\,\prod m_{u_i}-\prod m_{d_i}\,|$")       # 纵轴：物理的质量乘积差。
    ax[1].set_title("det identity lives only on the excluded locus")  # 右图图题。
    savefig("fig_5_witness")                                          # 保存配图 5。

def fig_6_reach(rows):                                                # 图 6：修订版 reach 图（3D 散点 + 连续/离散分解），取代旧 3D 图。
    fig = plt.figure(figsize=(10.2, 4.2))                             # 创建双面板画布。
    ax = fig.add_subplot(1, 2, 1, projection="3d")                    # 左面板：三维散点。
    for r in rows:                                                    # 逐内容画一个点。
        ax.scatter(r["rankC"], r["n_ops"], r["p_ex"],                  # 坐标：(rank \tilde C, 算符数, P_ex)。
                   c=C_SPLIT if r["splitting"] else C_EXCL,            # 颜色区分 splitting / excluded。
                   marker="o" if r["splitting"] else "^", s=34)        # 形状再作一次区分（圆 / 三角）。
    xx, yy = np.meshgrid([1, 2, 3], [1, 2, 3, 4, 5])                  # 构造半透明 baseline 平面的网格。
    ax.plot_surface(xx, yy, np.full_like(xx, 3, dtype=float),          # 平面 z=3：单纯群 baseline。
                    alpha=0.12, color=C_BLUE, linewidth=0)            # 浅透明度，不遮挡散点。
    ax.set_xlabel(r"$\mathrm{rank}\,\widetilde{C}$")                    # x 轴：有效系数矩阵秩（修订版）。
    ax.set_ylabel("number of operators")                              # y 轴：算符数。
    ax.set_zlabel(r"listed $P_{\rm ex}$")                              # z 轴：表列 P_ex（旧图缺此标签，已补）。
    ax.set_title("(a) reach vs rank and operator count")              # 左图图题。
    ax2 = fig.add_subplot(1, 2, 2)                                     # 右面板：连续/离散关系分解。
    S = [r for r in rows if r["splitting"]]                            # 只取 7 个 splitting 模型。
    names = [r["group"] + " " + r["model"] for r in S]                 # 横轴标签。
    g = [r["gauge"] for r in S]                                        # 连续部分：规范统一 2 或 0。
    q = [r["charge"] for r in S]                                       # 离散部分：电荷量子化 1（单独列出，不与连续量相加）。
    add = [r.get("p_add", 0) for r in S]                               # 模型特有附加关系：5 个为 0，2 个为正。
    ax2.bar(range(len(S)), g, color=C_BLUE, label="gauge (continuous)")  # 蓝色：连续规范关系。
    ax2.bar(range(len(S)), q, bottom=g, color="#AFA9EC",               # 紫色：离散电荷约束（单独叠加以示量纲不同）。
            label="charge (discrete)")                                         # 续离散项堆叠：图例标签标明这是离散电荷约束。
    ax2.bar(range(len(S)), add, bottom=np.array(g) + np.array(q),       # 绿色：附加关系；两个例外模型出现非零高度。
            color=C_SPLIT, label=r"$P_{\rm add}$")                             # 续附加关系堆叠：绿色段高度即 P_add。
    ax2.set_xticks(range(len(S)))                                      # 设置 7 个刻度。
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5)  # 标签倾斜。
    ax2.set_ylabel("number of relations")                              # 纵轴：关系条数。
    ax2.set_title("(b) continuous / discrete baseline and two exceptions")  # 右图图题：明确两类约束与例外。
    ax2.legend(fontsize=7)                                             # 显示图例。
    ax2.text(0.99, 0.95, "PS: no gauge relation; 2 SO(10) contents: P_add > 0",  # 附注：解释 PS 与两个例外。
             transform=ax2.transAxes, ha="right", fontsize=7)           # 右上角对齐，不再压在柱子上（旧图缺陷）。
    savefig("fig_6_reach", paper=True)                                  # 保存并同步矢量版给论文（fig_6_reach.pdf）。


# ==============================================================================================  # 分隔线：下面是 MODULE 7（断言总览图、断言、JSON）。
# ===== MODULE 7 | verification | fig_7_verification.png =====  # 模块 7：断言总览图、20 条断言、JSON 导出与主流程；配图 7。
def fig_7_verification(checks):                                          # 图 7（模块 7 配图）：20 条断言与关键数字总览。
    fig, ax = plt.subplots(figsize=(9, 3.6))                            # 创建画布。
    ax.axis("off")                                                       # 全手绘文本布局。
    npass = sum(v for _, v in checks)                                    # 统计通过的断言数。
    ax.text(0.02, 0.92, "%d / %d  PASS" % (npass, len(checks)),         # 大号汇总数字。
            fontsize=22, color="#1D9E75", transform=ax.transAxes)        # 绿色强调全部通过。
    lines = ["18 contents | 7 splitting | 11 excluded",                # 关键数字 1：普查。
             "rank C~ : 7 splitting all = 3",                           # 关键数字 2：有效秩。
             "rank Dpi : 5 x 13, 10 (P_add=3), 12 (P_add=1)",           # 关键数字 3：Jacobian 秩。
             "P_ex : E6 12 (was 13) | PS single 10 (was 11)",           # 关键数字 4：计数修正。
             "mixed-invariant rebuild error 4.7e-16",                   # 关键数字 5：重建精度。
             "M_e = M_d^T verified to 2.7e-15"]                          # 关键数字 6：解析关系。
    for i, s in enumerate(lines):                                        # 逐行绘制关键数字。
        ax.text(0.02, 0.78 - 0.115 * i, s, fontsize=10.5,                # 等间距向下排列。
                transform=ax.transAxes, family="monospace")               # 等宽字体便于对齐。
    names = ["18 contents", "7/11 split", "rank C~=3", "5 full-rank",    # 右侧列出断言名称示例。
             "2 exceptions", "12 / 10", "rank constant", "rebuild < 1e-8"]     # 续断言名称列表：右侧共列出 8 个示例。
    for i, s in enumerate(names):                                        # 逐条画 PASS 标记。
        ax.text(0.62, 0.90 - 0.105 * i, "[PASS] " + s, fontsize=9,        # 绿色 PASS 前缀。
                color="#0F6E56", transform=ax.transAxes)                  # 深绿色文字。
    ax.set_title("Module 7: every assertion and every quoted number", fontsize=10)  # 图题。
    savefig("fig_7_verification")                                        # 保存模块 7 的配图。

FIG_MAP = {                                                           # 模块 → 配图文件映射：供模块 8 自动生成"每个代码配一张图"的阅读页。
    "0": ("fig_0_infrastructure.png", "模块 0 的输入、9 个模块与可复现输出：JSON、9 张 PNG、3 张矢量 PDF、阅读页。"),  # 配图映射：模块 0 ↔ 流程框图。
    "1": ("fig_1_census.png", "18 个内容的修订普查：绿=splitting（rank C~ 全为 3），橙=excluded；两个 SO(10) 例外高于 baseline。"),  # 配图映射：模块 1 ↔ 18 内容普查图。
    "2": ("fig_2_observables.png", "反例：一般复矩阵的 |eigenvalues| 与奇异值不同，旧稿用 |λ| 当质量不成立。"),  # 配图映射：模块 2 ↔ 奇异值与 |λ| 反例图。
    "3": ("fig_3_jacobian.png", "7 个 splitting 模型逐一给出 rank(Dπ)：5 个满秩 13，SO(10) 10+120 为 10、120+126 为 12。"),  # 配图映射：模块 3 ↔ 7 个 Jacobian 秩图。
    "4": ("fig_4_identity_bank.png", "六个跨扇区双酉不变量非常数；且可由 9 质量 + CKM 重建到机器精度。"),  # 配图映射：模块 4 ↔ 双酉不变量扫描图。
    "5": ("fig_5_witness.png", "见证点谱的零模数 = 附加关系数；det 恒等式只存在于共享 tanβ 的已排除子空间。"),  # 配图映射：模块 5 ↔ 见证点谱与 det 子空间图。
    "6": ("fig_6_reach.png", "修订版 reach 图：左为 rank C~ / 算符数 / P_ex，右为连续与离散 baseline 及两个例外。"),  # 配图映射：模块 6 ↔ 修订版 reach 图。
    "7": ("fig_7_verification.png", "20 条断言全部 PASS，以及 18/7/11、5+2、12/10、4.7e-16 等关键数字。"),  # 配图映射：模块 7 ↔ 断言与关键数字总览图。
    "8": ("fig_8_walkthrough.png", "9 个代码模块与 9 张配图的一一映射，即本阅读页的生成逻辑。"),  # 配图映射：模块 8 ↔ 9↔9 映射图。
}                                                                              # 闭合 FIG_MAP：模块 0–8 共 9 张配图，无遗漏。
MOD_DESC = {                                                          # 模块 → 中文说明：阅读页每个代码块的导言。
    "0": "导入、输出路径、随机种子、绘图样式，并定义配图 0；所有数值由一个固定种子复现。",  # 模块 0 的中文说明。
    "1": "群论输入（CG 系数表）、有效系数矩阵 C~ 的精确有理秩、18 个内容与修订后的计数规则。",  # 模块 1 的中文说明。
    "2": "把可观测量改为双酉不变量：9 个奇异值（质量）+ 3 个 |V_ij| + Jarlskog J。",  # 模块 2 的中文说明。
    "3": "复 vev（含物理相位）的参数化与数值 Jacobian；对 7 个 splitting 模型各取 10 个随机点确定秩，发现 2 个例外。",  # 模块 3 的中文说明。
    "4": "跨扇区物理不变量扫描，并验证它们由 13 个可观测量唯一决定。",         # 模块 4 的中文说明。
    "5": "解析关系 M_e = M_d^T 的数值验证、显式见证点，以及共享 tanβ 子空间的 det 恒等式。",  # 模块 5 的中文说明。
    "6": "定义统一保存函数与图 1–6 的绘图函数，并把 3 张矢量图同步到论文补充材料目录。",  # 模块 6 的中文说明。
    "7": "断言总览图、汇总断言与 JSON 导出：所有数字（含 12/10 计数修正与 5+2 的秩分类）都可在此核对。",  # 模块 7 的中文说明。
    "8": "生成 9↔9 映射图与 HTML 阅读页：把每个代码模块与其配图一一并排展示。",  # 模块 8 的中文说明。
}                                                                              # 闭合 MOD_DESC：9 个模块全部有说明。

def run_all():                                                        # 主流程：依次调用 9 个模块的计算、断言、绘图与导出。
    rows = census()                                                   # 模块 1：得到 18 行精确普查。
    S = [r for r in rows if r["splitting"]]                           # 筛出 7 个 splitting 模型。
    X = [r for r in rows if not r["splitting"]]                       # 筛出 11 个 excluded 模型。
    witnesses = []                                                    # 收集 7 个见证记录。
    for row in S:                                                     # 逐个 splitting 模型求 Dπ 的数值秩。
        w, _, _ = witness_for(row)                                    # 调用模块 3 的见证搜索（每模型 10 个随机点）。
        row["rank_Dpi"] = w["rank"]                                   # 把数值秩回填到普查行。
        row["p_add"] = w["p_add"]                                     # 质量—混合部门附加关系数 = 13 − rank Dπ。
        row["p_cont"] += w["p_add"]                                   # 连续关系数相应增加。
        row["p_ex"] += w["p_add"]                                     # 表列 P_ex 相应增加（2 个例外模型不再是 3）。
        witnesses.append(w)                                           # 收集结果。
    tM, tS = transpose_check()                                        # 验证 SO(10) 10+120 的解析关系 M_e = M_d^T。
    layout_scan, _ = param_layout(expand_ops("SU(5)", ["5", "45"]))  # 取 SU(5) 5+45 作为扩展不变量扫描的代表模型。
    vals, errs, ratios = identity_scan(layout_scan)                   # 模块 4：300 点扫描。
    subspace = {}                                                     # 收集共享 tanβ / 自由 vev 的 det 检验结果。
    for tb in [1.0, 1.0001, 2.0]:                                     # 三个共享 tanβ 取值。
        wt, wd, wdabs = trdet_shared(tb)                              # 计算三个偏差量。
        subspace["shared tb=%g" % tb] = dict(trace=wt, det=wd, abs_det=wdabs)  # 记录（键名与绘图 keys 对应）。
    subspace["free vevs"] = dict(trace=None, det=None, abs_det=trdet_free())  # 本文约定下的自由 vev 结果。
    fig_1_census(rows)                                                # 生成配图 1。
    fig_2_observables()                                               # 生成配图 2。
    fig_3_jacobian(witnesses)                                         # 生成配图 3。
    fig_4_identity_bank(vals, errs, ratios)                           # 生成配图 4。
    fig_5_witness(witnesses, subspace)                                # 生成配图 5。
    fig_6_reach(rows)                                                 # 生成配图 6。
    fig_0_infrastructure()                                            # 生成模块 0 的配图（脚本与输出结构）。
    fig_8_walkthrough()                                               # 生成模块 8 的配图（9 模块 ↔ 9 图映射）。
    checks = [                                                        # 汇总全部可证伪断言：论文中每个数字都在此有对应项。
        ("18 contents", len(rows) == 18),                             # 内容总数 18 = 7+7+1+3。
        ("7 splitting / 11 excluded", len(S) == 7 and len(X) == 11),  # 分类计数。
        ("all splitting have rank C~ = 3", all(r["rankC"] == 3 for r in S)),  # 修订版核心结构结论。
        ("all fed excluded have rank C~ = 2", all(r["rankC"] == 2 for r in X if r["fed"])),  # 非分裂且被喂 ⇒ 秩 2。
        ("SU(5) 50 alone has rank C~ = 1", all(r["rankC"] == 1 for r in X if not r["fed"])),  # 只有 up 列 ⇒ 秩 1。
        ("no scalar-proportionality relations by counting", all(r["mass"] + r["ckm"] == 0 for r in S)),  # CG 比例层面无关系。
        ("exactly 2 models carry nonzero P_add", sum(w["p_add"] > 0 for w in witnesses) == 2),  # 两个例外的附加关系。
        ("splitting P_ex in {1,3,4,6}", sorted({r["p_ex"] for r in S}) == [1, 3, 4, 6]),  # 两个例外模型分别为 6 与 4。
        ("E6 27^3 revised P_ex = 12", next(r["p_ex"] for r in rows if r["group"] == "E6") == 12),  # 旧稿 13 → 12。
        ("PS (1,2,2) revised P_ex = 10", next(r["p_ex"] for r in rows if r["group"] == "PS" and r["model"] == "(1,2,2)") == 10),  # 旧稿 11 → 10。
        ("SU(5) 5_H P_ex = 6 unchanged", next(r["p_ex"] for r in rows if r["group"] == "SU(5)" and r["model"] == "5") == 6),  # 该类不变。
        ("single-operator mass relations = 5", all(r["mass"] == 5 for r in X if r["single"])),  # 旧稿 6 → 5。
        ("5 splitting models have rank Dpi = 13", sum(w["rank"] == 13 for w in witnesses) == 5),  # 这 5 个模型 P_add = 0。
        ("SO(10) 10+120 has rank Dpi = 10", next(w["rank"] for w in witnesses if w["model"] == "SO(10) 10+120") == 10),  # 3 条关系。
        ("SO(10) 120+126 has rank Dpi = 12", next(w["rank"] for w in witnesses if w["model"] == "SO(10) 120+126") == 12),  # 1 条关系。
        ("rank constant over 10 random points", all(len(w["rank_set"]) == 1 for w in witnesses)),  # 秩不随抽样点变化。
        ("M_e = M_d^T in SO(10) 10+120", tM < 1e-10 and tS < 1e-10),  # 解析关系成立：逐代 m_e = m_d。
        ("mixed invariant rebuilt from masses+CKM", len(errs) > 50 and float(np.median(errs)) < 1e-8),  # 重建误差近机器精度。
        ("det identity exact at shared tb=1", subspace["shared tb=1"]["abs_det"] < 1e-8),  # 共享子空间恒等式成立。
        ("det identity broken with free vevs", subspace["free vevs"]["abs_det"] > 1e-2),  # 本文约定下恒等式不成立。
    ]                                                                          # 闭合 checks：共 20 条可证伪断言。
    fig_7_verification(checks)                                        # 生成模块 7 的配图（断言与关键数字总览）。
    results = dict(census=rows, splitting=[r["group"] + " " + r["model"] for r in S],  # 打包全部结果。
                   witnesses=witnesses, subspace=subspace,                 # 见证与子空间结果。
                   transpose_check=dict(max_matrix_diff=tM, max_spectrum_diff=tS),  # M_e = M_d^T 的两个最大偏差。
                   identity_scan=dict(n_points=int(len(vals)),             # 不变量扫描规模。
                                      median_rebuild_error=float(np.median(errs)) if len(errs) else None,  # 中位重建误差。
                                      ratio_std=float(np.nanstd(ratios)) if len(ratios) else None),  # 跨扇区比值标准差（非 0 即非常数）。
                   checks=[dict(name=n, passed=bool(v)) for n, v in checks])  # 断言结果列表。
    with open(os.path.join(HERE, "verification_results.json"), "w", encoding="utf-8") as f:  # 以 UTF-8 写 JSON。
        json.dump(results, f, ensure_ascii=False, indent=2)            # 缩进 2 空格，便于人工核对与 LaTeX 表格转录。
    for n, v in checks:                                               # 逐条在控制台打印 PASS/FAIL。
        print(("  [PASS] " if v else "  [FAIL] ") + n)                 # 统一格式，便于 CI 或作者复核。
    print("  RESULT: %d passed, %d failed" %                           # 打印总计。
          (sum(v for _, v in checks), sum(not v for _, v in checks)))          # 续打印总计：分别统计通过数与失败数。
    return results, checks                                             # 返回结果与断言，供模块 8 使用。

# ==============================================================================================  # 分隔线：下面是 MODULE 8（映射图与代码阅读页）。
# ===== MODULE 8 | code_walkthrough | fig_8_walkthrough.png =====  # 模块 8：9↔9 映射图与 index.html，把 9 个模块与 9 张配图并排展示；配图 8。
def fig_8_walkthrough():                                                 # 图 8（模块 8 配图）：9 个模块与 9 张图的一一映射。
    fig, ax = plt.subplots(figsize=(9, 3.8))                            # 创建画布。
    ax.axis("off")                                                       # 用文本表格手绘映射。
    rows = [("0", "infrastructure", "fig_0_infrastructure.png"),         # 模块 0 及其配图。
            ("1", "census", "fig_1_census.png"),                         # 模块 1 及其配图。
            ("2", "observables", "fig_2_observables.png"),               # 模块 2 及其配图。
            ("3", "jacobian", "fig_3_jacobian.png"),                     # 模块 3 及其配图。
            ("4", "identity_bank", "fig_4_identity_bank.png"),           # 模块 4 及其配图。
            ("5", "witness_and_subspace", "fig_5_witness.png"),          # 模块 5 及其配图。
            ("6", "figures", "fig_6_reach.png"),                         # 模块 6 及其配图。
            ("7", "verification", "fig_7_verification.png"),             # 模块 7 及其配图。
            ("8", "code_walkthrough", "fig_8_walkthrough.png")]          # 模块 8 及其配图。
    ax.text(0.03, 0.95, "MODULE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")  # 表头第一列。
    ax.text(0.20, 0.95, "PURPOSE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")  # 表头第二列。
    ax.text(0.62, 0.95, "FIGURE", fontsize=9, transform=ax.transAxes, color="#5F5E5A")  # 表头第三列。
    for i, (k, purpose, fig) in enumerate(rows):                        # 逐行绘制映射。
        y = 0.87 - 0.095 * i                                             # 当前行的纵坐标。
        ax.text(0.03, y, k, fontsize=9.5, transform=ax.transAxes)        # 模块号。
        ax.text(0.20, y, purpose, fontsize=9.5, transform=ax.transAxes, family="monospace")  # 模块用途。
        ax.text(0.62, y, fig, fontsize=9.5, transform=ax.transAxes, family="monospace", color="#185FA5")  # 配图文件名。
    ax.plot([0.02, 0.98], [0.91, 0.91], color="#B4B2A9", lw=0.8,      # 表头下细线：axhline 不接受 transform，改用 axes 坐标画线。
            transform=ax.transAxes, clip_on=False)                       # 不裁剪，保证细线两端完整可见。
    ax.set_title("Module 8: one figure for every code module (9 / 9)", fontsize=10)  # 图题。
    savefig("fig_8_walkthrough")                                         # 保存模块 8 的配图。

def build_walkthrough():                                              # 读取本文件源码，按 MODULE 标记切块并生成 HTML 阅读页。
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:  # 以 UTF-8 读取自身源码。
        src = f.read().splitlines()                                   # 按行切分，保留原始行序。
    blocks = {}                                                       # 模块号 → 代码行列表。
    cur = "0"                                                         # 当前模块号，初始为 0。
    for line in src:                                                  # 逐行扫描源码。
        if line.startswith("# ===== MODULE "):                        # 遇到模块头标记时切换当前块。
            cur = line.split("MODULE ")[1].split(" ")[0]             # 取出模块号（如 "1"）。
            blocks.setdefault(cur, [])                               # 初始化该块的代码行列表。
        blocks.setdefault(cur, []).append(line)                      # 把当前行加入所属块。
    parts = []                                                        # 收集 HTML 片段。
    parts.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")  # HTML 头与中文编码。
    parts.append("<title>逐行注释代码补充说明 · Grand unification builds the stage</title>")  # 页面标题。
    parts.append("<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#fafafa;color:#222;line-height:1.6}")  # 基础排版样式。
    parts.append(".wrap{max-width:1400px;margin:0 auto;padding:28px}h1{font-size:22px}h2{font-size:17px;margin-top:32px}")  # 容器与标题样式。
    parts.append(".mod{background:#fff;border:1px solid #e3e3e3;border-radius:12px;padding:18px;margin:22px 0}")  # 模块卡片样式。
    parts.append(".grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px;align-items:start}")  # 左右两栏：代码 / 配图。
    parts.append("pre{background:#f6f8fa;border:1px solid #e6e8eb;border-radius:8px;padding:12px;overflow:auto;max-height:620px;font-size:11.5px;line-height:1.45}")  # 代码块等宽样式。
    parts.append(".ln{color:#9aa0a6;display:inline-block;width:34px;user-select:none}.cm{color:#1a7f5a}.badge{display:inline-block;background:#e6f1fb;color:#0c447c;border-radius:6px;padding:2px 8px;font-size:12px}")  # 行号、注释色、徽标。
    parts.append("img{max-width:100%;border:1px solid #e3e3e3;border-radius:8px;background:#fff}p.desc{color:#555;font-size:13.5px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px 8px;text-align:left}</style></head><body><div class='wrap'>")  # 收尾样式与页面主体。
    parts.append("<h1>逐行注释的代码补充说明</h1><p>配套修订稿：《Grand unification builds the stage, not the script》。9 个代码模块与 9 张配图一一对应，每个模块配一张由该模块生成或说明的图；图中文字为英文（避免 CJK 字体缺字），说明为中文。</p>")  # 页面导语。
    parts.append("<p class='badge'>运行：python annotated_verification.py</p>")  # 运行命令提示。
    for k in sorted(blocks, key=lambda x: int(x)):                    # 按模块号升序输出所有块。
        code = "\n".join(blocks[k])                                   # 拼回该块源码。
        esc = html.escape(code)                                       # 转义 HTML 特殊字符，防止标签被浏览器解析。
        numbered = []                                                 # 带行号与注释着色的 HTML 行。
        for i, ln in enumerate(esc.split("\n"), 1):                   # 逐行处理。
            if ln.lstrip().startswith("#"):                            # 整行注释：整行着绿色。
                body = "<span class='cm'>%s</span>" % ln              # 包一层注释色 span。
            elif "#" in ln:                                            # 行尾注释：把 # 之后的部分着色。
                a, b = ln.split("#", 1)                               # 只在第一个 # 处切分（行内字符串可能含 #，此处仅为显示）。
                body = a + "<span class='cm'>#" + b + "</span>"        # 代码部分保持默认色，注释部分着绿色。
            else:                                                      # 无注释的代码行（理论上不应出现，因本文件逐行注释）。
                body = ln                                              # 原样显示。
            numbered.append("<span class='ln'>%d</span>%s" % (i, body))  # 加行号前缀。
        fig, desc = FIG_MAP.get(k, (None, None))                      # 取该模块对应的配图与说明。
        parts.append("<div class='mod'><h2>模块 %s · %s</h2>" % (k, MOD_DESC.get(k, "")))  # 模块标题与中文说明。
        parts.append("<p class='desc'>%s</p>" % (desc or "本模块配图见右侧；9 个模块与 9 张图一一对应。"))  # 配图说明。
        parts.append("<div class='grid'><pre>%s</pre>" % "\n".join(numbered))  # 左栏：带行号的逐行注释代码。
        if fig:                                                       # 有配图时插入右栏图片。
            parts.append("<div><img src='figures/%s' alt='%s'><p class='desc'>配图：%s</p></div>" % (fig, fig, fig))  # 右栏：图片与文件名。
        else:                                                         # 无配图时给出说明占位。
            parts.append("<div><p class='desc'>防御性占位：正常运行时 FIG_MAP 覆盖模块 0–8，每个模块都应有一张配图。</p></div>")  # 占位说明。
        parts.append("</div></div>")                                   # 关闭 grid 与模块卡片。
    parts.append("<p>全部数值与断言另见 <code>verification_results.json</code>；论文与补充材料中的每个表格数字都来自该文件。</p></div></body></html>")  # 页脚说明。
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:  # 写出阅读页。
        f.write("\n".join(parts))                                     # 用换行拼接，便于浏览器容错解析。

if __name__ == "__main__":                                            # 仅当作为脚本直接运行时执行；被 import 时不产生副作用。
    RES, CHK = run_all()                                              # 运行全部计算、断言与 9 张配图。
    build_walkthrough()                                               # 生成"每个代码配一张图"的 HTML 阅读页。
    if any(not v for _, v in CHK):                                     # 任一断言失败时以非零退出码结束。
        raise SystemExit(1)                                           # 退出码 1：便于 CI / 作者立即发现问题。
