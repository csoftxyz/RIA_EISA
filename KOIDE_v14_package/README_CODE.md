# 论文 <-> 代码 对应说明

论文: `PAPER_KOIDE_NOTE_v14.pdf` / `.tex`  (Koide's relation is a pole-mass statement)

## 1. 权威脚本

| 脚本 | 对应论文章节/公式 | 输入 | 输出 |
|---|---|---|---|
| `z44_final2.py` | §1 (Q=(3+ρ²)/9 …), §2 (460σ), §3 (定理), §4 (方案表/185σ/6800×), §6 (闭式解), §5 (CF 窗口) | 三组质量 (pole/MSbar/PDG-current) | Q, r, φ (pole & MSbar), R_M0, 敏感度, δQ_exp, tanφ* 的 minimal polynomial, 一圈 QED 交叉检验 |
| `z44_final.py` | §6 闭式解 `√m_τ=2(a+b)+√3√(...)` | pole 或 MSbar 质量 | m_τ(pole)=1776.969, m_τ(MSbar)=1724.79 |
| `run_and_verify.py` | (本 LOG 生成器) | 上述脚本 | VERIFY_LOG.md |

## 2. 论文各节数据流

- **§1 恒等式 / 几何**: `Q=(1+r²/2)/3=(3+ρ²)/9`, `cosα=1/√(3Q)`, `α∈[0,54.7356°]` — 纯解析.
- **§2 经验问题 Q1**: `R_M0` 由 `(√2, 2π/3+2/9)` 直接算 (z44_final2 §2) = **206.77032**; 对比 pole 实测 206.7682830(44) ⇒ **≈460σ**.
- **§3 超越性定理**: `tan φ*` 代数 (sympy minimal_polynomial, **deg=8**) + Lindemann–Weierstrass ⇒ φ* 超越.
- **§4 基本问题 Q2**: pole vs MSbar 表 (z44_final2 §1); `ΔQ=+1.26e-3` = **185σ_Q** (=205× 经验偏差); `Δφ=−1.19e-3` rad = **6800×|φ*−2/9|**; 敏感度 **56.14 rad⁻¹**.
- **§5 离散对称**: CF 窗口 `δφ^{(μ/e)}=2.1e-8/56.14=3.7e-10` (pole 限定).
- **§6 条件式预言**: 闭式解 pole 1776.969 MeV (0.85σ); MSbar 1724.79 vs 1746.56 (偏 22 MeV).

## 3. 输入数据与出处

- pole 质量: Xing–Zhang [hep-ph/0602134] Eq.(1) (PDG 2004 时代); 当前 PDG/CODATA 差 0.14σ.
- MSbar(M_Z): Xing–Zhang Eq.(8): 0.486755106 / 102.740394 / 1746.56 MeV.
- 一圈 QED 交叉检验: `mbar/m_pole` = 0.9526 / 0.9724 / 0.9830 (符合到 0.3%).

## 4. 脚本状态表 (哪些支撑 v14, 哪些历史/撤回)

| 脚本 | 状态 |
|---|---|
| `z44_final2.py` | ★权威复现 (v14 §1-§4 全部数字) |
| `z44_final.py` | §6 闭式解渠道; 但含硬编码(六审B3/B4) — 已被 final2 取代 |
| `z44_koide.py` | 历史: Koide √2=root壳模长 (已被超越性定理取代) |
| `z44_koide2.py` | 历史: 2/9 相位拟合 (同上) |
| `z44_derive_2over9.py` | 历史: 2/9 格点来源搜索 (先天注定, 见定理) |
| `z44_derive_2over9b.py` | 历史: 2/9 动力学搜索 |
| `z44_derive_2over9c.py` | 历史: 2/9 六角离散化 (被否证) |
| `z44_eps_scheme_v3.py` | 历史: 方案模糊度 ε (前提已被六审更正) |
| `z44_rge_phi2.py` | 已撤回: §7 scale-scan 用非物理插值 (v12 起删除) |
| `z44_2loop_v2.py` | 历史: 两圈估计 (量级, 符号未定) |
| `run_and_verify.py` | 本 LOG 的生成脚本 |

## 5. 复现方法

```
cd z44_flavor_probe
python3 z44_final2.py        # 论文全部关键数字
python3 z44_final.py         # 闭式解
python3 run_and_verify.py    # 生成本 LOG
```

环境: Python 3, numpy, scipy, sympy, mpmath (mpmath 用 50 位精度).