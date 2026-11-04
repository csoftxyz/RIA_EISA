#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send the CPC supplement package.  The SMTP authorisation code is read from
memory at run time and never written to disk or printed."""
import os
import re
import smtplib
import sys
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

WS = "/root/.openclaw/workspace"
mem = open(os.path.join(WS, "memory/2026-09-05.md"), encoding="utf-8").read()
m = re.search(r"08-25\s*会话\s*([A-Za-z0-9]{8,})", mem)
if not m:
    print("ERROR: no SMTP auth code found")
    sys.exit(2)
AUTH = m.group(1)
SENDER = "3030136@qq.com"
RECV = "3030136@qq.com"

ATT = [
    os.path.join(WS, "CPC_radialflux_supplement_2026-11-04.zip"),
    os.path.join(WS, "CPC_radialflux/supplement/supplement.pdf"),
]

body = """张老师，

CPC 补充材料做好了，LaTeX 源 + PDF 都在附件里。56 页，三部分，逐条对应您的要求：

【1 代码逐行注释】
· src/ 全部 12 个文件、1383 条可执行行，267 条注释，覆盖率 100%。
· 不是我自己说"全覆盖"：supplement/check_annotations.py 逐行核验，有缺口就
  exit 1，运行输出原样印在 PDF 第 5 节。
· 外部函数来源单列一表：math.gamma 为什么不用 Stirling、np.expm1 为什么必须用
  （q-1=1e-8 时直接算 1-q^-a 会丢 8 位）、np.fft.fftfreq 的频率约定、
  np.linalg.lstsq 的 rcond 取法，都写清楚了。
· 代码里该批评的地方也照写：lattice96.py 第 66 行 rstar 是算而未用的死代码；
  exact_bias 的闭式在 q-1 < 1e-8 时会掉精度（稳定路径是 exact_radius）。没有替
  代码遮掩。

【2 三维可视化，9 张】
每张都由 supplement/make_figs3d.py 调用 src/ 里的函数现算，不复制公式，所以图不会
和代码走偏。表 3 是"代码 ←→ 图"的映射表：
  S1 一个壳层：r_n、r_n+1 与金色 ρ* 球，通量箭头；
  S2 偏差曲面 κ_L(q,d)，红面是精确规则；
  S3 等比网格七层壳，每层一颗金色采样球（自相似由此一眼可见）；
  S4 96³ 解的四分之一剖切点云，对数色标跨越六个数量级；
  S5 阈值曲面 q+q^(2-d)-2 与零等值线 q*_d（黄金比 / 塑料常数 / n-nacci）；
  S6 收敛漏斗：误差在 (log(q-1), log N) 上坍缩到二阶导板，实测阶 2.01/2.00/2.00；
  S7 通信类比示意（图上自带"仅示意、不进任何结论"字样）；
  S8 多极泄漏曲面 Σ q^-j，q=1.25、l=10 的 138% 标在金点上；
  S9 零检验"瀑布图"：四条估计量在同一 96³ 格点上对照真值平面。

【3 通信物理意义（第 5 节）】
· 开头就是加粗的状态声明：这是启发式类比，不是论文结果，论文任何结论都不依赖它。
· 给了"论文对象 ←→ 通信对象"对照表（壳层↔符号区间、q↔时钟比、ρ*↔无偏判决点、
  q*_d↔Nyquist 型阈值、Σq^-j↔ISI、背景↔直流偏置、running residual↔慢衰落）。
· 明确分成三层：(i) 严格成立的代数同构；(ii) 成立但平凡（阈值只是"点在不在区间内"，
  和 Nyquist 采样定理无关）；(iii) 完全没建立（本文没有信道、没有噪声、r 不是时间）。
· 写了作者研究生是通信专业、本文思路源于此 —— 作为出处记录，不作为论据。
· 最后给出"若真要当物理做，最小诚实方案是什么"（d 维波动方程 + 局部源 + 球面观测），
  说清楚那不是本文回答的问题。

附件：
  CPC_radialflux_supplement_2026-11-04.zip  （19 MB，50 个文件：tex/pdf/9 张图的
  PDF+PNG/注释库/核验脚本/五个日志/完整 src/）
  supplement.pdf                            （单文件，方便直接翻）

编译只需 pdflatex 两遍；依赖 NumPy，作图另需 Matplotlib，符号核验另需 SymPy，无数据文件。

如果要改口径（比如第 5 节想更收敛或更展开、图想要深色底/白底版本），说一声我改。
"""

msg = MIMEMultipart()
msg["From"] = formataddr((str(Header("Yuxuan Zhang", "utf-8")), SENDER))
msg["To"] = RECV
msg["Subject"] = Header("CPC 补充材料：逐行注释代码 + 9 张 3D 图 + 通信 outlook（tex+pdf）", "utf-8")
msg.attach(MIMEText(body, "plain", "utf-8"))

total = 0
for path in ATT:
    if not os.path.exists(path):
        print("missing", path)
        sys.exit(3)
    with open(path, "rb") as fh:
        part = MIMEApplication(fh.read(), Name=os.path.basename(path))
    part["Content-Disposition"] = 'attachment; filename="%s"' % os.path.basename(path)
    msg.attach(part)
    total += os.path.getsize(path)

print("attachments: %d files, %.1f MB" % (len(ATT), total / 1048576))
s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=120)
s.login(SENDER, AUTH)
s.sendmail(SENDER, [RECV], msg.as_string())
s.quit()
print("SENT to", RECV)
