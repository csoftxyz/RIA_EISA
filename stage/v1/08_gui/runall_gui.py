"""
runall_gui.py -- a friendly control panel for the whole verification suite.

Every program that checks a claim in the paper is a button here. Click a button
to read, in plain language, WHAT it does, WHY it matters, and WHAT a good result
looks like (with a picture). Press "Run" to execute it; the live output appears
in the console at the bottom. "Run all" executes the master verifier
(verify_prd.py, 70 assertions) which calls most of the others.

No physics knowledge is needed to read the panels: each one is written for a
curious non-specialist. The pictures are real result figures from the paper.

Run:  python runall_gui.py   (or double-click runall.bat)
"""
import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "04_verification_code"))
ICONS = os.path.join(HERE, "icons")
PYTHON = sys.executable

# ---------------------------------------------------------------- program catalogue
# Each entry: (key, title, icon, script, what, why, good_result)
PROGRAMS = [
    ("master", "RUN ALL CHECKS (master verifier)", "certificate.png", "verify_prd.py",
     "Runs the entire verification program: the census of 18 unified models, the exact "
     "rational ranks, the neutrino dimension certificate, and 70 yes/no checks (assertions). "
     "It also redrawing all 12 figures and writes results.json.",
     "This is the single source of truth. If every assertion passes, every number quoted in "
     "the paper is reproduced by code. If any fails, the program exits with an error.",
     "You want to see: '70 / 70 assertions passed' and 'viable 7 | excluded 11 | P_add set [0]'."),

    ("census", "Census: which models survive?", "census.png", "tests_validation/census_viability.py",
     "Looks at all 18 ways to build a unified model from the standard Higgs pieces and asks: "
     "can each one spread the quark and lepton masses apart enough to match reality?",
     "A model that forces, say, the down-quark and electron masses to be equal is dead on "
     "arrival -- nature says they are very different. This separates the living from the dead.",
     "7 contents can fit the spectrum (viable); 11 force relations nature refuses (excluded)."),

    ("pencil", "The pencil identity (exact)", "pencil.png", "pencil_analytic_origin.py",
     "In the simplest SO(10) model (120+126), the neutrino mass matrix and the down-quark "
     "matrix are linked. This checks an exact algebraic fact: a certain combination always "
     "has a repeated eigenvalue -- like a polynomial with a double root.",
     "It is the paper's one exact new equation. The repeated root is forced by the "
     "antisymmetry of the 120 Yukawa (det A = 0), not by lucky parameter choices.",
     "Every link of the derivation verified to ~1e-13; 'CHAIN VERIFIED: True'."),

    ("cert", "Neutrino dimension certificate", "certificate.png", "grade_a_fast.py",
     "Proves, with rigorous interval arithmetic, that the neutrino observable map has full "
     "rank 22 -- meaning the model forces ZERO exact low-energy relations. It does this by "
     "showing a certain safety margin (rho) stays below 1.",
     "This upgrades the claim from 'looks full-rank numerically' to 'provably full-rank'. "
     "rho < 1 is a mathematical guarantee, not a numerical guess.",
     "rho ~ 0.12 / 0.09 / 0.09 at three points, all < 1; rank 22 certified (grade a)."),

    ("attrib", "Control: is it really the pencil?", "attribution.png", "tests_validation/pencil_attribution_control.py",
     "A statistical preference (single-S models favour a weaker neutrino mass hierarchy) was "
     "once blamed on the pencil identity. This control repeats the test on 10+126, which has "
     "NO pencil identity -- and finds the SAME preference.",
     "It catches a misattribution: the preference comes from the single-S structure of M_R, "
     "not from the pencil. Honest science means killing your own favourite explanation.",
     "Preference present in 10+126 too (and in pure type-I) => NOT a pencil consequence."),

    ("score", "Declaration-cost scoreboard", "scoreboard.png", "tests_validation/seven_content_scoreboard.py",
     "All 7 viable models tie at 'zero forced flavour relations'. This breaks the tie by "
     "asking: how many EXTRA assumptions must each model declare to actually fit the data?",
     "It answers 'which group is more economical?' with a computable number instead of a "
     "slogan. Fewer declared inputs = more economical.",
     "5 contents fit with 0 declarations; 10+120 and 120+126 need >= 1. Minimal is least economical."),

    ("ckm", "Why 11 models die: CKM locking", "ckm.png", "tests_validation/exclusion_taxonomy.py",
     "If a single Yukawa matrix feeds several fermion sectors, their left-handed rotations "
     "line up and the quark mixing matrix becomes the identity (no mixing). Nature has large "
     "mixing, so such models are excluded.",
     "This shows the 11 exclusions are not random accidents: 9 of them die by this ONE "
     "mechanism, across all four groups. A classification theorem, not a head-count.",
     "9/11 excluded contents are CKM-locked; the minimal-content exclusion survives running."),

    ("rg", "Does running kill the wall?", "rg.png", "rg_lowenergy_match.py",
     "The 120+126 model cannot reach the observed tau/bottom mass ratio at the unification "
     "scale (a 'wall' at 2.76). But parameters change with energy. This evolves the MEASURED "
     "spectrum up to the GUT scale to check if the wall still excludes the model.",
     "A wall at high energy is only a real exclusion if measured data, run up to that energy, "
     "still hits it. Anchoring to data makes the conclusion scheme-independent.",
     "Measured (m_tau/m_b)_GUT ~ 1.07, far below the 2.76 wall => exclusion survives running."),

    ("seesaw", "Seesaw surjectivity theorem", "seesaw.png", "tests_validation/constrained_seesaw_test.py",
     "The type-I seesaw lets you pick almost any neutrino mass pattern by choosing the heavy "
     "Majorana matrix M_R freely. This checks the map is 'onto': any symmetric target can be hit.",
     "It is the baseline: a FREE Majorana sector adds no prediction. Predictivity must come "
     "from a declared structure on M_R, not from the seesaw itself.",
     "500 arbitrary symmetric targets reproduced to better than 1 part in 1e12."),

    ("prior", "Prior-sensitivity of the preference", "attribution.png", "tests_validation/pencil_prior_sensitivity.py",
     "The statistical preference might be an artefact of how we randomise the Yukawas. This "
     "re-runs it under 6 very different randomisation recipes and 3 breaking strengths.",
     "If the direction survives every recipe, it is robust; if it flips, it was an artefact. "
     "This is how you stress-test a statistical claim.",
     "Direction (single-S => weaker hierarchy) holds in 6/6 priors and 3/3 strengths."),

    ("rational", "Why no rational shortcut to 22", "pencil.png", "tests_validation/rational_route_exhaustive.py",
     "One might hope to prove rank 22 using only rational invariants (trace words). This "
     "exhaustively screens them and shows they stall at rank 17 -- they cannot see the "
     "Majorana-phase directions.",
     "It explains WHY the certificate must use the full non-rational observable map: the "
     "rational toolbox is provably too small (fraction field adds nothing).",
     "40 candidates -> 20 fibre-constant -> rank 16; fraction-field rank 17; never 22."),

    ("seed42", "The seed-42 anomaly explained", "certificate.png", "tests_validation/seed42_diagnosis.py",
     "At one random point (seed 42) the rank reads 21 instead of 22. This diagnoses why: the "
     "Jarlskog invariant (a measure of CP violation) happens to be near zero there, making "
     "the coordinate chart degenerate -- not a physical relation.",
     "A single low-rank point could look like a counterexample. Showing it is a coordinate "
     "artefact protects the generic rank-22 claim.",
     "J ~ -4.6e-7, d2J stable, mass gaps healthy => coordinate degeneracy, generic rank 22."),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("C21 Unified-Flavour Verification Suite -- Control Panel")
        root.geometry("1280x840")
        root.minsize(1100, 720)
        root.configure(bg="#f4f6f8")
        self.q = queue.Queue()
        self.proc = None
        self.photos = {}
        self._build_fonts()
        self._build_header()
        self._build_body()
        self._build_console()
        self._poll()
        self.select("master")

    def _build_fonts(self):
        self.f_title = font.Font(family="Segoe UI", size=15, weight="bold")
        self.f_h = font.Font(family="Segoe UI", size=11, weight="bold")
        self.f_b = font.Font(family="Segoe UI", size=10)
        self.f_s = font.Font(family="Segoe UI", size=9)
        self.f_mono = font.Font(family="Consolas", size=9)

    def _build_header(self):
        h = tk.Frame(self.root, bg="#1b2430", height=64)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h, text="C21  ·  Grand Unification Builds the Stage, Not the Script",
                 bg="#1b2430", fg="white", font=self.f_title).pack(side="left", padx=18, pady=12)
        tk.Label(h, text="Verification & audit control panel  ·  project started February 2025",
                 bg="#1b2430", fg="#9fb3c8", font=self.f_s).pack(side="right", padx=18)

    def _build_body(self):
        body = tk.Frame(self.root, bg="#f4f6f8", height=560)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # left: button list
        left = tk.Frame(body, bg="#f4f6f8", width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text="PROGRAMS  (click to inspect)", bg="#f4f6f8",
                 fg="#1b2430", font=self.f_h).pack(anchor="w", pady=(0, 6))
        canvas = tk.Canvas(left, bg="#f4f6f8", highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.btnframe = tk.Frame(canvas, bg="#f4f6f8")
        self.btnframe.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.btnframe, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.buttons = {}
        for key, title, icon, script, what, why, good in PROGRAMS:
            b = tk.Button(self.btnframe, text="  " + title, anchor="w",
                          font=self.f_b, bg="white", fg="#1b2430", relief="flat",
                          activebackground="#dce6f0", bd=0, padx=8, pady=7,
                          command=lambda k=key: self.select(k))
            b.pack(fill="x", pady=2)
            self.buttons[key] = b

        # right: detail panel
        right = tk.Frame(body, bg="white", bd=0)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.detail_title = tk.Label(right, text="", bg="white", fg="#1b2430",
                                     font=self.f_title, anchor="w", justify="left")
        self.detail_title.pack(fill="x", padx=16, pady=(12, 2))
        txt = tk.Frame(right, bg="white")
        txt.pack(fill="x", padx=16)
        self.what_lbl = self._section(txt, "WHAT it does", "#1f6f4a")
        self.why_lbl = self._section(txt, "WHY it matters", "#1f6f4a")
        self.good_lbl = self._section(txt, "WHAT a good result looks like", "#1f6f4a")
        imgrow = tk.Frame(right, bg="white")
        imgrow.pack(fill="x", padx=16, pady=(6, 0))
        tk.Label(imgrow, text="ILLUSTRATION", bg="white", fg="#6b7785",
                 font=self.f_s).pack(anchor="w")
        self.img_label = tk.Label(imgrow, bg="white")
        self.img_label.pack(anchor="w")
        runbar = tk.Frame(right, bg="white")
        runbar.pack(fill="x", padx=16, pady=8)
        self.run_btn = tk.Button(runbar, text="▶  Run this program", font=self.f_h,
                                 bg="#1f6f4a", fg="white", relief="flat", padx=16, pady=6,
                                 command=self.run_selected)
        self.run_btn.pack(side="left")
        self.script_lbl = tk.Label(runbar, text="", bg="white", fg="#6b7785", font=self.f_mono)
        self.script_lbl.pack(side="left", padx=12)

    def _section(self, parent, head, color):
        tk.Label(parent, text=head, bg="white", fg=color, font=self.f_h,
                 anchor="w").pack(fill="x", pady=(6, 1))
        lbl = tk.Label(parent, text="", bg="white", fg="#2b3440", font=self.f_b,
                       anchor="w", justify="left", wraplength=480)
        lbl.pack(fill="x")
        return lbl

    def _build_console(self):
        c = tk.Frame(self.root, bg="#1b2430", height=200)
        c.pack(fill="x", expand=False, padx=10, pady=(0, 10))
        c.pack_propagate(False)
        bar = tk.Frame(c, bg="#1b2430")
        bar.pack(fill="x")
        tk.Label(bar, text="LIVE OUTPUT", bg="#1b2430", fg="#9fb3c8",
                 font=self.f_s).pack(side="left", padx=8, pady=4)
        self.status = tk.Label(bar, text="idle", bg="#1b2430", fg="#7fc8a9", font=self.f_s)
        self.status.pack(side="right", padx=8)
        tk.Button(bar, text="Clear", font=self.f_s, bg="#2b3440", fg="white",
                  relief="flat", command=lambda: self.console.delete("1.0", "end")).pack(side="right", padx=4)
        self.console = tk.Text(c, height=12, bg="#0f1620", fg="#d6e2ee", font=self.f_mono,
                               relief="flat", wrap="word")
        self.console.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def select(self, key):
        self.current = key
        for k, b in self.buttons.items():
            b.configure(bg="#dce6f0" if k == key else "white")
        for p in PROGRAMS:
            if p[0] == key:
                _, title, icon, script, what, why, good = p
                self.detail_title.configure(text=title)
                self.what_lbl.configure(text=what)
                self.why_lbl.configure(text=why)
                self.good_lbl.configure(text=good)
                self.script_lbl.configure(text=script)
                path = os.path.join(ICONS, icon)
                if os.path.exists(path):
                    try:
                        im = Image.open(path).convert("RGB")
                        w, h = im.size
                        if w > 230:
                            im = im.resize((230, int(h * 230 / w)), Image.LANCZOS)
                        img = ImageTk.PhotoImage(im)
                        self.photos[key] = img
                        self.img_label.configure(image=img, text="")
                    except Exception as e:
                        self.img_label.configure(image="", text="[figure: %s]" % e)
                break

    def run_selected(self):
        for p in PROGRAMS:
            if p[0] == self.current:
                self._run(p[3])
                break

    def _run(self, script):
        if self.proc is not None:
            self._log("A program is already running. Please wait.\n")
            return
        path = os.path.join(CODE, script)
        if not os.path.exists(path):
            self._log("Script not found: %s\n" % path)
            return
        self.console.delete("1.0", "end")
        self._log(">>> Running %s\n\n" % script)
        self.status.configure(text="running: " + script, fg="#f0c674")
        self.run_btn.configure(state="disabled")
        env = dict(os.environ, PYTHONPATH="")
        def worker():
            try:
                p = subprocess.Popen([PYTHON, path], cwd=CODE, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)
                self.proc = p
                for line in p.stdout:
                    self.q.put(line)
                p.wait()
                self.q.put("\n[exit code %d]\n" % p.returncode)
            except Exception as e:
                self.q.put("\n[error: %s]\n" % e)
            finally:
                self.proc = None
                self.q.put(("__DONE__",))
        threading.Thread(target=worker, daemon=True).start()

    def _log(self, text):
        self.console.insert("end", text)
        self.console.see("end")

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    self.status.configure(text="idle", fg="#7fc8a9")
                    self.run_btn.configure(state="normal")
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
