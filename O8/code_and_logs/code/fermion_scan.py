#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver: exact beta-scan of the octahedral minimal-fermion model + diagnostics."""
import numpy as np
from fermion_exact import (expansion, expansion_minus, solve_face_charges,
                                series_sector, In_table, E, F, A, verts, vid, edges,
                                th, Gm, faces)
from collections import defaultdict
import time

# ---------- precompute: merged weight terms with their n0 solutions (beta-independent) ----------
def merged_pair_terms(m):
    P = expansion(m); Mm = expansion_minus(m)
    agg = defaultdict(float)
    for c1, nu1 in P:
        for c2, nu2 in Mm:
            agg[tuple(nu1 + nu2)] += c1 * c2
    out = []
    for nu, c in agg.items():
        n0 = solve_face_charges(np.array(nu, int))
        out.append((c, n0))
    return out

def merged_det_terms(m):
    agg = defaultdict(float)
    for c, nu in expansion(m):
        agg[tuple(nu)] += c
    return [(c, solve_face_charges(np.array(nu, int))) for nu, c in agg.items()]

def series_tot(n0, tab, N=70):
    out = np.zeros(3)
    Q0 = int(n0.sum()) % 3
    for k in range(-N, N + 1):
        idx = np.abs(n0 + k)
        if idx.max() > N: continue
        out[(Q0 + 8 * k) % 3] += np.prod(tab[idx])
    return out

def Zg_from_tab(tab, N=70):
    return sum(tab[abs(k)]**8 for k in range(-N, N + 1))

def run_beta_grid(betas, m=0.0, N=70):
    W = merged_pair_terms(m)          # |det|^2 weight terms
    D1 = merged_det_terms(m)          # det weight terms (signed)
    rows = []
    for b in betas:
        tab = In_table(b, N)
        Zg = Zg_from_tab(tab, N)
        Zf_sectors = np.zeros(3)
        Zdet_sectors = np.zeros(3)
        for c, n0 in W:
            Zf_sectors += c * series_tot(n0, tab, N)
        for c, n0 in D1:
            Zdet_sectors += c * series_tot(n0, tab, N)
        Zf = Zf_sectors.sum()
        Zdet = Zdet_sectors.sum()
        # pure gauge sectors (fermion-free) for comparison
        Zs_pure = np.zeros(3)
        for k in range(-N, N + 1):
            Zs_pure[(8 * k) % 3] += tab[abs(k)]**8
        f = Zf_sectors / Zf
        fp = Zs_pure / Zg
        R = (Zf_sectors[1] + Zf_sectors[2]) / Zf_sectors[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            dF = -np.log(R) / b
        rows.append(dict(beta=b, Zg=Zg, Zf=Zf, Zdet=Zdet, mean_det2=Zf / Zg,
                         mean_det=Zdet / Zg, f0=f[0], f1=f[1], f2=f[2],
                         f0p=fp[0], R=R, dF=dF))
    return rows

# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    grid = np.round(np.arange(0.5, 2.0001, 0.05), 4)
    fine = np.round(np.arange(0.9, 1.1001, 0.01), 4)
    betas = np.unique(np.concatenate([grid, fine]))
    rows = run_beta_grid(betas)
    print(f"terms: |det|^2 merged = {len(merged_pair_terms(0.0))}, det merged = {len(merged_det_terms(0.0))}")
    print(f"elapsed {time.time()-t0:.1f}s for {len(betas)} beta points\n")

    print(f"{'beta':>5} {'<det>':>10} {'<det^2>':>10} {'f0(ferm)':>10} {'f0(pure)':>10} {'R':>10} {'DeltaF':>10}")
    sel = [r for r in rows if abs(r['beta'] - 1.0) < 1e-9 or (r['beta'] in [0.5,0.75,0.95,1.05,1.25,1.5,2.0])]
    for r in rows:
        if r['beta'] in [0.5,0.75,0.9,0.95,0.99,1.0,1.01,1.05,1.1,1.25,1.5,2.0]:
            print(f"{r['beta']:5.2f} {r['mean_det']:10.6f} {r['mean_det2']:10.6f} "
                  f"{r['f0']:10.6f} {r['f0p']:10.6f} {r['R']:10.6f} {r['dF']:10.6f}")

    # ---- derivative diagnostics around beta=1 (fine grid) ----
    fb = np.array([r['beta'] for r in rows]); 
    msk = (fb >= 0.9) & (fb <= 1.1)
    bf = fb[msk]
    dFf = np.array([r['dF'] for r in rows])[msk]
    Rf  = np.array([r['R'] for r in rows])[msk]
    mdet= np.array([r['mean_det'] for r in rows])[msk]
    mdet2=np.array([r['mean_det2'] for r in rows])[msk]
    d1 = np.gradient(dFf, bf); d2 = np.gradient(d1, bf)
    i1 = int(np.argmin(np.abs(bf - 1.0)))
    print(f"\nfine-grid diagnostics at beta=1 (h={bf[1]-bf[0]}):")
    print(f"  DeltaF(1)    = {dFf[i1]:.8f}")
    print(f"  dDeltaF/db   = {d1[i1]:.6f}   (smooth across? check sides: {d1[i1-1]:.6f} | {d1[i1+1]:.6f})")
    print(f"  d2DeltaF/db2 = {d2[i1]:.6f}   (sides: {d2[i1-1]:.6f} | {d2[i1+1]:.6f})")
    print(f"  <det> slopes : d/db at 1^-/1^+ : {(mdet[i1]-mdet[i1-1])/(bf[i1]-bf[i1-1]):.6f} | "
          f"{(mdet[i1+1]-mdet[i1])/(bf[i1+1]-bf[i1]):.6f}")
    print(f"  <det^2>      : d/db at 1^-/1^+ : {(mdet2[i1]-mdet2[i1-1])/(bf[i1]-bf[i1-1]):.6f} | "
          f"{(mdet2[i1+1]-mdet2[i1])/(bf[i1+1]-bf[i1]):.6f}")
    # smoothness: is d2DeltaF continuous? max |jump| of d2 between adjacent fine points near 1
    jj = np.max(np.abs(np.diff(d2)))
    print(f"  max |jump of d2DeltaF/db2| on fine grid = {jj:.3e}  (analytic -> ~h^2 noise, no kink)")

    # ---- pure-gauge sector reference at beta=1 ----
    tab = In_table(1.0); Zg = Zg_from_tab(tab)
    Zs_pure = np.zeros(3)
    for k in range(-70, 71): Zs_pure[(8*k) % 3] += tab[abs(k)]**8
    print(f"\npure-gauge sector fractions at b=1: f0={Zs_pure[0]/Zg:.8f} f1=f2={Zs_pure[1]/Zg:.8f} "
          f"(paper Z3 char <T> = { (Zs_pure[0]-Zs_pure[1]/2-Zs_pure[2]/2)/Zg:.6f})")

    # ---- zero-mode probes at special (T-symmetric) configurations ----
    print("\nzero-mode probe: |lambda_min| of D at uniform-flux configs (all 8 faces = phi):")
    B = A.T  # 8x12 face->link incidence? A is 12x8; need link phases with B theta = phi*1
    Bf = A.T
    for phi in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        thl, *_ = np.linalg.lstsq(Bf, np.full(F, phi), rcond=None)
        if np.max(np.abs(Bf @ thl - phi)) > 1e-6:
            print(f"  phi={phi/np.pi:.2f}pi : unattainable by link phases"); continue
        M = np.zeros((6, 6), complex)
        for k, (u, v) in enumerate(edges):
            s, e = th(u, v)
            ph = s * thl[e]
            M[vid[u], vid[v]] = np.exp(1j * ph)
            M[vid[v], vid[u]] = -np.exp(-1j * ph)
        evs = np.linalg.eigvals(M)
        print(f"  phi={phi/np.pi:.3f}pi : min|lambda| = {np.abs(evs).min():.6f}   "
              f"det = {np.linalg.det(M).real:+.5f}")
    # random sample: distribution of min|lambda| (m=0): any near-zero?
    rng = np.random.default_rng(7)
    mins = []
    for _ in range(400):
        thl = rng.uniform(0, 2*np.pi, E)
        M = np.zeros((6, 6), complex)
        for k, (u, v) in enumerate(edges):
            s, e = th(u, v)
            M[vid[u], vid[v]] = np.exp(1j * s * thl[e])
            M[vid[v], vid[u]] = -np.exp(-1j * s * thl[e])
        mins.append(np.abs(np.linalg.eigvals(M)).min())
    mins = np.array(mins)
    print(f"  random configs: min|lambda| quantiles q0/q50/q90 = "
          f"{np.quantile(mins,[0,.5,.9]).round(4)}  (no exact zero encountered: {np.min(mins):.2e})")
