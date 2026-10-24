#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wilson loops on the octahedron: exact character-expansion result versus
direct Monte Carlo on the 12 link angles.

Exact result (closed surface, F faces, loop C separating A and F-A faces):
    <W(C)> = sum_n I_n(beta)^A I_{n+1}(beta)^{F-A} / sum_n I_n(beta)^F .

The Monte Carlo uses the raw link angles with no gauge fixing, so it is an
independent check of both the formula and the gauge-fixed counting.  Errors
are batch-means (20 batches).
"""
import json
import os
import numpy as np

BETA = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))

# vertices: (axis, sign) -> index 0..5 ; axes x,y,z = 0,1,2
def vid(axis, sign):
    return 2 * axis + (0 if sign > 0 else 1)


VERTS = [(a, s) for a in range(3) for s in (1, -1)]
EDGES = []
for i, (a1, s1) in enumerate(VERTS):
    for j, (a2, s2) in enumerate(VERTS):
        if j > i and a1 != a2:
            EDGES.append((i, j))
assert len(EDGES) == 12

FACES = []
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            FACES.append((vid(0, sx), vid(1, sy), vid(2, sz)))
assert len(FACES) == 8
EDGE_INDEX = {e: k for k, e in enumerate(EDGES)}


def orient(cycle):
    out = []
    for k in range(len(cycle)):
        a, b = cycle[k], cycle[(k + 1) % len(cycle)]
        if (a, b) in EDGE_INDEX:
            out.append((EDGE_INDEX[(a, b)], +1))
        else:
            out.append((EDGE_INDEX[(b, a)], -1))
    return out


FACE_LINKS = [orient(f) for f in FACES]

# boundary of two adjacent triangles sharing an edge: separates 2 faces from 6
LOOP_A2 = [vid(0, 1), vid(1, 1), vid(0, -1), vid(2, 1)]
# 4-cycle x+ y+ x- y- : separates 4 faces from 4
LOOP_A4 = [vid(0, 1), vid(1, 1), vid(0, -1), vid(1, -1)]

L2 = orient(LOOP_A2)
L4 = orient(LOOP_A4)

# link -> [(face, sign), ...]
LINK_FACES = [[] for _ in range(12)]
for fi, fl in enumerate(FACE_LINKS):
    for (l, s) in fl:
        LINK_FACES[l].append((fi, s))


def plaquette_angles(theta):
    return np.array([sum(s * theta[l] for l, s in fl) for fl in FACE_LINKS])


def mc(n_batch=20, n_per=12000, n_therm=3000, seed=2026, step=1.1):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, 12)
    plaq = plaquette_angles(theta)
    acc = 0
    props = 0
    batches = np.zeros((n_batch, 3))

    def dS_link(l, new):
        dS = 0.0
        for (fi, s) in LINK_FACES[l]:
            dS += np.cos(plaq[fi] + s * (new - theta[l])) - np.cos(plaq[fi])
        return BETA * dS

    def measure():
        w2 = sum(s * theta[l] for l, s in L2)
        w4 = sum(s * theta[l] for l, s in L4)
        return np.cos(plaq).mean(), np.cos(w2), np.cos(w4)

    for _ in range(n_therm):
        for l in range(12):
            new = theta[l] + rng.normal(0.0, step)
            dS = dS_link(l, new)
            if dS > 0 or rng.random() < np.exp(dS):
                theta[l] = new
                plaq = plaquette_angles(theta)

    for ib in range(n_batch):
        accb = 0
        s = np.zeros(3)
        for _ in range(n_per):
            for l in range(12):
                props += 1
                new = theta[l] + rng.normal(0.0, step)
                dS = dS_link(l, new)
                if dS > 0 or rng.random() < np.exp(dS):
                    theta[l] = new
                    plaq = plaquette_angles(theta)
                    acc += 1
                    accb += 1
            s += np.array(measure())
        batches[ib] = s / n_per

    mean = batches.mean(axis=0)
    err = batches.std(axis=0, ddof=1) / np.sqrt(n_batch)
    return dict(plaquette=mean[0], plaquette_err=err[0],
                W_A2=mean[1], W_A2_err=err[1],
                W_A4=mean[2], W_A4_err=err[2],
                acc=acc / props)


def main():
    import mpmath as mp
    mp.mp.dps = 40
    F = 8

    def exact(A):
        Z = sum(mp.besseli(n, BETA) ** F for n in range(-30, 31))
        num = sum(mp.besseli(n, BETA) ** A * mp.besseli(n + 1, BETA) ** (F - A)
                  for n in range(-30, 31))
        return num / Z

    mcres = mc()
    out = dict(beta=BETA,
               exact={f"A{A}": mp.nstr(exact(A), 12) for A in (1, 2, 3, 4)},
               mc={k: float(v) for k, v in mcres.items()})
    print("exact  <W(A=1)> =", out["exact"]["A1"])
    print("exact  <W(A=2)> =", out["exact"]["A2"])
    print("exact  <W(A=4)> =", out["exact"]["A4"])
    print("MC     <cos theta_p> = %.5f +- %.5f  (exact %.5f)"
          % (mcres["plaquette"], mcres["plaquette_err"], float(exact(1))))
    print("MC     <W(A=2)> = %.5f +- %.5f  (exact %.5f)"
          % (mcres["W_A2"], mcres["W_A2_err"], float(exact(2))))
    print("MC     <W(A=4)> = %.5f +- %.5f  (exact %.5f)"
          % (mcres["W_A4"], mcres["W_A4_err"], float(exact(4))))
    print("MC acceptance = %.3f" % mcres["acc"])
    with open(os.path.join(HERE, os.pardir, "logs", "wilson_loops.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
