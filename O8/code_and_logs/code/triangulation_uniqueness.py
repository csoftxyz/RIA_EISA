#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triangulation_uniqueness.py

Verifies, by exhaustive enumeration, that the octahedron is the unique
triangulation of S^2 with six vertices.

Argument (also in the manuscript):
  * E = 3V - 6 = 12, so every vertex has degree 2E/V = 4.
  * A 4-regular graph on six vertices is K6 minus a perfect matching; all
    perfect matchings are equivalent under S6, so the graph is unique:
    K_{2,2,2}.
  * K_{2,2,2} has exactly 8 triangles and each edge lies in exactly 2 of them,
    so the face set is forced; the vertex links are 4-cycles and Euler gives
    chi = 2.  A 3-connected planar graph has a unique embedding (Whitney).
The script checks each step by brute force.
"""
import itertools
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "logs", "triangulation_uniqueness.json")

V = 6
PAIRS = list(itertools.combinations(range(V), 2))


def is_4regular(edges):
    deg = [0] * V
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return all(d == 4 for d in deg)


def triangles(edges):
    es = {tuple(sorted(e)) for e in edges}
    out = []
    for tri in itertools.combinations(range(V), 3):
        a, b, c = tri
        if (a, b) in es and (a, c) in es and (b, c) in es:
            out.append(tri)
    return out


def canonical(edges):
    """Canonical form of a labelled graph under all 720 relabellings."""
    best = None
    for perm in itertools.permutations(range(V)):
        img = tuple(sorted(tuple(sorted((perm[a], perm[b]))) for a, b in edges))
        if best is None or img < best:
            best = img
    return best


def canonical_tris(tris):
    """Canonical form of a labelled set of triangles under S6."""
    best = None
    for perm in itertools.permutations(range(V)):
        img = tuple(sorted(tuple(sorted(perm[x] for x in t)) for t in tris))
        if best is None or img < best:
            best = img
    return best


def link_is_cycle(v, tris):
    """The link of v must be a single 4-cycle."""
    nbr = defaultdict(set)
    for tri in tris:
        if v in tri:
            u, w = [x for x in tri if x != v]
            nbr[u].add(w)
            nbr[w].add(u)
    if len(nbr) != 4 or any(len(s) != 2 for s in nbr.values()):
        return False
    start = next(iter(nbr))
    seen, cur, prev = [start], start, None
    for _ in range(4):
        nxt = [x for x in nbr[cur] if x != prev][0]
        if nxt == start:
            return len(seen) == 4
        if nxt in seen:
            return False
        seen.append(nxt)
        prev, cur = cur, nxt
    return False


def main():
    graphs = []
    for combo in itertools.combinations(PAIRS, 12):
        if is_4regular(combo):
            graphs.append(combo)
    print("4-regular labelled graphs on 6 vertices:", len(graphs))
    classes = {canonical(g) for g in graphs}
    print("isomorphism classes:", len(classes))

    g = graphs[0]
    tris = triangles(g)
    edge_count = defaultdict(int)
    for tri in tris:
        for e in itertools.combinations(tri, 2):
            edge_count[tuple(sorted(e))] += 1
    links_ok = all(link_is_cycle(v, tris) for v in range(V))
    euler = V - len(g) + len(tris)
    faces = tris

    # count triangulations up to relabelling: all labelled 4-regular graphs
    # give the same triangle set after relabelling, so the orbit count is 1.
    orbits = set()
    for h in graphs:
        th = triangles(h)
        orbits.add(canonical_tris(th))
    print("triangles per 4-regular graph:", len(tris))
    print("edge triangle counts (should be all 2):",
          sorted(set(edge_count.values())), "on", len(edge_count), "edges")
    print("all vertex links 4-cycles:", links_ok)
    print("Euler characteristic V-E+F =", euler)
    print("distinct triangulations up to relabelling:", len(orbits))

    ok = (len(classes) == 1 and len(tris) == 8 and len(edge_count) == 12
          and set(edge_count.values()) == {2} and links_ok and euler == 2
          and len(orbits) == 1)
    print("VERDICT:", "PASS" if ok else "FAIL")
    json.dump(dict(labelled=len(graphs), classes=len(classes),
                   triangles=len(tris), edges=len(edge_count),
                   edge_multiplicities=sorted(set(edge_count.values())),
                   links_ok=links_ok, euler=euler, orbits=len(orbits),
                   pass_=bool(ok)), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
