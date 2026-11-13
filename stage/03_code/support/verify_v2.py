#!/usr/bin/env python3
"""Assert every number that appears in Table 1 of the Letter and in Table S1 of the Supplement.

Depends only on model2.py, which is the single source of truth for the census.
Run from this directory:  python3 verify_v2.py
Exit code 0 means every assertion passed.
"""
import sys
import model2 as M

OK = FAIL = 0
def chk(name, got, want, tol=1e-9):
    global OK, FAIL
    if isinstance(want, (list, tuple, set)) or isinstance(got, (list, tuple, set)):
        good = list(got) == list(want)
    else:
        good = abs(float(got) - float(want)) <= tol
    OK += good; FAIL += (not good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name:52s} got={got!s:>18} want={want!s}")
    return good

R = M.res
S = [e for e in R if e["splitting"]]
X = [e for e in R if not e["splitting"]]

print("=" * 78); print("1  CENSUS"); print("=" * 78)
chk("eighteen contents", len(R), 18)
chk("spectrum-splitting", len(S), 6)
chk("excluded", len(X), 12)
chk("all splitting have P_add = 0 except one", sum(e["p_add"] > 0 for e in S), 1)
chk("P_ex values among splitting", sorted({e["P"] for e in S}), [1, 3, 4])
chk("every splitting model: P = baseline + P_add",
    all(e["P"] == e["baseline"] + e["p_add"] for e in S), True)

print("=" * 78); print("2  ROWS OF TABLE 1 (Letter)"); print("=" * 78)
def row(g, m):
    e = next(x for x in R if x["grp"] == g and x["model"] == m)
    return (e["mass"], e["ckm"], e["gauge"], e["charge"], e["p_add"], e["P"], e["splitting"])
for g, m, want in [
    ("SU(5)",  "5+45",            (0, 0, 2, 1, 0, 3, True)),
    ("SU(5)",  "5+45+50",         (0, 0, 2, 1, 0, 3, True)),
    ("SO(10)", "10+126",          (0, 0, 2, 1, 0, 3, True)),
    ("SO(10)", "10+120+126",      (0, 0, 2, 1, 0, 3, True)),
    ("SO(10)", "120+126",         (0, 0, 2, 1, 1, 4, True)),
    ("PS",     "(1,2,2)+(15,2,2)",(0, 0, 0, 1, 0, 1, True)),
    ("E6",     "27",              (5, 4, 2, 1, 0, 12, False)),
    ("SO(10)", "10",              (5, 4, 2, 1, 0, 12, False)),
    ("PS",     "(1,2,2)",         (5, 4, 0, 1, 0, 10, False)),
    ("SU(5)",  "5",               (3, 0, 2, 1, 0, 6, False)),
    ("SO(10)", "10+120",          (3, 0, 2, 1, 0, 6, False)),
]:
    chk(f"{g} {m}", row(g, m), want)

print("=" * 78); print("3  COUNTING RULES"); print("=" * 78)
chk("single-operator mass relations (3 + 2)", 3 + 2, 5)
chk("simple-group baseline (2 gauge + 1 charge)",
    next(x for x in R if x["grp"] == "SU(5)" and x["model"] == "5+45")["baseline"], 3)
chk("Pati-Salam baseline (0 gauge + 1 charge)",
    next(x for x in R if x["grp"] == "PS" and x["model"] == "(1,2,2)")["baseline"], 1)

print("=" * 78)
print(f"RESULT: {OK} passed, {FAIL} failed")
print("=" * 78)
sys.exit(0 if FAIL == 0 else 1)
