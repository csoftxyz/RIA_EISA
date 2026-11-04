"""LaTeX integrity check: duplicate labels, dangling refs, uncited bibitems, unused figures.

A duplicate \\label is only a Warning in LaTeX, so it survives an ordinary
"0 error / 0 undefined reference" check while silently redirecting every
\\ref to the last definition.  This script exits non-zero on any finding.
"""
import re, sys, os

TEX = os.path.join(os.path.dirname(__file__), "..", "radialflux.tex")

def main():
    s = open(TEX, encoding="utf-8").read()
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    seen, dup = set(), []
    for l in labels:
        if l in seen:
            dup.append(l)
        else:
            seen.add(l)
    refs = set(re.findall(r"\\(?:eq|auto)?ref\{([^}]+)\}", s))
    items = re.findall(r"\\bibitem\{([^}]+)\}", s)
    cites = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", s):
        cites.update(k.strip() for k in m.split(","))
    figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
    figlabel = set(re.findall(r"\\label\{(fig:[^}]+)\}", s))
    figrefs = set(r for r in refs if r.startswith("fig:"))

    problems = []
    if dup:
        problems.append("duplicate labels: " + ", ".join(sorted(set(dup))))
    dangling = refs - seen
    if dangling:
        problems.append("dangling refs: " + ", ".join(sorted(dangling)))
    uncited = [i for i in items if i not in cites]
    if uncited:
        problems.append("uncited bibitems: " + ", ".join(uncited))
    phantom = sorted(c for c in cites if c not in items)
    if phantom:
        problems.append("phantom cites: " + ", ".join(phantom))
    unreferenced = sorted(figlabel - figrefs)
    if unreferenced:
        problems.append("figures never referenced: " + ", ".join(unreferenced))
    for f in sorted(figs):
        if not os.path.exists(os.path.join(os.path.dirname(TEX), f)):
            problems.append("missing figure file: " + f)

    print(f"labels {len(labels)} ({len(seen)} distinct), refs {len(refs)}, "
          f"bibitems {len(items)} ({len(items)-len(uncited)} cited), figures {len(figs)}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("TEX OK: no duplicate labels, no dangling refs, no uncited bibitems, "
          "every figure referenced and present")
    sys.exit(0)

if __name__ == "__main__":
    main()
