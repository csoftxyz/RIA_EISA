# -*- coding: utf-8 -*-
"""Verify that the annotation table covers every non-blank source line.

Exit code 0 only if, for each file, every line that contains anything other
than whitespace is inside at least one annotated range.  Also reports ranges
that run past the end of the file and notes attached to blank-only blocks.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, HERE)
from annotations import A  # noqa: E402


def main():
    failures = 0
    total_lines = total_code = total_notes = 0
    for fname, rows in A.items():
        base = HERE if fname == "make_figs3d.py" else SRC
        path = os.path.join(base, fname)
        lines = open(path, encoding="utf-8").read().splitlines()
        n = len(lines)
        covered = set()
        problems = []
        prev_end = 0
        for a, b, note in rows:
            if not (1 <= a <= b):
                problems.append("bad range %d-%d" % (a, b))
                continue
            if b > n:
                problems.append("range %d-%d exceeds file length %d" % (a, b, n))
            if a <= prev_end:
                problems.append("range %d-%d overlaps previous (ends %d)" % (a, b, prev_end))
            prev_end = max(prev_end, b)
            for i in range(a, b + 1):
                covered.add(i)
        missing = [i for i in range(1, n + 1)
                   if lines[i - 1].strip() and i not in covered]
        nonblank = sum(1 for l in lines if l.strip())
        code = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        total_lines += n
        total_code += code
        total_lines_ = 0
        total_notes += len(rows)
        status = "OK " if not missing and not problems else "FAIL"
        if missing or problems:
            failures += 1
        print("%s %-18s lines %3d  code %3d  (non-blank %3d)  notes %3d  coverage %5.1f%%"
              % (status, fname, n, code, nonblank, len(rows),
                 100.0 * (nonblank - len(missing)) / max(nonblank, 1)))
        for p in problems:
            print("      ! " + p)
        if missing:
            print("      uncovered: " + ", ".join(map(str, missing[:20]))
                  + (" ..." if len(missing) > 20 else ""))
    print("\n%d files, %d source lines, %d code lines, %d notes, %d file(s) with gaps"
          % (len(A), total_lines, total_code, total_notes, failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
