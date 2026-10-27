#!/bin/sh
# Build both illustrated guides (xelatex twice each).
set -u
cd "$(dirname "$0")/.." || exit 1
python3 code/make_guide_figures.py
for g in zh en; do
  (cd guide && xelatex -interaction=nonstopmode "guide_$g.tex" > "build_$g.log" 2>&1 && \
   xelatex -interaction=nonstopmode "guide_$g.tex" >> "build_$g.log" 2>&1)
  echo "### guide_$g exit=$?"
done
