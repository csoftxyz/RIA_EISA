#!/usr/bin/env bash
# Run every script, capture a log per step, fail loudly on the first error.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
fail=0

run() {
  name="$1"; shift
  echo "== $name =="
  start=$(date +%s)
  "$@" > "logs/${name}.log" 2>&1
  code=$?
  end=$(date +%s)
  echo "   exit=$code  time=$((end-start))s  log=logs/${name}.log"
  if [ $code -ne 0 ]; then fail=1; tail -20 "logs/${name}.log"; fi
}

run exact_gauge          python3 code/exact_gauge.py
run finite_size          python3 code/finite_size_asymptotic.py
run wilson_loops         python3 code/wilson_loops.py
run villain_general      python3 code/villain_general.py
run triangulation        python3 code/triangulation_uniqueness.py
run dirac_checks         python3 code/dirac_operator_checks.py
run fermion_spectrum     python3 code/fermion_spectrum.py
run fermion_exact        python3 code/fermion_exact.py
run fermion_scan         python3 code/fermion_scan.py
run algebra_census       python3 code/algebra_census.py
run verify_jacobi        python3 code/verify_jacobi.py
run figures              python3 code/make_figures.py
run guide_figures        python3 code/make_guide_figures.py
run verify_all           python3 code/verify_all.py

if command -v pdflatex >/dev/null; then
  run latex_paper1 pdflatex -interaction=nonstopmode octahedron_NPB.tex
  run latex_paper2 pdflatex -interaction=nonstopmode octahedron_NPB.tex
fi
if command -v xelatex >/dev/null; then
  run latex_guide_zh1 xelatex -interaction=nonstopmode guide_zh.tex
  run latex_guide_zh2 xelatex -interaction=nonstopmode guide_zh.tex
fi
if command -v pdflatex >/dev/null; then
  run latex_guide_en1 pdflatex -interaction=nonstopmode guide_en.tex
  run latex_guide_en2 pdflatex -interaction=nonstopmode guide_en.tex
fi

echo
if [ $fail -eq 0 ]; then
  echo "ALL STEPS PASSED"
else
  echo "SOME STEPS FAILED — inspect the logs above"
fi
exit $fail
