#!/bin/sh
# Full verification suite: every test writes logs/<name>.log and logs/<name>.json
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
status=0
for t in t1_exact_gauss t3_lattice_green t4_running t5_marginality t6_exponents; do
  echo "### running $t"
  /usr/bin/python3 "code/$t.py" > "logs/$t.log" 2>&1
  rc=$?
  echo "### $t exit=$rc"
  [ $rc -ne 0 ] && status=1
done
echo "### running make_figures"
/usr/bin/python3 code/make_figures.py > logs/make_figures.log 2>&1
rc=$?; echo "### make_figures exit=$rc"; [ $rc -ne 0 ] && status=1
echo "### suite exit=$status"
exit $status
