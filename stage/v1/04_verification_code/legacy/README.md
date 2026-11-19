# Legacy / DEPRECATED scripts

These scripts implement methods that have been **withdrawn** and must NOT be used to
judge the physical observable-space dimension. They are kept only for provenance.

- `decisive_rank.py`, `rank_crisis.py`, `codim_correct.py`, `relations_complete.py`,
  `solve_relations.py`, `pheno_check2.py`: use the **orbit-subtraction** method
  (matrix rank minus "trivial orbit directions") and/or print "four relations /
  22-18=4 / 18 / 14 / 34". The orbit construction was wrong (six of its ten
  directions change physical observables). All their dimension numbers are withdrawn.

The correct, single source of truth for observable-space dimensions is
`../strict_obs.py` (strict, basis-independent Autonne-Takagi observables), which gives
generic rank 22 (zero observable relations) for every content and 13 for the charged
sector. The exact matrix-level pencil identities remain valid and live in
`../exact_core.py` (grade a); its `run()` no longer emits the withdrawn orbit numbers.

Do not cite any number from this directory.
