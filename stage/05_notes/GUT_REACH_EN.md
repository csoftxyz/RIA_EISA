> **SUPERSEDED.** English edition of a working note kept for provenance. The final results are in
> the Letter and its Supplement. One claim in this note (a negative correlation between group rank
> and reach) was later withdrawn; see the note at the end.

# Explanatory reach of grand unification, measured

**Question.** Every unified model is sold on the same aesthetic: a bigger group explains more.
Nobody has put a number on "how much does this group actually explain?" This note defines that
number, states the counting rules, verifies the group theory, and computes it for six standard
models.

## Definition

Fix a group G, a breaking chain, and a Higgs content H. The observable set at the unification scale
is nine charged-fermion masses, four CKM parameters and three gauge couplings, sixteen in total.

Let C be the matrix whose rows are the sectors u, d, e and whose columns are the allowed Yukawa
operators, with Clebsch-Gordan entries. Its rank is the structural quantity.

**Reach** P is the number of independent relations among the sixteen observables that hold for every
admissible choice of Yukawa matrices and, crucially, of vacuum expectation values.

Counting rules, declared rather than fitted: gauge unification 3 to 1 gives 2; a proportionality
M_d = k M_e with k fixed by the coefficients gives 3; M_u = t M_d with t = tan(beta) free gives 2;
a trivial CKM matrix gives 4; charge quantisation gives 1.

## The minimal case

SU(5) with one 5_H has two operators, 10.10.5 and 10.5b.5b. The second feeds down quarks and
charged leptons through the same matrix, so M_d = M_e^T identically and three mass relations follow.
Adding a 45_H introduces the Georgi-Jarlskog operator with coefficients +1 and -3, which destroys
the identity.

## The trade-off law

Each added Yukawa matrix that splits M_d from M_e costs exactly three relations; splitting M_u from
M_d as well costs four more. Reach is spent by model building, not extended by it.

## Withdrawn claim

An earlier draft quoted a negative correlation between group rank and reach, r = -0.688. It was
computed on a sample that contained two models which are in fact excluded. It is withdrawn. The
correct statement is stronger: across the models that survive, the reach does not vary at all, so
there is no correlation to measure.

*Reproduce with `model2.py` in `03_code/support/`.*
