> **SUPERSEDED.** English edition of a working note. The final results are in the Letter and its
> Supplement.

# The vev loophole closes

Could a symmetry that fixes a vacuum expectation value ratio restore a proportionality, and so break
the ceiling? This was the only escape that does not carry a 1/M suppression, so it was the real
threat. It closes, and the closure is an invariance argument confirmed numerically.

## The argument

A Yukawa contribution enters as M_s = sum_j c_sj r_j Y_j with r_j the vev ratio. Changing a ratio
multiplies one column of the coefficient matrix by a non-zero scalar. Row rank is invariant under
non-zero column scaling. Hence:

> A symmetry that fixes a vev ratio — to any value, pure number or not — cannot create a
> proportionality between two sectors, and therefore cannot create an observable relation.

The only way to change the dimension of the image this way is to send a vev to zero, which deletes an
operator, drops the rank, restores M_d proportional to M_e, and returns the model to the excluded
branch.

## Numerical confirmation

The image dimension was measured as the numerical rank of the Jacobian at random points, with
degenerate sample points rejected. Over 240 scans, six models, forty ratios each, the dimension is
constant. Switching off both 45 operators of SU(5) with 5 + 45 returns the dimension from 13 to 10,
i.e. to SU(5) with a single 5_H, which is excluded.

## An anomaly that was my own error

A first run reported a dimension of 12 for SO(10) with 10 + 120 + 126 at some ratios, which looked
like an extra relation. It was a bug: the 120 Yukawa had been typed as *symmetric* in generation
indices. It is antisymmetric, since 16 x 16 = 10 + 120 + 126 with the 120 in the antisymmetric part.
With the correct type the anomaly disappears.

This is the second error the checks caught in my own code, and both ran in the same direction: they
made the structure look more constrained than it is.
