# Quadratic point-symmetry showcase

`conformal_inverse_square.py` discovers the three point generators of

\[
u_{xx}=g/u^3,
\]

including the quadratic special-conformal generator
\(K=x^2\partial_x+xu\partial_u\). It also recovers the nonconstant
relative-invariance multiplier, compiles the invariant \(u/x\), constructs a
rectifying coordinate proportional to \(1/x\), and checks bracket closure.

Run it from the repository root:

```bash
python -m examples.quadratic_symmetry.conformal_inverse_square
```
