# Poisson-geometry showcases

Three examples exercise discovery rather than merely checking hard-coded
identities:

- `cyclic_lotka_volterra.py` recovers a shared quadratic log-canonical
  bracket, three linear Hamiltonians, generic rank two, exact polynomial
  Jacobi, and the cubic Casimir \(xyz\).
- `translated_euler_top.py` rejects constant and homogeneous-linear lanes,
  then recovers an affine Lie--Poisson bracket shared by three inertia
  tensors, quadratic Hamiltonians, the translated quadratic Casimir, and the
  rank drop at the translated origin.
- `casimir_taxonomy.py` contrasts a physical Poisson Casimir on `so(3)*` with
  the rotation-algebra Casimir on full-rank canonical phase space, demonstrates
  the Hamiltonian gauge quotient, and pulls the algebra invariant back to
  `|q x p|^2`.

Run them from the repository root:

```bash
python -m examples.poisson_geometry.cyclic_lotka_volterra
python -m examples.poisson_geometry.translated_euler_top
python -m examples.poisson_geometry.casimir_taxonomy
```
