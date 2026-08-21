# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Unit tests for the unified AtomNode inputs field."""

import pytest
import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    Mul,
    Pow,
    Var,
    clone_ast,
    eval_inputs,
)


class TestAtomNodeInputsField:
    """Tests for the new inputs field on AtomNode."""

    def test_simple_atom_inputs_auto_populated(self):
        """Simple atoms should have inputs auto-populated from var_idxs."""
        atom = AtomNode(kind="nn", var_idxs=(0, 1))

        assert atom.inputs is not None
        assert len(atom.inputs) == 2
        # Each input should be a Var node
        assert atom.inputs[0].kind == "var"
        assert atom.inputs[0].var_idxs == (0,)
        assert atom.inputs[1].kind == "var"
        assert atom.inputs[1].var_idxs == (1,)

    def test_compound_atom_inputs_auto_populated(self):
        """Compound atoms should have inputs from compound expr + extra vars."""
        # z = x0 * x1, extra: x2
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1, 2),
            inputs=(expr, Var(2)),
        )

        assert atom.inputs is not None
        assert len(atom.inputs) == 2  # compound expr + 1 extra var
        # First input is the compound expression
        assert str(atom.inputs[0]) == "(x0 * x1)"
        # Second is the extra variable
        assert atom.inputs[1].kind == "var"
        assert atom.inputs[1].var_idxs == (2,)

    def test_is_simple_true_for_var_inputs(self):
        """is_simple() should return True for atoms with only Var inputs."""
        atom = AtomNode(kind="nn", var_idxs=(0, 1, 2))
        assert atom.is_simple() is True

    def test_is_simple_false_for_compound(self):
        """is_simple() should return False for compound atoms."""
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1),
            inputs=(expr,),
        )

        assert atom.is_simple() is False

    def test_simple_var_idxs_for_simple_atom(self):
        """simple_var_idxs() should return tuple of indices for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(2, 0, 1))
        idxs = atom.simple_var_idxs()

        assert idxs is not None
        assert idxs == (2, 0, 1)  # Preserves order

    def test_simple_var_idxs_none_for_compound(self):
        """simple_var_idxs() should return None for compound atoms."""
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1),
            inputs=(expr,),
        )

        assert atom.simple_var_idxs() is None

    def test_raw_var_idxs_simple(self):
        """raw_var_idxs should return var_idxs for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(1, 3, 5))
        assert atom.raw_var_idxs == (1, 3, 5)

    def test_raw_var_idxs_compound(self):
        """raw_var_idxs should collect all var indices from compound expr."""
        # z = x0 * x1, extra: x3
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1, 3),
            inputs=(expr, Var(3)),
        )

        # Should include all vars from expr and extra vars
        assert set(atom.raw_var_idxs) == {0, 1, 3}

    def test_n_in_simple(self):
        """n_in should match number of var_idxs for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(0, 1, 2))
        assert atom.n_in == 3

    def test_n_in_compound(self):
        """n_in should be 1 + len(extra_var_idxs) for compound atoms."""
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1, 2, 3),
            inputs=(expr, Var(2), Var(3)),
        )

        assert atom.n_in == 3  # 1 compound + 2 extras


class TestEvalInputs:
    """Tests for the eval_inputs() function."""

    def test_eval_simple_atom(self):
        """eval_inputs should select columns for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(0, 2))

        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])

        x_in, grad, hess = eval_inputs(atom, x)

        assert x_in.shape == (2, 2)
        assert torch.allclose(x_in, x[:, [0, 2]])
        assert grad is None
        assert hess is None

    def test_eval_simple_atom_with_grad(self):
        """eval_inputs should compute identity Jacobian for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(1, 2))

        x = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        x_in, grad, hess = eval_inputs(atom, x, need_grad=True)

        assert x_in.shape == (2, 2)
        assert grad.shape == (2, 2, 3)  # (B, n_in, Nx)

        # Jacobian should be identity-like at the selected columns
        expected_grad = torch.zeros(2, 2, 3)
        expected_grad[:, 0, 1] = 1.0  # d(x_in[0])/d(x[1]) = 1
        expected_grad[:, 1, 2] = 1.0  # d(x_in[1])/d(x[2]) = 1

        assert torch.allclose(grad, expected_grad)

    def test_eval_compound_product(self):
        """eval_inputs should evaluate z = x0 * x1 for compound atoms."""
        # z = x0 * x1
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1),
            inputs=(expr,),
        )

        x = torch.tensor([
            [2.0, 3.0],
            [4.0, 5.0],
        ])

        x_in, grad, hess = eval_inputs(atom, x)

        assert x_in.shape == (2, 1)
        assert torch.allclose(x_in, torch.tensor([[6.0], [20.0]]))

    def test_eval_compound_product_with_grad(self):
        """eval_inputs should compute correct Jacobian for z = x0 * x1."""
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1),
            inputs=(expr,),
        )

        x = torch.tensor([[2.0, 3.0]])  # x0=2, x1=3

        x_in, grad, hess = eval_inputs(atom, x, need_grad=True)

        assert x_in.shape == (1, 1)
        assert grad.shape == (1, 1, 2)

        # dz/dx0 = x1 = 3, dz/dx1 = x0 = 2
        assert torch.allclose(grad[0, 0, 0], torch.tensor(3.0))
        assert torch.allclose(grad[0, 0, 1], torch.tensor(2.0))

    def test_eval_compound_with_extra_vars(self):
        """eval_inputs should handle compound + extra variables."""
        # z = x0 * x1, extra: x2
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1, 2),
            inputs=(expr, Var(2)),
        )

        x = torch.tensor([
            [2.0, 3.0, 10.0],
            [4.0, 5.0, 20.0],
        ])

        x_in, grad, hess = eval_inputs(atom, x)

        # x_in should be [z, x2] = [x0*x1, x2]
        assert x_in.shape == (2, 2)
        assert torch.allclose(x_in[:, 0], torch.tensor([6.0, 20.0]))
        assert torch.allclose(x_in[:, 1], torch.tensor([10.0, 20.0]))

    def test_eval_compound_ratio(self):
        """eval_inputs should handle z = x0 / x1 = x0 * x1^(-1)."""
        # z = x0 / x1 = x0 * x1^(-1)
        expr = Mul(Var(0), Pow(Var(1), -1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1),
            inputs=(expr,),
        )

        x = torch.tensor([[6.0, 3.0]])  # z = 6/3 = 2

        x_in, grad, hess = eval_inputs(atom, x)

        assert x_in.shape == (1, 1)
        assert torch.allclose(x_in[0, 0], torch.tensor(2.0))

    def test_eval_empty_inputs(self):
        """eval_inputs should handle atoms with no inputs (e.g., DE features)."""
        atom = AtomNode(kind="u", var_idxs=())

        x = torch.tensor([[1.0, 2.0, 3.0]])

        x_in, grad, hess = eval_inputs(atom, x)

        assert x_in.shape == (1, 0)


class TestCloneAstWithInputs:
    """Tests that clone_ast properly handles the inputs field."""

    def test_clone_simple_atom_preserves_inputs(self):
        """clone_ast should clone inputs for simple atoms."""
        atom = AtomNode(kind="nn", var_idxs=(0, 1))

        cloned = clone_ast(atom)

        assert cloned.inputs is not None
        assert len(cloned.inputs) == 2
        # Verify they are new objects
        assert cloned.inputs[0] is not atom.inputs[0]
        # But have same values
        assert cloned.inputs[0].var_idxs == atom.inputs[0].var_idxs

    def test_clone_compound_atom_preserves_inputs(self):
        """clone_ast should deep clone compound inputs."""
        expr = Mul(Var(0), Var(1))

        atom = AtomNode(
            kind="nn",
            var_idxs=(0, 1, 2),
            inputs=(expr, Var(2)),
        )

        cloned = clone_ast(atom)

        assert cloned.inputs is not None
        assert len(cloned.inputs) == len(atom.inputs)
        # Verify deep clone (different objects)
        assert cloned.inputs[0] is not atom.inputs[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
