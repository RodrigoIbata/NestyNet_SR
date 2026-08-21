# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_core.bridges import AddNode, ConstNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.training import pretrain_compound_leaf_from_teacher


class _RadialTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_input = None

    def forward(self, inputs):
        self.seen_input = inputs.detach().clone()
        assert inputs.shape[1] == 2
        return torch.sqrt(inputs[:, :1] ** 2 + inputs[:, 1:2] ** 2)


class _IdentityStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))

    def forward(self, inputs):
        return self.scale * inputs[:, :1]


def _difference(left, right):
    return AddNode(Var(left), MulNode(ConstNode(-1.0), Var(right)))


def test_teacher_pretrain_preserves_ordered_multi_compound_inputs():
    z0 = _difference(2, 3)
    z1 = _difference(0, 1)
    radial = PowNode(AddNode(PowNode(z0, 2.0), PowNode(z1, 2.0)), 0.5)
    x_data = torch.tensor(
        [
            [2.0, 0.5, 4.0, 1.0],
            [-1.0, 2.0, 0.5, -2.5],
            [3.0, -1.0, -2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    teacher = _RadialTeacher()
    student = _IdentityStudent()
    compound_model = object()

    returned = pretrain_compound_leaf_from_teacher(
        compound_model=compound_model,
        original_leaf=teacher,
        compound_leaf=student,
        z_ast=radial,
        x_data=x_data,
        original_var_idxs=[0, 1, 2, 3],
        device=torch.device("cpu"),
        dtype=torch.float64,
        original_input_asts=(z0, z1),
        epochs=1,
    )

    assert returned is compound_model
    assert teacher.seen_input is not None
    assert teacher.seen_input.shape == (3, 2)
    torch.testing.assert_close(
        teacher.seen_input,
        torch.column_stack((x_data[:, 2] - x_data[:, 3], x_data[:, 0] - x_data[:, 1])),
    )
    torch.testing.assert_close(student.scale.detach(), torch.ones((), dtype=torch.float64))


def test_teacher_pretrain_retains_legacy_single_compound_raw_extra_path():
    z0 = _difference(0, 1)
    x_data = torch.tensor(
        [
            [2.0, 0.5, 4.0],
            [-1.0, 2.0, 0.5],
            [3.0, -1.0, -2.0],
        ],
        dtype=torch.float64,
    )

    class _LegacyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_input = None

        def forward(self, inputs):
            self.seen_input = inputs.detach().clone()
            return inputs[:, :1] + inputs[:, 1:2]

    class _LegacyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, inputs):
            return inputs[:, :1] + self.bias

    teacher = _LegacyTeacher()
    student = _LegacyStudent()
    pretrain_compound_leaf_from_teacher(
        compound_model=object(),
        original_leaf=teacher,
        compound_leaf=student,
        z_ast=AddNode(z0, Var(2)),
        x_data=x_data,
        original_var_idxs=[0, 1, 2],
        device=torch.device("cpu"),
        dtype=torch.float64,
        original_compound_z_ast=z0,
        original_compound_extra_idxs=[2],
        epochs=1,
    )

    assert teacher.seen_input is not None
    torch.testing.assert_close(
        teacher.seen_input,
        torch.column_stack((x_data[:, 0] - x_data[:, 1], x_data[:, 2])),
    )


def test_teacher_pretrain_preserves_mixed_input_ast_order():
    z0 = _difference(2, 3)
    z1 = _difference(0, 1)
    x_data = torch.tensor(
        [
            [2.0, 0.5, 4.0, 1.0, 7.0],
            [-1.0, 2.0, 0.5, -2.5, -3.0],
            [3.0, -1.0, -2.0, 1.0, 0.25],
        ],
        dtype=torch.float64,
    )

    class _MixedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_input = None

        def forward(self, inputs):
            self.seen_input = inputs.detach().clone()
            assert inputs.shape[1] == 3
            return inputs.sum(dim=1, keepdim=True)

    teacher = _MixedTeacher()
    student = _IdentityStudent()
    combined = AddNode(AddNode(z0, Var(4)), z1)
    pretrain_compound_leaf_from_teacher(
        compound_model=object(),
        original_leaf=teacher,
        compound_leaf=student,
        z_ast=combined,
        x_data=x_data,
        original_var_idxs=[0, 1, 2, 3, 4],
        device=torch.device("cpu"),
        dtype=torch.float64,
        original_input_asts=(z0, Var(4), z1),
        epochs=1,
    )

    assert teacher.seen_input is not None
    torch.testing.assert_close(
        teacher.seen_input,
        torch.column_stack(
            (
                x_data[:, 2] - x_data[:, 3],
                x_data[:, 4],
                x_data[:, 0] - x_data[:, 1],
            )
        ),
    )
