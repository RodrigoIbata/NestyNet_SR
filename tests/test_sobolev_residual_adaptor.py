import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.adaptors.sobolev_residual import SobolevGradientAdaptor
from nestynet.optimizer.gauss_newton_ops import ResidualsModule


class ToyQuadraticProvider(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([0.7, -1.3, 0.2], dtype=torch.float64))

    def forward(self, x):
        return (
            self.theta[0] * x[:, :1]
            + self.theta[1] * x[:, 1:2].square()
            + self.theta[2]
        )

    def num_parameters(self):
        return 3

    def build_cache(self, data, **_kw):
        x, y, *_ = data
        f = self.forward(x)
        return {
            "x": x,
            "y": y,
            "f": f.detach(),
            "SegmentedModel": False,
            "S": 1,
            "Pseg": 3,
            "O": 1,
        }

    def residuals(self, cache, data=None, *, track_grad=False):
        x, y = (cache["x"], cache["y"]) if data is None else data[:2]
        with torch.set_grad_enabled(track_grad):
            r = y - self.forward(x)
        return r if track_grad else r.detach()

    def jvp(self, cache, v, out_dim=None):
        x = cache["x"]
        out = -(v[0] * x[:, 0] + v[1] * x[:, 1].square() + v[2])
        return out if out_dim is not None else out.unsqueeze(1)

    def vjp(self, cache, w, out_dim=None):
        x = cache["x"]
        ww = w.reshape(-1)
        return -torch.stack(
            [
                torch.sum(ww * x[:, 0]),
                torch.sum(ww * x[:, 1].square()),
                torch.sum(ww),
            ]
        ).reshape(1, -1)

    def grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        g = torch.stack(
            [
                self.theta[0].expand_as(x[:, 0]),
                2.0 * self.theta[1] * x[:, 1],
            ],
            dim=1,
        )
        return g if out_dim is not None else g.unsqueeze(1)

    def grad_grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        h = x.new_zeros(x.shape[0], 2, 2)
        h[:, 1, 1] = 2.0 * self.theta[1]
        return h if out_dim is not None else h.unsqueeze(1)

    def grad_jvp(self, cache, v, out_dim=None):
        x = cache["x"]
        out = torch.stack(
            [
                -torch.full_like(x[:, 0], v[0]),
                -(2.0 * v[1] * x[:, 1]),
            ],
            dim=1,
        )
        return out if out_dim is not None else out.unsqueeze(1)

    def grad_vjp(self, cache, v, out_dim=None):
        x = cache["x"]
        vv = v[:, 0, :] if v.ndim == 3 else v
        return -torch.stack(
            [
                torch.sum(vv[:, 0]),
                torch.sum(vv[:, 1] * 2.0 * x[:, 1]),
                torch.zeros((), dtype=x.dtype, device=x.device),
            ]
        ).reshape(1, -1)

    def jacobian(self, cache, out_dim=None):
        x = cache["x"]
        J = -torch.stack(
            [
                x[:, 0],
                x[:, 1].square(),
                torch.ones_like(x[:, 0]),
            ],
            dim=1,
        ).unsqueeze(1)
        return J

    def blocks(self):
        idx = torch.arange(3, dtype=torch.long)
        yield {
            "owner": id(self),
            "segments": None,
            "global_map": idx,
            "analytic_map": idx,
            "dimension_map": torch.zeros(3, dtype=torch.long),
            "param_idx": idx,
            "weight": 1.0,
        }


def test_sobolev_residuals_jvp_and_vjp():
    torch.set_default_dtype(torch.float64)
    x = torch.tensor(
        [
            [-0.4, 0.2],
            [0.1, -0.7],
            [0.8, 0.5],
            [1.2, -0.3],
        ],
        dtype=torch.float64,
    )
    base = ToyQuadraticProvider()
    y_value = base(x).detach() + 0.05
    y_grad = base.grad(x, out_dim=0).detach() + torch.tensor([0.01, -0.02])
    y_aug = torch.cat([y_value, y_grad], dim=1)

    adaptor = SobolevGradientAdaptor(
        base,
        axes=(0, 1),
        value_weight=2.0,
        grad_weight=0.5,
        value_scale=3.0,
        grad_scales=torch.tensor([5.0, 7.0], dtype=torch.float64),
    )
    cache = adaptor.build_cache((x, y_aug))
    r = adaptor.residuals(cache)

    expected_value = (y_value - base(x).detach()) * (2.0**0.5) * 3.0
    expected_grad = (y_grad - base.grad(x, out_dim=0).detach())
    expected_grad = expected_grad * (0.5**0.5) * torch.tensor([5.0, 7.0], dtype=torch.float64)
    assert torch.allclose(r, torch.cat([expected_value, expected_grad], dim=1))

    v = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    eps = 1e-6
    with torch.no_grad():
        theta0 = base.theta.detach().clone()
        base.theta.copy_(theta0 + eps * v)
    r_plus = adaptor.residuals(adaptor.build_cache((x, y_aug)))
    with torch.no_grad():
        base.theta.copy_(theta0 - eps * v)
    r_minus = adaptor.residuals(adaptor.build_cache((x, y_aug)))
    with torch.no_grad():
        base.theta.copy_(theta0)
    fd = (r_plus - r_minus) / (2.0 * eps)
    assert torch.allclose(adaptor.jvp(cache, v), fd, atol=2e-8, rtol=2e-8)

    w = torch.randn(x.shape[0], 3, dtype=torch.float64)
    rows = adaptor.vjp(cache, w)
    assert rows.shape == (3, 3)
    lhs = torch.sum(adaptor.jvp(cache, v) * w)
    rhs = torch.sum(rows.sum(dim=0) * v)
    assert torch.allclose(lhs, rhs, atol=2e-10, rtol=2e-10)

    for out_dim in range(3):
        lhs_k = torch.sum(adaptor.jvp(cache, v, out_dim=out_dim) * w[:, out_dim])
        rhs_k = torch.sum(adaptor.vjp(cache, w[:, out_dim], out_dim=out_dim).reshape(-1) * v)
        assert torch.allclose(lhs_k, rhs_k, atol=2e-10, rtol=2e-10)


def test_sobolev_residuals_module_uses_gradient_channels_in_jtj():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    x = torch.tensor(
        [
            [-0.4, 0.2],
            [0.1, -0.7],
            [0.8, 0.5],
            [1.2, -0.3],
        ],
        dtype=torch.float64,
        device=device,
    )
    base = ToyQuadraticProvider().to(device)
    y_value = base(x).detach() + 0.05
    y_grad = base.grad(x, out_dim=0).detach() + torch.tensor([0.01, -0.02], dtype=torch.float64)
    y_aug = torch.cat([y_value, y_grad], dim=1)

    adaptor = SobolevGradientAdaptor(
        base,
        axes=(0, 1),
        value_weight=1.7,
        grad_weight=0.9,
        value_scale=2.0,
        grad_scales=torch.tensor([3.0, 5.0], dtype=torch.float64),
    )
    loader = DataLoader(TensorDataset(x, y_aug), batch_size=x.shape[0], shuffle=False)
    rm = ResidualsModule([adaptor], dataloader=loader, device=device, normalization="sum")

    cache = adaptor.build_cache((x, y_aug))
    J = adaptor.jacobian(cache)
    P = int(adaptor.num_parameters())
    H_expected = J.reshape(-1, P).t().matmul(J.reshape(-1, P))
    H_by_channel = adaptor.dense(cache, out_dim=None)
    D_by_channel = adaptor.diag(cache, out_dim=None)
    assert H_by_channel.shape == (3, P, P)
    assert D_by_channel.shape == (3, P)
    assert torch.allclose(H_by_channel.sum(dim=0), H_expected, atol=1e-12, rtol=1e-12)

    idx = torch.arange(P, dtype=torch.long, device=device)
    dim_map = torch.full((P,), -1, dtype=torch.long, device=device)
    g_total, ops_total, separable, _amap = rm.accumulate(
        None,
        device,
        owner=id(adaptor),
        global_map=idx,
        analytic_map=idx,
        dimension_map=dim_map,
        need_ops=True,
    )

    assert not separable
    assert g_total.shape == (1, P)
    assert len(ops_total) == 1
    H_used = sum((op.dense() for op in ops_total[0]), start=torch.zeros_like(H_expected))
    assert torch.allclose(H_used, H_expected, atol=1e-10, rtol=1e-10)
