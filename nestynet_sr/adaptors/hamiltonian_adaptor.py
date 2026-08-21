# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch


class HamiltonianResidualAdaptor(torch.nn.Module):
    def __init__(self, H, q_idx, p_idx):
        super().__init__()
        self.H = H
        self.q_idx = torch.as_tensor(q_idx, dtype=torch.long)
        self.p_idx = torch.as_tensor(p_idx, dtype=torch.long)

    def blocks(self, *a, **k):
        return self.H.blocks(*a, **k)

    def build_cache(self, data, **kw):
        x, y, *_ = data
        y0 = x.new_zeros((x.size(0), 1))
        Hc = self.H.build_cache((x, y0), **kw)
        return {"x": x, "y": y, "Hc": Hc}

    def residuals(self, cache, data=None, track_grad=False):
        _x, y = cache["x"], cache["y"]
        g = self.H.grad(cache["Hc"]).squeeze(1)
        qhat = g[:, self.p_idx]
        phat = -g[:, self.q_idx]
        fhat = torch.cat([qhat, phat], dim=1)
        r = y - fhat
        return r if track_grad else r.detach()

    def jvp(self, cache, v, out_dim=None):
        Gjv = self.H.grad_jvp(cache["Hc"], v).squeeze(1)
        j1 = Gjv[:, self.p_idx]
        j2 = -Gjv[:, self.q_idx]
        return torch.cat([j1, j2], dim=1)

    def vjp(self, cache, w, out_dim=None):
        if w.ndim == 1:
            w = w.unsqueeze(1)
        B, Nx = cache["x"].size(0), cache["x"].size(1)
        n = int(self.q_idx.numel())
        vgrad = w.new_zeros((B, 1, Nx))
        vgrad[:, 0, self.p_idx] = w[:, :n]
        vgrad[:, 0, self.q_idx] = -w[:, n:]
        return self.H.grad_vjp(cache["Hc"], vgrad, out_dim=out_dim)
