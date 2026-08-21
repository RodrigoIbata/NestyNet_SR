# SPDX-License-Identifier: MPL-2.0
"""Coherent uncertainty and structural-stability adapters.

The statistical-selection core deliberately does not know how a NestyNet
posterior/sister draw is produced.  This module defines the interchange
contract used by the DE pipeline: one draw must contain a *coherent* set of
losses for every frozen candidate and every audit unit.  Values and all
required derivative jets therefore come from the same surrogate draw.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CoherentLossDraws:
    losses: np.ndarray
    candidate_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self):
        x=np.asarray(self.losses,float)
        if x.ndim!=3: raise ValueError("coherent losses must have shape (draw, unit, candidate)")
        if x.shape!=(len(self.draw_ids),len(self.unit_ids),len(self.candidate_ids)):
            raise ValueError("coherent-loss dimensions do not match identifiers")
        if x.shape[0]<2: raise ValueError("at least two coherent surrogate draws are required")
        if not np.all(np.isfinite(x)): raise ValueError("coherent losses must be finite; encode failures with the declared failure loss")
        object.__setattr__(self,"losses",x)

    @classmethod
    def from_npz(cls,path: str|Path) -> "CoherentLossDraws":
        with np.load(path,allow_pickle=False) as z:
            required={"losses","candidate_ids","unit_ids"}
            missing=required-set(z.files)
            if missing: raise ValueError(f"coherent-loss bundle missing arrays: {sorted(missing)}")
            losses=np.asarray(z["losses"],float)
            c=tuple(str(v) for v in z["candidate_ids"].tolist())
            u=tuple(str(v) for v in z["unit_ids"].tolist())
            d=tuple(str(v) for v in (z["draw_ids"].tolist() if "draw_ids" in z.files else range(losses.shape[0])))
            meta={}
            if "metadata_json" in z.files:
                raw=z["metadata_json"].tolist()
                if isinstance(raw,bytes): raw=raw.decode()
                meta=json.loads(str(raw))
        return cls(losses,c,u,d,meta)

    def aligned(self,candidate_ids: Sequence[str],unit_ids: Sequence[str]) -> "CoherentLossDraws":
        ci={v:i for i,v in enumerate(self.candidate_ids)}; ui={v:i for i,v in enumerate(self.unit_ids)}
        missing_c=[v for v in candidate_ids if v not in ci]; missing_u=[v for v in unit_ids if v not in ui]
        if missing_c or missing_u: raise ValueError(f"coherent-loss bundle is not aligned; missing candidates={missing_c}, units={missing_u}")
        x=self.losses[:,[ui[v] for v in unit_ids],:][:,:,[ci[v] for v in candidate_ids]]
        return CoherentLossDraws(x,tuple(candidate_ids),tuple(unit_ids),self.draw_ids,self.metadata)

    def summary(self) -> dict[str,Any]:
        unit_draw_mean=self.losses.mean(axis=0)
        draw_risk=self.losses.mean(axis=1)
        q=np.quantile(draw_risk,[.025,.5,.975],axis=0)
        return {
            "n_draws":len(self.draw_ids),
            "coherent_draw_contract":True,
            "mean_loss_by_unit_candidate":unit_draw_mean.tolist(),
            "risk_by_candidate":draw_risk.mean(axis=0).tolist(),
            "risk_draw_sd_by_candidate":draw_risk.std(axis=0,ddof=1).tolist(),
            "risk_draw_quantiles_025_50_975_by_candidate":q.T.tolist(),
            "metadata":dict(self.metadata),
        }


def structural_rediscovery_summary(report_paths: Sequence[str|Path], *, support_key_fn) -> dict[str,Any]:
    """Summarize full-pipeline rediscovery reports without treating reruns as IID rows."""
    counts: dict[str,int]={}; selected=[]; failures=[]
    for p in report_paths:
        try:
            report=json.loads(Path(p).read_text())
            node=report.get("de_discovery",report)
            candidate=node.get("selected") or node.get("best") or node.get("result") or node
            if not isinstance(candidate,Mapping): raise ValueError("no mapping-valued selected candidate")
            key=str(support_key_fn(candidate)); counts[key]=counts.get(key,0)+1; selected.append(key)
        except Exception as exc:
            failures.append({"path":str(p),"error":f"{type(exc).__name__}: {exc}"})
    n=len(report_paths); ok=len(selected)
    rows=[{"support_key":k,"count":v,"frequency_over_successful_runs":v/ok if ok else 0.0,
           "frequency_over_all_runs":v/n if n else 0.0} for k,v in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))]
    return {"n_requested_runs":n,"n_successful_runs":ok,"failures":failures,"support_frequencies":rows,
            "modal_support_key":rows[0]["support_key"] if rows else None,
            "modal_frequency_over_all_runs":rows[0]["frequency_over_all_runs"] if rows else 0.0}


def calibration_smoke_test(*, n_repetitions=2000,n_units=12,alpha=.05,seed=12345) -> dict[str,float]:
    """Fast deterministic calibration check for paired Gaussian mean differences."""
    rng=np.random.default_rng(seed); z=1.959963984540054
    covered=0; power=0
    for _ in range(int(n_repetitions)):
        d=rng.normal(0,1,int(n_units)); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n_units); covered += (m-z*se<=0<=m+z*se)
        d=rng.normal(-1,1,int(n_units)); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n_units); power += (m+z*se<0)
    return {"null_coverage":covered/n_repetitions,"dominance_power_at_minus_one_sd":power/n_repetitions,
            "n_repetitions":int(n_repetitions),"n_units":int(n_units),"alpha":float(alpha)}
