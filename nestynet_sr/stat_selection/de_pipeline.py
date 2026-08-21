# SPDX-License-Identifier: MPL-2.0
"""Trajectory-level statistical selection for discovered ODE candidates.

Search may use any heuristic it likes.  This adapter freezes the serialized
candidate slate before opening held-out trajectory files, evaluates every
candidate on every trajectory with the same rollout contract, and constructs
paired simultaneous confidence Pareto fronts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .archive import CandidateArchive
from .audit import AuditDesign, LossAudit
from .certificate import build_certificate
from .complexity import ComplexityVector
from .pareto import confidence_pareto, bootstrap_front_inclusion_frequencies
from .uncertainty import CoherentLossDraws, calibration_smoke_test, structural_rediscovery_summary
from nestynet_sr.sr_de.de_validation import candidate_to_rhs_callable, validate_by_simulation


@dataclass(frozen=True)
class DEAuditPlan:
    search_paths: tuple[str, ...]
    audit_paths: tuple[str, ...]
    search_sha256: tuple[str, ...]
    audit_sha256: tuple[str, ...]
    source_kind: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)

    @property
    def fingerprint(self) -> str:
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def revalidate(self) -> None:
        for path, expected in zip(self.search_paths,self.search_sha256):
            if _sha256(path)!=expected: raise RuntimeError(f"search trajectory changed after firewall creation: {path}")
        for path, expected in zip(self.audit_paths,self.audit_sha256):
            if _sha256(path)!=expected: raise RuntimeError(f"audit trajectory changed after firewall creation: {path}")


def prepare_de_audit_plan(filepaths: Sequence[str], *, external_audit_paths: Sequence[str]|None=None,
                          reserve_trajectories: int=1) -> DEAuditPlan:
    paths=tuple(str(Path(p).resolve()) for p in filepaths)
    if not paths: raise ValueError("DE statistical selection requires at least one trajectory")
    for p in paths:
        if not Path(p).is_file(): raise FileNotFoundError(p)
    external=tuple(str(Path(p).resolve()) for p in (external_audit_paths or ()))
    if external:
        for p in external:
            if not Path(p).is_file(): raise FileNotFoundError(p)
        search=paths; audit=external; kind="external_trajectories"
    else:
        n=int(reserve_trajectories)
        if n<2:
            raise ValueError("DE confidence selection needs at least two independent audit trajectories; use --stat-audit-filepaths or --stat-audit-trajectories >=2")
        if len(paths)<=n:
            raise ValueError("whole-trajectory firewall would leave no search trajectories")
        search=paths[:-n]; audit=paths[-n:]; kind="internal_whole_trajectory_tail"
    if set(search)&set(audit): raise ValueError("search and audit trajectory paths must be disjoint")
    sh=tuple(_sha256(p) for p in search); ah=tuple(_sha256(p) for p in audit)
    if set(sh)&set(ah): raise ValueError("an audit trajectory is byte-identical to a search trajectory")
    return DEAuditPlan(search, audit, sh, ah, kind)


def _sha256(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def _walk_candidates(report: Mapping[str,Any]):
    def visit(obj, source):
        if isinstance(obj, Mapping):
            if (obj.get("engine") is not None or obj.get("canonical_equation")) and ("validation_candidate" in obj or "term_asts_json" in obj) and obj.get("order") in (1,2,"1","2"):
                yield dict(obj),source
            for k,v in obj.items():
                if k in {"diagnostics","config","metadata","statistical_selection"}: continue
                yield from visit(v,f"{source}.{k}")
        elif isinstance(obj,list):
            for i,v in enumerate(obj): yield from visit(v,f"{source}[{i}]")
    yield from visit(report.get("de_discovery",report),"de_discovery")


def _canonical(candidate: Mapping[str,Any]) -> str:
    eq=str(candidate.get("canonical_equation") or "").strip()
    if eq: return " ".join(eq.split())
    payload=candidate.get("validation_candidate") or {
        "order":candidate.get("order"),"coefficients":candidate.get("coefficients"),
        "term_asts_json":candidate.get("term_asts_json")}
    return json.dumps(payload,sort_keys=True,separators=(",",":"),default=str)


def _ast_nodes(value: Any) -> int:
    if isinstance(value,Mapping): return 1+sum(_ast_nodes(v) for v in value.values())
    if isinstance(value,list): return sum(_ast_nodes(v) for v in value)
    return 0


def build_de_archive(report: Mapping[str,Any], *, max_candidates: int=256) -> CandidateArchive:
    archive=CandidateArchive(archive_label="de-global-candidate-archive",metadata={"adapter":"trajectory-rollout-v1"})
    rows=[]
    for cand,source in _walk_candidates(report):
        try:
            candidate_to_rhs_callable(cand,engine=cand.get("engine"))
        except Exception:
            continue
        structure=_canonical(cand); order=int(cand.get("order",1))
        validation=cand.get("validation_candidate") if isinstance(cand.get("validation_candidate"),Mapping) else cand
        coeffs=list(validation.get("coefficients",[]) or [])
        terms=list(validation.get("term_asts_json",[]) or [])
        active=sum(abs(float(c))>0 for c in coeffs) if coeffs else max(1,len(terms))
        nodes=max(1,_ast_nodes(terms))
        rows.append((order,active,nodes,structure,cand,source))
    rows.sort(key=lambda r:(r[0],r[1],r[2],r[3],r[5]))
    for order,active,nodes,structure,cand,source in rows[:max(1,int(max_candidates))]:
        archive.add_structure(structure,ComplexityVector.from_mapping({"differential_order":order,"active_terms":active,"ast_nodes":nodes}),
            grammar_version="nestynet-de-serialized-v1",metadata={"candidate":cand,"support_key":_support_key(cand)},provenance=[{"source":source,"engine":cand.get("engine")}])
    if not len(archive): raise ValueError("no validation-ready DE candidates were found in the serialized proposal slate")
    return archive.freeze()


def _support_key(c: Mapping[str,Any]) -> str:
    v=c.get("validation_candidate") if isinstance(c.get("validation_candidate"),Mapping) else c
    return json.dumps({"order":v.get("order",c.get("order")),"terms":v.get("term_asts_json",c.get("term_asts_json"))},sort_keys=True,default=str)


def _runs(paths: Sequence[str]):
    out=[]
    for p in paths:
        arr=np.genfromtxt(p,delimiter=",",names=True,dtype=float,encoding=None)
        names=list(arr.dtype.names or ())
        if len(names)<2: raise ValueError(f"trajectory CSV needs at least coordinate and state columns: {p}")
        u=np.asarray(arr[names[0]],float); x=np.asarray(arr[names[1]],float)
        if x.size<3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)): raise ValueError(f"invalid audit trajectory: {p}")
        v0=float((u[1]-u[0])/(x[1]-x[0])) if x[1]!=x[0] else 0.0
        out.append(SimpleNamespace(csv_path=Path(p),traj_id=Path(p).stem,u0=float(u[0]),v0=v0))
    return out


def run_de_statistical_selection(report: Mapping[str,Any], plan: DEAuditPlan, *, alpha=.05,delta=.01,
                                 n_resamples=4000,seed=12345,multiplier="normal",failure_loss=100.0,
                                 max_candidates=256,rollout_window_fraction=1.0,rollout_max_span=None,
                                 traj_time_budget_s=20.0, coherent_loss_draws_path=None,
                                 rediscovery_report_paths: Sequence[str]|None=None,
                                 calibration_repetitions: int=0) -> dict[str,Any]:
    plan.revalidate(); archive=build_de_archive(report,max_candidates=max_candidates); plan.revalidate()
    runs=_runs(plan.audit_paths); losses=np.full((len(runs),len(archive.candidates)),float(failure_loss))
    failures={}
    for j,spec in enumerate(archive.candidates):
        cand=dict(spec.metadata["candidate"])
        try:
            order,rhs=candidate_to_rhs_callable(cand,engine=cand.get("engine"))
            _,_,scores=validate_by_simulation(runs,rhs_fn=rhs,order=order,pass_nrmse=float("inf"),partial_nrmse=float("inf"),
                traj_time_budget_s=traj_time_budget_s,rollout_window_fraction=rollout_window_fraction,rollout_max_span=rollout_max_span)
            for i,row in enumerate(scores):
                val=row.get("nrmse",row.get("rollout_nrmse"))
                if row.get("status") not in {"ERROR","FAIL"} and val is not None and math.isfinite(float(val)):
                    losses[i,j]=min(float(failure_loss),float(val)**2)
                else: failures.setdefault(spec.candidate_id,[]).append(row)
        except Exception as exc:
            failures[spec.candidate_id]=[{"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}]
    coherent_summary=None
    if coherent_loss_draws_path:
        bundle=CoherentLossDraws.from_npz(coherent_loss_draws_path).aligned(archive.candidate_ids,[Path(p).stem for p in plan.audit_paths])
        coherent_summary=bundle.summary()
        losses=np.asarray(coherent_summary["mean_loss_by_unit_candidate"],float)
        coherent_summary["bundle_path"]=str(coherent_loss_draws_path)
    eligible=[c.candidate_id for c in archive.candidates if c.candidate_id not in failures]
    design=AuditDesign(loss_name="coherent_draw_mean_rollout_nrmse_squared" if coherent_summary else "full_trajectory_rollout_nrmse_squared",unit_kind="independent_trajectory",
        fit_protocol="candidate frozen before audit; no audit refit",evaluation_domain={"audit_paths":[Path(p).name for p in plan.audit_paths]},
        sampling_assumptions=("audit trajectories are independent experimental units","candidate archive was frozen before audit responses were evaluated","all candidates share one solver and failure penalty"))
    failure_mask=np.zeros_like(losses,dtype=bool)
    for j,c in enumerate(archive.candidates):
        if c.candidate_id in failures: failure_mask[:,j]=True
    audit=LossAudit.from_matrix(candidate_ids=archive.candidate_ids,unit_ids=[Path(p).stem for p in plan.audit_paths],design=design,
        losses=losses,failure_mask=failure_mask,nonfinite="penalize",failure_loss=float(failure_loss),archive=archive)
    pareto=confidence_pareto(audit,archive,alpha=alpha,delta=delta,n_resamples=n_resamples,seed=seed,multiplier=multiplier,eligible_candidate_ids=eligible)
    frequencies=bootstrap_front_inclusion_frequencies(audit,archive,delta=delta,n_resamples=max(200,min(n_resamples,2000)),seed=seed+1,eligible_candidate_ids=eligible)
    cert=build_certificate(archive,audit,pareto).to_dict()
    cert["front_inclusion_frequencies"]=frequencies
    cert["firewall"]=plan.to_dict()|{"fingerprint":plan.fingerprint}
    cert["failures_by_candidate"]=failures
    cert["support_classes"]=_classes(archive,"support_key")
    if coherent_summary is not None:
        coherent_summary["candidate_ids"]=list(archive.candidate_ids)
        coherent_summary["unit_ids"]=[Path(p).stem for p in plan.audit_paths]
        cert["coherent_surrogate_uncertainty"]=coherent_summary
    if rediscovery_report_paths:
        cert["structural_rediscovery"]=structural_rediscovery_summary(rediscovery_report_paths,support_key_fn=_support_key)
    if int(calibration_repetitions)>0:
        cert["calibration_smoke_test"]=calibration_smoke_test(n_repetitions=int(calibration_repetitions),n_units=max(3,len(runs)),alpha=alpha,seed=seed+2)
    cert["campaign_action"]=_campaign_action(pareto,failures,len(runs))
    return cert


def _classes(archive: CandidateArchive,key: str):
    groups={}
    for c in archive.candidates: groups.setdefault(str(c.metadata.get(key,"")),[]).append(c.candidate_id)
    return [v for _,v in sorted(groups.items())]


def _campaign_action(result,failures,n_units):
    if n_units<2: return "acquire_independent_trajectories_or_excitations"
    if failures and len(failures)==len(result.candidate_ids): return "escalate_symbolic_search"
    if len(result.practical_front)>1: return "acquire_independent_trajectories_or_excitations"
    return "stop"
