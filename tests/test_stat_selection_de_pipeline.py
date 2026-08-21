import json
from pathlib import Path
import numpy as np
import pytest

from nestynet_sr.stat_selection.de_pipeline import prepare_de_audit_plan, build_de_archive, run_de_statistical_selection


def _write(path, rate=1.0, amp=1.0):
    x=np.linspace(0,2,81); u=amp*np.exp(-rate*x)
    path.write_text('u,x\n'+'\n'.join(f'{a:.17g},{b:.17g}' for a,b in zip(u,x))+'\n')
    return str(path)


def _candidate(c, name):
    v={"order":1,"x_axis":0,"coefficients":[c],"term_asts_json":[{"type":"atom","kind":"u","var_idxs":[],"kwargs":{}}]}
    return {"engine":"stlsq","order":1,"canonical_equation":name,"validation_candidate":v}


def _report():
    a=_candidate(1.0,'du + u = 0'); b=_candidate(.7,'du + 0.7*u = 0')
    return {"de_discovery":{"selected":a,"proposal_slate":[a,b],"first_line":a}}


def test_whole_trajectory_firewall_and_duplicate_rejection(tmp_path):
    paths=[_write(tmp_path/f't{i}.csv',amp=1+i/10) for i in range(4)]
    plan=prepare_de_audit_plan(paths,reserve_trajectories=2)
    assert plan.search_paths==tuple(str(Path(p).resolve()) for p in paths[:2])
    assert plan.audit_paths==tuple(str(Path(p).resolve()) for p in paths[2:])
    clone=tmp_path/'clone.csv'; clone.write_bytes(Path(paths[0]).read_bytes())
    with pytest.raises(ValueError,match='byte-identical'):
        prepare_de_audit_plan([paths[0]],external_audit_paths=[str(clone)])


def test_archive_merges_duplicate_provenance():
    archive=build_de_archive(_report())
    assert len(archive)==2
    exact=[c for c in archive.candidates if 'du + u' in c.canonical_structure][0]
    assert len(exact.provenance)>=2


def test_end_to_end_de_confidence_front(tmp_path):
    search=[_write(tmp_path/f's{i}.csv',amp=1+i/10) for i in range(2)]
    audit=[_write(tmp_path/f'a{i}.csv',amp=.8+i/7) for i in range(4)]
    plan=prepare_de_audit_plan(search,external_audit_paths=audit)
    out=run_de_statistical_selection(_report(),plan,n_resamples=300,seed=7,delta=.002)
    assert out['audit']['design']['unit_kind']=='independent_trajectory'
    assert len(out['pareto']['candidate_ids'])==2
    risks=out['pareto']['risks']
    best=min(risks,key=risks.get)
    assert 'du + u' in next(c['canonical_structure'] for c in out['archive']['candidates'] if c['candidate_id']==best)
    assert out['campaign_action'] in {'stop','acquire_independent_trajectories_or_excitations'}
    json.dumps(out,allow_nan=False)


def test_failed_candidate_is_retained_but_ineligible(tmp_path):
    bad={"engine":"stlsq","order":1,"canonical_equation":"bad","validation_candidate":{"order":1,"x_axis":0,"coefficients":[1.0],"term_asts_json":[{"type":"atom","kind":"unsupported","kwargs":{}}]}}
    report=_report(); report['de_discovery']['proposal_slate'].append(bad)
    archive=build_de_archive(report)
    assert len(archive)==3

from nestynet_sr.stat_selection.uncertainty import CoherentLossDraws, structural_rediscovery_summary, calibration_smoke_test
from nestynet_sr.stat_selection.de_pipeline import _support_key


def test_coherent_loss_bundle_alignment_and_summary(tmp_path):
    x=np.arange(24,dtype=float).reshape(3,4,2)/10
    p=tmp_path/'draws.npz'
    np.savez(p,losses=x,candidate_ids=np.array(['c0','c1']),unit_ids=np.array(['u0','u1','u2','u3']),draw_ids=np.array(['d0','d1','d2']),metadata_json=np.array(json.dumps({'jet_order':2})))
    b=CoherentLossDraws.from_npz(p).aligned(['c1','c0'],['u3','u1'])
    assert b.losses.shape==(3,2,2)
    out=b.summary()
    assert out['coherent_draw_contract'] is True
    assert out['metadata']['jet_order']==2


def test_coherent_draws_are_embedded_in_de_certificate(tmp_path):
    search=[_write(tmp_path/f's{i}.csv',amp=1+i/10) for i in range(2)]
    audit=[_write(tmp_path/f'a{i}.csv',amp=.8+i/7) for i in range(3)]
    plan=prepare_de_audit_plan(search,external_audit_paths=audit)
    archive=build_de_archive(_report())
    losses=np.empty((4,3,len(archive.candidate_ids)))
    for b in range(4):
        losses[b,:,0]=.01+b*.0001
        losses[b,:,1]=.3+b*.001
    p=tmp_path/'coherent.npz'
    np.savez(p,losses=losses,candidate_ids=np.array(archive.candidate_ids),unit_ids=np.array([Path(v).stem for v in audit]))
    out=run_de_statistical_selection(_report(),plan,n_resamples=100,seed=2,coherent_loss_draws_path=p)
    assert out['coherent_surrogate_uncertainty']['n_draws']==4
    assert out['audit']['design']['loss_name']=='coherent_draw_mean_rollout_nrmse_squared'


def test_structural_rediscovery_and_calibration(tmp_path):
    paths=[]
    for i,c in enumerate((1.0,1.0,.7)):
        p=tmp_path/f'r{i}.json'; p.write_text(json.dumps({'de_discovery':{'selected':_candidate(c,str(c))}})); paths.append(p)
    out=structural_rediscovery_summary(paths,support_key_fn=_support_key)
    assert out['n_successful_runs']==3
    assert out['modal_frequency_over_all_runs']==1.0
    cal=calibration_smoke_test(n_repetitions=400,n_units=20,seed=3)
    assert .85<cal['null_coverage']<=1.0
    assert cal['dominance_power_at_minus_one_sd']>.8
