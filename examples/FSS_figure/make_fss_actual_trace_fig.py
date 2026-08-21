#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "xdg-cache"),
)

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nestynet_sr.sr_search.factorized_search.explorer import run_explorer_core
from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str, eval_node, build_pool
from nestynet_sr.sr_search.factorized_search.expr_mapping import eval_mapping

torch.set_num_threads(1)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "FSS_figure" / "actual_trace"
DEFAULT_SEED = 7
DEFAULT_N_FIT = 256
DEFAULT_N_PROBE = 512
DTYPE = torch.float64

def target(X):
    return 2*torch.sin(3*X[:,0:1]*X[:,1:2]+0.4)+0.1*X[:,2:3]

def make_data(n, g):
    X = torch.rand((n,3), generator=g, dtype=DTYPE)*2 - 1
    return X, target(X)

def jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k,v in x.items() if k not in {'_basis_transition'}}
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], str):
            try: return node_str(x)
            except Exception: pass
        return [jsonable(v) for v in x]
    try:
        if isinstance(x, (np.floating, np.integer)): return x.item()
    except Exception:
        pass
    return x

def run_actual_trace(seed: int = DEFAULT_SEED, n_fit: int = DEFAULT_N_FIT, n_probe: int = DEFAULT_N_PROBE):
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    x_fit, y_fit = make_data(int(n_fit), g)
    x_probe, y_probe = make_data(int(n_probe), g)
    events = []
    orig_update = ResidualBasinArchive.update
    def logged_update(self, key, mse, expr, z, mapping, raw_mse=None):
        before = self.best(1)[0].best_mse if self.d else float('inf')
        ok = orig_update(self, key, mse, expr, z, mapping, raw_mse)
        after = self.best(1)[0].best_mse if self.d else float('inf')
        events.append({
            'n_eval': int(getattr(self, 'n_eval', -1)),
            'expr': expr,
            'expr_str': node_str(expr),
            'mse': float(mse),
            'raw_mse': float(raw_mse) if raw_mse is not None else float(mse),
            'new_basin': bool(ok),
            'global_improve': bool(after < before*(1-1e-12)),
            'best_after': float(after),
            'mapping': copy.deepcopy(mapping),
            'residual_key': key,
        })
        return ok
    ResidualBasinArchive.update = logged_update
    try:
        arch = run_explorer_core(
            target, 3,
            n_iter=0, max_depth=5, poly_degree=5, seed=int(seed), dtype=DTYPE,
            x_fit_data=x_fit, y_fit_data=y_fit, x_probe_data=x_probe, y_probe_data=y_probe,
            brute_depth=2, brute_max_expressions=1000, early_stop_mse=1e-15,
            score_head_enable=True, score_head_vars_enable=False, score_head_untyped_enable=True,
            score_head_omp_enable=True, score_head_omp_max_terms=1, score_head_omp_topk_try=8,
            score_head_min_rel_improve=0.0,
            score_mapping_family_mode='full', brute_score_mapping_family_mode='full',
            score_prescreen_enable=False, score_pade_structural_enable=False,
            refine_enable=False, closure_search_enable=False,
            no_residual=True, no_crossover=True, degenerate_abort_enable=False,
            verbose=False, print_every=0,
        )
    finally:
        ResidualBasinArchive.update = orig_update
    return arch, events

def strip_head(mapping):
    return {k:v for k,v in mapping.items() if k not in {'_lin_head','_score_decomp','_score_ladder','_acceptance_basis'}}

def find_event(events):
    candidates = []
    for e in events:
        m = e.get('mapping') or {}
        h = m.get('_lin_head') if isinstance(m, dict) else None
        if e.get('expr_str') == '(x0*x1)' and isinstance(h, dict):
            terms = [node_str(t) for t in h.get('terms', [])]
            if 'x2' in terms:
                candidates.append(e)
    if not candidates:
        raise RuntimeError('Did not find the intended x0*x1 + x2 head event.')
    return min(candidates, key=lambda row: float(row.get('mse', float('inf'))))

def residual_pool_scores(expr, mapping, x_fit, y_fit):
    core_map = strip_head(mapping)
    s_fit = eval_node(expr, x_fit)
    core_fit = eval_mapping(s_fit, core_map)
    resid = (y_fit - core_fit).squeeze(-1)
    rows = []
    for i, t in enumerate(build_pool(3)):
        if isinstance(t, tuple) and t and t[0] == 'const':
            continue
        try:
            v = eval_node(t, x_fit).squeeze(-1)
        except Exception:
            continue
        if not torch.isfinite(v).all():
            continue
        norm = float((v*v).sum())
        if norm <= 1e-30:
            continue
        score = float(abs(torch.dot(v, resid)) / math.sqrt(norm))
        rows.append({'idx': i, 'expr': node_str(t), 'score': score})
    rows.sort(key=lambda r: r['score'], reverse=True)
    return rows

def finite_text(value, precision='.3g'):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(x):
        return 'n/a'
    return format(x, precision)

def make_figure(arch, events, output_dir: Path, seed: int = DEFAULT_SEED):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = find_event(events)
    x_probe, y_probe = arch.x_probe, arch.y_probe
    x_fit, y_fit = arch.x_fit, arch.y_fit
    expr = chosen['expr']
    mapping = chosen['mapping']
    core_map = strip_head(mapping)
    head = mapping['_lin_head']
    ladder = mapping.get('_score_ladder', {})
    core_mse = ladder.get('mapped', {}).get('probe_mse', mapping.get('_score_decomp', {}).get('mse_core'))
    final_mse = chosen['mse']
    carrier_mse = ladder.get('carrier', {}).get('probe_mse_identity')

    s_probe = eval_node(expr, x_probe)
    core_probe = eval_mapping(s_probe, core_map)
    residual_probe = (y_probe - core_probe).squeeze(-1)
    b = float(head['coeffs'][0]); a = float(head['coeffs'][1])
    head_probe = b + a*x_probe[:,2]
    final_resid = residual_probe - head_probe

    scores = residual_pool_scores(expr, mapping, x_fit, y_fit)[:8]
    def fmt_label(lab):
        repl = {
            '(x2*cos((x0*x1)))': r'$x_2\cos(x_0x_1)$',
            '(x2*cos((x1*x2)))': r'$x_2\cos(x_1x_2)$',
            '(x2*cos((x0*x2)))': r'$x_2\cos(x_0x_2)$',
            '(x2*cos(x0))': r'$x_2\cos(x_0)$',
            '(x2*cos(x1))': r'$x_2\cos(x_1)$',
            '((x2*x2)*x2)': r'$x_2^3$',
            'sin(x2)': r'$\sin(x_2)$',
            'x2': r'$x_2$',
        }
        return repl.get(lab, lab)
    top_raw = [r['expr'] for r in scores][::-1]
    top_labels = [fmt_label(r) for r in top_raw]
    top_scores = [r['score'] for r in scores][::-1]

    xs = np.array([e['n_eval'] for e in events], dtype=float)
    mses = np.array([e['mse'] for e in events], dtype=float)
    best = np.minimum.accumulate(mses)
    basis = [str((e.get('mapping') or {}).get('_acceptance_basis', '')) for e in events]
    is_head = np.array(['head_augmented' in b0 for b0 in basis])
    chosen_n = chosen['n_eval']

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.2,
        'axes.titlesize': 9.0,
        'axes.labelsize': 8.2,
        'xtick.labelsize': 7.3,
        'ytick.labelsize': 7.3,
        'legend.fontsize': 7.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    c_bg = '#fbfaf7'; c_grid = '#d9d3c7'; c_trace = '#283845'; c_dot = '#7c8792'
    c_head = '#b2533e'; c_map = '#227c9d'; c_gold = '#d39b2a'
    fig = plt.figure(figsize=(7.25, 5.7), dpi=220, facecolor=c_bg)
    gs = GridSpec(2, 2, figure=fig, left=0.085, right=0.982, top=0.875, bottom=0.125, wspace=0.28, hspace=0.43)
    fig.text(0.085, 0.965, 'FSS steering in a real NestyNet_SR trace', weight='bold', fontsize=12.0, color='#1f2529')
    fig.text(0.085, 0.932, rf'target samples: $y=2\sin(3x_0x_1+0.4)+0.1x_2$  |  seed {int(seed)}, brute depth 2, residual OMP head enabled', fontsize=8.3, color='#3e464c')

    ax = fig.add_subplot(gs[0,0], facecolor='white')
    ax.scatter(xs[~is_head], mses[~is_head], s=22, color=c_dot, alpha=0.75, edgecolor='none', label='scored candidate')
    ax.scatter(xs[is_head], mses[is_head], s=28, color=c_head, alpha=0.85, edgecolor='white', linewidth=0.35, label='head-augmented score')
    ax.plot(xs, best, color=c_trace, lw=1.7, label='archive best')
    ax.scatter([chosen_n], [final_mse], s=95, marker='*', color=c_gold, edgecolor='#47360f', linewidth=0.5, zorder=5)
    ax.set_yscale('log')
    ax.set_ylim(5e-16, 4)
    ax.set_xlim(0.2, max(xs)+0.8)
    ax.set_xlabel('archive update')
    ax.set_ylabel('probe MSE')
    ax.set_title('A. Archive trace', loc='left', pad=5)
    ax.grid(True, which='both', color=c_grid, lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc='upper right', handlelength=1.8)
    ax.annotate(r'$x_0x_1$ accepted'+'\n'+r'$+\;0.1000x_2$ head', xy=(chosen_n, final_mse), xytext=(14.0, 1.2e-7),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#333333'), ha='left', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', fc='#fff7df', ec='#d1a842', lw=0.6))

    ax = fig.add_subplot(gs[0,1], facecolor='white')
    s_np = s_probe.squeeze(-1).detach().cpu().numpy(); y_np = y_probe.squeeze(-1).detach().cpu().numpy()
    ax.scatter(s_np, y_np, s=9, color='#8d99a6', alpha=0.36, edgecolor='none', rasterized=True)
    grid = torch.linspace(float(s_probe.min()), float(s_probe.max()), 500, dtype=DTYPE).unsqueeze(-1)
    ygrid = eval_mapping(grid, core_map).squeeze(-1).detach().cpu().numpy()
    ax.plot(grid.squeeze(-1).detach().cpu().numpy(), ygrid, color=c_map, lw=2.0, label=r'learned $M_\theta(s)$')
    ax.set_xlabel(r'carrier $s=x_0x_1$')
    ax.set_ylabel(r'target $y$')
    ax.set_title('B. Partial credit for the carrier', loc='left', pad=5)
    ax.grid(True, color=c_grid, lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc='lower right')
    ax.text(0.03, 0.95, f'carrier identity MSE = {finite_text(carrier_mse)}\nouter-map MSE = {finite_text(core_mse)}', transform=ax.transAxes, ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.28', fc='#eef7fb', ec='#79aeca', lw=0.6))

    ax = fig.add_subplot(gs[1,0], facecolor='white')
    x2 = x_probe[:,2].detach().cpu().numpy(); r_np = residual_probe.detach().cpu().numpy()
    ax.scatter(x2, r_np, s=10, color='#7b8794', alpha=0.45, edgecolor='none', rasterized=True)
    xx = np.linspace(float(x2.min()), float(x2.max()), 200)
    ax.plot(xx, b + a*xx, color=c_head, lw=2.0, label=rf'head $h={b:+.3g}{a:+.4f}x_2$')
    ax.axhline(0, color='#222222', lw=0.6, alpha=0.35)
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'residual $r$')
    ax.set_title('C. Residual reveals the missing term', loc='left', pad=5)
    ax.grid(True, color=c_grid, lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc='upper left')
    ax.text(0.04, 0.08, f'after head MSE = {final_mse:.2e}\nresidual std after head = {float(final_resid.std()):.2e}', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.28', fc='#fff1ee', ec='#cc8b7a', lw=0.6))

    ax = fig.add_subplot(gs[1,1], facecolor='white')
    y_pos = np.arange(len(top_labels))
    colors = [c_head if raw == 'x2' else '#9aa3aa' for raw in top_raw]
    ax.barh(y_pos, top_scores, color=colors, edgecolor='white', linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels)
    ax.set_xlabel(r'residual score $|r^T\phi|/\|\phi\|$')
    ax.set_title('D. OMP residual scores', loc='left', pad=5)
    ax.grid(True, axis='x', color=c_grid, lw=0.45, alpha=0.65)
    ax.text(0.98, 0.08, 'selected term: '+r'$x_2$'+'\n'+f'coefficient = {a:.7f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.28', fc='#fff7df', ec='#d1a842', lw=0.6))
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

    for label, axis in zip(['A','B','C','D'], fig.axes):
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('#8f969b')

    fig.text(0.085, 0.035, 'Instrumented NestyNet_SR FSS run: plotted losses, residuals, scores, and coefficients are real trace data.', fontsize=7.2, color='#4b5358')

    png = output_dir/'fss_actual_trace_single_equation.png'
    svg = output_dir/'fss_actual_trace_single_equation.svg'
    pdf = output_dir/'fss_actual_trace_single_equation.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'target': '2*sin(3*x0*x1 + 0.4) + 0.1*x2',
        'seed': int(seed),
        'n_fit': int(arch.x_fit.shape[0]),
        'n_probe': int(arch.x_probe.shape[0]),
        'chosen_event': {
            'n_eval': chosen['n_eval'],
            'expr': chosen['expr_str'],
            'mse': final_mse,
            'mapping_kind': mapping.get('kind'),
            'head_terms': [node_str(t) for t in head.get('terms', [])],
            'head_coeffs': [float(v) for v in head.get('coeffs', [])],
            'carrier_identity_mse': float(carrier_mse) if carrier_mse is not None else None,
            'outer_mapping_mse': float(core_mse) if core_mse is not None else None,
            'final_head_mse': float(final_mse),
            'top_residual_scores': scores,
        },
        'events': [{k: (jsonable(v) if k != 'expr' else node_str(v)) for k,v in e.items() if k != 'mapping'} | {'mapping': jsonable(e.get('mapping'))} for e in events],
    }
    summary_path = output_dir/'fss_actual_trace_single_equation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return png, svg, pdf, summary_path, summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an instrumented FSS trace figure from a real NestyNet_SR search run.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--n-fit', type=int, default=DEFAULT_N_FIT)
    parser.add_argument('--n-probe', type=int, default=DEFAULT_N_PROBE)
    args = parser.parse_args()
    arch, events = run_actual_trace(seed=args.seed, n_fit=args.n_fit, n_probe=args.n_probe)
    png, svg, pdf, summary_path, summary = make_figure(arch, events, args.output_dir, seed=args.seed)
    print(json.dumps({
        'png': str(png), 'svg': str(svg), 'pdf': str(pdf),
        'summary': str(summary_path),
        'chosen_event': summary['chosen_event'],
        'n_events': len(events),
        'n_basins': len(arch.d),
    }, indent=2))
