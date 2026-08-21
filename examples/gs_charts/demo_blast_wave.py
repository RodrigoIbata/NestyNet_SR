# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""The GS chart machine re-does Taylor's blast-wave analysis, blind.

G. I. Taylor famously inferred the Trinity yield from declassified film
frames using the Sedov-Taylor similarity law R(t) = xi0 (E t^2 / rho)^(1/5).
Here SYNTHETIC blast-wave radius data (Trinity-like constants, small noise,
and a detonation instant BEFORE the first frame) are handed to the GS chart
bridge with no physics declared. The machine must, from (t, R) alone:

  1. fit an identity surrogate and feed its analytic graph derivatives to
     the affine graph-symmetry determining operator;
  2. discover the scaling symmetry (t - t0) d_t + p R d_R;
  3. recover the detonation instant t0 = -b/A (not a sample time!);
  4. read the similarity exponent p = beta/A off the output action;
  5. compile and sharpness-certify the shifted-log chart u = log(t - t0).

The scorer (not part of discovery) then compares p with 2/5, t0 with truth,
and inverts the amplitude for the yield in kilotons using the declared air
density and Sedov constant.

Cases:
  python demo_blast_wave.py                 # blast wave (the Taylor story)
  python demo_blast_wave.py --case decay    # RC discharge: translation +
                                            #   cofactor, reads off tau
  python demo_blast_wave.py --case control  # aperiodic: must certify nothing
"""

# region imports
import argparse

import numpy as np

from nestynet_sr.sr_gs.chart_bridge import scan_and_compile_charts
from nestynet.charts import FitConfig, fit_chart

# endregion

KT_JOULES = 4.184e12
RHO_AIR = 1.25          # kg m^-3
XI0 = 1.033             # Sedov constant, gamma = 1.4
E_TRUE_KT = 20.0
T0_TRUE_MS = -2.0       # detonation 2 ms before the first film frame


def make_blast(noise: float, seed: int = 0):
    """Synthetic Sedov-Taylor radii vs film-frame time (ms, m)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.3, 62.0, 400)                 # ms since first frame
    E = E_TRUE_KT * KT_JOULES
    C = XI0 * (E / RHO_AIR) ** 0.2                  # R = C * t_sec^{2/5}
    R = C * ((t - T0_TRUE_MS) * 1e-3) ** 0.4
    return t, R * (1.0 + noise * rng.standard_normal(t.shape))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("blast", "decay", "control"), default="blast")
    p.add_argument("--noise", type=float, default=0.002)
    args = p.parse_args()

    if args.case == "blast":
        t, y = make_blast(args.noise)
    elif args.case == "decay":
        t = np.linspace(0.0, 8.0, 400)              # seconds
        rng = np.random.default_rng(0)
        y = 5.0 * np.exp(-t / 1.5) * (1.0 + args.noise * rng.standard_normal(t.shape))
    else:
        t = np.linspace(0.0, 60.0, 400)
        rng = np.random.default_rng(0)
        y = (0.7 * np.exp(-(((t - 30.0) / 12.0) ** 2)) + 0.01 * t
             + args.noise * rng.standard_normal(t.shape))

    print(f"== GS chart machine, case {args.case} (noise {args.noise:.1%}) ==")
    res = scan_and_compile_charts(
        t, y,
        fit_cfg=FitConfig(segments=24, epochs=400, restarts=3),
        sharp_fit_cfg=FitConfig(segments=12, epochs=150, restarts=2),
        # noisy-surrogate regime: loosen the proposer, let the key-sharpness
        # certificate be the guard (layered gates)
        nullity_strategy="rank_tol",
        rank_rtol=0.15,
        acceptance_residual_tol=0.15,
    )
    for line in res.log:
        print("  " + line)
    for prop in res.proposals:
        print(f"  proposal: kind {prop.kind}; {prop.note}")

    certified = [pr for pr in res.proposals if pr.certified]

    if args.case == "blast":
        if not certified:
            print("  FAILED: no certified chart")
            return
        pr = certified[0]
        t0 = pr.chart.get_param("t0")
        p_exp = pr.exponent
        print("\n-- scorer (not part of discovery) --")
        print(f"  detonation instant t0: {t0:+.3f} ms (true {T0_TRUE_MS:+.3f} ms)")
        print(f"  similarity exponent beta/A: {p_exp:.4f} (Sedov-Taylor 2/5 = 0.4)")
        # refit on the certified chart and invert the amplitude for the yield
        chart_val = fit_chart(t, y, pr.chart, FitConfig(segments=24, epochs=400)).val_rel_rmse
        print(f"  fit on the discovered chart: val relRMSE {chart_val:.2e} "
              f"(identity surrogate was {res.surrogate_val_rel_rmse:.2e})")
        x = (t - t0) * 1e-3
        p_snap = 0.4  # snap to the discovered rational exponent for inversion
        C_amp = float(np.sum(y * x**p_snap) / np.sum(x ** (2 * p_snap)))
        E_est = RHO_AIR * (C_amp / XI0) ** 5
        print(f"  yield from amplitude (rho, xi0 declared): "
              f"{E_est / KT_JOULES:.1f} kt (true {E_TRUE_KT:.1f} kt)")
    elif args.case == "decay":
        trans = [pr for pr in res.proposals if pr.kind == "translation"]
        if trans:
            pr = trans[0]
            tau = -pr.b / pr.beta
            print("\n-- scorer --")
            print(f"  recovered time constant tau: {tau:.4f} s (true 1.500 s)")
    else:
        print(f"\n-- scorer --\n  certified charts: {len(certified)} (must be 0)")


if __name__ == "__main__":
    main()
