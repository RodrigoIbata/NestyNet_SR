# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""The GS chart machine dates the explosion of SN 1993J from real VLBI radii.

SN 1993J in M81 is the textbook self-similar radio supernova: a decade of
VLBI imaging measured its angular radius growing as R ~ t^m, with
m = 0.845 +/- 0.005 before day ~1500 and 0.788 +/- 0.015 after
(Marcaide et al. 2009, A&A 505, 927). The explosion date, 1993 March 28,
is known from the optical discovery.

The machine is given only (elapsed time since the FIRST VLBI epoch, radius).
It must discover the scaling symmetry of the expansion and locate its fixed
point -- the explosion -- 182 days before any of its data begin, and read
the deceleration exponent m off the output action of the generator.

Data: outer angular radii digitized from Marcaide et al. (2009), Table 1
(ar5iv rendering; mixed observing wavelengths; verify against the published
table before any paper-grade use). Uncertainties 0.4-2%.

Run: python demo_sn1993j.py
"""

# region imports
import numpy as np

from nestynet.charts import FitConfig
from nestynet_sr.sr_gs.chart_bridge import scan_and_compile_charts

# endregion

# (age in days since explosion, outer radius in mas, sigma in mas)
# digitized from Marcaide et al. (2009) Table 1
SN1993J_TABLE = [
    (182, 0.488, 0.004), (239, 0.628, 0.014), (329, 0.818, 0.015),
    (427, 1.02, 0.02), (541, 1.15, 0.03), (697, 1.48, 0.04),
    (774, 1.66, 0.04), (917, 1.92, 0.04), (1096, 2.21, 0.03),
    (1177, 2.31, 0.04), (1304, 2.61, 0.02), (1430, 2.81, 0.05),
    (1638, 3.09, 0.03), (1788, 3.37, 0.03), (1889, 3.48, 0.04),
    (2066, 3.74, 0.04), (2073, 3.84, 0.06), (2265, 4.06, 0.04),
    (2369, 4.20, 0.04), (2627, 4.48, 0.06), (2794, 4.98, 0.07),
    (2798, 4.71, 0.10), (2880, 4.84, 0.07), (3157, 5.21, 0.10),
    (3511, 5.67, 0.10), (3521, 6.06, 0.05), (3867, 6.15, 0.11),
]
FIRST_EPOCH_AGE = 182  # days between explosion and the first VLBI epoch


def main() -> None:
    age = np.array([row[0] for row in SN1993J_TABLE], dtype=np.float64)
    radius = np.array([row[1] for row in SN1993J_TABLE], dtype=np.float64)

    # the machine sees only elapsed time since the first VLBI epoch
    t = age - FIRST_EPOCH_AGE

    print(f"== SN 1993J: {len(t)} real VLBI radii, "
          f"{t.max() / 365.25:.1f} yr of elapsed time ==")
    res = scan_and_compile_charts(
        t, radius,
        # N=27: the bootstrap must be HEAVILY smoothed (>=4 segments lets
        # f' oscillate through the noise and poisons the scan); the warped
        # coordinate is nearly linear, so the warp refits can afford 4.
        fit_cfg=FitConfig(segments=3, epochs=400, restarts=3),
        sharp_fit_cfg=FitConfig(segments=4, epochs=300, restarts=3),
        nullity_strategy="rank_tol",
        rank_rtol=0.15,
        acceptance_residual_tol=0.15,
    )
    for line in res.log:
        print("  " + line)

    certified = [p for p in res.proposals if p.certified]
    if not certified:
        print("  no certified chart")
        return
    p = certified[0]
    t0 = p.chart.get_param("t0")
    print("\n-- scorer (not part of discovery) --")
    print(f"  recovered explosion epoch: {t0:+.0f} d before the first VLBI "
          f"epoch is {-t0:.0f} d (true {FIRST_EPOCH_AGE} d; "
          f"error {abs(-t0 - FIRST_EPOCH_AGE):.0f} d over a "
          f"{t.max() / 365.25:.1f} yr baseline)")
    print(f"  deceleration exponent m: {p.exponent:.3f} "
          f"(Marcaide et al. 2009: 0.845 before day ~1500, 0.788 after)")


if __name__ == "__main__":
    main()
