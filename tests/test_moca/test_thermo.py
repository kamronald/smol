import numpy as np

from smol.constants import kB
from smol.moca.analysis.thermodynamics import calculate_s_inf, free_en_s_from_enthalpy


def test_free_en_s_from_enthalpy():

    arr1 = [0, 2]
    arr2 = [1, 3]
    therm_d = free_en_s_from_enthalpy(arr1, arr2, 0, ref_type="high_t", interp=False)
    assert np.isclose(therm_d["free_energy"][0], 2 / 3, atol=1e-5)

    therm_d2 = free_en_s_from_enthalpy(arr1, arr2, 0, interp=False, ref_type="low_t")
    assert np.isclose(therm_d2["free_energy"][0], -2, atol=1e-5)

    temps = [2500, 15000, 1800, 10000, 2000, 25000, 1600, 4000, 6000, 3000, 58000]

    avg_ens = [
        -8910.51018,
        -8889.17218,
        -8915.03381,
        -8893.78547,
        -8913.60124,
        -8884.05332,
        -8916.889,
        -8904.62279,
        -8899.63863,
        -8908.18733,
        -8876.26348,
    ]

    s_inf = np.log(2) * 480

    thermo_d = free_en_s_from_enthalpy(
        enthalpies=avg_ens,
        temperatures=temps,
        beta0_f0=-s_inf,
        interp=True,
        ref_type="high_t",
    )

    assert np.isclose(thermo_d["free_energy"][0], -8950.001512688506, atol=1e-5)
    assert np.isclose(round(thermo_d["temperature"][0]), 1590)

    # Add test for low-T limit


def test_calculate_s_inf():

    # binary
    subl_comps1 = [[120, 120]]
    assert np.isclose(calculate_s_inf(subl_comps1) / kB, np.log(2) * 240)

    # coupled binary sublattice
    subl_comps2 = [[120, 120], [120, 120]]
    assert np.isclose(calculate_s_inf(subl_comps2) / kB, np.log(2) * 480)

    # quaternary sublattice
    subl_comps3 = [[120, 120, 120, 120]]
    assert np.isclose(calculate_s_inf(subl_comps3) / kB, np.log(4) * 480)

    # inhomogeneous ternary
    subl_comps4 = [[240, 120, 120]]
    test_4 = 480 * np.log(480) - 240 * np.log(240) - 2 * 120 * np.log(120)
    assert np.isclose(calculate_s_inf(subl_comps4) / kB, test_4)

    # ternary-binary
    subl_comps5 = [[240, 120, 120], [120, 120]]
    test_5 = (
        480 * np.log(480) - 240 * np.log(240) - 2 * 120 * np.log(120) + np.log(2) * 240
    )
    assert np.isclose(calculate_s_inf(subl_comps5) / kB, test_5)
