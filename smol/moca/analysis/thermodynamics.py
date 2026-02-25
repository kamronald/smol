"""Post-processing functions to perform thermodynamic integration.

Integration of ensemble-averaged properties (i.e. enthalpies, compositions)
which can be obtained from SampleContainer.
"""

__author__ = "Ronald L. Kam"

import warnings

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline

from smol.constants import kB


def free_en_s_from_enthalpy(
    enthalpies,
    temperatures,
    beta0_f0,
    ref_type,
    interp=True,
    interp_order=2,
    interp_delta_beta=0.1,
):
    """Calculate free energy and entropy from average enthalpy.

    Free energy (F) obtained from integrating a generalized enthalpy across
    beta (1/kT).

    beta*F - beta_0*F_0 = integral(avg_enthalpy)d_beta           (eq 1)
    Where beta_0*F_0 is a user-specified reference state.
    "Generalized enthalpy" is an energy describing a microstate of a specified ensemble.
    Generalized enthalpy of canonical ensemble is E, grand-canonical is E - mu*N, etc.

    When integrating average enthalpies from canonical MC, this gives the
    Helmholtz free energy (F = U - TS).
    Similarly when integrating average enthalpies from grand-canonical MC
    (at fixed chemical potentials), this yields the grand-canonical free
    (omega = U - TS - mu*N).

    Also evaluates the entropy using the free energy and enthalpy, e.g. S = (U - F)/T

    Refer to A. van de Walle and M. Asta 2002 Modelling Simul. Mater. Sci. Eng. 10 521
    for more details.

    Args:
        enthalpies (ndarray):
            averaged enthalpies from each temperature
        temperatures (ndarray):
            list of temperatures in K
        beta0_f0 (float):
            value of reference state (beta_0 * F_0) from eq 1.
        ref_type (str):
            Whether the reference state is from high T or low T. Can only be
            'high_t' or 'low_t'
        interp (bool):
            Whether to interpolate the enthalpies onto a fine grid of beta.
        interp_order (int):
            Order of spline interpolation (2 = quadratic, 3 = cubic, etc)
        interp_delta_beta (float):
            Spacing of beta values to interpolate over

    Returns:
        thermal_props_d (dict):
            Dict of ndarrays containing temperature, free energy, average
            enthalpy, and entropy.
            {'temperature': ndarray, 'free_energy': ndarray, 'enthalpy': ndarray,
            'entropy': ndarray
            }

    """
    if ref_type not in ["high_t", "low_t"]:
        raise ValueError('ref_type must be either "high_t" or "low_t"')

    # convert from temperature to beta (1/kT) for integration
    beta_enths = [(1 / kB / temp, enth) for temp, enth in zip(temperatures, enthalpies)]
    beta_enths = sorted(beta_enths, key=lambda x: x[0])

    if interp:
        avg_enths_beta_interp = make_interp_spline(
            [t[0] for t in beta_enths], [t[1] for t in beta_enths], k=interp_order
        )
        if ref_type == "high_t":  # make beta grid start from zero (infinite T)
            min_beta = beta_enths[0][0]
            if min_beta > 2:
                warnings.warn(
                    f"You are integrating from high T, but highest simulated T is only "
                    f"{1/kB/min_beta} K, or beta = {min_beta} eV^-1. Might want to go "
                    f"to higher T > 12000 K (beta < 1 eV^-1)."
                )
            beta_grid = np.arange(
                0, beta_enths[-1][0] + interp_delta_beta / 2, interp_delta_beta
            )
        else:
            beta_grid = np.arange(
                beta_enths[0][0],
                beta_enths[-1][0] + interp_delta_beta / 2,
                interp_delta_beta,
            )

        # include the original temperatures for more convenient post-processing
        beta_grid = np.concatenate(
            [
                beta_grid,
                [
                    1 / kB / t
                    for t in temperatures
                    if not np.any(np.isclose(1 / kB / beta_grid, t))
                ],
            ]
        )
        beta_grid = sorted(beta_grid)

        avg_enths_beta_integrate = [
            (beta, enth)
            for beta, enth in zip(beta_grid, avg_enths_beta_interp(beta_grid))
        ]

    else:
        warnings.warn(
            "You have turned off interpolation which is not recommended, be wary of results!"
        )
        avg_enths_beta_integrate = beta_enths

    if ref_type == "low_t":
        avg_enths_beta_integrate = sorted(
            avg_enths_beta_integrate, key=lambda t: t[0], reverse=True
        )

    temp_free_ens = []
    for i, (beta, avg_enth) in enumerate(avg_enths_beta_integrate):
        if i == 0:
            continue
        betas_integrate = np.array([t[0] for t in avg_enths_beta_integrate[: i + 1]])
        enth_integrate = np.array([t[1] for t in avg_enths_beta_integrate[: i + 1]])
        this_int = simpson(x=betas_integrate, y=enth_integrate)
        this_f = (beta0_f0 + this_int) / beta
        temp_free_ens.append((1 / kB / beta, this_f, avg_enth))

    temp_free_ens = sorted(temp_free_ens, key=lambda x: x[0])

    thermal_props_d = {}
    thermal_props_d["temperature"] = np.array([t[0] for t in temp_free_ens])
    thermal_props_d["free_energy"] = np.array([t[1] for t in temp_free_ens])
    thermal_props_d["enthalpy"] = np.array([t[2] for t in temp_free_ens])

    thermal_props_d["entropy"] = np.array(
        np.divide(
            thermal_props_d["enthalpy"] - thermal_props_d["free_energy"],
            thermal_props_d["temperature"],
        )
    )

    return thermal_props_d


def free_en_integrate_comp(
    compositions,
    chem_pots,
    omega_0,
    ref_type,
    interp=True,
    interp_order=2,
    interp_delta_mu=0.005,
):
    """Calculate grand-canonical free energy (omega) at fixed temperature.

    Integrate average composition (N) of one species across range of chemical potential
    (mu) of that species.

    omega = omega_0 - integral(N)d_mu, at fixed T.
    Where omega_0 is the free energy at a reference state.

    Args:
        compositions (ndarray):
            Averaged compositions of one species at each chemical potential
        chem_pots (ndarray):
            Chemical potentials (mu) corresponding to the compositions.
        omega_0 (float):
            Free energy of the reference state.
        ref_type (str): 'high_mu' or 'low_mu'
            Whether the reference state is at high mu or low mu.
        interp (bool):
            Whether to interpolate compositions onto a fine grid of chemical potentials.
        interp_order (integer):
            Order of spline interpolation: 2 is quadratic, 3 is cubic...
        interp_delta_mu (float):
            Interval of chemical potentials to interpolate over.

    Returns:
        mu_free_ens_d (dict):
            Free energy, enthalpy, and composition for each chem pot

    """
    mus_comps = [(mu, comp) for mu, comp in zip(chem_pots, compositions)]
    mus_comps = sorted(mus_comps, key=lambda x: x[0])

    if interp:
        mus_comps_interp = make_interp_spline(
            x=[mu for mu, comp in mus_comps],
            y=[comp for mu, comp in mus_comps],
            k=interp_order,
        )

        mu_grid = np.arange(
            mus_comps[0][0], mus_comps[-1][0] + interp_delta_mu / 2, interp_delta_mu
        )

        mus_comps_integrate = [
            (mu, comp) for mu, comp in zip(mu_grid, mus_comps_interp(mu_grid))
        ]
    else:
        mus_comps_integrate = mus_comps

    if ref_type == "high_mu":
        mus_comps_integrate = sorted(
            mus_comps_integrate, key=lambda x: x[0], reverse=True
        )
    elif ref_type == "low_mu":
        mus_comps_integrate = sorted(mus_comps_integrate, key=lambda x: x[0])
    else:
        raise ValueError("ref_type must be either 'high_mu' or 'low_mu'!")

    mu_free_ens = [(mus_comps_integrate[0][0], omega_0, mus_comps_integrate[0][1])]
    for i, (mu, comp) in enumerate(mus_comps_integrate):
        if i == 0:
            continue

        mus_integrate = np.array([t[0] for t in mus_comps_integrate[: i + 1]])
        comp_integrate = np.array([t[1] for t in mus_comps_integrate[: i + 1]])
        this_int = simpson(x=mus_integrate, y=-comp_integrate)
        this_omega = omega_0 + this_int
        mu_free_ens.append((mu, this_omega, comp))

    mu_free_ens = sorted(mu_free_ens, key=lambda x: x[0])

    mu_free_ens_d = {
        "chemical_potential": np.array([mu for mu, omega, comp in mu_free_ens])
    }
    mu_free_ens_d["free_energy"] = np.array([omega for mu, omega, comp in mu_free_ens])
    mu_free_ens_d["composition"] = np.array([comp for mu, omega, comp in mu_free_ens])

    return mu_free_ens_d


def calculate_s_inf(subl_comps):
    """Calculate entropy in the infinite temperature limit (S_inf).

    S_inf = kB * ln(sigma), where sigma is the number of possible
    configurations on the lattice for the specified species compositions.

    Stirling's approximation used to evaluate expressions of the form
    ln(N!) = N * ln(N) + N

    Args:
        subl_comps (list of ndarray):
            List of ndarrays, each containing the composition
            (number of each species) within an active sublattice.
            [ndarray(n_li, n_mn...), ndarray(n_li, n_vac...)]
            Order of the species and sublattices do not matter

    Returns:
        s_inf (float):
            Entropy (eV/K) in the infinite temperature limit.
    """
    sigma = 0
    for comp in subl_comps:
        num_sites = sum(comp)
        sigma += num_sites * np.log(num_sites)
        for comp in comp:
            sigma -= comp * np.log(comp)

    s_inf = kB * sigma
    return s_inf
