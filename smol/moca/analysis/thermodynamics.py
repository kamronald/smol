"""Utility functions to perform thermodynamic integration.

Integration of ensemble-averaged properties, which can be extracted
from SampleContainer.
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
    ref_type="high_t",
    interp=True,
    interp_order=2,
    interp_delta_beta=0.1,
):
    """Calculate free energy and entropy from average enthalpy.

    Free energy (F) obtained from integrating enthalpy across beta (1/kT).

    beta*F - beta_0*F_0 = integral(avg_enthalpy)d_beta (eq 1)
    Where beta_0*F_0 is a user-specified reference state.

    Also evaluates the entropy using the free energy and enthalpy.

    When integrating average enthalpies from canonical MC, this gives the
    Helmholtz free energy (F = U - TS).
    Similarly when integrating average enthalpies from grand-canonical MC
    (at fixed chemical potentials), this yields the grand-canonical free
    energy (omega = U - TS - mu*N).
    Args:
        enthalpies (ndarray):
            averaged enthalpies from specified temperatures
        temperatures (ndarray):
            list of temperatures in K
        beta0_f0 (float):
            value of reference state (beta_0 * F_0) from eq 1.
        ref_type (str):
            Whether the boundary condition is from high T or low T. Can only be
            'high_t' or 'low_t'
        interp (bool):
            Whether to interpolate the enthalpies onto a fine grid of beta.
        interp_order (int):
            Order of spline interpolation (2 = quadratic, 3 = cubic, etc)
        interp_delta_beta (float):
            Spacing of beta values to interpolate over

    Returns:
        temp_free_ens (dict):
            Dict of ndarrays containing temperature, free energy, average
            enthalpy, and entropy.
            {'temperature': ndarray, 'free_energy': ndarray, 'enthalpy': ndarray}

    """
    if ref_type not in ["high_t", "low_t"]:
        raise ValueError('bound_type must be either "high_t" or "low_t"')

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
                    f"{1/kB/min_beta} K, or beta = {min_beta} eV^-1. Might be better to go "
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

    temp_free_ens_d = {}
    temp_free_ens_d["temperature"] = np.array([t[0] for t in temp_free_ens])
    temp_free_ens_d["free_energy"] = np.array([t[1] for t in temp_free_ens])
    temp_free_ens_d["enthalpy"] = np.array([t[2] for t in temp_free_ens])

    temp_free_ens_d["entropy"] = np.array(
        np.divide(
            temp_free_ens_d["enthalpy"] - temp_free_ens_d["free_energy"],
            temp_free_ens_d["temperature"],
        )
    )

    return temp_free_ens_d


def calculate_s_inf(compositions):
    """Calculate entropy in the infinite temperature limit (S_inf).

    S_inf = kB * ln(sigma), where sigma is the number of possible
    arrangements for the specified species compositions.

    We use Stirling's approximation to evaluate ln(sigma):
    e.g. ln(N!) = N * ln(N)

    Args:
        compositions (ndarray):
            Number of each species -- order does not matter.

    Returns:
        s_inf (float):
            Entropy in the infinite temperature limit in eV/K.
    """
    num_sites = sum(compositions)
    s_inf = num_sites * np.log(num_sites)
    for comp in compositions:
        s_inf -= comp * np.log(comp)

    return kB * s_inf
