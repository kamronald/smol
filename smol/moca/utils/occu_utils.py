"""Utility functions to handle encoded occupation arrays."""

__author__ = "Fengyu Xie"

import numpy as np


# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_list(occupancy, sublattices,
                         active_only=False):
    """Get occupancy status of each sub-lattice.

    Get table of the indices of sites that are occupied by each specie on
    sub-lattices, from an encoded occupancy array.

    Args:
        occupancy(1d Arraylike[int]):
            An array representing encoded occupancy, can be list.
        sublattices(smol.moca.Sublattice):
            All sub-lattices, active or not.
        active_only(Boolean):
            If true, will count un-restricted sites on all
            sub-lattices only. Default to false, will count
            all sites and sub-lattices.

    Return:
        Index of sites occupied by each species, sublattices concatenated:
            List[List[int]]
    """
    occu = np.array(occupancy, dtype=int)

    # Encodings is not necessarily range(len)
    if active_only:
        return [s.active_sites[occu[s.active_sites] == sp_id]
                .tolist() for s in sublattices
                for sp_id in s.encoding]
    else:
        return [s.sites[occu[s.sites] == sp_id].tolist()
                for s in sublattices for sp_id in s.encoding]


def occu_to_species_n(occupancy, sublattices,
                      active_only=False):
    """Count number of species from occupation array.

    Get a statistics table of each specie on sub-lattices from an encoded
    occupancy array.
    Args:
        occupancy(1D ArrayLike[int]):
            An array representing encoded occupancy, can be list.
        sublattices(smol.moca.Sublattice):
            All sub-lattices, active or not.
        active_only(Bool): optional
            If true, will count un-restricted sites on active
            sub-lattices only. Default to false, will count
            all sites and sub-lattices.

    Return:
        Amount of each species, sublattices concatenated:
            1D np.ndarray[int]
    """
    return np.array([len(sp_sites) for sp_sites in
                     occu_to_species_list(occupancy, sublattices,
                                          active_only=active_only)],
                    dtype=int)


def get_dim_ids_by_sublattice(bits):
    """Get the component index of each species in vector n.

    Args:
        bits(List[List[Specie|Vacancy|Element]]):
           Species on each sub-lattice.
    Returns:
        Component index of each species on each sublattice in vector n:
           List[List[int]]
    """
    dim_ids = []
    dim_id = 0
    for species in bits:
        dim_ids.append(list(range(dim_id, dim_id + len(species))))
        dim_id += len(species)
    return dim_ids


def delta_n_from_step(occu, step, sublattices):
    """Get the change of species amounts from MC step.

    Args:
        occu(1D arrayLike[int]):
            Encoded occupation array.
        step(List[tuple(int,int)]):
            List of tuples recording (site_id, code_of_species_to
            _replace_with).
        sublattices(smol.moca.Sublattice):
            All sublattices, active or not.

    Return:
        Change of species amounts (delta_n):
            1D np.ndarray[int]
    """
    occu = np.array(occu, dtype=int)
    d = sum([len(s.site_space) for s in sublattices])
    sublattice_ids = np.zeros(len(occu), dtype=int) - 1
    for sl_id, s in enumerate(sublattices):
        sublattice_ids[s.sites] = sl_id
    if np.any(np.isclose(sublattice_ids, -1)):
        raise ValueError("Number of sites in sub-lattices cannot be "
                         "fewer than total number of sites!")

    delta_n = np.zeros(d, dtype=int)
    dim_ids = get_dim_ids_by_sublattice([s.species for s in sublattices])
    for site_id, code in step:
        sl_id = sublattice_ids[site_id]
        ori_code = occu[site_id]

        code_id = sublattices[sl_id].encoding.tolist().index(code)
        ori_code_id = sublattices[sl_id].encoding.tolist().index(ori_code)

        ori_dim_id = dim_ids[sl_id][ori_code_id]
        dim_id = dim_ids[sl_id][code_id]
        delta_n[ori_dim_id] -= 1
        delta_n[dim_id] += 1

    return delta_n


def delta_x_from_step(occu, step, sublattices, comp_space):
    """Get the change of constrained coordinates from MC step.

    Args:
        occu(1D arrayLike):
            Encoded occupation array.
        step(List[type(int.int)]):
            List of tuples corresponding to single flips
            in a step.
        sublattices(smol.moca.Sublattice):
            All sublattices, active or not.
        comp_space(smol.CompSpace):
            composition space object.

        comp_space must be generated from sublattices.
    Return:
        Change of constraint lattice coordinates:
            1D np.ndarray[int]
    """
    delta_n = delta_n_from_step(occu, step, sublattices)
    return comp_space.transform_format(delta_n,
                                       from_format='n',
                                       to_format='x',
                                       check_bounded=False)


def flip_weights_mask(flip_vectors, n, max_n=None):
    """Mark feasibility of flip vectors.

    If a flip direction leads to any n+v < 0, then it is marked
    infeasible. Generates a boolean mask, every two components
    marks whether a flip direction and its inverse is feasible
    given n at the current occupancy.
    Will be used by Tableflipper.

    Args:
        flip_vectors(1D ArrayLike[int]):
            Flip directions in the table (inverses not included).
        n(1D ArrayLike[int]):
            Amount of each specie on sublattices. Same as returned
            by occu_to_species_n.
        max_n(1D ArrayLike[int]): optional
            Maximum number of each species allowed. This is needed
            When the number of active sites != number of sites in
            some sub-lattice.

    Return:
        Direction and its inverse are feasible or not:
           1D np.ndarray[bool]
    """
    flip_vectors = np.array(flip_vectors, dtype=int)
    directions = np.concatenate([(u, -u) for u in flip_vectors])
    if max_n is None:
        max_n = np.ones(len(n)) * np.inf
    elif isinstance(max_n, (int, np.int32, np.int64)):
        max_n = np.ones(len(n), dtype=int) * max_n
    else:
        max_n = np.array(max_n, dtype=int)
    return (np.any(directions + n < 0, axis=-1)
            | np.any(directions + n) > max_n)
