""" Implementation of low temperature expansion (LTE) to treat the configurational free
energy at low temperatures, where it is likely that MC will get "stuck" on ground
states, yielding unconverged thermodynamic quantities. The principle of LTE is to
enumerate the possible excitations from a ground state and construct a partition
function with their associated energies.

For reference:
Kohan, A. F.; Tepesch, P. D.; Ceder, G.; Wolverton, C. Computation of Alloy Phase
Diagrams at Low Temperatures. Comp Mater Sci 1998, 9 (3–4), 389–396.

"""

__author__ = "Ronald Kam"

import copy
import math

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from smol.constants import kB


def gen_canon_excitations(ensemble, gs_occu):
    """ Generate all single swap perturbations from the ground state to include in a
    canonical partition function.

    Args:
        processor:
        gs_occu: Encoded occupancy array of ground state structure

    Returns:
        all_swap_energies (Dict): dictionary mapping energy of excitation from a ground
        state to its multiplicity
    """
    gs_struc = ensemble.processor.structure_from_occupancy(gs_occu,
                                                           include_vacancies=True
                                                           )
    sga = SpacegroupAnalyzer(structure=gs_struc)
    sym_gs = sga.get_symmetrized_structure()

    all_swap_energies = {}
    for subl in ensemble.active_sublattices:
        # determine the sets of sym equivalent sites that fit in this sublattice
        species_site_map = get_subl_equiv_sites(subl, sym_gs)
        # now perform the necessary swaps
        subl_swap_energies = gen_swap_energies(
            species_site_map, ensemble.processor, gs_occu
        )
        for en, mult in subl_swap_energies.items():
            if en in all_swap_energies:
                all_swap_energies[en] += mult
            else:
                all_swap_energies[en] = mult

    return all_swap_energies


def eval_canon_z(energy_mult_d, temp, min_e):
    """ Evaluate canonical partition function at a temperature, given information on
    energies. Values are centered on the ground state energy, such that exponential
    functions can be nicely computed.

    Args:
        energy_mult_d (Dict): {energy: multiplicity}

    Returns:
        (float) evaluated value of partition function at a given temperature

    """
    return 1 + sum(
        [math.exp(-(e-min_e) / (kB*temp)) * mult for e, mult in energy_mult_d.items()]
    )


def get_subl_equiv_sites(sublattice, symmetrized_structure):
    """ Obtain the sets of symmetrically distinct sites that pertain to a sublattice,
    grouped by species.

    Args:
        sublattice: active sublattice in a given ensemble
        symmetrized_structure: SymmetrizedStructure (from Pymatgen)

    Returns:
        Dictionary mapping species to list of list of equivalent sites
        {species: List[List[site_indices]]}

    """
    species_site_map = {}
    for equiv_inds in symmetrized_structure.equivalent_indices:
        spec = symmetrized_structure[equiv_inds[0]].specie
        if spec not in sublattice.species:
            continue

        relevant_inds = [ind for ind in equiv_inds if ind in sublattice.sites]

        if spec not in species_site_map:
            species_site_map[spec] = [relevant_inds]
        else:
            species_site_map[spec].append(relevant_inds)

    return species_site_map


def gen_swap_energies(species_site_map, processor, occu):
    """ Generate all the symmetrically distinct swap energies from the ground state and
    their multiplicities.

    Args:
        species_site_map: Dictionary mapping species to lists of symmetrically
        equivalent sites

    Returns:
        swap_en_mults: Dictionary mapping energy of a swap to its multiplicity
        {energy (float): multiplicity (int)}

    """
    all_species = [s for s in species_site_map.keys()]
    swap_en_mults = {}
    for i in range(len(all_species)-1):
        for equiv_sites_i in species_site_map[all_species[i]]:
            # Perform swaps with all the other species, keeping in mind not to duplicate
            site_i_ind = equiv_sites_i[0]
            # only need to swap one of these sites, the rest are symmetrically equiv.
            multiplicity = len(equiv_sites_i)
            remaining_species = all_species[i+1:]
            sites_to_swap = [
                sites for species, sites_list in species_site_map.items()
                for sites in sites_list if species in remaining_species
            ]
            for equiv_sites_j in sites_to_swap:
                for site_j_ind in equiv_sites_j:
                    swap_en = compute_swap(processor, site_i_ind, site_j_ind, occu)
                    if swap_en in swap_en_mults:
                        swap_en_mults[swap_en] += multiplicity
                    else:
                        swap_en_mults[swap_en] = multiplicity

    return swap_en_mults


def compute_swap(processor, s1_ind, s2_ind, occu):
    """ Compute energy after a given swap.

    Args:
        processor:
        s1_ind (int): site index 1
        s2_ind (int): site index 2
        occu (np.array of ints): encoded occupancy array

    Returns:
        swap_energy (float): energy of new configuration after swap.

    """
    new_occu = copy.deepcopy(occu)
    new_occu[s1_ind] = occu[s2_ind]
    new_occu[s2_ind] = occu[s1_ind]
    swap_energy = processor.compute_property(new_occu)[0]
    return swap_energy
