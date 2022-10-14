""" Implementation of low temperature expansion to treat the configurational free
energy at low temperatures, where it is likely that MC will get "stuck" on ground
states, yielding unconverged thermodynamic quantities.
"""
import copy
import math
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, PeriodicSite
from smol.cofe.space import Vacancy, get_allowed_species, get_site_spaces
from smol.moca.sublattice import Sublattice
from smol.constants import kB


def canonical_low_t_exp(ensemble, gs_occu):
    """ Implement low temperature expansion of a binary system on canonical ensemble.
    Include all single swap perturbations from the ground state in the partition
    function.

    Args:
        processor:
        gs_occu: Encoded occupancy of ground state structure

    Returns:

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
            all_swap_energies[en] = mult

    return all_swap_energies


def get_subl_equiv_sites(sublattice, symmetrized_structure):
    """ Obtain the sets of symmetrically distinct sites that pertain to a sublattice,
    grouped by species.

    Args:
        sublattice:
        symmetrized_structure:

    Returns:
        Dictionary mapping species to list of list of equivalent sites
        {species: List[List[site_indices]]}

    """
    species_site_map = {}
    for equiv_inds in symmetrized_structure.equivalent_indices:
        spec = symmetrized_structure[equiv_inds[0]].specie
        if spec not in sublattice.species:
            continue

        if spec not in species_site_map:
            species_site_map[spec] = [equiv_inds]
        else:
            species_site_map[spec].append(equiv_inds)

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
    for i in range(len(all_species)):
        for equiv_sites_i in species_site_map[all_species[i]]:
            # Perform swaps with all the other sites, keeping in mind not to duplicate
            site_i_ind = equiv_sites_i[0]
            # only need to swap one of these sites, the rest are symmetrically equiv.
            multiplicity = len(equiv_sites_i)
            remaining_species = all_species[i:]
            sites_to_swap = [
                sites for species, sites_list in species_site_map.items()
                for sites in sites_list if species in remaining_species
            ]
            for equiv_sites_j in sites_to_swap:
                for site_j_ind in equiv_sites_j:
                    swap_en = compute_swap(processor, site_i_ind, site_j_ind, occu)
                    swap_en_mults[swap_en] = multiplicity

    return swap_en_mults


def canon_partition_function(energy_mult_d, temp, min_e):
    """ Evaluate canonical partition function at a temperature, given list of energies.

    Args:
        energy_mult_d: {energy: multiplicity}

    Returns:
        evaluated canonical partition function at a temperature

    """
    return 1 + sum(
        [math.exp(-(e-min_e) / (kB*temp)) * mult for e, mult in energy_mult_d.items()]
    )


def compute_swap(processor, s1_ind, s2_ind, occu):
    new_occu = copy.deepcopy(occu)
    new_occu[s1_ind] = occu[s2_ind]
    new_occu[s2_ind] = occu[s1_ind]
    swap_energy = processor.compute_property(new_occu)[0]
    return swap_energy
