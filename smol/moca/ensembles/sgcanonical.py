"""Implementation of a Semi-Grand Canonical Ensemble Classes for running Monte
Carlo simulations for fixed number of sites but variable concentration of
species.

Two classes are different SGC ensembles implemented:
* MuSemiGrandEnsemble - for which relative chemical potentials are fixed
* FuSemiGrandEnsemble - for which relative fugacity fractions are fixed.
"""

__author__ = "Luis Barroso-Luque"

from collections import defaultdict
import random
from abc import ABCMeta, abstractmethod
from math import exp
import numpy as np
from smol.moca.processor import CEProcessor
from smol.moca.ensembles.canonical import CanonicalEnsemble


class BaseSemiGrandEnsemble(CanonicalEnsemble, metaclass=ABCMeta):
    """
    Abstract Semi-Grand Canonical Base Ensemble. Total number of species are
    fixed but composition of "active" (with partial occupancies) sublattices is
    allowed to change.

    This class can not be instantiated. See MuSemiGrandEnsemble and
    FuSemiGrandEnsemble below.
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sample_interval (int):
                interval of steps to save the current occupancy and property
            initial_occupancy (ndarray or list):
                Initial occupancy vector. The occupancy can be encoded
                according to the processor or the species names directly.
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, temperature, sample_interval,
                         initial_occupancy=initial_occupancy, seed=seed)

    @property
    def current_species_counts(self):
        """
        Counts of species. This excludes species in "static" sublattices.
        Those with no partial occupancy.
        """
        counts = self._get_counts()
        species_counts = {}

        for name in self._sublattices.keys():
            cons = {sp: count for sp, count
                    in zip(self._sublattices[name]['species'], counts[name])}
            species_counts.update(cons)

        return species_counts

    @property
    def current_composition(self):
        """Composition of all active sites. Excludes inactive sites."""
        comps = defaultdict(lambda: 0)
        sublatt_comps = self.current_sublattice_compositions
        for sub_comp in sublatt_comps:
            for sp, val in sub_comp.items():
                comps[sp] += val
        return {sp: val/len(sublatt_comps) for sp, val in comps.items()}

    @property
    def composition_samples(self):
        """Samples of all active site compositions."""
        n = len(self.data[self._prod_start:])
        comp_samples = {sp: np.empty(n) for sp in self.current_composition}
        for i, sample in enumerate(self.data[self._prod_start:]):
            for sp, comp in sample['composition'].items():
                comp_samples[sp][i] = comp
        return comp_samples

    @property
    def average_composition(self):
        """Average composition of all active sites."""
        return {sp: comp.mean() for sp, comp
                in self.composition_samples.items()}

    @property
    def composition_variance(self):
        """Composition variance."""
        return {sp: comp.var() for sp, comp in self.composition_samples}

    @property
    def current_sublattice_compositions(self):
        """Composition for each "active" sublattice."""
        comps = self._get_sublattice_comps()
        return tuple(comps.values())

    @property
    def sublattice_composition_samples(self):
        """Samples of sublattice compositions"""
        n = len(self.data[self._prod_start:])
        comp_samples = tuple({sp: np.empty(n) for sp in comps}
                             for comps in self.current_sublattice_compositions)
        for i, sample in enumerate(self.data[self._prod_start:]):
            for j, sublat_comps in enumerate(sample['sublattice_compositions']):  # noqa
                for sp, comp in sublat_comps.items():
                    comp_samples[j][sp][i] = comp
        return comp_samples

    @property
    def average_sublattice_compositions(self):
        """Average sublattice compositions."""
        return tuple({sp: comp.mean() for sp, comp in comps.items()}
                     for comps in self.sublattice_composition_samples)

    @property
    def sublattice_composition_variance(self):
        """sublattice composition variance."""
        return tuple({sp: comp.var() for sp, comp in comps.items()}
                     for comps in self.sublattice_composition_samples)

    @abstractmethod
    def _attempt_step(self, sublattices):
        """
        Attempts a MC step and returns 0, 1 based on whether the step was
        accepted or not.
        """
        pass

    def _get_flips(self, sublattices=None):
        """
        Gets a possible semi-grand canonical flip, and the corresponding
        change in chemical potential

        Args:
            sublattices (list of str): optional
                If only considering a subset of the active sublattices.
        Returns: flip
            tuple
        """
        if sublattices is None:
            sublattices = self.sublattices

        sublattice_name = random.choice(sublattices)
        sublattice = self._sublattices[sublattice_name]
        species = tuple(sublattice['species'].keys())

        site = random.choice(sublattice['sites'])
        old_bit = self._occupancy[site]
        choices = set(range(len(species))) - {old_bit}
        new_bit = random.choice(list(choices))
        old_species = species[old_bit]
        new_species = species[new_bit]

        return (site, new_bit), sublattice, new_species, old_species

    def _get_counts(self):
        """
        Get the total count of each species for current occupation

        Returns: dict of sublattices with corresponding species concentrations
            dict
        """
        counts = {}
        for name, sublattice in self._sublattices.items():
            occupancy = self._occupancy[sublattice['sites']]
            counts[name] = [len(occupancy[occupancy == i]) for i
                            in range(len(sublattice['species']))]
        return counts

    def _get_sublattice_comps(self):
        """
        Get the current composition (species concentration) for each
        sublattice.
        """
        composition = {}
        counts = self._get_counts()
        for name, sublattice in self._sublattices.items():
            occupancy = self._occupancy[sublattice['sites']]
            composition[name] = {sp: ct/len(occupancy) for sp, ct
                                 in zip(sublattice['species'], counts[name])}
        return composition

    def _get_current_data(self):
        """
        Get ensemble specific data for current MC step
        """
        data = super()._get_current_data()
        data['counts'] = self.current_species_counts
        data['composition'] = self.current_composition
        data['sublattice_compositions'] = self.current_sublattice_compositions
        return data


class MuSemiGrandEnsemble(BaseSemiGrandEnsemble):
    """
    A Semi-Grand Canonical Ensemble for Monte Carlo Simulations where species
    chemical potentials are predefined. Note that in the SGC Ensemble
    implemented here, only the differences in chemical potentials with
    respect to a reference species on each sublattice are fixed, and not the
    absolute values. To obtain the absolute values you must calculate the
    reference chemical potential and then simply subtract it from the given
    values.
    """

    def __init__(self, processor, temperature, chemical_potentials,
                 sample_interval, initial_occupancy, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            chemical_potentials (dict):
                dictionary with species names and chemical potentials. If the
                chemical potential for one species is not zero (reference), one
                will be chosen and all other values will be shifted accordingly
            sample_interval (int):
                interval of steps to save the current occupancy and property
            inital_occupancy (array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, temperature, sample_interval,
                         initial_occupancy=initial_occupancy,
                         seed=seed)

        # check that species are valid
        species = [sp for sps in processor.unique_bits for sp in sps]
        for sp in chemical_potentials.keys():
            if sp not in species:
                raise ValueError(f'Species {sp} in provided chemical '
                                 'potentials is not a specie in the expansion'
                                 f': {species}')
        for sp in species:
            if sp not in chemical_potentials.keys():
                raise ValueError(f'Species {sp} was not assigned a chemical '
                                 ' potential, a value must be provided.')

        # Add chemical potentials to sublattice dictionary
        for sublatt in self._sublattices.values():
            sublatt['mu'] = {sp: mu for sp, mu in chemical_potentials.items()
                             if sp in sublatt['species']}
            # This can be removed since it really doesn't affect results...
            # If no reference species is set, then set and recenter others
            mus = list(sublatt['mu'].values())
            if not any([mu == 0 for mu in mus]):
                ref_mu = mus[0]
                sublatt['mu'] = {sp: mu - ref_mu for sp, mu
                                 in sublatt['mu'].items()}

    @property
    def chemical_potentials(self):
        """Relative chemical potentials. Reference species have 0."""
        chem_pots = {}
        for sublattice in self._sublattices.values():
            chem_pots.update(sublattice['mu'])
        return chem_pots

    def _attempt_step(self, sublattices=None):
        """
        Attempts flips corresponding to a semi canonical swap (a single site
        identity flip).
        Args:
            sublattices (list of str): optional
                If only considering a subset of the active sublattices.

        Returns:
            bool: Flip acceptance
        """
        flip, sublattice, new_sp, old_sp = self._get_flips(sublattices)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         [flip])
        delta_mu = sublattice['mu'][new_sp] - sublattice['mu'][old_sp]
        delta_phi = delta_e - delta_mu
        accept = self._accept(delta_phi, self.beta)

        if accept:
            self._property += delta_e
            self._occupancy[flip[0]] = flip[1]
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()

        return accept

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['chem_pots'] = self.chemical_potentials
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Creates a CanonicalEnsemble from MSONable dict representation.
        """
        eb = cls(CEProcessor.from_dict(d['processor']),
                 temperature=d['temperature'],
                 chemical_potentials=d['chem_pots'],
                 sample_interval=d['sample_interval'],
                 initial_occupancy=d['initial_occupancy'],
                 seed=d['seed'])
        eb._min_energy = d['_min_energy']
        eb._min_occupancy = np.array(d['_min_occupancy'])
        eb._sublattices = d['_sublattices']
        eb._data = d['_data']
        eb._step = d['_step']
        eb._ssteps = d['_ssteps']
        eb._property = d['_energy']
        eb._occupancy = np.array(d['_occupancy'])
        return eb


class FuSemiGrandEnsemble(BaseSemiGrandEnsemble):
    """
    A Semi-Grand Canonical Ensemble for Monte Carlo simulations where the
    species fugacity ratios are set constant. This implicitly sets the chemical
    potentials, albeit for a specific temperature. Since one species per
    sublattice is the reference species, to calculate actual fugacities the
    reference fugacity must be computed as an ensemble average and all other
    fugacities can then be calculated. From the fugacities and the set
    temperature the corresponding chemical potentials can then be calculated.
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, fugacity_fractions=None, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sample_interval (int):
                interval of steps to save the current occupancy and property
            inital_occupancy (list or array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            fugacity_fractions (list/tuple of dicts): optional
                dictionary of species name and fugacity fraction for each
                sublattice (ie think of it as the sublattice concentrations
                for random structure). If not given this will be taken from the
                prim structure used in the CE.
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, temperature, sample_interval,
                         initial_occupancy=initial_occupancy,
                         seed=seed)

        if fugacity_fractions is not None:
            # check that species are valid
            species = [sp for sps in processor.unique_bits for sp in sps]
            for sublatt in fugacity_fractions:
                if sum([f for f in sublatt.values()]) != 1:
                    raise ValueError('Fugacity ratios must add to one.')
                for sp in sublatt.keys():
                    if sp not in species:
                        raise ValueError(f'Species {sp} in provided fugacity '
                                         'ratios is not a species in the'
                                         f'expansion: {species}')

            # Add fugacities to sublattice dictionary
            # Note that in the strange cases where you want sublattices
            # with the same allowed species but different concentrations this
            # will mess it up and give both of them the first dictionary...
            for sublatt in self._sublattices.values():
                ind = [sl.keys() for sl
                       in fugacity_fractions].index(sublatt['species'].keys())
                sublatt['species'] = fugacity_fractions[ind]

    def _attempt_step(self, sublattices=None):
        """
        Attempts flips corresponding to a canonical swap.

        Args:
            sublattices (list of str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """
        flip, sublattice, new_sp, old_sp = self._get_flips(sublattices)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         [flip])
        ratio = sublattice['species'][new_sp]/sublattice['species'][old_sp]
        accept = self._accept(delta_e, ratio, self.beta)

        if accept:
            self._property += delta_e
            self._occupancy[flip[0]] = flip[1]
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()

        return accept

    @staticmethod
    def _accept(delta_e, ratio, beta=1.0):
        """
        Fugacity based Semi-Grand Canonical Metropolis acceptance criteria.

        Args:
            ratio: ratio of fugacity fractions for new and old configuration

        Returns:
            bool
        """
        condition = ratio*exp(-beta*delta_e)
        return True if condition >= 1 else condition >= random.random()

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Creates a CanonicalEnsemble from MSONable dict representation.
        """
        eb = cls(CEProcessor.from_dict(d['processor']),
                 temperature=d['temperature'],
                 sample_interval=d['sample_interval'],
                 initial_occupancy=d['initial_occupancy'],
                 seed=d['seed'])
        eb._min_energy = d['_min_energy']
        eb._min_occupancy = np.array(d['_min_occupancy'])
        eb._sublattices = d['_sublattices']
        eb._data = d['_data']
        eb._step = d['_step']
        eb._ssteps = d['_ssteps']
        eb._property = d['_energy']
        eb._occupancy = np.array(d['_occupancy'])
        return eb
