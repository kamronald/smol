"""Implementation of CE processor class for a fixed size super cell.

A CE processor is optimized to compute correlation vectors and local changes in
correlation vectors. This class allows the use a cluster expansion hamiltonian
to run Monte Carlo based simulations.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a CEProcessor and an EwaldProcessor
class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from collections import defaultdict
from smol.cofe import ClusterExpansion
from src.mc_utils import (corr_from_occupancy, general_delta_corr_single_flip,
                          indicator_delta_corr_single_flip)

from smol.moca.processors.base import BaseProcessor


class CEProcessor(BaseProcessor):
    """CEProcessor class to use a ClusterExpansion in MC simulations.

    A processor allows an ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a cluster expansion
    Hamiltonian.

    Attributes:
    """

    def __init__(self, cluster_expansion, supercell_matrix,
                 optimize_indicator=False):
        """Initialize a CEProcessor.

        Args:
            cluster_expansion (ClusterExpansion):
                A fitted cluster expansion representing a Hamiltonian
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            optimize_indicator (bool):
                When using an indicator basis, set the delta_corr function to
                the indicator optimize function. This can make MC steps faster.
                Make sure your cluster expansion was indeed fit with an
                indicator basis set, otherwise your MC results are no good.
        """
        super().__init__(cluster_expansion.cluster_subspace, supercell_matrix)
        # the only reason to keep the CE is for the MSONable from_dict
        self.cluster_expansion = cluster_expansion
        self.coefs = cluster_expansion.coefs
        self.n_orbit_functions = self.cluster_subspace.n_bit_orderings

        # set the dcorr_single_flip function
        self.indicator_opt = optimize_indicator
        self._dcorr_single_flip = indicator_delta_corr_single_flip \
            if optimize_indicator \
            else general_delta_corr_single_flip

        # Prepare necssary information for local updates
        self._orbit_inds = self.subspace.supercell_orbit_mappings(supercell_matrix)  # noqa
        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = []
        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        for orbit, inds in self._orbit_inds:
            self._orbit_list.append((orbit.bit_id, orbit.bit_combos,
                                     orbit.bases_array, inds))
            for site_ind in np.unique(inds):
                in_inds = np.any(inds == site_ind, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append((orbit.bit_id, ratio,
                                                        orbit.bit_combos,
                                                        orbit.bases_array,
                                                        inds[in_inds]))

    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        The property fitted to the corresponding to the CE.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return np.dot(self.compute_correlation(occupancy), self.coefs) * self.size  # noqa

    def compute_property_change(self, occupancy, flips):
        """Compute change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float: property difference between inital and final states
        """
        return np.dot(self._delta_corr(flips, occupancy), self.coefs) * self.size  # noqa

    def compute_correlation(self, occu):
        """Compute the correlation vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occu (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return corr_from_occupancy(occu, self.n_orbit_functions,
                                   self._orbit_list)

    def get_average_drift(self, iterations=1000):
        """Get the average forward and reverse drift for the given property.

        This is a sanity check function. The drift value should be very, very,
        very small, the smaller the better...think machine precision values.

        The average drift is the difference between the quick routine for used
        for MC to get a property difference from a single flip and the
        change in that property from explicitly calculating it fully for the
        initial state and the flipped state.

        Args:
            iterations (int): optional
                number of iterations/flips to compute.

        Returns:
            tuple: (float, float) forward and reverse average property drift
        """
        forward_drift, reverse_drift = 0.0, 0.0
        trajectory = []
        occu = [np.random.choice(species) for species in self.allowed_species]
        occu = self.encode_occupancy(occu)
        for _ in range(iterations):
            site = np.random.randint(self.size)
            species = set(range(len(self.allowed_species[site])))-{occu[site]}
            species = np.random.choice(list(species))
            delta_prop = self.compute_property_change(occu, [(site, species)])
            new_occu = occu.copy()
            new_occu[site] = species
            prop = self.compute_property(occu)
            new_prop = self.compute_property(new_occu)
            forward_drift += (new_prop - prop) - delta_prop
            reverse_flips = [(site, occu[site])]
            trajectory.append((prop - new_prop, new_occu, reverse_flips))
            occu = new_occu

        forward_drift /= iterations
        reverse_drift = sum(dp - self.compute_property_change(o, f)
                            for dp, o, f in trajectory) / iterations
        return forward_drift, reverse_drift

    def _delta_corr(self, flips, occu):
        """
        Compute the change in the correlation vector from a list of flips.

        Args:
            flips list(tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.
            occu (ndarray):
                encoded occupancy array

        Returns:
            array: change in correlation vector
        """
        occu_i = occu
        delta_corr = np.zeros(self.n_orbit_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            orbits = self._orbits_by_sites[f[0]]
            delta_corr += self._dcorr_single_flip(occu_f, occu_i,
                                                  self.n_orbit_functions,
                                                  orbits)
            occu_i = occu_f

        return delta_corr

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_expansion': self.cluster_expansion.as_dict(),
             'indicator': self.indicator_opt}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        return cls(ClusterExpansion.from_dict(d['cluster_expansion']),
                   np.array(d['supercell_matrix']),
                   optimize_indicator=d['indicator'])
