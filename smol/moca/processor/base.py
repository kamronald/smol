"""Implementation of base processor classes for a fixed size supercell.

Processor classes are used to represent a configuration domain for a fixed
size supercell and should implement a "fast" way to compute the property
they represent or changes in said property from site flips. The processor
also contains other things necessary to run Monte Carlo sampling.

Processor classes should inherit from the BaseProcessor class. Processors
can be combined into composite processor for mixed models.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
from functools import lru_cache

import numpy as np
from monty.json import MSONable
from pymatgen.core import PeriodicSite, Structure, Element

from smol.cofe.space import Vacancy, get_allowed_species, get_site_spaces
from smol.moca.sublattice import Sublattice
from smol.utils import get_subclasses


class Processor(MSONable, metaclass=ABCMeta):
    """Abstract base class for processors.

    A processor is used to provide a quick way to calculated energy differences
    (probability ratio's) between two configurational states for a fixed system
    size/supercell.

    Attributes:
        unique_site_spaces (tuple):
            Tuple of all the distinct site spaces.
        allowed_species (list):
            A list of tuples of the allowed species at each site.
        size (int):
            Number of prims in the supercell structure.
    """

    def __init__(self, cluster_subspace, supercell_matrix, coefficients):
        """Initialize a BaseProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                ClusterExpansion prim structure.
            coefficients:
                single or array of fit coefficients.
        """
        self._subspace = cluster_subspace
        self._structure = self._subspace.structure.copy()
        self._structure.make_supercell(supercell_matrix)
        self._scmatrix = np.array(supercell_matrix)

        self.coefs = np.array(coefficients)
        # if scalar force array to have 1 dimension (1,)
        if len(self.coefs.shape) == 0:
            self.coefs = self.coefs[np.newaxis]

        # this can be used (maybe should) to check if a flip is valid
        site_spaces = set(get_site_spaces(self.structure))
        self.unique_site_spaces = tuple(site_spaces)
        self.active_site_spaces = tuple(
            space for space in self.unique_site_spaces if len(space) > 1
        )

        self.allowed_species = get_allowed_species(self.structure)
        self.size = self._subspace.num_prims_from_matrix(supercell_matrix)

    @property
    def cluster_subspace(self):
        """Get the underlying cluster subspace."""
        return self._subspace

    @property
    def structure(self):
        """Get the underlying disordered supercell structure."""
        return self._structure

    @property
    @lru_cache(maxsize=None)
    def num_sites(self):
        """Get total number of sites in supercell."""
        return len(self.cluster_subspace.structure) * self.size

    @property
    def supercell_matrix(self):
        """Get the supercell matrix."""
        return self._scmatrix

    @abstractmethod
    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return

    @abstractmethod
    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the feature vector from a list of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in correlation vector
        """
        return

    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return np.dot(self.coefs, self.compute_feature_vector(occupancy))

    def compute_property_change(self, occupancy, flips):
        """Compute change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of (index of site, specie code to set) tuples

        Returns:
            float:  property difference between initial and final states
        """
        return np.dot(self.coefs, self.compute_feature_vector_change(occupancy, flips))

    def occupancy_from_structure(self, structure):
        """Get the occupancy array for a given structure.

        The structure must strictly be a supercell of the prim according to the
        processor's supercell matrix.

        Args:
            structure (Structure):
                a pymatgen structure (related to the cluster expansion prim
                by the supercell matrix passed to the processor)
        Returns: encoded occupancy string
            np.ndarray[int]
        """
        occu = self._subspace.occupancy_from_structure(
            structure, scmatrix=self.supercell_matrix
        )
        return self.encode_occupancy(occu)

    def structure_from_occupancy(self, occupancy):
        """Get Structure from an occupancy string.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            Structure
        """
        occupancy = self.decode_occupancy(occupancy)
        sites = []
        for spec, site in zip(occupancy, self.structure):
            if spec != Vacancy():
                site = PeriodicSite(spec, site.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occupancy(self, occupancy):
        """Encode occupancy string of Species object to ints."""
        # TODO check if setting to np.intc improves speed
        return np.array(
            [
                species.index(spec)
                for species, spec in zip(self.allowed_species, occupancy)
            ],
            dtype=int,
        )

    def decode_occupancy(self, encoded_occupancy):
        """Decode an encoded occupancy string of int to Species object."""
        return [
            species[i] for i, species in zip(encoded_occupancy, self.allowed_species)
        ]

    def get_sublattices(self):
        """Get a list of sublattices from the processor.

        Initialized as the default encoding, but encoding can be changed
        in Ensemble (for example, when a sub-lattice is split by occupancy,
        usually seen in a de-lithiation MC). Therefore, these sub-lattices,
        and self.unique_site_spaces are not always consistent with the
        sub-lattices in the ensemble. Use them carefully!
        Returns:
            list of Sublattice
        """
        return [
            Sublattice(
                site_space,
                np.array(
                    [
                        i
                        for i, spec in enumerate(self.allowed_species)
                        if spec == list(site_space.keys())
                    ]
                ),
            )
            for site_space in self.unique_site_spaces
        ]

    def gen_random_occu(self, rng=None):
        """Generate a random encoded occupancy according to a list of sublattices.

            Args:
                sublattices (Sequence of Sublattice):
                    A sequence of sublattices
                rng (optional): {None, int, array_like[ints], SeedSequence,
                                BitGenerator, Generator}
                                A RNG, seed or otherwise to initialize default_rng

            Returns:
                ndarray: encoded occupancy
            """
        sublattices = self.get_sublattices()
        num_sites = sum(len(sl.sites) for sl in sublattices)
        rand_occu = np.zeros(num_sites, dtype=int)
        rng = np.random.default_rng(rng)
        for sublatt in sublattices:
            rand_occu[sublatt.sites] = rng.choice(
                sublatt.encoding, size=len(sublatt.sites), replace=True
            )
        return rand_occu

    def gen_random_neutral_occu(self, lam, max_it=10000, rng=None):
        """Generate a random encoded occupancy according to a list of sublattices.

            Args:
                sublattices (Sequence of Sublattice):
                    A sequence of sublattices
                rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator
                                , Generator},
                    A RNG, seed or otherwise to initialize default_rng

            Returns:
                ndarray: encoded occupancy
            """

        rng = np.random.default_rng(rng)
        sublattices = self.get_sublattices()

        def get_charge(sp):
            if isinstance(sp, (Element, Vacancy)):
                return 0
            else:
                return sp.oxi_state

        def charge(occu, sublattices):
            charge = 0
            for sl in sublattices:
                for site in sl.sites:
                    sp_id = sl.encoding.tolist().index(occu[site])
                    charge += get_charge(sl.species[sp_id])
            return charge

        def flip(occu, sublattices, lam=10):
            actives = [s for s in sublattices if s.is_active]
            sl = rng.choice(actives)
            site = rng.choice(sl.sites)
            code = rng.choice(list(set(sl.encoding) - {occu[site]}))
            occu_next = occu.copy()
            occu_next[site] = code
            C = charge(occu, sublattices)
            C_next = charge(occu_next, sublattices)
            accept = np.log(rng.random()) < -lam * (C_next ** 2 - C ** 2)
            if accept and C != 0:
                return occu_next.copy(), C_next
            else:
                return occu.copy(), C

        occu = self.gen_random_occu(rng=rng)

        for _ in range(max_it):
            occu, C = flip(occu, sublattices, lam=lam)
            if C == 0:
                return occu.copy()

        raise TimeoutError(f"Cannot generate a neutral occupancy in {max_it} flips!")

    def compute_average_drift(self, iterations=1000):
        """Compute average forward and reverse drift for the given property.

        This is a sanity check function. The drift value should be very, very,
        very small, the smaller the better (think machine precision values).

        The average drift is the difference between the quick routine used
        for MC to get a property difference from a single flip and the
        change in that property from explicitly calculating it fully for the
        initial state and the flipped state.

        Args:
            iterations (int): optional
                number of iterations/flips to compute.

        Returns:
            tuple: (float, float) forward and reverse average property drift
        """
        rng = np.random.default_rng()
        forward_drift, reverse_drift = 0.0, 0.0
        trajectory = []
        occu = [rng.choice(species) for species in self.allowed_species]
        occu = self.encode_occupancy(occu)
        for _ in range(iterations):
            site = rng.integers(self.size)
            species = set(range(len(self.allowed_species[site]))) - {occu[site]}
            species = rng.choice(list(species))
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
        reverse_drift = (
            sum(dp - self.compute_property_change(o, f) for dp, o, f in trajectory)
            / iterations
        )
        return forward_drift, reverse_drift

    def __len__(self):
        """Get number of sites processor supercell."""
        return self.num_sites

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        proc_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "cluster_subspace": self.cluster_subspace.as_dict(),
            "supercell_matrix": self.supercell_matrix.tolist(),
            "coefficients": np.array(self.coefs).tolist(),
        }
        return proc_d

    @classmethod
    def from_dict(cls, d):
        """Create a processor from serialized MSONable dict."""
        # is this good design?
        try:
            subclass = get_subclasses(cls)[d["@class"]]
        except KeyError as err:
            raise NameError(
                f"{d['@class']} is not implemented or is not a subclass of " f"{cls}."
            ) from err
        return subclass.from_dict(d)
