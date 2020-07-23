"""Implementation of base processor classes for a fixed size super cell.

Processor classes are used to represent a configuration domain for a fixed
sized supercell and should implement a "fast" way to compute the property
they represent or changes in said property from site flips. Things necessary
to run Monte Carlo sampling.

Processor classes should inherit from the BaseProcessor class. Processors
can be combined into composite processors for mixed models.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

from pymatgen import Structure, PeriodicSite
from monty.json import MSONable
from smol.cofe.configspace.basis import get_site_spaces, get_allowed_species


class BaseProcessor(MSONable, metaclass=ABCMeta):
    """Abstract base class for processors.

    A processor is used to provide a quick way to calculated energy differences
    (probability ratio's) between two adjacent configurational states for a
    fixed system size/supercell.

    Attributes:
        unique_site_spaces (tuple):
            Tuple of all the distinct site spaces.
        allowed_species (list):
            A list of tuples of the allowed species at each site.
        size (int):
            Number of prims in the supercell structure.
    """
    def __init__(self, cluster_subspace, supercell_matrix):
        """Initialize a BaseProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
        """
        self._subspace = cluster_subspace
        self._structure = self._subspace.structure.copy()
        self._structure.make_supercell(supercell_matrix)
        self._scmatrix = supercell_matrix

        # this can be used (maybe should) to check if a flip is valid
        site_spaces = get_site_spaces(self._subspace.expansion_structure)
        self.unique_site_spaces = tuple(OrderedDict(space) for space in
                                        set(tuple(spaces.items())
                                            for spaces in site_spaces))

        self.allowed_species = get_allowed_species(self.structure)
        self.size = self._subspace.num_prims_from_matrix(supercell_matrix)

    @property
    def cluster_subspace(self):
        """Get the underlying cluster subspace."""
        return self._subspace

    @property
    def structure(self):
        """Get the underlying supercell disordered structure."""
        return self._structure

    @property
    def supercell_matrix(self):
        """Get the give supercell matrix."""
        return self._scmatrix

    @property
    def size(self):
        """Return the size of the processor (number of prims)"""
        return

    @abstractmethod
    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return

    @abstractmethod
    def compute_property_change(self, occupancy, flips):
        """Compute change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return

    def occupancy_from_structure(self, structure):
        """Get the occupancy array for a given structure.

        The structure must strictly be a supercell of the prim according to the
        processor's supercell matrix.

        Args:
            structure (Structure):
                A pymatgen structure (related to the cluster-expansion prim
                by the supercell matrix passed to the processor)
        Returns: encoded occupancy array
            list
        """
        occu = self._subspace.occupancy_from_structure(structure,
                                                      scmatrix=self.supercell_matrix)  # noqa
        return self.encode_occupancy(occu)

    def structure_from_occupancy(self, occu):
        """Get pymatgen Structure from an occupancy array.

        Args:
            occu (ndarray):
                encoded occupancy array

        Returns:
            Structure
        """
        occu = self.decode_occupancy(occu)
        sites = []
        for sp, s in zip(occu, self.structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occupancy(self, occu):
        """Encode occupancy array of species str to ints."""
        return np.array([species.index(sp) for species, sp
                        in zip(self.allowed_species, occu)])

    def decode_occupancy(self, enc_occu):
        """Decode an encoded occupancy array of int to species str."""
        return [species[i] for i, species in
                zip(enc_occu, self.allowed_species)]

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_subspace': self.cluster_subspace.as_dict(),
             'supercell_matrix': self.supercell_matrix.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a processor from serialized MSONable dict."""
        # is this good design?
        try:
            for derived in cls.__subclasses__():
                if derived.__name__ == d['@class']:
                    return derived.from_dict(d)
        except KeyError:
            raise NameError(f"Unable to instantiate {d['@class']}.")
