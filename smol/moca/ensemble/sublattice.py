"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. It more rigourously represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

from collections import OrderedDict
import numpy as np
from monty.json import MSONable
from pymatgen import Specie, DummySpecie, Element
from smol.cofe.configspace.domain import Vacancy

def get_sublattices(processor):
    """Get a list of sublattices from a processor

    Args:
        processor (Processor):
            A processor object to extract sublattices from
    Returns:
        list of Sublattice
    """
    return [Sublattice(site_space,
                       np.array([i for i, sp in
                                 enumerate(processor.allowed_species)
                                if sp == list(site_space.keys())]))
            for site_space in processor.unique_site_spaces]


# TODO consider adding the inactive sublattices?
class Sublattice(MSONable):
    """Sublattice class.

     A Sublattice is used to represent a subset of supercell sites that have
     the same site space.

     Attributes:
         site_space (OrderedDict):
            Ordered dict with the allowed species and their random
            state composition. See definitions in cofe.cofigspace.basis
         sites (ndarray):
            array of site indices for all sites in sublattice
         active_sites (ndarray):
            array of site indices for all unrestricted sites in the sublattice.
         restricted_sites (ndarray):
            list of site indices for all restricted sites in the sublattice.
            restricted sites are excluded from flip proposals.

    """

    def __init__(self, site_space, sites):
        """Initialize Sublattice.

        Args:
            site_space (OrderedDict):
                An ordered dict with the allowed species and their random
                state composition. See definitions in cofe.cofigspace.basis
            sites (ndarray):
                array with the site indices
        """
        self.sites = sites
        self.site_space = site_space
        self.active_sites = sites.copy()
        self.restricted_sites = []

    @property
    def species(self):
        """Get allowed species for sites in sublattice."""
        return tuple(self.site_space.keys())

    @property
    def encoding(self):
        """Get the encoding for the allowed species."""
        return list(range(len(self.site_space)))

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.
        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        self.active_sites = np.array([i for i in self.active_sites
                                      if i not in sites])
        self.restricted_sites += [i for i in sites
                                  if i not in self.restricted_sites]

    def reset_restricted_sites(self):
        """Resets all restricted sites to active."""
        self.active_sites = self.sites.copy()
        self.restricted_sites = []

    def __str__(self):
        """Pretty print the sublattice species."""
        string = f'Sublattice\n Site space: {dict(self.site_space)}\n'
        string += f' Number of sites: {len(self.sites)}\n'
        return string

    def __repr__(self):
        """Repr for nice viewing."""
        rep = f'Sublattice Summary \n\n   site_space: {self.site_space}\n\n'
        rep += f'   sites: {self.sites}\n\n active_sites: {self.active_sites}'
        return rep

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'site_space': tuple((s.as_dict(), m)
                                 for s, m in self.site_space.items()),
             'sites': self.sites.tolist(),
             'active_sites': self.active_sites.tolist(),
             'restricted_sites': self.restricted_sites}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            Sublattice
        """
        site_space = []
        for sp, m in d['site_space']:
            if ("oxidation_state" in sp
                    and Element.is_valid_symbol(sp["element"])):
                sp = Specie.from_dict(sp)
            elif "oxidation_state" in sp:
                if sp['@class'] == 'Vacancy':
                    sp = Vacancy.from_dict(sp)
                else:
                    sp = DummySpecie.from_dict(sp)
            else:
                sp = Element(sp["element"])
            site_space.append((sp, m))
        sublattice = cls(OrderedDict(site_space),
                         sites=np.array(d['sites']))
        sublattice.active_sites = np.array(d['active_sites'])
        sublattice.restricted_sites = d['restricted_sites']
        return sublattice
