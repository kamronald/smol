import os
import pytest
import numpy as np
from pymatgen.core import Structure
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace, StructureWrangler
from smol.moca import CanonicalEnsemble, SemiGrandEnsemble, \
    ClusterExpansionProcessor, EwaldProcessor, CompositeProcessor
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.basis import BasisIterator
from smol.utils import get_subclasses
from tests.utils import gen_fake_training_data

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures and other parameters for creating fixtures
files = ['AuPd_prim.json', 'CrFeW_prim.json', 'LiCaBr_prim.json',
         'LiMOF_prim.json', 'LiMnTiVOF_prim.json']

test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]
basis_iterator_names = list(get_subclasses(BasisIterator))
ensembles = [CanonicalEnsemble, SemiGrandEnsemble]

# fixture definitions used in several tests


@pytest.fixture(scope='package')
def cluster_cutoffs():
    # parametrize this if we ever want to test on different cutoffs
    return {2: 6, 3: 5, 4: 4}


@pytest.fixture(params=test_structures, scope='package')
def structure(request):
    return request.param


@pytest.fixture(scope='package')
def expansion_structure(structure):
    sites = [site for site in structure if site.species.num_atoms < 0.99
             or len(site.species) > 1]
    return Structure.from_sites(sites)


@pytest.fixture(scope='package')
def single_structure():
    # this is the LiCaBr structure used for some fixed tests
    return test_structures[2]


@pytest.fixture(params=test_structures, scope='package')
def cluster_subspace(cluster_cutoffs, request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs=cluster_cutoffs, supercell_size='volume')
    return subspace


@pytest.fixture(params=test_structures, scope='package')
def cluster_subspace_ewald(cluster_cutoffs, request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs=cluster_cutoffs, supercell_size='volume')
    subspace.add_external_term(EwaldTerm())
    return subspace


@pytest.fixture(scope='package')
def single_subspace(single_structure):
    # this is a subspace with the LiCaBr structure for some fixed tests
    subspace = ClusterSubspace.from_cutoffs(
        single_structure, cutoffs={2: 6, 3: 5}, supercell_size='volume')
    return subspace


@pytest.fixture(scope='module')
def ce_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    return ClusterExpansionProcessor(cluster_subspace, supercell_matrix=scmatrix,
                       coefficients=coefs)


@pytest.fixture(scope='module')
def composite_processor(cluster_subspace_ewald):
    coefs = 2 * np.random.random(cluster_subspace_ewald.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace_ewald, supercell_matrix=scmatrix)
    proc.add_processor(ClusterExpansionProcessor(cluster_subspace_ewald, scmatrix,
                                   coefficients=coefs[:-1]))
    proc.add_processor(EwaldProcessor(cluster_subspace_ewald, scmatrix, EwaldTerm(),
                                      coefficient=coefs[-1]))
    return proc


@pytest.fixture(params=ensembles, scope='module')
def ensemble(composite_processor, request):
    if request.param is SemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in composite_processor.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(composite_processor, **kwargs)


@pytest.fixture(scope='module')
def single_canonical_ensemble(single_subspace):
    coefs = np.random.random(single_subspace.num_corr_functions)
    proc = ClusterExpansionProcessor(single_subspace, 4 * np.eye(3), coefs)
    return CanonicalEnsemble(proc)


@pytest.fixture(params=basis_iterator_names, scope='package')
def basis_name(request):
    return request.param.split('Iterator')[0]


@pytest.fixture
def supercell_matrix():
    m = np.random.randint(-3, 3, size=(3, 3))
    while abs(np.linalg.det(m)) < 1E-6:  # make sure not singular
        m = np.random.randint(-3, 3, size=(3, 3))
    return m


@pytest.fixture(scope='module')
def structure_wrangler(single_subspace):
    wrangler = StructureWrangler(single_subspace)
    for struct, energy in gen_fake_training_data(
            wrangler.cluster_subspace.structure, n=10):
        wrangler.add_data(struct, {'energy': energy}, weights={'random': 2.0})
    return wrangler
