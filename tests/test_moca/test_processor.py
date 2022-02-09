import pytest
import numpy as np
import numpy.testing as npt
from tests.utils import assert_msonable, gen_random_occupancy
from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca.processor import ClusterExpansionProcessor, OrbitDecompositionProcessor, \
    EwaldProcessor, CompositeProcessor
from smol.moca.processor.base import Processor
from smol.moca.ensemble.sublattice import get_sublattices

RTOL = 0.0  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 2E4 * np.finfo(float).eps
DRIFT_TOL = 10 * np.finfo(float).eps  # tolerance of average drift


@pytest.fixture
def ce_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    return ClusterExpansionProcessor(
        cluster_subspace, supercell_matrix=scmatrix, coefficients=coefs)


@pytest.fixture(params=['real', 'reciprocal', 'point'])
def ewald_processor(cluster_subspace, request):
    coef = np.random.random(1)
    scmatrix = 3 * np.eye(3)
    ewald_term = EwaldTerm(use_term=request.param)
    return EwaldProcessor(
        cluster_subspace, supercell_matrix=scmatrix, coefficient=coef,
        ewald_term=ewald_term)


@pytest.fixture(params=['CE', 'OD'])
def composite_processor(cluster_subspace, request):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    ewald_term = EwaldTerm()
    cluster_subspace.add_external_term(ewald_term)
    proc = CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
    if request.param == 'CE':
        proc.add_processor(
            ClusterExpansionProcessor(cluster_subspace, scmatrix, coefficients=coefs[:-1])
        )
    else:
        expansion = ClusterExpansion(cluster_subspace, coefs)
        proc.add_processor(
            OrbitDecompositionProcessor(cluster_subspace, scmatrix,
                                        expansion.orbit_factor_tensors)
        )
    proc.add_processor(EwaldProcessor(cluster_subspace, scmatrix, ewald_term,
                                      coefficient=coefs[-1]))
    # bind raw coefficients since OD processors do not store them
    # and be able to test computing properties, hacky but oh well
    proc.raw_coefs = coefs
    return proc


# General tests for all processors
# Currently being done only on composites because I can not for the life of
# me figure out a clean way to parametrize with parametrized fixtures or use a
# fixture union from pytest_cases that works.
def test_encode_decode_property(composite_processor):
    occu = gen_random_occupancy(get_sublattices(composite_processor),
                                composite_processor.num_sites)
    decoccu = composite_processor.decode_occupancy(occu)
    for species, space in zip(decoccu, composite_processor.allowed_species):
        assert species in space
    npt.assert_equal(occu, composite_processor.encode_occupancy(decoccu))


def test_get_average_drift(composite_processor):
    forward, reverse = composite_processor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(composite_processor):
    sublattices = get_sublattices(composite_processor)
    occu = gen_random_occupancy(sublattices, composite_processor.num_sites)
    for _ in range(100):
        sublatt = np.random.choice(sublattices)
        site = np.random.choice(sublatt.sites)
        new_sp = np.random.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = composite_processor.compute_property(new_occu)
        prop_i = composite_processor.compute_property(occu)
        dprop = composite_processor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = composite_processor.compute_property_change(new_occu, [(site, old_sp)])
        assert dprop == -1 * rdprop


# TODO implement these
def test_structure_from_occupancy():
    pass


def test_occupancy_from_structure():
    pass


def test_compute_feature_change(composite_processor):
    sublattices = get_sublattices(composite_processor)
    occu = gen_random_occupancy(sublattices, composite_processor.num_sites)
    composite_processor.cluster_subspace.change_site_bases('indicator')
    for _ in range(10):
        sublatt = np.random.choice(sublattices)
        site = np.random.choice(sublatt.sites)
        new_sp = np.random.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = composite_processor.compute_property(new_occu)
        prop_i = composite_processor.compute_property(occu)
        dprop = composite_processor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = composite_processor.compute_property_change(new_occu, [(site, old_sp)])
        assert dprop == -1 * rdprop


def test_compute_property(composite_processor):
    occu = gen_random_occupancy(get_sublattices(composite_processor),
                                composite_processor.num_sites)
    struct = composite_processor.structure_from_occupancy(occu)
    pred = np.dot(composite_processor.raw_coefs,
                  composite_processor.cluster_subspace.corr_from_structure(struct, False))
    assert composite_processor.compute_property(occu) == pytest.approx(pred, abs=ATOL)


def test_msonable(composite_processor):
    occu = gen_random_occupancy(get_sublattices(composite_processor),
                                composite_processor.num_sites)
    d = composite_processor.as_dict()
    pr = Processor.from_dict(d)
    assert composite_processor.compute_property(occu) == pr.compute_property(occu)
    npt.assert_array_equal(composite_processor.coefs, pr.coefs)
    # send in pr bc composite_processor is scoped for function and new random
    # coefficients will be created.
    assert_msonable(pr)


# ClusterExpansionProcessor only tests
def test_compute_feature_vector(ce_processor):
    occu = gen_random_occupancy(get_sublattices(ce_processor),
                                ce_processor.num_sites)
    struct = ce_processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(ce_processor.compute_feature_vector(occu) / ce_processor.size,
                        ce_processor.cluster_subspace.corr_from_structure(struct))

# orbit decomp processor
def test_compute_orbit_factors(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    expansion = ClusterExpansion(cluster_subspace, coefs)
    processor = OrbitDecompositionProcessor(
        cluster_subspace, scmatrix, expansion.orbit_factor_tensors)
    occu = gen_random_occupancy(get_sublattices(processor),
                                processor.num_sites)
    struct = processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(processor.compute_feature_vector(occu) / processor.size,
                        expansion.compute_orbit_factors(struct))


def test_bad_coef_length(cluster_subspace):
    coefs = np.random.random(cluster_subspace.num_corr_functions - 1)
    with pytest.raises(ValueError):
        ClusterExpansionProcessor(cluster_subspace, 5 * np.eye(3), coefficients=coefs)


def test_bad_composite(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
    with pytest.raises(AttributeError):
        proc.add_processor(CompositeProcessor(cluster_subspace,
                                              supercell_matrix=scmatrix))
    with pytest.raises(ValueError):
        proc.add_processor(
            ClusterExpansionProcessor(cluster_subspace, 2 * scmatrix, coefficients=coefs))
    with pytest.raises(ValueError):
        new_cs = cluster_subspace.copy()
        new_cs.change_site_bases("chebyshev")
        proc.add_processor(ClusterExpansionProcessor(new_cs, scmatrix, coefficients=coefs))
    with pytest.raises(ValueError):
        new_cs = cluster_subspace.copy()
        ids = range(1, new_cs.num_corr_functions)
        new_cs.remove_orbit_bit_combos(np.random.choice(ids, size=10))
        proc.add_processor(ClusterExpansionProcessor(new_cs, scmatrix, coefficients=coefs))

