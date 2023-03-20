"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"

cimport numpy as np

from smol.utils.cluster_utils.container cimport (
    FloatArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)


cdef class ClusterSpaceEvaluator(OrbitContainer):
    """ClusterSpaceEvaluator class is used to compute the correlation vectors.

    This extension type should rarely be used directly. Instead, use the
    ClusterSubspace class to create a cluster subspace instance and compute correlation
    vectors using the ClusterSubspace.corr_from_occupancy method.
    """

    cpdef np.ndarray[np.float64_t, ndim=1] corr_from_occupancy(
            self,
            const long[::1] occu,
            const int num_corr_functions,  # TODO save this in the evaluator
            IntArray2DContainer cluster_indices,
    )

    cpdef np.ndarray[np.float64_t, ndim=1] interactions_from_occupancy(
            self, const long[::1] occu,
            const double offset,
            FloatArray1DContainer cluster_interaction_tensors,
            IntArray2DContainer cluster_indices,
    )
