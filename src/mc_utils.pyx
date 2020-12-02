"""
General cython utility functions to calculate correlation vectors and local
changes in correlation vectors a tad bit faster than pure python.
"""

__author__ = "Luis Barroso-Luque, William D. Richards"

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef corr_from_occupancy(const long[::1] occu,
                          const int n_bit_orderings,
                          orbit_list):
    """Computes the correlation vector for a given encoded occupancy string.

    Args:
        occu (ndarray):
            encoded occupancy vector
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        orbit_list:
            Information of all orbits that include the flip site.
            (bit_combos, orbit id, site indices, bases array)

    Returns: array
        correlation vector difference
    """
    cdef int i, j, k, I, J, K, n
    cdef double p, pi
    cdef const long[:, ::1] inds
    cdef const long[:, ::1] bits
    cdef const double[:, :, ::1] bases
    out = np.zeros(n_bit_orderings)
    cdef double[:] o_view = out
    o_view[0] = 1  # empty cluster

    for n, combos, bases, inds in orbit_list:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        for bits in combos:
            J = bits.shape[0]
            p = 0
            for i in range(I):
                for j in range(J):
                    pi = 1
                    for k in range(K):
                        pi *= bases[k, bits[j, k], occu[inds[i, k]]]
                    p += pi
            o_view[n] = p / (I * J)
            n += 1
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef general_delta_corr_single_flip(const long[::1] occu_f,
                                     const long[::1] occu_i,
                                     const int n_bit_orderings,
                                     site_orbit_list):
    """Computes the correlation difference between two occupancy vectors.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (bit_combos, orbit id, site indices, ratio, bases array)


    Returns:
        ndarray: correlation vector difference
    """
    cdef int i, j, k, I, J, K, n
    cdef double p, pi, pf, r
    cdef const long[:, ::1] inds
    cdef const long[:, ::1] bits
    cdef const double[:, :, ::1] bases
    out = np.zeros(n_bit_orderings)
    cdef double[::1] o_view = out

    for n, r, combos, bases, inds in site_orbit_list:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        for bits in combos:
            J = bits.shape[0]
            p = 0
            for i in range(I):
                for j in range(J):
                    pf = 1
                    pi = 1
                    for k in range(K):
                        pf *= bases[k, bits[j, k], occu_f[inds[i, k]]]
                        pi *= bases[k, bits[j, k], occu_i[inds[i, k]]]
                    p += (pf - pi)
            o_view[n] = p / r / (I * J)
            n += 1
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef indicator_delta_corr_single_flip(const long[::1] occu_f,
                                       const long[::1] occu_i,
                                       const int n_bit_orderings,
                                       site_orbit_list):
    """Local change in indicator basis correlation vector from single flip.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (bit_combos, orbit id, site indices, ratio, bases array)

    Returns:
        ndarray: correlation vector difference
    """
    cdef int i, j, k, I, J, K, n
    cdef bint ok
    cdef const long[:, ::1] b, inds
    out = np.zeros(n_bit_orderings)
    cdef double[::1] o_view = out
    cdef double r, o

    for n, r, combos, _, inds in site_orbit_list:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        for b in combos:
            J = b.shape[0] # index within bit array
            o = 0
            for i in range(I):
                for j in range(J):
                    ok = True
                    for k in range(K):
                        if occu_f[inds[i, k]] != b[j, k]:
                            ok = False
                            break
                    if ok:
                        o += 1

                    ok = True
                    for k in range(K):
                        if occu_i[inds[i, k]] != b[j, k]:
                            ok = False
                            break
                    if ok:
                        o -= 1

            o_view[n] = o / r / (I * J)
            n += 1
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef delta_ewald_single_flip(const long[::1] occu_f,
                              const long[::1] occu_i,
                              const double[:, ::1] ewald_matrix,
                              const long[:, ::1] ewald_inds,
                              const int site_ind):
    """Compute the change in electrostatic interaction energy from a flip.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        ewald_matrix (ndarray):
            Ewald matrix for electrostatic interactions
        ewald_inds (ndarray):
            2D array of indices corresponding to a specific site occupation
            in the ewald matrix
        site_ind (int):
            site index for site being flipped

    Returns:
        float: electrostatic interaction energy difference
    """
    cdef int i, j, k, add, sub
    cdef bint ok
    cdef double out = 0
    cdef double out_k

    # values of -1 are vacancies and hence don't have ewald indices
    add = ewald_inds[site_ind, occu_f[site_ind]]
    sub = ewald_inds[site_ind, occu_i[site_ind]]

    for k in range(occu_f.shape[0]):
        i = ewald_inds[k, occu_f[k]]
        out_k = 0
        if i != -1 and add != -1:
            if i != add:
                out_k = out_k + 2 * ewald_matrix[i, add]
            else:
                out_k = out_k + ewald_matrix[i, add]

        j = ewald_inds[k, occu_i[k]]
        if j != -1 and sub != -1:
            if j != sub:
                out_k = out_k - 2 * ewald_matrix[j, sub]
            else:
                out_k = out_k - ewald_matrix[j, sub]
        out += out_k
    return out
