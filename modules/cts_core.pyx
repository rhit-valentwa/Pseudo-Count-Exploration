# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport log

ctypedef cnp.uint32_t UINT32_t
ctypedef cnp.int64_t INT64_t
ctypedef cnp.float32_t FLOAT32_t

cdef int OUT_H = 42
cdef int OUT_W = 42
cdef int N_LEVELS = 8
cdef int CTS_DEPTH = 4
cdef int N_NODES = 4681  # 1 + 8 + 8^2 + 8^3 + 8^4


cdef inline int level_start(int depth) noexcept:
    if depth == 0:
        return 0
    elif depth == 1:
        return 1
    elif depth == 2:
        return 9
    elif depth == 3:
        return 73
    else:
        return 585


cdef inline int child_index(int node, int depth, int ctx_symbol) noexcept:
    cdef int pos = node - level_start(depth)
    return level_start(depth + 1) + pos * N_LEVELS + ctx_symbol


cdef double prob_node(
    int loc,
    int node,
    int depth,
    int[:] ctx,
    int symbol,
    UINT32_t[:, :, :] counts,
    UINT32_t[:, :] totals,
    FLOAT32_t[:, :] w_base,
    FLOAT32_t[:, :] w_split,
) noexcept:
    cdef double c = <double>counts[loc, node, symbol]
    cdef double t = <double>totals[loc, node]
    cdef double p_base = (c + 0.5) / (t + 0.5 * N_LEVELS)
    cdef int child
    cdef double p_split

    if depth >= CTS_DEPTH:
        return p_base

    child = child_index(node, depth, ctx[depth])
    p_split = prob_node(
        loc,
        child,
        depth + 1,
        ctx,
        symbol,
        counts,
        totals,
        w_base,
        w_split,
    )

    return (<double>w_base[loc, node]) * p_base + (<double>w_split[loc, node]) * p_split


cdef double update_node(
    int loc,
    int node,
    int depth,
    int[:] ctx,
    int symbol,
    UINT32_t[:, :, :] counts,
    UINT32_t[:, :] totals,
    UINT32_t[:, :] node_updates,
    FLOAT32_t[:, :] w_base,
    FLOAT32_t[:, :] w_split,
) noexcept:
    """
    Recursive CTS update.

    Returns the predictive probability assigned before the update.
    """
    cdef double c = <double>counts[loc, node, symbol]
    cdef double t = <double>totals[loc, node]
    cdef double p_base = (c + 0.5) / (t + 0.5 * N_LEVELS)
    cdef double p_split = 0.0
    cdef double old_base, old_split, p_mix
    cdef double post_base, post_split
    cdef double switch_rate
    cdef double new_base, new_split, norm
    cdef int child
    cdef unsigned int m

    if depth >= CTS_DEPTH:
        counts[loc, node, symbol] += 1
        totals[loc, node] += 1
        node_updates[loc, node] += 1
        return p_base

    child = child_index(node, depth, ctx[depth])
    p_split = update_node(
        loc,
        child,
        depth + 1,
        ctx,
        symbol,
        counts,
        totals,
        node_updates,
        w_base,
        w_split,
    )

    old_base = <double>w_base[loc, node]
    old_split = <double>w_split[loc, node]
    p_mix = old_base * p_base + old_split * p_split

    if p_mix < 1e-300:
        p_mix = 1e-300

    # Posterior over base-vs-split experts after seeing the symbol.
    post_base = old_base * p_base / p_mix
    post_split = old_split * p_split / p_mix

    # Update this node's own KT estimator after using its pre-update prob.
    counts[loc, node, symbol] += 1
    totals[loc, node] += 1
    node_updates[loc, node] += 1
    m = node_updates[loc, node]

    # CTS-style decreasing switch schedule.
    switch_rate = 1.0 / (<double>m + 1.0)

    new_base = (1.0 - switch_rate) * post_base + switch_rate * post_split
    new_split = switch_rate * post_base + (1.0 - switch_rate) * post_split
    norm = new_base + new_split

    if norm > 0.0:
        w_base[loc, node] = <FLOAT32_t>(new_base / norm)
        w_split[loc, node] = <FLOAT32_t>(new_split / norm)
    else:
        w_base[loc, node] = <FLOAT32_t>0.5
        w_split[loc, node] = <FLOAT32_t>0.5

    return p_mix


def cts_log_prob_update(
    cnp.ndarray[INT64_t, ndim=2] frame,
    cnp.ndarray[UINT32_t, ndim=3] counts,
    cnp.ndarray[UINT32_t, ndim=2] totals,
    cnp.ndarray[UINT32_t, ndim=2] node_updates,
    cnp.ndarray[FLOAT32_t, ndim=2] w_base,
    cnp.ndarray[FLOAT32_t, ndim=2] w_split,
):
    """
    Compute log rho_n(x), update CTS with x, then compute log rho'_n(x).

    frame:        int64  shape (42, 42), values 0..7
    counts:       uint32 shape (1764, 4681, 8)
    totals:       uint32 shape (1764, 4681)
    node_updates: uint32 shape (1764, 4681)
    w_base:       float32 shape (1764, 4681)
    w_split:      float32 shape (1764, 4681)
    """
    cdef int i, j, loc, symbol
    cdef int up, left, up_left, down_left
    cdef int ctx_arr[4]
    cdef int[:] ctx = ctx_arr
    cdef double p_before, p_after
    cdef double log_before = 0.0
    cdef double log_after = 0.0

    for i in range(OUT_H):
        for j in range(OUT_W):
            loc = i * OUT_W + j
            symbol = <int>frame[i, j]

            up = <int>frame[i - 1, j] if i > 0 else 0
            left = <int>frame[i, j - 1] if j > 0 else 0
            up_left = <int>frame[i - 1, j - 1] if i > 0 and j > 0 else 0
            down_left = <int>frame[i + 1, j - 1] if i + 1 < OUT_H and j > 0 else 0

            # Parent order from pseudo-count paper Appendix C.1:
            #   (i-1,j), (i,j-1), (i-1,j-1), (i+1,j-1)
            ctx_arr[0] = up
            ctx_arr[1] = left
            ctx_arr[2] = up_left
            ctx_arr[3] = down_left

            p_before = update_node(
                loc,
                0,
                0,
                ctx,
                symbol,
                counts,
                totals,
                node_updates,
                w_base,
                w_split,
            )

            if p_before < 1e-300:
                p_before = 1e-300

            log_before += log(p_before)

            p_after = prob_node(
                loc,
                0,
                0,
                ctx,
                symbol,
                counts,
                totals,
                w_base,
                w_split,
            )

            if p_after < 1e-300:
                p_after = 1e-300

            log_after += log(p_after)

    return log_before, log_after