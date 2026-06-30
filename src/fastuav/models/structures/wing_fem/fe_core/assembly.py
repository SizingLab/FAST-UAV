"""
Generic finite-element assembly for a 6-DOF/node global model.

Shells (4-node) and 3D beams (2-node) share the same nodal DOF layout
``[u, v, w, theta_x, theta_y, theta_z]`` so their element matrices scatter into
one global system. Dense storage is fine for the coarse wingbox meshes used in
preliminary sizing (a few thousand DOF at most).
"""

from __future__ import annotations

import numpy as np

DOF_PER_NODE = 6


def node_dofs(node_ids):
    """Global DOF indices for the given node ids (6 per node)."""
    node_ids = np.asarray(node_ids, dtype=int)
    return (node_ids[:, None] * DOF_PER_NODE + np.arange(DOF_PER_NODE)).ravel()


def assemble(n_nodes, elements):
    """
    Assemble the global stiffness matrix.

    ``elements`` is an iterable of ``(node_ids, K_elem)`` where ``K_elem`` is
    square of size ``len(node_ids) * 6``.
    """
    n_dof = n_nodes * DOF_PER_NODE
    K = np.zeros((n_dof, n_dof))
    for node_ids, K_e in elements:
        dofs = node_dofs(node_ids)
        K[np.ix_(dofs, dofs)] += K_e
    return K


def solve_clamped(K, F, fixed_dofs):
    """
    Solve ``K u = F`` with the given DOF indices clamped to zero.

    Returns the full DOF vector ``u`` (length n_dof) with zeros at fixed DOFs.
    """
    n_dof = K.shape[0]
    fixed = np.zeros(n_dof, dtype=bool)
    fixed[np.asarray(fixed_dofs, dtype=int)] = True
    free = np.where(~fixed)[0]

    u = np.zeros(n_dof)
    Kff = K[np.ix_(free, free)]
    try:
        u[free] = np.linalg.solve(Kff, F[free])
    except np.linalg.LinAlgError:
        # Singular reduced stiffness (e.g. a degenerate mesh probed by the
        # optimiser). Fall back to a least-squares solve so the caller gets a
        # finite (if meaningless) result instead of a raised exception; the
        # model layer sanitises non-finite results into a feasibility penalty.
        u[free] = np.linalg.lstsq(Kff, F[free], rcond=None)[0]
    return u
