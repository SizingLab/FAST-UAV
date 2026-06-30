"""
1D Euler-Bernoulli bending solver (pure NumPy).

This is the shared backbone for the ``tube_spar_foam`` wing model. It solves a
clamped-root cantilever under a distributed transverse load q(y), given the
*effective* bending stiffness EI(y) sampled at element midpoints. Decoupling
the solver from the section model lets the bare tube spar and the composite
tube+sandwich section reuse exactly the same (verified) FE assembly.

The element stiffness/recovery routines accept ``(E, Iy)``; we pass ``E = EI``
and ``Iy = 1`` so that ``E*Iy`` equals the supplied effective EI.
"""

from __future__ import annotations

import numpy as np

from .beam_element import BeamElement2D, tube_section_properties


def solve_bending_beam(y_nodes, EI_mid, q_nodes, element_class=BeamElement2D):
    """
    Solve K u = F for a clamped-root bending beam.

    Parameters
    ----------
    y_nodes : node coordinates root->tip, shape (n_nodes,) [m] (ascending).
    EI_mid  : effective bending stiffness at each element midpoint,
              shape (n_nodes - 1,) [N*m^2].
    q_nodes : distributed transverse load at nodes, shape (n_nodes,) [N/m].
    element_class : 2-DOF/node bending element (default BeamElement2D).

    Returns
    -------
    dict with 'u' (nodal DOFs), 'M_bending' (per-element midpoint moment [N*m]),
    and 'w_tip' (tip transverse displacement [m]).
    """
    y_nodes = np.asarray(y_nodes, dtype=float)
    EI_mid = np.asarray(EI_mid, dtype=float)
    q_nodes = np.asarray(q_nodes, dtype=float)

    n_nodes = y_nodes.size
    n_elem = n_nodes - 1
    dpn = element_class.dof_per_node
    n_dof = dpn * n_nodes

    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    for e in range(n_elem):
        L = y_nodes[e + 1] - y_nodes[e]
        # E := EI_mid[e], Iy := 1  =>  E*Iy = EI_mid[e]
        K_e = element_class.stiffness_matrix(L, EI_mid[e], 0.0, 0.0, 1.0, 1.0, 0.0)
        F_e = element_class.consistent_load_vector(L, q_nodes[e], q_nodes[e + 1])
        i0, i1 = e * dpn, (e + 2) * dpn
        K[i0:i1, i0:i1] += K_e
        F[i0:i1] += F_e

    # Clamp the root node (all its DOFs).
    free = np.arange(dpn, n_dof)
    u = np.zeros(n_dof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    M_bending = np.zeros(n_elem)
    for e in range(n_elem):
        L = y_nodes[e + 1] - y_nodes[e]
        i0, i1 = e * dpn, (e + 2) * dpn
        M_bending[e] = element_class.bending_moment_at_midpoint(L, EI_mid[e], 1.0, u[i0:i1])

    w_tip = u[(n_nodes - 1) * dpn]
    return {"u": u, "M_bending": M_bending, "w_tip": w_tip}


def solve_beam_fem(
    semi_span: float,
    R_root: float,
    R_tip: float,
    t_root: float,
    t_tip: float,
    E: float,
    G: float,
    q_nodes: np.ndarray,
    n_elements: int = 20,
):
    """
    Bare tapered-tube spar convenience wrapper around :func:`solve_bending_beam`.

    Kept for verification against the standalone prototype and for the
    spar-only limit of the ``tube_spar_foam`` model. Returns the tube geometry
    sampled at nodes plus the bending solution.
    """
    n_nodes = n_elements + 1
    y_nodes = np.linspace(0.0, semi_span, n_nodes)
    eta = y_nodes / semi_span
    R_nodes = R_root + (R_tip - R_root) * eta
    t_nodes = t_root + (t_tip - t_root) * eta

    R_mid = 0.5 * (R_nodes[:-1] + R_nodes[1:])
    t_mid = 0.5 * (t_nodes[:-1] + t_nodes[1:])
    Iy_mid = np.array([tube_section_properties(R, t)[1] for R, t in zip(R_mid, t_mid)])
    EI_mid = E * Iy_mid

    sol = solve_bending_beam(y_nodes, EI_mid, q_nodes, element_class=BeamElement2D)
    sol.update({"y_nodes": y_nodes, "R_nodes": R_nodes, "t_nodes": t_nodes})
    return sol
