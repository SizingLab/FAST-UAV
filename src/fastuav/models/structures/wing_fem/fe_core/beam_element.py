"""
Beam element formulations for the wing-structure FEM.

Ported from the standalone ``fast_uav_fem`` prototype and extended with a fully
implemented 3D spatial frame element (``BeamElement3D``), used for the spar caps
of the shell wingbox model.

All element classes share a common interface so the assembly logic stays
element-agnostic:

* ``dof_per_node``                              -- DOF count per node
* ``stiffness_matrix(L, E, G, A, Iy, Iz, J)``   -- local element stiffness
* ``consistent_load_vector(L, q_left, q_right)``-- equivalent nodal loads
* ``bending_moment_at_midpoint(L, E, Iy, u_e)`` -- stress-recovery helper

Sign / coordinate convention
----------------------------
Local element axis is ``x`` (along the beam, root->tip for a spanwise spar).
For the 2D element ``w`` is the transverse displacement (positive up) and
``theta = dw/dx``. For the 3D element the local DOFs per node are
``[u, v, w, theta_x, theta_y, theta_z]`` and the element is oriented in 3D via
:func:`BeamElement3D.transformation_matrix`.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cross-section properties (tubular spar, exact thick-walled formula)
# ---------------------------------------------------------------------------


def tube_section_properties(R: float, t: float):
    """
    Cross-section properties of a circular tube with outer radius R and
    wall thickness t (exact, not thin-walled approximation).

    Returns
    -------
    A : area [m^2]
    Iy : second moment of area about the chordwise axis (bending) [m^4]
    Iz : second moment of area about the vertical axis (chordwise bending) [m^4]
    J  : torsion constant for a circular tube (= Iy + Iz) [m^4]
    """
    R_in = R - t
    A = np.pi * (R**2 - R_in**2)
    Iy = (np.pi / 4.0) * (R**4 - R_in**4)
    Iz = Iy  # axisymmetric tube
    J = Iy + Iz  # exact for circular tube
    return A, Iy, Iz, J


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BeamElementBase:
    """
    2-node beam element interface. Subclasses define DOF count and matrices.
    """

    n_nodes: int = 2
    dof_per_node: int = 0  # set by subclass
    n_load_components: int = 0  # number of distributed-load components per span station

    @classmethod
    def n_dof(cls) -> int:
        return cls.n_nodes * cls.dof_per_node

    @classmethod
    def stiffness_matrix(cls, L, E, G, A, Iy, Iz, J):
        raise NotImplementedError

    @classmethod
    def consistent_load_vector(cls, L, q_left, q_right):
        raise NotImplementedError

    @classmethod
    def bending_moment_at_midpoint(cls, L, E, Iy, u_e):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2D Euler-Bernoulli element  (2 DOF/node: w, theta)
# ---------------------------------------------------------------------------


class BeamElement2D(BeamElementBase):
    """
    2-node Euler-Bernoulli beam element, bending only.

    DOFs per node: [w, theta]
        w     -- transverse displacement (positive up)
        theta -- rotation about the chordwise axis (dw/dy)

    Element stiffness matrix (4x4) is the standard EI/L^3 form.
    """

    dof_per_node = 2
    n_load_components = 1  # only transverse distributed force q_w(y)

    @classmethod
    def stiffness_matrix(cls, L, E, G, A, Iy, Iz, J):
        # Only E*Iy enters in pure bending. The other args are accepted so the
        # interface matches the 3D element.
        c = E * Iy / L**3
        K = c * np.array(
            [
                [12.0, 6.0 * L, -12.0, 6.0 * L],
                [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
                [-12.0, -6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
            ]
        )
        return K

    @classmethod
    def consistent_load_vector(cls, L, q_left, q_right):
        """
        Equivalent nodal force/moment vector for a transverse load varying
        linearly from q_left at xi=0 to q_right at xi=1.

        For uniform q this reduces to the textbook (qL/2)[1, L/6, 1, -L/6].
        """
        return (L / 60.0) * np.array(
            [
                21.0 * q_left + 9.0 * q_right,
                L * (3.0 * q_left + 2.0 * q_right),
                9.0 * q_left + 21.0 * q_right,
                -L * (2.0 * q_left + 3.0 * q_right),
            ]
        )

    @classmethod
    def bending_moment_at_midpoint(cls, L, E, Iy, u_e):
        """
        Bending moment M = E*Iy * w'' at the element midpoint.

        At xi = 0.5 the second-derivative shape-function row is
            B_mid = [0, -1/L, 0, 1/L]
        so M_mid = (E*Iy / L) * (theta_right - theta_left). The midpoint is the
        Gauss point of the cubic-Hermite element (super-convergent recovery).
        """
        B_mid = np.array([0.0, -1.0 / L, 0.0, 1.0 / L])
        return E * Iy * (B_mid @ u_e)


# ---------------------------------------------------------------------------
# 3D spatial frame element  (6 DOF/node: u, v, w, theta_x, theta_y, theta_z)
# ---------------------------------------------------------------------------


class BeamElement3D(BeamElementBase):
    """
    2-node spatial Euler-Bernoulli frame element.

    Superposes axial (EA), St-Venant torsion (GJ), and two independent bending
    planes (E*Iz in the local x-y plane, E*Iy in the local x-z plane). DOFs per
    node, in the *local* frame, are ``[u, v, w, theta_x, theta_y, theta_z]``.

    The element is placed in 3D by :meth:`transformation_matrix`, which builds
    the 12x12 block-diagonal rotation from a local triad. For a spanwise spar
    cap the local x-axis points root->tip; the local z-axis is taken "up"
    (global +Z) by default via the reference vector.
    """

    dof_per_node = 6
    n_load_components = 2  # distributed transverse force in local y and z

    @classmethod
    def stiffness_matrix(cls, L, E, G, A, Iy, Iz, J):
        """
        12x12 local stiffness matrix (McGuire/Bathe standard frame element).

        Iz -> bending in the local x-y plane (v, theta_z).
        Iy -> bending in the local x-z plane (w, theta_y).
        """
        K = np.zeros((12, 12))

        ax = E * A / L  # axial
        gj = G * J / L  # torsion

        # Bending about local z (x-y plane): v (1,7), theta_z (5,11)
        z1 = 12.0 * E * Iz / L**3
        z2 = 6.0 * E * Iz / L**2
        z3 = 4.0 * E * Iz / L
        z4 = 2.0 * E * Iz / L
        # Bending about local y (x-z plane): w (2,8), theta_y (4,10)
        y1 = 12.0 * E * Iy / L**3
        y2 = 6.0 * E * Iy / L**2
        y3 = 4.0 * E * Iy / L
        y4 = 2.0 * E * Iy / L

        # Axial
        K[0, 0] = ax
        K[0, 6] = -ax
        K[6, 0] = -ax
        K[6, 6] = ax
        # Torsion
        K[3, 3] = gj
        K[3, 9] = -gj
        K[9, 3] = -gj
        K[9, 9] = gj
        # Bending in x-y plane (v, theta_z)
        K[1, 1] = z1
        K[1, 5] = z2
        K[1, 7] = -z1
        K[1, 11] = z2
        K[5, 1] = z2
        K[5, 5] = z3
        K[5, 7] = -z2
        K[5, 11] = z4
        K[7, 1] = -z1
        K[7, 5] = -z2
        K[7, 7] = z1
        K[7, 11] = -z2
        K[11, 1] = z2
        K[11, 5] = z4
        K[11, 7] = -z2
        K[11, 11] = z3
        # Bending in x-z plane (w, theta_y) -- note opposite coupling sign
        K[2, 2] = y1
        K[2, 4] = -y2
        K[2, 8] = -y1
        K[2, 10] = -y2
        K[4, 2] = -y2
        K[4, 4] = y3
        K[4, 8] = y2
        K[4, 10] = y4
        K[8, 2] = -y1
        K[8, 4] = y2
        K[8, 8] = y1
        K[8, 10] = y2
        K[10, 2] = -y2
        K[10, 4] = y4
        K[10, 8] = y2
        K[10, 10] = y3

        return K

    @classmethod
    def transformation_matrix(cls, xi, xj, up=None):
        """
        Build the 12x12 transformation T (local = T @ global) for an element
        from node ``xi`` to node ``xj`` (3-vectors).

        ``up`` is a reference direction (default global +Z) used to orient the
        local z-axis; the local y-axis completes the right-handed triad. The
        3x3 direction-cosine block is repeated 4 times along the diagonal (one
        per translational/rotational triad at each node).
        """
        xi = np.asarray(xi, dtype=float)
        xj = np.asarray(xj, dtype=float)
        ex = xj - xi
        L = np.linalg.norm(ex)
        if L == 0.0:
            raise ValueError("Zero-length 3D beam element.")
        ex = ex / L

        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
        # If the element is (nearly) parallel to `up`, pick another reference.
        if abs(np.dot(ex, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])

        ey = np.cross(up, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)

        R = np.vstack([ex, ey, ez])  # rows are local axes in global coords
        T = np.zeros((12, 12))
        for b in range(4):
            T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
        return T

    @classmethod
    def consistent_load_vector(cls, L, q_left, q_right):
        """
        Equivalent nodal loads for transverse distributed loads given as
        ``q_left = (qy_l, qz_l)`` and ``q_right = (qy_r, qz_r)`` (local frame),
        each varying linearly along the element. Returns a length-12 vector in
        the local DOF order. Axial and torsional distributed loads are assumed
        zero (added later if needed).
        """
        q_left = np.atleast_1d(q_left).astype(float)
        q_right = np.atleast_1d(q_right).astype(float)
        qy_l, qz_l = q_left[0], q_left[-1] if q_left.size > 1 else q_left[0]
        qy_r, qz_r = q_right[0], q_right[-1] if q_right.size > 1 else q_right[0]

        f = np.zeros(12)
        # x-y plane bending DOFs: v(1), theta_z(5), v(7), theta_z(11)
        f[[1, 5, 7, 11]] = (L / 60.0) * np.array(
            [
                21.0 * qy_l + 9.0 * qy_r,
                L * (3.0 * qy_l + 2.0 * qy_r),
                9.0 * qy_l + 21.0 * qy_r,
                -L * (2.0 * qy_l + 3.0 * qy_r),
            ]
        )
        # x-z plane bending DOFs: w(2), theta_y(4), w(8), theta_y(10)
        # theta_y couples with -w, so the moment rows flip sign.
        f[[2, 4, 8, 10]] = (L / 60.0) * np.array(
            [
                21.0 * qz_l + 9.0 * qz_r,
                -L * (3.0 * qz_l + 2.0 * qz_r),
                9.0 * qz_l + 21.0 * qz_r,
                L * (2.0 * qz_l + 3.0 * qz_r),
            ]
        )
        return f

    @classmethod
    def axial_force(cls, L, E, A, u_e_local):
        """
        Axial force N = E*A * du/dx (constant over the element), from the local
        DOF vector ``u_e_local`` (length 12). Positive in tension.
        """
        return E * A * (u_e_local[6] - u_e_local[0]) / L

    @classmethod
    def bending_moment_at_midpoint(cls, L, E, Iy, u_e_local):
        """
        Bending moment about the local y-axis (x-z plane) at the element
        midpoint, for stress recovery on a spar cap. Uses the w/theta_y DOFs
        ``[w1, theta_y1, w2, theta_y2] = u_e_local[[2, 4, 8, 10]]``.
        """
        w1, ty1, w2, ty2 = u_e_local[2], u_e_local[4], u_e_local[8], u_e_local[10]
        # In the x-z plane theta_y = -dw/dx, so M = E*Iy*w'' uses the flipped row.
        B_mid = np.array([0.0, 1.0 / L, 0.0, -1.0 / L])
        return E * Iy * (B_mid @ np.array([w1, ty1, w2, ty2]))
