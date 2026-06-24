"""
Beam element formulations for the wing-spar FEM.

Designed to scale from 2D Euler-Bernoulli (2 DOF/node: w, theta) to a 3D
spatial frame element (6 DOF/node: u, v, w, theta_x, theta_y, theta_z).

All element classes share a common interface so the assembly logic in
beam_fem.py is element-agnostic. Adding the 3D element later only requires
implementing BeamElement3D below; nothing in BeamFEM should need to change
beyond declaring more components on the distributed-load input.

Sign / coordinate convention
----------------------------
y : along the half-span, root at y=0, tip at y=b/2
w : transverse displacement of the spar, positive up
theta : rotation about the chordwise axis x, dw/dy = theta (small angles)
M : bending moment about the chordwise axis x, M = E*Iy * w''
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
    Iz = Iy                # axisymmetric tube
    J = Iy + Iz            # exact for circular tube
    return A, Iy, Iz, J


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BeamElementBase:
    """
    2-node beam element interface. Subclasses define DOF count and matrices.

    The convention is that each element knows how many DOFs per node it has,
    so the assembler in BeamFEM can stay element-agnostic.
    """
    n_nodes: int = 2
    dof_per_node: int = 0       # set by subclass
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

    Hermite cubic shape functions for w(xi), xi in [0,1]:
        N1(xi) = 1 - 3 xi^2 + 2 xi^3
        N2(xi) = L (xi - 2 xi^2 + xi^3)
        N3(xi) = 3 xi^2 - 2 xi^3
        N4(xi) = L (-xi^2 + xi^3)

    Element stiffness matrix (4x4) is the standard EI/L^3 form.
    """
    dof_per_node = 2
    n_load_components = 1   # only transverse distributed force q_w(y)

    @classmethod
    def stiffness_matrix(cls, L, E, G, A, Iy, Iz, J):
        # Only E*Iy enters in pure bending. The other args are accepted so the
        # interface matches the 3D element.
        c = E * Iy / L**3
        K = c * np.array([
            [12.0,    6.0*L,    -12.0,    6.0*L   ],
            [6.0*L,   4.0*L*L,  -6.0*L,   2.0*L*L ],
            [-12.0,   -6.0*L,    12.0,   -6.0*L   ],
            [6.0*L,   2.0*L*L,  -6.0*L,   4.0*L*L ],
        ])
        return K

    @classmethod
    def consistent_load_vector(cls, L, q_left, q_right):
        """
        Equivalent nodal force/moment vector for a transverse load varying
        linearly along the element from q_left at xi=0 to q_right at xi=1.

            F = (L/60) [21 q_l + 9 q_r,
                        L (3 q_l + 2 q_r),
                        9 q_l + 21 q_r,
                       -L (2 q_l + 3 q_r)]

        For uniform q (q_l = q_r = q) this reduces to the textbook
            F = (qL/2) [1, L/6, 1, -L/6]
        i.e. the classical fixed-end moment qL^2/12.
        """
        return (L / 60.0) * np.array([
            21.0 * q_left + 9.0 * q_right,
            L * (3.0 * q_left + 2.0 * q_right),
            9.0 * q_left + 21.0 * q_right,
            -L * (2.0 * q_left + 3.0 * q_right),
        ])

    @classmethod
    def bending_moment_at_midpoint(cls, L, E, Iy, u_e):
        """
        Bending moment M = E*Iy * w'' evaluated at the element midpoint.

        At xi = 0.5 the second-derivative shape-function row is
            B_mid = [0, -1/L, 0, 1/L]
        so M_mid = (E*Iy / L) * (theta_right - theta_left).

        Midpoint is preferred over end-points for piecewise-cubic w because
        it is the Gauss point of the cubic-Hermite element and gives
        super-convergent stress recovery.
        """
        B_mid = np.array([0.0, -1.0 / L, 0.0, 1.0 / L])
        return E * Iy * (B_mid @ u_e)


# ---------------------------------------------------------------------------
# 3D spatial frame element  (6 DOF/node) -- "BEAM3"
# ---------------------------------------------------------------------------

class BeamElement3D(BeamElementBase):
    """
    2-node spatial frame element ("BEAM3"): 6 DOF/node, 12 DOF/element.

    DOFs per node, in this fixed order::

        [u, v, w, theta_x, theta_y, theta_z]
          0  1  2     3        4        5
        u        -- axial displacement along y (spanwise)
        v        -- chordwise transverse displacement
        w        -- vertical transverse displacement (positive up)
        theta_x  -- torsional rotation about the spanwise axis
        theta_y  -- bending rotation of the vertical-bending (x-z) plane
        theta_z  -- bending rotation of the chordwise-bending (x-y) plane

    The element superposes four uncoupled sub-problems (the spar is straight
    along the span, so local and global frames coincide and no rotation matrix
    is needed):

      * axial      : E*A/L stiffness on the u DOFs
      * torsion    : G*J/L stiffness on the theta_x DOFs
      * x-z bending: the validated Euler-Bernoulli 2D block with E*Iy on the
                     (w, theta_y) DOFs -- this is the vertical/sizing direction
      * x-y bending: the same 2D block with E*Iz on the (v, theta_z) DOFs

    Embedding the verified :class:`BeamElement2D` bending block guarantees the
    vertical-bending response is identical to the 2D element under a purely
    transverse load, so switching elements does not change the sizing result.
    """
    dof_per_node = 6
    n_load_components = 1   # transverse vertical force q_w(y); v/torsion unloaded

    # Local DOF offsets within a node block.
    _U, _V, _W, _TX, _TY, _TZ = range(6)
    # (w, theta_y) DOF indices for the two nodes -> vertical-bending sub-vector.
    _ZBEND = (_W, _TY, 6 + _W, 6 + _TY)
    # (v, theta_z) DOF indices for the two nodes -> chordwise-bending sub-vector.
    _YBEND = (_V, _TZ, 6 + _V, 6 + _TZ)

    @classmethod
    def stiffness_matrix(cls, L, E, G, A, Iy, Iz, J):
        K = np.zeros((12, 12))

        # Axial: u DOFs of both nodes.
        ka = E * A / L
        K[cls._U, cls._U] += ka
        K[6 + cls._U, 6 + cls._U] += ka
        K[cls._U, 6 + cls._U] += -ka
        K[6 + cls._U, cls._U] += -ka

        # Torsion: theta_x DOFs of both nodes.
        kt = G * J / L
        K[cls._TX, cls._TX] += kt
        K[6 + cls._TX, 6 + cls._TX] += kt
        K[cls._TX, 6 + cls._TX] += -kt
        K[6 + cls._TX, cls._TX] += -kt

        # Vertical (x-z) bending block, E*Iy, on (w, theta_y).
        K_zb = BeamElement2D.stiffness_matrix(L, E, G, A, Iy, Iz, J)
        idz = cls._ZBEND
        for a in range(4):
            for b in range(4):
                K[idz[a], idz[b]] += K_zb[a, b]

        # Chordwise (x-y) bending block, E*Iz, on (v, theta_z).
        K_yb = BeamElement2D.stiffness_matrix(L, E, G, A, Iz, Iy, J)
        idy = cls._YBEND
        for a in range(4):
            for b in range(4):
                K[idy[a], idy[b]] += K_yb[a, b]

        return K

    @classmethod
    def consistent_load_vector(cls, L, q_left, q_right):
        """
        Equivalent nodal load for the transverse vertical distributed force,
        placed on the (w, theta_y) DOFs; all other DOFs are unloaded.
        """
        F = np.zeros(12)
        F_zb = BeamElement2D.consistent_load_vector(L, q_left, q_right)
        for a, dof in enumerate(cls._ZBEND):
            F[dof] = F_zb[a]
        return F

    @classmethod
    def bending_moment_at_midpoint(cls, L, E, Iy, u_e):
        """
        Vertical-plane bending moment at the element midpoint, recovered from
        the (w, theta_y) sub-vector exactly as for the 2D element.
        """
        u_zb = np.array([u_e[i] for i in cls._ZBEND])
        return BeamElement2D.bending_moment_at_midpoint(L, E, Iy, u_zb)
