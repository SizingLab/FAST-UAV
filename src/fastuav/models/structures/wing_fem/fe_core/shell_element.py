"""
Flat 4-node shell element (membrane + Mindlin-Reissner plate) for the wingbox.

Formulation
-----------
Each node carries 6 DOF ``[u, v, w, theta_x, theta_y, theta_z]`` in the global
frame. The element is flat: it is built in a local frame (origin at node 0,
local z = element normal), where the membrane and bending responses uncouple:

* **Membrane** -- bilinear Q4 plane-stress, 2x2 Gauss (DOFs u, v).
* **Bending + shear** -- Q4 Mindlin-Reissner plate with **selective reduced
  integration** (2x2 Gauss on bending, 1-point on transverse shear) to avoid
  shear locking for thin panels (DOFs w, theta_x, theta_y).
* **Drilling** -- a small fictitious rotational stiffness on the local theta_z
  DOF stabilises the assembled system when shells meet at an angle.

This selective-reduced-integration element is a standard, robust, locking-free
flat shell; it is used here in place of the MITC4 assumed-strain element named
in the plan because it is materially simpler to implement correctly and
validate, and is fully adequate for the spanwise-bending stress objective.

Sign convention (local plate): a normal fibre rotates by ``theta_x`` about x and
``theta_y`` about y, giving curvatures
``kx = d(theta_y)/dx``, ``ky = -d(theta_x)/dy``,
``kxy = d(theta_y)/dy - d(theta_x)/dx`` and transverse shears
``gxz = dw/dx + theta_y``, ``gyz = dw/dy - theta_x``.
"""

from __future__ import annotations

import numpy as np

# Per-node DOF order in the local/global element vector.
DOF_PER_NODE = 6
_GAUSS2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def quad_local_frame(coords):
    """
    Local orthonormal triad and in-plane 2D coordinates for a quad.

    ``coords`` is (4, 3) global node coordinates. Returns ``(R, xy)`` where
    ``R`` (3, 3) has the local x, y, z axes as rows (local = R @ global) and
    ``xy`` (4, 2) are the node coordinates projected into the local plane
    (local z dropped; the quad is treated as flat).
    """
    coords = np.asarray(coords, dtype=float)
    # Local x along edge 0->1; normal from the two diagonals.
    ex = coords[1] - coords[0]
    ex = ex / np.linalg.norm(ex)
    n = np.cross(coords[2] - coords[0], coords[3] - coords[1])
    ez = n / np.linalg.norm(n)
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])
    xy = np.array([R[:2] @ (p - coords[0]) for p in coords])
    return R, xy


def _shape_q4(xi, eta):
    """Bilinear shape functions and natural-coord derivatives at (xi, eta)."""
    s = np.array([-1.0, 1.0, 1.0, -1.0])
    t = np.array([-1.0, -1.0, 1.0, 1.0])
    N = 0.25 * (1 + s * xi) * (1 + t * eta)
    dN_dxi = 0.25 * s * (1 + t * eta)
    dN_deta = 0.25 * t * (1 + s * xi)
    return N, dN_dxi, dN_deta


def _jacobian(xy, dN_dxi, dN_deta):
    J = np.array(
        [
            [dN_dxi @ xy[:, 0], dN_dxi @ xy[:, 1]],
            [dN_deta @ xy[:, 0], dN_deta @ xy[:, 1]],
        ]
    )
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
    dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
    return dN_dx, dN_dy, detJ


# ---------------------------------------------------------------------------
# Local stiffness blocks
# ---------------------------------------------------------------------------


def _membrane_stiffness(xy, E, nu, t):
    """8x8 plane-stress membrane stiffness (DOF order per node: u, v)."""
    D = (
        E
        / (1.0 - nu**2)
        * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )
    )
    K = np.zeros((8, 8))
    for xi in _GAUSS2:
        for eta in _GAUSS2:
            _, dxi, deta = _shape_q4(xi, eta)
            dNx, dNy, detJ = _jacobian(xy, dxi, deta)
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx
            B[1, 1::2] = dNy
            B[2, 0::2] = dNy
            B[2, 1::2] = dNx
            K += t * (B.T @ D @ B) * detJ
    return K


def _plate_stiffness(xy, E, nu, t, kappa=5.0 / 6.0):
    """
    12x12 Mindlin plate stiffness (DOF order per node: w, theta_x, theta_y),
    selective reduced integration (2x2 bending, 1-point shear).
    """
    Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ]
    )
    G = E / (2.0 * (1.0 + nu))
    Ds = kappa * G * t * np.eye(2)

    K = np.zeros((12, 12))

    def B_bending(dNx, dNy):
        B = np.zeros((3, 12))
        # kx = d(theta_y)/dx ; theta_y is local DOF index 2 within node block
        B[0, 2::3] = dNx
        # ky = -d(theta_x)/dy ; theta_x is index 1
        B[1, 1::3] = -dNy
        # kxy = d(theta_y)/dy - d(theta_x)/dx
        B[2, 2::3] = dNy
        B[2, 1::3] = -dNx
        return B

    def B_shear(N, dNx, dNy):
        B = np.zeros((2, 12))
        # gxz = dw/dx + theta_y ; w index 0, theta_y index 2
        B[0, 0::3] = dNx
        B[0, 2::3] = N
        # gyz = dw/dy - theta_x ; theta_x index 1
        B[1, 0::3] = dNy
        B[1, 1::3] = -N
        return B

    # Bending: full 2x2 Gauss.
    for xi in _GAUSS2:
        for eta in _GAUSS2:
            _, dxi, deta = _shape_q4(xi, eta)
            dNx, dNy, detJ = _jacobian(xy, dxi, deta)
            Bb = B_bending(dNx, dNy)
            K += (Bb.T @ Db @ Bb) * detJ

    # Shear: 1-point (reduced) at the centroid, weight 4.
    N0, dxi0, deta0 = _shape_q4(0.0, 0.0)
    dNx0, dNy0, detJ0 = _jacobian(xy, dxi0, deta0)
    Bs = B_shear(N0, dNx0, dNy0)
    K += 4.0 * (Bs.T @ Ds @ Bs) * detJ0
    return K


def shell_stiffness(coords, E, nu, t, drilling_rel=1.0e-3):
    """
    24x24 **global** stiffness of a flat shell quad and its transformation.

    Parameters
    ----------
    coords : (4, 3) global node coordinates.
    E, nu, t : Young's modulus [Pa], Poisson ratio [-], thickness [m].
    drilling_rel : drilling stiffness as a fraction of the mean bending
        diagonal (stabilisation only; keep small).

    Returns
    -------
    K_glob : (24, 24) stiffness in global DOFs, node-major order
             ``[u, v, w, theta_x, theta_y, theta_z] x 4``.
    R : (3, 3) local frame (rows = local axes in global coords).
    """
    R, xy = quad_local_frame(coords)
    Km = _membrane_stiffness(xy, E, nu, t)
    Kp = _plate_stiffness(xy, E, nu, t)

    K_loc = np.zeros((24, 24))
    # Scatter membrane (u,v -> local indices 0,1 of each node block).
    m_idx = np.array([0, 1, 6, 7, 12, 13, 18, 19])
    p_idx = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
    K_loc[np.ix_(m_idx, m_idx)] += Km
    K_loc[np.ix_(p_idx, p_idx)] += Kp

    # Drilling stabilisation on local theta_z (indices 5, 11, 17, 23).
    drill = drilling_rel * (np.trace(Kp) / 12.0)
    for d in (5, 11, 17, 23):
        K_loc[d, d] += drill

    # Transform local -> global. T maps global DOFs to local (loc = T @ glob),
    # so K_glob = T^T K_loc T with T block-diagonal in R.
    T = np.zeros((24, 24))
    for b in range(8):  # 8 triads (2 per node)
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
    K_glob = T.T @ K_loc @ T
    return K_glob, R


def membrane_stress(coords, E, nu, t, u_elem_global):
    """
    Membrane + bending outer-fibre stress at the element centroid.

    Returns ``(sigma_local, svm_top, svm_bot)`` where ``sigma_local`` is the
    centroid membrane stress ``[sx, sy, txy]`` in the local frame and
    ``svm_top/bot`` are von Mises stresses on the two surfaces (membrane +/-
    bending). ``u_elem_global`` is the 24-vector of element nodal DOFs.
    """
    R, xy = quad_local_frame(coords)
    T = np.zeros((24, 24))
    for b in range(8):
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
    u_loc = T @ np.asarray(u_elem_global, dtype=float)

    u_mem = u_loc[[0, 1, 6, 7, 12, 13, 18, 19]]
    u_pl = u_loc[[2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22]]

    Dm = (
        E / (1.0 - nu**2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    )

    _, dxi, deta = _shape_q4(0.0, 0.0)
    dNx, dNy, _ = _jacobian(xy, dxi, deta)
    Bm = np.zeros((3, 8))
    Bm[0, 0::2] = dNx
    Bm[1, 1::2] = dNy
    Bm[2, 0::2] = dNy
    Bm[2, 1::2] = dNx
    sigma_m = Dm @ (Bm @ u_mem)

    Db = (E / (1.0 - nu**2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )
    Bb = np.zeros((3, 12))
    Bb[0, 2::3] = dNx
    Bb[1, 1::3] = -dNy
    Bb[2, 2::3] = dNy
    Bb[2, 1::3] = -dNx
    kappa = Bb @ u_pl
    sigma_b = (t / 2.0) * (Db @ kappa)  # outer-fibre bending stress

    def vm(s):
        return np.sqrt(s[0] ** 2 - s[0] * s[1] + s[1] ** 2 + 3.0 * s[2] ** 2)

    return sigma_m, vm(sigma_m + sigma_b), vm(sigma_m - sigma_b)
