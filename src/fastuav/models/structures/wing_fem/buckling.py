"""
Closed-form buckling allowables for the FEM wing models (pure NumPy).

These complement the material-stress check. Thin walls, panels and slender caps
buckle in *compression* well below their material strength, and buckling is
typically the sizing driver for light, thin-walled wing structures: without it
the optimiser drives every gauge to its lower bound and the wing comes out
unrealistically thin. Each helper returns a **critical compressive stress**
[Pa]; the calling model forms a utilisation ``sigma_compression / sigma_cr`` and
KS-aggregates them into a single dimensionless buckling margin (``<= 0``
feasible), mirroring the material failure margin.

Conservative classical coefficients are used, with the imperfection knockdown
folded into a single tunable coefficient per mode so the models can be
calibrated against test / higher-fidelity data without touching the algebra.
"""

from __future__ import annotations

import numpy as np


def tube_local_buckling_stress(E, t, R, coeff=0.3):
    """
    Local (shell) buckling stress of a thin-walled circular tube in axial
    compression / bending.

    The classical critical stress is ``sigma_cl = E / sqrt(3 (1 - nu^2)) * t/R``
    (= 0.605 E t/R for nu = 0.3). Real cylinders are strongly imperfection
    sensitive, so a knockdown of ~0.5 is applied; the default ``coeff = 0.3``
    is ~ 0.5 * 0.605.
    """
    R = np.maximum(R, 1e-9)
    return coeff * E * t / R


def sandwich_wrinkling_stress(E_face, E_core, coeff=0.5, nu_core=0.3):
    """
    Face-wrinkling stress of a foam-cored sandwich face (Hoff / Hodgkinson):

        sigma_wr = coeff * (E_face * E_core * G_core)^(1/3)

    with the core shear modulus ``G_core = E_core / (2 (1 + nu_core))``. The
    coefficient ``coeff ~ 0.5`` lumps the classical constant and a knockdown.
    """
    G_core = E_core / (2.0 * (1.0 + nu_core))
    return coeff * (E_face * E_core * G_core) ** (1.0 / 3.0)


def plate_buckling_stress(E, t, b, nu=0.3, kc=4.0):
    """
    Critical compressive stress of a flat rectangular plate under uniaxial
    in-plane compression:

        sigma_cr = kc * pi^2 E / (12 (1 - nu^2)) * (t / b)^2

    ``b`` is the panel width transverse to the load and ``kc ~ 4`` for a long,
    simply-supported plate.
    """
    b = np.maximum(b, 1e-9)
    return kc * np.pi**2 * E / (12.0 * (1.0 - nu**2)) * (t / b) ** 2


def column_buckling_stress(E, I, A, L, coeff=1.0):
    """
    Euler column-buckling stress of a beam of section ``(I, A)`` and free
    length ``L``:

        sigma_cr = coeff * pi^2 E I / (A L^2)

    ``coeff`` captures the end fixity (1.0 = pinned-pinned, conservative for a
    cap that runs continuously over the ribs).
    """
    L = np.maximum(L, 1e-9)
    A = np.maximum(A, 1e-12)
    return coeff * np.pi**2 * E * I / (A * L**2)
