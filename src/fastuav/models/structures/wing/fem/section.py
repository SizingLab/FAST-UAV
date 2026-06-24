"""
Spar cross-section properties for the wing-spar FEM.

Two cross-section configurations are supported, selected by ``spar_model``:

    "pipe"   : circular hollow tube, parameterised by (outer radius R, wall t)
    "I_beam" : symmetric I-section,  parameterised by (total depth H, thickness t)

Both return the same 5-tuple so the FEM, mass and stress-recovery code can stay
section-agnostic::

    A   : cross-section area                                   [m^2]
    Iy  : second moment of area about the chordwise axis       [m^4]
          (drives vertical bending -- the sizing direction)
    Iz  : second moment of area about the vertical axis        [m^4]
          (chordwise bending)
    J   : torsion constant                                     [m^4]
    c   : distance from neutral axis to the extreme fibre      [m]
          (used for the bending-stress recovery sigma = M*c/Iy)

The two taper dimensions (``d1`` root/tip and ``d2`` root/tip) keep the same
OpenMDAO variable names for both models (``data:geometry:wing:spar:R_*`` and
``data:geometry:wing:spar:t_*``); their meaning depends on ``spar_model``:

    pipe   -> d1 = outer radius R [m], d2 = wall thickness t [m]
    I_beam -> d1 = total depth  H [m], d2 = web/flange thickness t [m]
"""

from __future__ import annotations

import numpy as np

SPAR_MODELS = ("pipe", "I_beam")

# I-beam shape ratios (held fixed; the two free dimensions are depth and
# thickness, consistent with the single-DOF-per-dimension FEM taper).
I_BEAM_FLANGE_WIDTH_RATIO = 0.6   # flange width  b_f = ratio * total depth H


def tube_section_properties(R: float, t: float):
    """
    Circular hollow tube, exact (thick-walled) properties.

    Returns ``(A, Iy, Iz, J)`` -- kept 4-valued for backward compatibility with
    existing callers. Use :func:`section_properties` for the 5-tuple that also
    returns the extreme-fibre distance.
    """
    R_in = R - t
    A = np.pi * (R**2 - R_in**2)
    Iy = (np.pi / 4.0) * (R**4 - R_in**4)
    Iz = Iy                # axisymmetric tube
    J = Iy + Iz            # exact for a circular tube
    return A, Iy, Iz, J


def _tube_properties(R: float, t: float):
    A, Iy, Iz, J = tube_section_properties(R, t)
    c = R                  # extreme fibre at the outer radius
    return A, Iy, Iz, J, c


def _i_beam_properties(H: float, t: float,
                       flange_width_ratio: float = I_BEAM_FLANGE_WIDTH_RATIO):
    """
    Symmetric I-section of total depth ``H`` and wall thickness ``t``.

    Flange thickness is taken equal to the web thickness ``t`` and the flange
    width is ``flange_width_ratio * H``. The chordwise axis (Iy) is the strong
    bending axis, matching the tube convention used by the rest of the FEM.
    """
    t_f = t                                 # flange thickness = web thickness
    b_f = flange_width_ratio * H            # flange width
    h_w = max(H - 2.0 * t_f, 0.0)           # clear web height between flanges
    t_w = t                                 # web thickness

    # Area.
    A = 2.0 * b_f * t_f + h_w * t_w

    # Strong-axis (vertical bending, about the chordwise centroidal axis):
    # full b_f x H rectangle minus the two voids beside the web.
    Iy = (b_f * H**3 - (b_f - t_w) * h_w**3) / 12.0

    # Weak-axis (chordwise bending, about the vertical centroidal axis):
    # two flanges about their own centroid + web.
    Iz = 2.0 * (t_f * b_f**3) / 12.0 + (h_w * t_w**3) / 12.0

    # Thin-walled open-section torsion constant: J ~ (1/3) sum(b_i * t_i^3).
    J = (1.0 / 3.0) * (2.0 * b_f * t_f**3 + h_w * t_w**3)

    c = 0.5 * H                             # extreme fibre at the outer flange
    return A, Iy, Iz, J, c


def section_properties(spar_model: str, d1, d2):
    """
    Dispatch to the section model and return ``(A, Iy, Iz, J, c)``.

    Parameters
    ----------
    spar_model : "pipe" or "I_beam"
    d1, d2 : the two taper dimensions (see module docstring for their meaning).
    """
    if spar_model == "pipe":
        return _tube_properties(d1, d2)
    elif spar_model == "I_beam":
        return _i_beam_properties(d1, d2)
    raise ValueError(
        f"Unknown spar_model {spar_model!r}; expected one of {SPAR_MODELS}."
    )