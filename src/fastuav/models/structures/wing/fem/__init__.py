"""
Finite-element wing-spar structural model (vendored from the fast-uav-fem archive).

This subpackage provides an FEM-based alternative to the analytical/estimation
wing structures model. The pipeline is:

    WingLoadDistribution  ->  BeamFEM  ->  StressRecovery  ->  SparMass

All components use FAST-UAV-style ``data:...`` variable naming and are wired
together by :class:`WingStructure`.

Public API:
    BeamElement2D, BeamElement3D : element formulations
    solve_beam_fem               : pure-NumPy FEM solver
    BeamFEM                      : OpenMDAO ExplicitComponent for the FEM
    WingLoadDistribution         : VLM-driven q(y) at FE nodes
    StressRecovery               : sigma(y) and KS-aggregated failure margin
    SparMass                     : trapezoidal integration of rho*A(y)
    WingStructure                : top-level group wiring all the above
    tube_section_properties      : tube cross-section helper
"""

from .beam_element import (
    BeamElement2D,
    BeamElement3D,
    BeamElementBase,
    tube_section_properties,
)
from .beam_fem import solve_beam_fem
from .load_distribution import build_q_distribution
from .stress_recovery import recover_stress
from .spar_mass import compute_spar_mass
from .section import section_properties, SPAR_MODELS

try:
    from .beam_fem import BeamFEM
    from .load_distribution import WingLoadDistribution
    from .stress_recovery import StressRecovery
    from .spar_mass import SparMass
    from .wing_structure_group import WingStructure
except ImportError:
    BeamFEM = None
    WingLoadDistribution = None
    StressRecovery = None
    SparMass = None
    WingStructure = None
