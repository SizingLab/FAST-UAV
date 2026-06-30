"""
Fixed Wing Structures

Two registered systems share this module:

* ``fastuav.structures.fixedwing``     -- analytic estimation-model wing
  (:class:`Structures`).
* ``fastuav.structures_fem.fixedwing`` -- finite-element wing
  (:class:`StructuresFEM`); reuses the analytic tails + fuselage and swaps only
  the wing for an FE model (see :mod:`fastuav.models.structures.wing_fem`).
"""

import fastoad.api as oad
import openmdao.api as om

from fastuav.models.structures.fuselage import FuselageStructures
from fastuav.models.structures.tails import (
    HorizontalTailStructures,
    VerticalTailStructures,
)
from fastuav.models.structures.wing.wing import WingStructuresFW
from fastuav.models.structures.wing_fem.wing_group import WingStructureFEM


@oad.RegisterOpenMDAOSystem("fastuav.structures.fixedwing")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem(
            "wing",
            WingStructuresFW(spar_model=self.options["spar_model"]),
            promotes=["*"],
        )
        self.add_subsystem("horizontal_tail", HorizontalTailStructures(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailStructures(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageStructures(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("fastuav.structures_fem.fixedwing")
class StructuresFEM(om.Group):
    """
    Airframe structural analysis and weights with an FEM wing model.

    Reuses the analytic tail and fuselage models unchanged and swaps **only the
    wing** for a finite-element model (see
    :mod:`fastuav.models.structures.wing_fem`), which sizes the wing for minimum
    structural mass under a bending-stress constraint driven by the top-level
    MDO.

    Options
    -------
    wing_model : "tube_spar_foam" (default) or "wingbox_shell".
    n_elements : beam elements for the tube_spar_foam model.
    n_span, n_chord : shell mesh refinement for the wingbox_shell model.
    ks_rho : KS aggregation sharpness for the stress constraint.
    use_aero_vectors : drive the lift shape from VLM spanwise vectors (else
        an elliptical stand-in built from geometry).
    """

    def initialize(self):
        self.options.declare(
            "wing_model", default="tube_spar_foam", values=["tube_spar_foam", "wingbox_shell"]
        )
        self.options.declare("n_elements", types=int, default=20)
        self.options.declare("n_span", types=int, default=10)
        self.options.declare("n_chord", types=int, default=6)
        self.options.declare("ks_rho", types=float, default=100.0)
        self.options.declare("use_aero_vectors", types=bool, default=False)

    def setup(self):
        self.add_subsystem(
            "wing",
            WingStructureFEM(
                wing_model=self.options["wing_model"],
                n_elements=self.options["n_elements"],
                n_span=self.options["n_span"],
                n_chord=self.options["n_chord"],
                ks_rho=self.options["ks_rho"],
                use_aero_vectors=self.options["use_aero_vectors"],
            ),
            promotes=["*"],
        )
        self.add_subsystem("horizontal_tail", HorizontalTailStructures(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailStructures(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageStructures(), promotes=["*"])
