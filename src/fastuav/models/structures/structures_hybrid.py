"""
Hybrid VTOL Structures
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.structures.structures_fixedwing import WingWeight, HorizontalTailWeight, VerticalTailWeight, FuselageWeight
from fastuav.models.structures.structures_multirotor import ArmsWeight


@oad.RegisterOpenMDAOSystem("fastuav.structures.hybrid")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem(
            "wing", WingWeight(spar_model=self.options["spar_model"]), promotes=["*"]
        )
        self.add_subsystem("horizontal_tail", HorizontalTailWeight(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailWeight(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageWeight(), promotes=["*"])
        self.add_subsystem("arms", ArmsWeight(), promotes=["*"])

