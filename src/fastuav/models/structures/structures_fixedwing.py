"""
Fixed Wing Structures
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.structures.wing.wing import WingStructuresFW
from fastuav.models.structures.tails import HorizontalTailStructures, VerticalTailStructures
from fastuav.models.structures.fuselage import FuselageStructures


@oad.RegisterOpenMDAOSystem("fastuav.structures.fixedwing")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem(
            "wing", WingStructuresFW(spar_model=self.options["spar_model"]), promotes=["*"]
        )
        self.add_subsystem("horizontal_tail", HorizontalTailStructures(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailStructures(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageStructures(), promotes=["*"])
