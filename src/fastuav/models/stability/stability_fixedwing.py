"""
Fixed wing UAV stability
"""
import fastoad.api as oad
import openmdao.api as om

from fastuav.models.stability.fixedwing.static_margin import StaticMargin
from fastuav.models.stability.fixedwing.neutral_point import NeutralPoint
from fastuav.models.stability.fixedwing.center_of_gravity import CenterOfGravity


@oad.RegisterOpenMDAOSystem("stability.fixedwing")
class StabilityFixedWing(om.Group):
    """
    Group containing the fixed wing stability calculations
    """

    def setup(self):
        self.add_subsystem("center_of_gravity", CenterOfGravity(), promotes=["*"])
        self.add_subsystem("neutral_point", NeutralPoint(), promotes=["*"])
        self.add_subsystem("static_margin", StaticMargin(), promotes=["*"])
