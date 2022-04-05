"""
UAV stability
"""
import fastoad.api as oad
import openmdao.api as om

from models.Stability.fixedwing.static_margin import StaticMargin
from models.Stability.fixedwing.neutral_point import NeutralPoint
from models.Stability.fixedwing.center_of_gravity import CenterOfGravity


@oad.RegisterOpenMDAOSystem("stability.fixedwing")
class StabilityFixedWing(om.Group):
    """
    Group containing the fixed wing stability calculations
    """

    def setup(self):
        self.add_subsystem("center_of_gravity", CenterOfGravity(), promotes=["*"])
        self.add_subsystem("neutral_point", NeutralPoint(), promotes=["*"])
        self.add_subsystem("static_margin", StaticMargin(), promotes=["*"])
