"""
Fixed wing UAV stability
"""
import fastoad.api as oad
import openmdao.api as om

from fastuav.models.stability.static_longitudinal.static_margin import StaticMargin
from fastuav.models.stability.static_longitudinal.neutral_point import NeutralPoint
from fastuav.models.stability.static_longitudinal.center_of_gravity.cog import CenterOfGravity
from fastuav.utils.constants import FW_PROPULSION


@oad.RegisterOpenMDAOSystem("fastuav.stability.fixedwing")
class StabilityFixedWing(om.Group):
    """
    Group containing the fixed wing stability calculations
    """

    def setup(self):
        self.add_subsystem("center_of_gravity", CenterOfGravity(propulsion_id_list=[FW_PROPULSION]), promotes=["*"])
        self.add_subsystem("neutral_point", NeutralPoint(), promotes=["*"])
        self.add_subsystem("static_margin", StaticMargin(), promotes=["*"])
