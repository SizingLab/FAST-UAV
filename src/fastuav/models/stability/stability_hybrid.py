"""
Hybrid VTOL UAV stability
"""
import fastoad.api as oad
import openmdao.api as om

from fastuav.models.stability.static_longitudinal.static_margin import StaticMargin, StaticMarginConstraints
from fastuav.models.stability.static_longitudinal.neutral_point import NeutralPoint
from fastuav.models.stability.static_longitudinal.center_of_gravity.cog import CenterOfGravity
from fastuav.constants import PROPULSION_ID_LIST


@oad.RegisterOpenMDAOSystem("fastuav.stability.hybrid")
class StabilityFixedWing(om.Group):
    """
    Group containing the fixed wing stability calculations
    """

    def setup(self):
        self.add_subsystem("center_of_gravity", CenterOfGravity(propulsion_id_list=PROPULSION_ID_LIST), promotes=["*"])
        self.add_subsystem("neutral_point", NeutralPoint(), promotes=["*"])
        self.add_subsystem("static_margin", StaticMargin(), promotes=["*"])
        self.add_subsystem("constraints", StaticMarginConstraints(), promotes=["*"])
