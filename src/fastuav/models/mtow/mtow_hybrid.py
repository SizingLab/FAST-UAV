"""
MTOW for hybrid (fixed wing VTOL) UAVs
"""

import fastoad.api as oad
from fastuav.constants import FW_PROPULSION, MR_PROPULSION
from fastuav.models.mtow.mtow import MTOW


@oad.RegisterOpenMDAOSystem("fastuav.mtow.hybrid")
class MTOWHybrid(MTOW):
    """
    Group containing the MTOW module for hybrid UAVs
    """

    def initialize(self):
        MTOW.initialize(self)
        self.options["propulsion_id_list"] = [MR_PROPULSION, FW_PROPULSION]
