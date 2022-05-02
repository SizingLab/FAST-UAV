"""
MTOW for fixed wing UAVs
"""

import fastoad.api as oad
from fastuav.utils.constants import FW_PROPULSION
from fastuav.models.mtow.mtow import MTOW


@oad.RegisterOpenMDAOSystem("fastuav.mtow.fixedwing")
class MTOWFixedWing(MTOW):
    """
    Group containing the MTOW module for fixed wing UAVs
    """

    def initialize(self):
        MTOW.initialize(self)
        self.options["propulsion_id"] = [FW_PROPULSION]