"""
MTOW for multirotor UAVs
"""

import fastoad.api as oad
from fastuav.utils.constants import MR_PROPULSION
from fastuav.models.mtow.mtow import MTOW


@oad.RegisterOpenMDAOSystem("fastuav.mtow.multirotor")
class MTOWMultirotor(MTOW):
    """
    Group containing the MTOW module for multirotor UAVs
    """

    def initialize(self):
        MTOW.initialize(self)
        self.options["propulsion_id"] = [MR_PROPULSION]