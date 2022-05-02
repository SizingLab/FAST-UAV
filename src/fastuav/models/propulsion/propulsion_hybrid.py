"""
Hybrid (fixed wing VTOL) propulsion system
"""

import fastoad.api as oad
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION
from fastuav.models.propulsion.propulsion import Propulsion


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.hybrid")
class PropulsionHybrid(Propulsion):
    """
    Group containing the hybrid propulsion system calculations
    """

    def initialize(self):
        Propulsion.initialize(self)
        self.options["propulsion_id"] = [MR_PROPULSION, FW_PROPULSION]
