"""
Fixed wing propulsion system
"""

import fastoad.api as oad
from fastuav.utils.constants import FW_PROPULSION
from fastuav.models.propulsion.propulsion import Propulsion


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.fixedwing")
class PropulsionFixedWing(Propulsion):
    """
    Group containing the fixed wing propulsion system calculations
    """

    def initialize(self):
        Propulsion.initialize(self)
        self.options["propulsion_id"] = [FW_PROPULSION]
