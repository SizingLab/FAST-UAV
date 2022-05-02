"""
Fixed wing wires
"""
import fastoad.api as oad
from fastuav.utils.constants import FW_PROPULSION
from fastuav.models.wires.wires import Wires


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.wires.fixedwing")
class WiresFixedWing(Wires):
    """
    Group containing the fixed wing wires calculations
    """

    def initialize(self):
        Wires.initialize(self)
        self.options["propulsion_id"] = [FW_PROPULSION]