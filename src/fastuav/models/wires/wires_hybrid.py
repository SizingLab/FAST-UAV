"""
Hybrid (fixed wing + VTOL) wires
"""
import fastoad.api as oad
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION
from fastuav.models.wires.wires import Wires


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.wires.hybrid")
class WiresHybrid(Wires):
    """
    Group containing the hybrid (fixed wing + VTOL) wires calculations
    """

    def initialize(self):
        Wires.initialize(self)
        self.options["propulsion_id"] = [MR_PROPULSION, FW_PROPULSION]