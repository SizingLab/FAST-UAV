"""
Multirotor wires
"""
import fastoad.api as oad
from fastuav.utils.constants import MR_PROPULSION
from fastuav.models.wires.wires import Wires


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.wires.multirotor")
class WiresMultirotor(Wires):
    """
    Group containing the multirotor wires calculations
    """

    def initialize(self):
        Wires.initialize(self)
        self.options["propulsion_id"] = [MR_PROPULSION]