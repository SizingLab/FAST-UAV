"""
Multirotor propulsion system
"""

import fastoad.api as oad
from fastuav.utils.constants import MR_PROPULSION
from fastuav.models.propulsion.propulsion import Propulsion


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.multirotor")
class PropulsionMultirotor(Propulsion):
    """
    Group containing the multirotor propulsion system calculations
    """

    def initialize(self):
        Propulsion.initialize(self)
        self.options["propulsion_id"] = [MR_PROPULSION]