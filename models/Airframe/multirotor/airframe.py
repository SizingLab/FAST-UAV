"""
Multirotor Airframe
"""
import fastoad.api as oad
from weights import *
from geometry import *


@oad.RegisterOpenMDAOSystem("airframe.multirotor")
class Airframe(om.Group):
    """
    Group containing the MDA of the multirotor airframe
    """

    def setup(self):
        self.add_subsystem("geometry", Geometry(), promotes=["*"])
        self.add_subsystem("weights", Weights(), promotes=["*"])
