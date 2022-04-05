"""
Airframe definition
"""
import fastoad.api as oad
import openmdao.api as om
from models.Airframe.fixedwing.weights import StructuresAndWeightsFW
from models.Airframe.fixedwing.geometry import GeometryFW
from models.Airframe.fixedwing.aerodynamics import AerodynamicsFW
from models.Airframe.multirotor.weights import StructuresAndWeightsMR
from models.Airframe.multirotor.geometry import GeometryMR


@oad.RegisterOpenMDAOSystem("airframe.multirotor")
class AirframeMultirotor(om.Group):
    """
    Group containing the MDA of the multirotor airframe
    """

    def setup(self):
        self.add_subsystem("geometry", GeometryMR(), promotes=["*"])
        self.add_subsystem("structures_and_weights", StructuresAndWeightsMR(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("airframe.fixedwing")
class AirframeFixedWing(om.Group):
    """
    Group containing the MDA of the fixed wing airframe
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem("geometry", GeometryFW(), promotes=["*"])
        self.add_subsystem("structures_and_weights",
                           StructuresAndWeightsFW(spar_model=self.options["spar_model"]),
                           promotes=["*"])
        self.add_subsystem("aero", AerodynamicsFW(), promotes=["*"])
