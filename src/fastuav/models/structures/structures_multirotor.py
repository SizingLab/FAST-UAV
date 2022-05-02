"""
Multirotor Structures
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("fastuav.structures.multirotor")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def setup(self):
        self.add_subsystem("arms", ArmsWeight(), promotes=["*"])
        self.add_subsystem("body", BodyWeight(), promotes=["*"])


class ArmsWeight(om.ExplicitComponent):
    """
    Computes arms weight
    """

    def setup(self):
        self.add_input("data:structures:arms:diameter:k", val=np.nan, units=None)
        self.add_input("data:structures:arms:diameter:outer", val=np.nan, units="m")
        self.add_input("data:geometry:arms:number", val=np.nan, units=None)
        self.add_input("data:geometry:arms:length", val=np.nan, units="m")
        self.add_input("data:weights:arms:density", val=np.nan, units="kg/m**3")
        self.add_output("data:weights:airframe:arms:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        D_ratio = inputs["data:structures:arms:diameter:k"]
        Narm = inputs["data:geometry:arms:number"]
        Larm = inputs["data:geometry:arms:length"]
        Dout = inputs["data:structures:arms:diameter:outer"]
        rho = inputs["data:weights:arms:density"]

        Marms = (
            np.pi / 4 * (Dout**2 - (D_ratio * Dout) ** 2) * Larm * rho * Narm
        )  # [kg] mass of the arms

        outputs["data:weights:airframe:arms:mass"] = Marms


class BodyWeight(om.ExplicitComponent):
    """
    Computes body weight
    """

    def setup(self):
        self.add_input("data:weights:airframe:arms:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:body:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:arms:mass", val=np.nan, units="kg")
        self.add_output("data:weights:airframe:body:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Marm_ref = inputs["data:weights:airframe:arms:mass:reference"]
        Mbody_ref = inputs["data:weights:airframe:body:mass:reference"]
        Marms = inputs["data:weights:airframe:arms:mass"]

        Mbody = Mbody_ref * (Marms / Marm_ref)  # [kg] mass of the frame

        outputs["data:weights:airframe:body:mass"] = Mbody
