"""
Multirotor Airframe Weights
"""
import openmdao.api as om
import numpy as np



class StructuresAndWeightsMR(om.Group):
    """
    Group containing the airframe weights calculation
    """

    def setup(self):
        self.add_subsystem("arms", WeightArms(), promotes=["*"])
        self.add_subsystem("body", WeightBody(), promotes=["*"])


class WeightArms(om.ExplicitComponent):
    """
    Computes arms weight
    """

    def setup(self):
        self.add_input(
            "data:airframe:arms:diameter:k", val=np.nan, units=None
        )
        self.add_input("data:airframe:arms:number", val=np.nan, units=None)
        self.add_input("data:airframe:arms:length", val=np.nan, units="m")
        self.add_input("data:airframe:arms:diameter:outer", val=np.nan, units="m")
        self.add_input(
            "data:airframe:arms:material:density", val=1700, units="kg/m**3"
        )
        self.add_output("data:airframe:arms:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        D_ratio = inputs["data:airframe:arms:diameter:k"]
        Narm = inputs["data:airframe:arms:number"]
        Larm = inputs["data:airframe:arms:length"]
        Dout = inputs["data:airframe:arms:diameter:outer"]
        rho = inputs["data:airframe:arms:material:density"]

        Marms = (
            np.pi / 4 * (Dout**2 - (D_ratio * Dout) ** 2) * Larm * rho * Narm
        )  # [kg] mass of the arms

        outputs["data:airframe:arms:mass"] = Marms


class WeightBody(om.ExplicitComponent):
    """
    Computes body weight
    """

    def setup(self):
        self.add_input("data:airframe:arms:reference:mass", val=np.nan, units="kg")
        self.add_input("data:airframe:body:reference:mass", val=np.nan, units="kg")
        self.add_input("data:airframe:arms:mass", val=np.nan, units="kg")
        self.add_output("data:airframe:body:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Marm_ref = inputs["data:airframe:arms:reference:mass"]
        Mbody_ref = inputs["data:airframe:body:reference:mass"]
        Marms = inputs["data:airframe:arms:mass"]

        Mbody = Mbody_ref * (Marms / Marm_ref)  # [kg] mass of the frame

        outputs["data:airframe:body:mass"] = Mbody
