"""
Multirotor Airframe Weights
"""
import openmdao.api as om
import numpy as np



class Weights(om.Group):
    """
    Group containing the airframe weights calculation
    """

    def setup(self):
        self.add_subsystem("weight_arms", WeightArms(), promotes=["*"])
        self.add_subsystem("weight_body", WeightBody(), promotes=["*"])


class WeightArms(om.ExplicitComponent):
    """
    Computes arms weight
    """

    def setup(self):
        self.add_input(
            "data:structure:arms:settings:diameter:k", val=np.nan, units=None
        )
        self.add_input("data:structure:arms:number", val=np.nan, units=None)
        self.add_input("data:structure:arms:length", val=np.nan, units="m")
        self.add_input("data:structure:arms:diameter:outer", val=np.nan, units="m")
        self.add_input(
            "data:structure:arms:material:density", val=1700, units="kg/m**3"
        )
        self.add_output("data:structure:arms:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        D_ratio = inputs["data:structure:arms:settings:diameter:k"]
        Narm = inputs["data:structure:arms:number"]
        Larm = inputs["data:structure:arms:length"]
        Dout = inputs["data:structure:arms:diameter:outer"]
        rho = inputs["data:structure:arms:material:density"]

        Marms = (
            np.pi / 4 * (Dout**2 - (D_ratio * Dout) ** 2) * Larm * rho * Narm
        )  # [kg] mass of the arms

        outputs["data:structure:arms:mass"] = Marms


class WeightBody(om.ExplicitComponent):
    """
    Computes body weight
    """

    def setup(self):
        self.add_input("data:structure:reference:arms:mass", val=np.nan, units="kg")
        self.add_input("data:structure:reference:body:mass", val=np.nan, units="kg")
        self.add_input("data:structure:arms:mass", val=np.nan, units="kg")
        self.add_output("data:structure:body:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Marm_ref = inputs["data:structure:reference:arms:mass"]
        Mbody_ref = inputs["data:structure:reference:body:mass"]
        Marms = inputs["data:structure:arms:mass"]

        Mbody = Mbody_ref * (Marms / Marm_ref)  # [kg] mass of the frame

        outputs["data:structure:body:mass"] = Mbody
