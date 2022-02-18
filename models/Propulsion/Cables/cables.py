"""
Cables component
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("propulsion.cables")
class Cables(om.Group):
    """
    Group containing the Cables MDA.
    """

    def setup(self):
        self.add_subsystem("radius", Radius(), promotes=["*"])
        self.add_subsystem("weight", Weight(), promotes=["*"])


class Radius(om.ExplicitComponent):
    """
    Computes cables radius.
    """

    def setup(self):
        self.add_input("data:cables:reference:radius", val=np.nan, units="m")
        self.add_input("data:cables:reference:current", val=np.nan, units="A")
        self.add_input("data:motor:current:hover", val=np.nan, units="A")
        self.add_output("data:cables:radius", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        r_ref = inputs["data:cables:reference:radius"]  # [m] radius of reference cable
        I_ref = inputs[
            "data:cables:reference:current"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:motor:current:hover"]  # [A] max current (continuous)

        r = r_ref * (I / I_ref) ** (1 / 1.5)  # [m] radius of cable

        outputs["data:cables:radius"] = r


class Weight(om.ExplicitComponent):
    """
    Computes cables weight.
    """

    def setup(self):
        self.add_input("data:cables:reference:density", val=np.nan, units="kg/m")
        self.add_input("data:cables:reference:current", val=np.nan, units="A")
        self.add_input("data:motor:current:hover", val=np.nan, units="A")
        self.add_input("data:airframe:arms:length", val=np.nan, units="m")
        self.add_input("data:airframe:arms:number", val=np.nan, units=None)
        self.add_output("data:cables:density", units="kg/m")
        self.add_output("data:cables:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        mu_ref = inputs[
            "data:cables:reference:density"
        ]  # [kg/m] linear mass of reference cable
        I_ref = inputs[
            "data:cables:reference:current"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:motor:current:hover"]  # [A] max current (continuous)
        Larm = inputs["data:airframe:arms:length"]  # [m] arms length
        Narm = inputs[
            "data:airframe:arms:number"
        ]  # [-] number of arms (i.e. number of cables)

        mu = mu_ref * (I / I_ref) ** (2 / 1.5)  # [kg/m] linear mass of cable
        M = mu * Larm * Narm * 2  # [kg] mass of cables (two cables per arm)

        outputs["data:cables:density"] = mu
        outputs["data:cables:mass"] = M

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mu_ref = inputs[
            "data:cables:reference:density"
        ]  # [kg/m] linear mass of reference cable
        I_ref = inputs[
            "data:cables:reference:current"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:motor:current:hover"]  # [A] max current (continuous)
        Larm = inputs["data:airframe:arms:length"]  # [m] arms length
        Narm = inputs[
            "data:airframe:arms:number"
        ]  # [-] number of arms (i.e. number of cables)

        mu = mu_ref * (I / I_ref) ** (2 / 1.5)

        partials["data:cables:density", "data:motor:current:hover"] = (
            mu_ref / I_ref ** (2 / 1.5) * (2 / 1.5) * I ** (1 / 3)
        )
        partials["data:cables:mass", "data:motor:current:hover"] = (
            mu_ref / I_ref ** (2 / 1.5) * (2 / 1.5) * I ** (1 / 3) * Larm * Narm * 2
        )
        partials["data:cables:mass", "data:airframe:arms:length"] = mu * Narm * 2
