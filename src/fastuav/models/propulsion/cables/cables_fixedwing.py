"""
Cables component
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.models.propulsion.cables.radius import Radius


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.cables.fixedwing")
class Cables(om.Group):
    """
    Group containing the Cables MDA, for fixed wing configurations.
    """

    def setup(self):
        self.add_subsystem("radius", Radius(), promotes=["*"])
        self.add_subsystem("weight", Weight(), promotes=["*"])


class Weight(om.ExplicitComponent):
    """
    Computes cables weight for fixed wing configurations
    """

    def setup(self):
        self.add_input("data:weights:cables:density:reference", val=np.nan, units="kg/m")
        self.add_input("data:propulsion:cables:current:reference", val=np.nan, units="A")
        self.add_input("data:propulsion:motor:current:hover", val=np.nan, units="A")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_output("data:weights:cables:density", units="kg/m")
        self.add_output("data:weights:cables:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        mu_ref = inputs[
            "data:weights:cables:density:reference"
        ]  # [kg/m] linear mass of reference cable
        I_ref = inputs[
            "data:propulsion:cables:current:reference"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:propulsion:motor:current:hover"]  # [A] max current (continuous)
        Lfus = inputs["data:geometry:fuselage:length"]  # [m] fuselage length
        Npro = inputs[
            "data:propulsion:propeller:number"
        ]  # [-] number of arms (i.e. number of cables)

        mu = mu_ref * (I / I_ref) ** (2 / 1.5)  # [kg/m] linear mass of cable
        M = mu * (Lfus / 2) * Npro * 3  # [kg] mass of cables (three cables per motor)

        outputs["data:weights:cables:density"] = mu
        outputs["data:weights:cables:mass"] = M

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mu_ref = inputs[
            "data:weights:cables:density:reference"
        ]  # [kg/m] linear mass of reference cable
        I_ref = inputs[
            "data:propulsion:cables:current:reference"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:propulsion:motor:current:hover"]  # [A] max current (continuous)
        Lfus = inputs["data:geometry:fuselage:length"]  # [m] fuselage length
        Npro = inputs[
            "data:propulsion:propeller:number"
        ]  # [-] number of propellers (i.e. number of cables)

        mu = mu_ref * (I / I_ref) ** (2 / 1.5)

        partials["data:weights:cables:density", "data:propulsion:motor:current:hover"] = (
            mu_ref / I_ref ** (2 / 1.5) * (2 / 1.5) * I ** (1 / 3)
        )
        partials["data:weights:cables:mass", "data:propulsion:motor:current:hover"] = (
            mu_ref / I_ref ** (2 / 1.5) * (2 / 1.5) * I ** (1 / 3) * (Lfus / 2) * Npro * 3
        )
        partials["data:weights:cables:mass", "data:geometry:fuselage:length"] = mu / 2 * Npro * 3
