"""
Multirotor Structures
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.constants import MR_PROPULSION


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

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]

        self.add_input("data:structures:arms:diameter:k", val=np.nan, units=None)
        self.add_input("data:geometry:arms:number", val=np.nan, units=None)
        self.add_input("data:geometry:arms:prop_per_arm", val=np.nan, units=None)
        self.add_input("data:geometry:arms:length", val=np.nan, units="m")
        self.add_input("data:weight:arms:density", val=np.nan, units="kg/m**3")
        self.add_input("data:structures:arms:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, val=np.nan, units="N")

        self.add_output("data:structures:arms:diameter:outer", units="m", lower=0.0)
        self.add_output("data:structures:arms:diameter:inner", units="m", lower=0.0)
        self.add_output("data:weight:airframe:arms:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        D_ratio = inputs["data:structures:arms:diameter:k"]
        Narm = inputs["data:geometry:arms:number"]
        Npro_arm = inputs["data:geometry:arms:prop_per_arm"]
        Larm = inputs["data:geometry:arms:length"]
        rho = inputs["data:weight:arms:density"]
        Sigma_max = inputs["data:structures:arms:stress:max"]
        F_pro_to = inputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id]

        # Inner and outer diameters
        Dout = (F_pro_to * Npro_arm * Larm * 32 / (np.pi * Sigma_max * (1 - D_ratio ** 4))) ** (
                1 / 3
        )  # [m] outer diameter of the beam (sized from max thrust)
        Din = D_ratio * Dout  # [m] inner diameter of the beam

        # Mass calculation
        Marms = (
            np.pi / 4 * (Dout**2 - (D_ratio * Dout) ** 2) * Larm * rho * Narm
        )  # [kg] mass of the arms

        outputs["data:structures:arms:diameter:outer"] = Dout
        outputs["data:structures:arms:diameter:inner"] = Din
        outputs["data:weight:airframe:arms:mass"] = Marms


class BodyWeight(om.ExplicitComponent):
    """
    Computes body weight
    """

    def setup(self):
        self.add_input("data:weight:airframe:arms:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:body:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units="kg")
        self.add_output("data:weight:airframe:body:mass", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Marm_ref = inputs["data:weight:airframe:arms:mass:reference"]
        Mbody_ref = inputs["data:weight:airframe:body:mass:reference"]
        Marms = inputs["data:weight:airframe:arms:mass"]

        Mbody = Mbody_ref * (Marms / Marm_ref)  # [kg] mass of the frame

        outputs["data:weight:airframe:body:mass"] = Mbody
