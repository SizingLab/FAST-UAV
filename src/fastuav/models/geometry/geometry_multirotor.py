"""
Multirotor Airframe Geometry
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("geometry.multirotor")
class Geometry(om.ExplicitComponent):
    """
    Computes Multi-Rotor geometry
    """

    def setup(self):
        self.add_input("data:structures:arms:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:structures:arms:diameter:k", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:arms:number", val=np.nan, units=None)
        self.add_input("data:geometry:arms:prop_per_arm", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_output("data:geometry:arms:length", units="m", lower=0.0)
        self.add_output("data:structures:arms:diameter:outer", units="m", lower=0.0)
        self.add_output("data:structures:arms:diameter:inner", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Frame sized from max thrust
        Sigma_max = inputs["data:structures:arms:stress:max"]
        D_ratio = inputs["data:structures:arms:diameter:k"]
        Dpro = inputs["data:propulsion:propeller:diameter"]
        Narm = inputs["data:geometry:arms:number"]
        F_pro_to = inputs["data:propulsion:propeller:thrust:takeoff"]
        Npro_arm = inputs["data:geometry:arms:prop_per_arm"]

        Larm = Dpro / 2 / (np.sin(np.pi / Narm))  # [m] length of the arm
        Dout = (F_pro_to * Npro_arm * Larm * 32 / (np.pi * Sigma_max * (1 - D_ratio**4))) ** (
            1 / 3
        )  # [m] outer diameter of the beam
        Din = D_ratio * Dout  # [m] inner diameter of the beam

        outputs["data:geometry:arms:length"] = Larm
        outputs["data:structures:arms:diameter:outer"] = Dout
        outputs["data:structures:arms:diameter:inner"] = Din
