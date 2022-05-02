"""
Multirotor Airframe Geometry
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import MR_PROPULSION


@oad.RegisterOpenMDAOSystem("fastuav.geometry.multirotor")
class Geometry(om.Group):
    """
    Computes Multi-Rotor geometry
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_subsystem("arms", Arms(propulsion_id=propulsion_id), promotes=["*"])

        # self.add_subsystem("body",
        # BodyAreas(),
        # promotes=["*"])  # The body areas calculation is called prior to sizing scenarios to break the algebraic loop


class Arms(om.ExplicitComponent):
    """
    Computes arms geometry
    """
    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:structures:arms:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:structures:arms:diameter:k", val=np.nan, units=None)
        self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, val=np.nan, units="N")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:propulsion:%s:propeller:is_coaxial" % propulsion_id, val=np.nan, units=None)
        self.add_output("data:geometry:arms:prop_per_arm", units=None)
        self.add_output("data:geometry:arms:number", units=None)
        self.add_output("data:geometry:arms:length", units="m", lower=0.0)
        self.add_output("data:structures:arms:diameter:outer", units="m", lower=0.0)
        self.add_output("data:structures:arms:diameter:inner", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        Sigma_max = inputs["data:structures:arms:stress:max"]
        D_ratio = inputs["data:structures:arms:diameter:k"]
        Dpro = inputs["data:propulsion:%s:propeller:diameter" % propulsion_id]
        F_pro_to = inputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        is_coaxial = inputs["data:propulsion:%s:propeller:is_coaxial" % propulsion_id]

        Npro_arm = 1 + is_coaxial
        Narm = N_pro / Npro_arm

        Larm = Dpro / 2 / (np.sin(np.pi / Narm))  # [m] length of the arm (minimum volume allocation)
        Dout = (F_pro_to * Npro_arm * Larm * 32 / (np.pi * Sigma_max * (1 - D_ratio**4))) ** (
            1 / 3
        )  # [m] outer diameter of the beam (sized from max thrust)
        Din = D_ratio * Dout  # [m] inner diameter of the beam

        outputs["data:geometry:arms:prop_per_arm"] = Npro_arm
        outputs["data:geometry:arms:number"] = Narm
        outputs["data:geometry:arms:length"] = Larm
        outputs["data:structures:arms:diameter:outer"] = Dout
        outputs["data:structures:arms:diameter:inner"] = Din


class BodyAreas(om.ExplicitComponent):
    """
    Computes body's top and front areas with scaling laws based on the MTOW.
    This is used as a preliminary calculation for sizing scenarios.
    """

    def setup(self):
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:body:surface:top:reference", val=np.nan, units="m**2")
        self.add_input("data:geometry:body:surface:front:reference", val=np.nan, units="m**2")
        self.add_input("data:weights:mtow:reference", val=np.nan, units="kg")
        self.add_output("data:geometry:body:surface:top", units="m**2")
        self.add_output("data:geometry:body:surface:front", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:mtow:guess"]
        S_top_ref = inputs["data:geometry:body:surface:top:reference"]
        S_front_ref = inputs["data:geometry:body:surface:front:reference"]
        MTOW_ref = inputs["data:weights:mtow:reference"]

        S_top_estimated = S_top_ref * (Mtotal_guess / MTOW_ref) ** (
            2 / 3
        )  # [m2] top surface estimation
        S_front_estimated = S_front_ref * (Mtotal_guess / MTOW_ref) ** (
            2 / 3
        )  # [m2] front surface estimation

        outputs["data:geometry:body:surface:top"] = S_top_estimated
        outputs["data:geometry:body:surface:front"] = S_front_estimated
