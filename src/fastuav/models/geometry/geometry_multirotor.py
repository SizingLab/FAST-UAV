"""
Multirotor Airframe Geometry
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.constants import MR_PROPULSION


@oad.RegisterOpenMDAOSystem("fastuav.geometry.multirotor")
class Geometry(om.Group):
    """
    Computes Multi-Rotor geometry
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_subsystem("arms", ArmsGeometry(propulsion_id=propulsion_id), promotes=["*"])
        self.add_subsystem("body", BodyGeometry(), promotes=["*"])


class ArmsGeometry(om.ExplicitComponent):
    """
    Computes arms geometry
    """
    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:propulsion:%s:propeller:is_coaxial" % propulsion_id, val=np.nan, units=None)
        self.add_output("data:geometry:arms:prop_per_arm", units=None)
        self.add_output("data:geometry:arms:number", units=None)
        self.add_output("data:geometry:arms:length", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        Dpro = inputs["data:propulsion:%s:propeller:diameter" % propulsion_id]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        is_coaxial = inputs["data:propulsion:%s:propeller:is_coaxial" % propulsion_id]

        Npro_arm = 1 + is_coaxial
        Narm = N_pro / Npro_arm
        Larm = Dpro / 2 / (np.sin(np.pi / Narm))  # [m] length of the arm (minimum volume allocation)

        outputs["data:geometry:arms:prop_per_arm"] = Npro_arm
        outputs["data:geometry:arms:number"] = Narm
        outputs["data:geometry:arms:length"] = Larm


class BodyGeometry(om.ExplicitComponent):
    """
    Computes body's top and front areas.
    For now this is a simple duplicate of the projected areas estimates,
    such that no consistency constraint for the projected areas is needed.
    """

    def setup(self):
        self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
        self.add_input("data:geometry:projected_area:front", val=np.nan, units="m**2")
        self.add_output("data:geometry:body:surface:top", units="m**2")
        self.add_output("data:geometry:body:surface:front", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["data:geometry:body:surface:top"] = inputs["data:geometry:projected_area:top"]
        outputs["data:geometry:body:surface:front"] = inputs["data:geometry:projected_area:front"]


class ProjectedAreasGuess(om.ExplicitComponent):
    """
    Computes a rough estimate of the projected areas with scaling laws.
    This is used as a preliminary calculation for sizing scenarios.
    """

    def setup(self):
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:body:surface:top:reference", val=np.nan, units="m**2")
        self.add_input("data:geometry:body:surface:front:reference", val=np.nan, units="m**2")
        self.add_input("data:weight:mtow:reference", val=np.nan, units="kg")
        self.add_input("data:geometry:projected_area:top:k", val=1.0, units=None)
        self.add_input("data:geometry:projected_area:front:k", val=1.0, units=None)
        self.add_output("data:geometry:projected_area:top", units="m**2")
        self.add_output("data:geometry:projected_area:front", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        m_uav_guess = inputs["data:weight:mtow:guess"]
        S_top_ref = inputs["data:geometry:body:surface:top:reference"]
        S_front_ref = inputs["data:geometry:body:surface:front:reference"]
        MTOW_ref = inputs["data:weight:mtow:reference"]
        k_top = inputs["data:geometry:projected_area:top:k"]
        k_front = inputs["data:geometry:projected_area:front:k"]

        S_top = k_top * S_top_ref * (m_uav_guess / MTOW_ref) ** (
            2 / 3
        )  # [m2] top surface estimation
        S_front = k_front * S_front_ref * (m_uav_guess / MTOW_ref) ** (
            2 / 3
        )  # [m2] front surface estimation

        outputs["data:geometry:projected_area:top"] = S_top
        outputs["data:geometry:projected_area:front"] = S_front