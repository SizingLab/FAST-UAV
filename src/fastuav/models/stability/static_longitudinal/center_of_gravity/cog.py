"""
Center of gravity for fixed wing UAVs.
"""
import openmdao.api as om
import numpy as np
from fastuav.models.stability.static_longitudinal.center_of_gravity.components.cog_airframe import CoG_airframe
from fastuav.models.stability.static_longitudinal.center_of_gravity.components.cog_propulsion import CoG_propulsion_FW, CoG_propulsion_MR
from fastuav.constants import FW_PROPULSION, MR_PROPULSION, PROPULSION_ID_LIST


class CenterOfGravity(om.Group):
    """
    Group containing the center of gravity calculations for fixed wing or hybrid VTOL UAVs
    """
    def initialize(self):
        self.options.declare("propulsion_id_list",
                             default=None,
                             values=[[FW_PROPULSION], PROPULSION_ID_LIST])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]

        self.add_subsystem("airframe", CoG_airframe(propulsion_id_list=propulsion_id_list), promotes=["*"])

        if FW_PROPULSION in propulsion_id_list:
            self.add_subsystem("propulsion_fw", CoG_propulsion_FW(), promotes=["*"])
        if MR_PROPULSION in propulsion_id_list:
            self.add_subsystem("propulsion_mr", CoG_propulsion_MR(), promotes=["*"])

        self.add_subsystem("UAV", CoG_UAV(propulsion_id_list=propulsion_id_list), promotes=["*"])


class CoG_UAV(om.ExplicitComponent):
    """
    Computes the center of gravity of a fixed wing or hybrid VTOL UAV.
    """
    def initialize(self):
        self.options.declare("propulsion_id_list",
                             default=None,
                             values=[[FW_PROPULSION], PROPULSION_ID_LIST])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]

        # Airframe
        self.add_input("data:stability:CoG:airframe", val=np.nan, units="m")
        self.add_input("data:weight:airframe", val=np.nan, units="kg")

        # Propulsion systems
        for propulsion_id in propulsion_id_list:
            self.add_input("data:stability:CoG:propulsion:%s" % propulsion_id, val=np.nan, units="m")
            self.add_input("data:weight:propulsion:%s" % propulsion_id, val=np.nan, units="kg")

        self.add_output("data:stability:CoG", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]

        # Airframe
        x_cg_airframe = inputs["data:stability:CoG:airframe"]
        m_airframe = inputs["data:weight:airframe"]

        # Propulsion systems
        m_propulsion = sum(inputs["data:weight:propulsion:%s" % propulsion_id] for propulsion_id in propulsion_id_list)
        x_cg_propulsion = sum(inputs["data:stability:CoG:propulsion:%s" % propulsion_id]
                              * inputs["data:weight:propulsion:%s" % propulsion_id]
                              for propulsion_id in propulsion_id_list) / m_propulsion

        # UAV
        x_cg_uav = (x_cg_airframe * m_airframe + x_cg_propulsion * m_propulsion) / (m_propulsion + m_airframe)

        outputs["data:stability:CoG"] = x_cg_uav