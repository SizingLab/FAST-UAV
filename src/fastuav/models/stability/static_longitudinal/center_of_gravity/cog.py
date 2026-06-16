"""
Center of gravity for fixed wing UAVs.
"""

import numpy as np
import openmdao.api as om

from fastuav.constants import FW_PROPULSION, MR_PROPULSION, PROPULSION_ID_LIST
from fastuav.models.stability.static_longitudinal.center_of_gravity.components.cog_airframe import (
    CoG_airframe,
)
from fastuav.models.stability.static_longitudinal.center_of_gravity.components.cog_load import (
    CoG_load_FW,
    CoG_load_MR,
)
from fastuav.models.stability.static_longitudinal.center_of_gravity.components.cog_propulsion import (
    CoG_propulsion_FW,
    CoG_propulsion_MR,
)


class CenterOfGravity(om.Group):
    """
    Group containing the center of gravity calculations for fixed wing or hybrid VTOL UAVs
    """

    def initialize(self):
        self.options.declare(
            "propulsion_id_list",
            default=None,
            values=[[FW_PROPULSION], PROPULSION_ID_LIST],
        )

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]

        self.add_subsystem(
            "airframe",
            CoG_airframe(propulsion_id_list=propulsion_id_list),
            promotes=["*"],
        )

        if FW_PROPULSION in propulsion_id_list:
            self.add_subsystem("propulsion_fw", CoG_propulsion_FW(), promotes=["*"])
            self.add_subsystem("load_fw", CoG_load_FW(), promotes=["*"])
        if MR_PROPULSION in propulsion_id_list:
            self.add_subsystem("propulsion_mr", CoG_propulsion_MR(), promotes=["*"])
            self.add_subsystem("load_mr", CoG_load_MR(), promotes=["*"])

        self.add_subsystem("UAV", CoG_UAV(propulsion_id_list=propulsion_id_list), promotes=["*"])


class CoG_UAV(om.ExplicitComponent):
    """
    Computes the center of gravity of a fixed wing or hybrid VTOL UAV.
    """

    def initialize(self):
        self.options.declare(
            "propulsion_id_list",
            default=None,
            values=[[FW_PROPULSION], PROPULSION_ID_LIST],
        )

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]

        # Airframe
        self.add_input("data:stability:CoG:airframe", val=np.nan, units="m")
        self.add_input("data:weight:airframe", val=np.nan, units="kg")

        # Propulsion systems
        for propulsion_id in propulsion_id_list:
            self.add_input(
                "data:stability:CoG:propulsion:%s" % propulsion_id,
                val=np.nan,
                units="m",
            )
            self.add_input("data:weight:propulsion:%s" % propulsion_id, val=np.nan, units="kg")
            self.add_input("data:weight:load:%s" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:stability:CoG:load:%s" % propulsion_id, val=0.0, units="m")

        self.add_output("data:stability:CoG:x", units="m")
        self.add_output("data:stability:CoG:y", units="m")
        self.add_output("data:stability:CoG:z", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]

        # Airframe
        x_cg_airframe = inputs["data:stability:CoG:airframe"]
        m_airframe = inputs["data:weight:airframe"]

        # Propulsion systems
        m_propulsion = sum(
            inputs["data:weight:propulsion:%s" % propulsion_id]
            for propulsion_id in propulsion_id_list
        )
        x_cg_propulsion = (
            sum(
                inputs["data:stability:CoG:propulsion:%s" % propulsion_id]
                * inputs["data:weight:propulsion:%s" % propulsion_id]
                for propulsion_id in propulsion_id_list
            )
            / m_propulsion
        )

        # Load (payload and non-modeled parts)
        m_load = sum(
            inputs["data:weight:load:%s" % propulsion_id] for propulsion_id in propulsion_id_list
        )
        x_cg_load = (
            sum(
                inputs["data:stability:CoG:load:%s" % propulsion_id]
                * inputs["data:weight:load:%s" % propulsion_id]
                for propulsion_id in propulsion_id_list
            )
            / m_load
        )

        # UAV
        x_cg_uav = (
            x_cg_airframe * m_airframe + x_cg_propulsion * m_propulsion + x_cg_load * m_load
        ) / (m_propulsion + m_airframe + m_load)

        outputs["data:stability:CoG:x"] = x_cg_uav
        outputs["data:stability:CoG:y"] = 0
        outputs["data:stability:CoG:z"] = 0
