"""
Module containing the center of gravity calculations for the payload and the non-modeled parts
"""

import numpy as np
import openmdao.api as om

from fastuav.constants import FW_PROPULSION, MR_PROPULSION


class CoG_load_FW(om.ExplicitComponent):
    """
    Computes the center of gravity of the fixed wing payload and the non-modeled parts
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:geometry:fuselage:length:nose", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length:mid", val=np.nan, units="m")
        self.add_input("mission:sizing:payload:mass", val=np.nan, units="kg")
        self.add_input("data:weight:misc:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:propulsion:%s:wires:mass" % propulsion_id, val=np.nan, units="kg"
        )
        self.add_input("data:weight:propulsion:%s:esc:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_output("data:weight:load:%s" % propulsion_id, units="kg")
        self.add_output("data:stability:CoG:load:%s" % propulsion_id, units="m")

    def setup_partials(self):
        propulsion_id = self.options["propulsion_id"]

        # CoG of the load is a linear function of the fuselage geometry
        self.declare_partials(
            "data:stability:CoG:load:%s" % propulsion_id,
            "data:geometry:fuselage:length:nose",
            val=1.0,
        )
        self.declare_partials(
            "data:stability:CoG:load:%s" % propulsion_id,
            "data:geometry:fuselage:length:mid",
            val=0.5,
        )

        # Load mass is a plain sum of the contributing masses
        self.declare_partials(
            "data:weight:load:%s" % propulsion_id,
            [
                "data:weight:propulsion:%s:wires:mass" % propulsion_id,
                "mission:sizing:payload:mass",
                "data:weight:misc:mass",
                "data:weight:propulsion:%s:esc:mass" % propulsion_id,
            ],
            val=1.0,
        )

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        lf_nose = inputs["data:geometry:fuselage:length:nose"]
        lf_mid = inputs["data:geometry:fuselage:length:mid"]
        m_payload = inputs["mission:sizing:payload:mass"]
        m_misc = inputs["data:weight:misc:mass"]
        m_wires = inputs["data:weight:propulsion:%s:wires:mass" % propulsion_id]
        m_esc = inputs["data:weight:propulsion:%s:esc:mass" % propulsion_id]

        m_load = m_wires + m_payload + m_misc + m_esc
        x_cg_load = lf_nose + lf_mid / 2

        outputs["data:weight:load:%s" % propulsion_id] = m_load
        outputs["data:stability:CoG:load:%s" % propulsion_id] = x_cg_load


class CoG_load_MR(om.ExplicitComponent):
    """
    Computes the center of gravity of the multirotor non-modeled parts
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:geometry:fuselage:length:nose", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length:mid", val=np.nan, units="m")
        self.add_input(
            "data:weight:propulsion:%s:wires:mass" % propulsion_id, val=np.nan, units="kg"
        )
        self.add_input("data:weight:propulsion:%s:esc:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("mission:sizing:payload:mass", val=np.nan, units="kg")
        self.add_input("data:weight:misc:mass", val=np.nan, units="kg")
        self.add_output("data:weight:load:%s" % propulsion_id, units="kg")
        self.add_output("data:stability:CoG:load:%s" % propulsion_id, units="m")

    def setup_partials(self):
        propulsion_id = self.options["propulsion_id"]

        # CoG of the load is a linear function of the fuselage geometry
        self.declare_partials(
            "data:stability:CoG:load:%s" % propulsion_id,
            "data:geometry:fuselage:length:nose",
            val=1.0,
        )
        self.declare_partials(
            "data:stability:CoG:load:%s" % propulsion_id,
            "data:geometry:fuselage:length:mid",
            val=0.5,
        )

        # Load mass sums the wires and the four ESCs
        self.declare_partials(
            "data:weight:load:%s" % propulsion_id,
            "data:weight:propulsion:%s:wires:mass" % propulsion_id,
            val=1.0,
        )
        self.declare_partials(
            "data:weight:load:%s" % propulsion_id,
            "data:weight:propulsion:%s:esc:mass" % propulsion_id,
            val=4.0,
        )

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        lf_nose = inputs["data:geometry:fuselage:length:nose"]
        lf_mid = inputs["data:geometry:fuselage:length:mid"]
        m_wires = inputs["data:weight:propulsion:%s:wires:mass" % propulsion_id]
        m_esc = inputs["data:weight:propulsion:%s:esc:mass" % propulsion_id]

        m_load = m_wires + 4 * m_esc
        x_cg_load = lf_nose + lf_mid / 2

        outputs["data:weight:load:%s" % propulsion_id] = m_load
        outputs["data:stability:CoG:load:%s" % propulsion_id] = x_cg_load
