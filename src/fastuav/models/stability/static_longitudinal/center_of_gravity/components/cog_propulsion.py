"""
Module containing the center of gravity calculations for all components, on the longitudinal axis
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION


class CoG_propulsion_FW(om.ExplicitComponent):
    """
    Computes the center of gravity of the fixed wing propulsion system
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])
        self.options.declare("propulsion_conf", default="tractor", values=["tractor", "pusher"])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        propulsion_conf = self.options["propulsion_conf"]

        self.add_input("data:geometry:wing:root:LE:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:TE:x", val=np.nan, units="m")
        self.add_input("data:propulsion:%s:motor:length:estimated" % propulsion_id, val=np.nan, units="m")
        if propulsion_conf == "pusher":
            self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:weights:propulsion:%s:propeller:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:motor:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:battery:mass" % propulsion_id, val=np.nan, units="kg")

        self.add_output("data:weights:propulsion:%s" % propulsion_id, units="kg")
        self.add_output("data:stability:CoG:propulsion:%s" % propulsion_id, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        propulsion_conf = self.options["propulsion_conf"]
        x_root_LE_w = inputs["data:geometry:wing:root:LE:x"]
        x_root_TE_w = inputs["data:geometry:wing:root:TE:x"]
        l_mot = inputs["data:propulsion:%s:motor:length:estimated" % propulsion_id]
        m_pro = inputs["data:weights:propulsion:%s:propeller:mass" % propulsion_id]
        m_mot = inputs["data:weights:propulsion:%s:motor:mass" % propulsion_id]
        m_bat = inputs["data:weights:propulsion:%s:battery:mass" % propulsion_id]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        if propulsion_conf == "pusher":
            l_fus = inputs["data:geometry:fuselage:length"]
            x_cg_pro = l_fus  # propeller located at fuselage rear [m]
            x_cg_mot = l_fus - l_mot / 2  # motor located at fuselage rear [m]
        else:
            x_cg_pro = 0  # propeller located at nose tip [m]
            x_cg_mot = l_mot / 2  # motor located at the nose tip [m]

        x_cg_bat = (x_root_LE_w + x_root_TE_w) / 2  # [m] wing-integrated or centered at wing position

        m_propulsion = N_pro * m_pro + N_pro * m_mot + m_bat
        x_cg_propulsion = (N_pro * x_cg_pro * m_pro + N_pro * x_cg_mot * m_mot + x_cg_bat * m_bat) / m_propulsion

        outputs["data:weights:propulsion:%s" % propulsion_id] = m_propulsion
        outputs["data:stability:CoG:propulsion:%s" % propulsion_id] = x_cg_propulsion


class CoG_propulsion_MR(om.ExplicitComponent):
    """
    Computes the center of gravity of the VTOL propulsion system
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]

        self.add_input("data:geometry:%s:propeller:x:front" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:x:rear" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:LE:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:TE:x", val=np.nan, units="m")
        self.add_input("data:weights:propulsion:%s:propeller:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:motor:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:battery:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)

        self.add_output("data:weights:propulsion:%s" % propulsion_id, units="kg")
        self.add_output("data:stability:CoG:propulsion:%s" % propulsion_id, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        x_pro_front = inputs["data:geometry:%s:propeller:x:front" % propulsion_id]
        x_pro_rear = inputs["data:geometry:%s:propeller:x:rear" % propulsion_id]
        x_root_LE_w = inputs["data:geometry:wing:root:LE:x"]
        x_root_TE_w = inputs["data:geometry:wing:root:TE:x"]
        m_pro = inputs["data:weights:propulsion:%s:propeller:mass" % propulsion_id]
        m_mot = inputs["data:weights:propulsion:%s:motor:mass" % propulsion_id]
        m_bat = inputs["data:weights:propulsion:%s:battery:mass" % propulsion_id]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        x_cg_pro = x_cg_mot = (x_pro_front + x_pro_rear) / 2  # [m] average position of the propellers / motors
        x_cg_bat = (x_root_LE_w + x_root_TE_w) / 2  # [m] wing-integrated or centered at wing position

        m_propulsion = N_pro * m_pro + N_pro * m_mot + m_bat
        x_cg_propulsion = (N_pro * x_cg_pro * m_pro + N_pro * x_cg_mot * m_mot + x_cg_bat * m_bat) / m_propulsion

        outputs["data:weights:propulsion:%s" % propulsion_id] = m_propulsion
        outputs["data:stability:CoG:propulsion:%s" % propulsion_id] = x_cg_propulsion
