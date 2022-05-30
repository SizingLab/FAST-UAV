"""
Module containing the center of gravity calculations for all components.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION, PROPULSION_ID_LIST


class CoG_airframe(om.Group):
    """
    Computes the center of gravity of the airframe
    """

    def initialize(self):
        self.options.declare("propulsion_id_list", default=None, values=[[FW_PROPULSION], PROPULSION_ID_LIST])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]
        self.add_subsystem("fuselage", CoG_fuselage(), promotes=["*"])
        self.add_subsystem("wing", CoG_wing(), promotes=["*"])
        self.add_subsystem("horizontal_tail", CoG_tail(tail="horizontal"), promotes=["*"])
        self.add_subsystem("vertical_tail", CoG_tail(tail="vertical"), promotes=["*"])

        if MR_PROPULSION in propulsion_id_list:
            self.add_subsystem("arms_VTOL", CoG_arms_VTOL(), promotes=["*"])

        self.add_subsystem("airframe",
                           CoG_airframe_component(propulsion_id_list=propulsion_id_list),
                           promotes=["*"])


class CoG_airframe_component(om.ExplicitComponent):
    """
    Computes the center of gravity of the airframe
    """

    def initialize(self):
        self.options.declare("propulsion_id_list", default=None, values=[[FW_PROPULSION], PROPULSION_ID_LIST])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]

        self.add_input("data:stability:CoG:airframe:fuselage", val=np.nan, units="m")
        self.add_input("data:stability:CoG:airframe:wing", val=np.nan, units="m")
        self.add_input("data:stability:CoG:airframe:tail:horizontal", val=np.nan, units="m")
        self.add_input("data:stability:CoG:airframe:tail:vertical", val=np.nan, units="m")
        self.add_input("data:weights:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:tail:horizontal:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:tail:vertical:mass", val=np.nan, units="kg")

        if MR_PROPULSION in propulsion_id_list:
            self.add_input("data:stability:CoG:arms", val=np.nan, units="m")
            self.add_input("data:weights:airframe:arms:mass", val=np.nan, units="kg")

        self.add_output("data:weights:airframe", units="kg")
        self.add_output("data:stability:CoG:airframe", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]

        x_cg_fus = inputs["data:stability:CoG:airframe:fuselage"]
        x_cg_w = inputs["data:stability:CoG:airframe:wing"]
        x_cg_ht = inputs["data:stability:CoG:airframe:tail:horizontal"]
        x_cg_vt = inputs["data:stability:CoG:airframe:tail:vertical"]
        m_fus = inputs["data:weights:airframe:fuselage:mass"]
        m_wing = inputs["data:weights:airframe:wing:mass"]
        m_ht = inputs["data:weights:airframe:tail:horizontal:mass"]
        m_vt = inputs["data:weights:airframe:tail:vertical:mass"]

        if MR_PROPULSION in propulsion_id_list:
            x_cg_arms = inputs["data:stability:CoG:arms"]
            m_arms = inputs["data:weights:airframe:arms:mass"]
        else:
            x_cg_arms = .0
            m_arms = .0

        m_airframe = m_fus + m_wing + m_ht + m_vt + m_arms
        x_cg_airframe = (x_cg_fus * m_fus
                         + x_cg_w * m_wing
                         + x_cg_ht * m_ht
                         + x_cg_vt * m_vt
                         + x_cg_arms * m_arms) / m_airframe

        outputs["data:weights:airframe"] = m_airframe
        outputs["data:stability:CoG:airframe"] = x_cg_airframe


class CoG_fuselage(om.ExplicitComponent):
    """
    Computes the center of gravity of the fuselage.
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:length:nose", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length:mid", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length:rear", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:tip", val=np.nan, units="m")
        self.add_input("data:weights:airframe:fuselage:mass:nose", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:fuselage:mass:mid", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:fuselage:mass:rear", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_output("data:stability:CoG:airframe:fuselage", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        l_nose = inputs["data:geometry:fuselage:length:nose"]
        l_mid = inputs["data:geometry:fuselage:length:mid"]
        l_rear = inputs["data:geometry:fuselage:length:rear"]
        d_fus_mid = inputs["data:geometry:fuselage:diameter:mid"]
        d_fus_tip = inputs["data:geometry:fuselage:diameter:tip"]
        m_nose = inputs["data:weights:airframe:fuselage:mass:nose"]
        m_mid = inputs["data:weights:airframe:fuselage:mass:mid"]
        m_rear = inputs["data:weights:airframe:fuselage:mass:rear"]
        m_fus = inputs["data:weights:airframe:fuselage:mass"]

        x_cg_nose = l_nose / 2  # [m]
        x_cg_mid = l_nose + l_mid / 2  # [m]
        x_cg_rear = (
            l_nose + l_mid + l_rear / 3 * (d_fus_mid + 2 * d_fus_tip) / (d_fus_mid + d_fus_tip)
        )  # [m]
        x_cg_fus = (m_nose * x_cg_nose + m_mid * x_cg_mid + m_rear * x_cg_rear) / m_fus  # [m]

        outputs["data:stability:CoG:airframe:fuselage"] = x_cg_fus


class CoG_wing(om.ExplicitComponent):
    """
    Computes the center of gravity of the wing.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:LE:x", val=np.nan, units="m")
        self.add_output("data:stability:CoG:airframe:wing", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        x_MAC_LE = inputs["data:geometry:wing:MAC:LE:x"]

        x_cg_w = x_MAC_LE + 0.4 * c_MAC  # [m]

        outputs["data:stability:CoG:airframe:wing"] = x_cg_w


class CoG_tail(om.ExplicitComponent):
    """
    Computes the center of gravity of the tail (horizontal or vertical).
    """

    def initialize(self):
        self.options.declare("tail", default=None, values=["horizontal", "vertical"])

    def setup(self):
        tail = self.options["tail"]
        self.add_input("data:geometry:tail:%s:MAC:length" % tail, val=np.nan, units="m")
        self.add_input("data:geometry:tail:%s:MAC:LE:x" % tail, val=np.nan, units="m")
        self.add_output("data:stability:CoG:airframe:tail:%s" % tail, units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        tail = self.options["tail"]
        c_MAC = inputs["data:geometry:tail:%s:MAC:length" % tail]
        x_MAC_LE = inputs["data:geometry:tail:%s:MAC:LE:x" % tail]

        x_cg_tail = x_MAC_LE + 0.4 * c_MAC  # [m]

        outputs["data:stability:CoG:airframe:tail:%s" % tail] = x_cg_tail


class CoG_arms_VTOL(om.ExplicitComponent):
    """
    Computes the center of gravity of the VTOL arms.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:geometry:%s:propeller:x:front" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:x:rear" % propulsion_id, val=np.nan, units="m")
        self.add_output("data:stability:CoG:arms", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        x_front = inputs["data:geometry:%s:propeller:x:front" % propulsion_id]
        x_rear = inputs["data:geometry:%s:propeller:x:rear" % propulsion_id]

        x_cg_arms = (x_front + x_rear) / 2  # [m]

        outputs["data:stability:CoG:arms"] = x_cg_arms


