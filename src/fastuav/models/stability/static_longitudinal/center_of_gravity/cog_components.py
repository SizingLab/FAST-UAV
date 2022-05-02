"""
Module containing the center of gravity calculations for all components.
"""
import openmdao.api as om
import numpy as np


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
        self.add_output("data:stability:CoG:fuselage", units="m")

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

        outputs["data:stability:CoG:fuselage"] = x_cg_fus


class CoG_wing(om.ExplicitComponent):
    """
    Computes the center of gravity of the wing.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:LE:x", val=np.nan, units="m")
        self.add_output("data:stability:CoG:wing", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        x_MAC_LE = inputs["data:geometry:wing:MAC:LE:x"]

        x_cg_w = x_MAC_LE + 0.4 * c_MAC  # [m]

        outputs["data:stability:CoG:wing"] = x_cg_w


class CoG_ht(om.ExplicitComponent):
    """
    Computes the center of gravity of the horizontal tail.
    """

    def setup(self):
        self.add_input("data:geometry:tail:horizontal:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:MAC:LE:x", val=np.nan, units="m")
        self.add_output("data:stability:CoG:tail:horizontal", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        c_MAC = inputs["data:geometry:tail:horizontal:MAC:length"]
        x_MAC_LE = inputs["data:geometry:tail:horizontal:MAC:LE:x"]

        x_cg_ht = x_MAC_LE + 0.4 * c_MAC  # [m]

        outputs["data:stability:CoG:tail:horizontal"] = x_cg_ht


class CoG_vt(om.ExplicitComponent):
    """
    Computes the center of gravity of the vertical tail.
    """

    def setup(self):
        self.add_input("data:geometry:tail:vertical:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:vertical:MAC:LE:x", val=np.nan, units="m")
        self.add_output("data:stability:CoG:tail:vertical", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        c_MAC = inputs["data:geometry:tail:vertical:MAC:length"]
        x_MAC_LE = inputs["data:geometry:tail:vertical:MAC:LE:x"]

        x_cg_vt = x_MAC_LE + 0.4 * c_MAC  # [m]

        outputs["data:stability:CoG:tail:vertical"] = x_cg_vt


class CoG_propeller_tractor(om.ExplicitComponent):
    """
    Computes the center of gravity of a tractor propeller located at the nose tip.
    """

    def setup(self):
        self.add_output("data:stability:CoG:propeller", units="m")

    # def setup_partials(self):
    # Finite difference all partials.
    #     self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x_cg_prop = 0  # propeller located at nose tip [m]

        outputs["data:stability:CoG:propeller"] = x_cg_prop


class CoG_motor_tractor(om.ExplicitComponent):
    """
    Computes the center of gravity of the motor located in the nose.
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:length:nose", val=np.nan, units="m")
        self.add_output("data:stability:CoG:motor", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        l_nose = inputs["data:geometry:fuselage:length:nose"]

        x_cg_mot = l_nose / 2  # motor located in the middle of the nose [m]

        outputs["data:stability:CoG:motor"] = x_cg_mot


class CoG_battery(om.ExplicitComponent):
    """
    Computes the center of gravity of the battery pack.
    """

    def setup(self):
        self.add_input("data:geometry:wing:root:LE:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:TE:x", val=np.nan, units="m")
        # self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_output("data:stability:CoG:battery", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x_root_LE_w = inputs["data:geometry:wing:root:LE:x"]
        x_root_TE_w = inputs["data:geometry:wing:root:TE:x"]
        # l_fus = inputs["data:geometry:fuselage:length"]

        # Assume that battery is wing-integrated or centered at wing position
        x_cg_bat = (x_root_LE_w + x_root_TE_w) / 2  # [m]
        # x_cg_bat = l_fus / 2  # [m]

        outputs["data:stability:CoG:battery"] = x_cg_bat



