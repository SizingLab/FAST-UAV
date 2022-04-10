"""
Center of gravity module.
"""
import openmdao.api as om
import numpy as np


class CenterOfGravity(om.Group):
    """
    Group containing the fixed wing center of gravity calculations
    """

    def setup(self):
        self.add_subsystem("fuselage", CoG_fuselage(), promotes=["*"])
        self.add_subsystem("wing", CoG_wing(), promotes=["*"])
        self.add_subsystem("horizontal_tail", CoG_ht(), promotes=["*"])
        self.add_subsystem("vertical_tail", CoG_vt(), promotes=["*"])
        self.add_subsystem("propeller", CoG_propeller(), promotes=["*"])
        self.add_subsystem("motor", CoG_motor(), promotes=["*"])
        self.add_subsystem("battery", CoG_battery(), promotes=["*"])
        self.add_subsystem("UAV", CoG_UAV(), promotes=["*"])


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
        self.add_input("data:weights:fuselage:mass:nose", val=np.nan, units="kg")
        self.add_input("data:weights:fuselage:mass:mid", val=np.nan, units="kg")
        self.add_input("data:weights:fuselage:mass:rear", val=np.nan, units="kg")
        self.add_input("data:weights:fuselage:mass", val=np.nan, units="kg")
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
        m_nose = inputs["data:weights:fuselage:mass:nose"]
        m_mid = inputs["data:weights:fuselage:mass:mid"]
        m_rear = inputs["data:weights:fuselage:mass:rear"]
        m_fus = inputs["data:weights:fuselage:mass"]

        x_cg_nose = l_nose / 2  # [m]
        x_cg_mid = l_nose + l_mid / 2  # [m]
        x_cg_rear = l_nose + l_mid + l_rear / 3 * (d_fus_mid + 2 * d_fus_tip) / (d_fus_mid + d_fus_tip)  # [m]
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


class CoG_propeller(om.ExplicitComponent):
    """
    Computes the center of gravity of the nose propeller.
    """

    def setup(self):
        self.add_output("data:stability:CoG:propeller", units="m")

    # def setup_partials(self):
        # Finite difference all partials.
    #     self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x_cg_prop = 0  # propeller located at nose tip [m]

        outputs["data:stability:CoG:propeller"] = x_cg_prop


class CoG_motor(om.ExplicitComponent):
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


class CoG_UAV(om.ExplicitComponent):
    """
    Computes the center of gravity of a fixed wing UAV.
    """

    def setup(self):
        self.add_input("data:stability:CoG:fuselage", val=np.nan, units="m")
        self.add_input("data:stability:CoG:wing", val=np.nan, units="m")
        self.add_input("data:stability:CoG:tail:horizontal", val=np.nan, units="m")
        self.add_input("data:stability:CoG:tail:vertical", val=np.nan, units="m")
        self.add_input("data:stability:CoG:propeller", val=np.nan, units="m")
        self.add_input("data:stability:CoG:motor", val=np.nan, units="m")
        self.add_input("data:stability:CoG:battery", val=np.nan, units="m")
        self.add_input("data:weights:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weights:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weights:tail:horizontal:mass", val=np.nan, units="kg")
        self.add_input("data:weights:tail:vertical:mass", val=np.nan, units="kg")
        self.add_input("data:weights:propeller:mass", val=np.nan, units="kg")
        self.add_input("data:weights:motor:mass", val=np.nan, units="kg")
        self.add_input("data:weights:battery:mass", val=np.nan, units="kg")
        # self.add_input("data:weights:MTOW", val=np.nan, units="kg")
        self.add_output("data:stability:CoG", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x_cg_fus = inputs["data:stability:CoG:fuselage"]
        x_cg_w = inputs["data:stability:CoG:wing"]
        x_cg_ht = inputs["data:stability:CoG:tail:horizontal"]
        x_cg_vt = inputs["data:stability:CoG:tail:vertical"]
        x_cg_prop = inputs["data:stability:CoG:propeller"]
        x_cg_mot = inputs["data:stability:CoG:motor"]
        x_cg_bat = inputs["data:stability:CoG:battery"]
        m_fus = inputs["data:weights:fuselage:mass"]
        m_wing = inputs["data:weights:wing:mass"]
        m_ht = inputs["data:weights:tail:horizontal:mass"]
        m_vt = inputs["data:weights:tail:vertical:mass"]
        m_prop = inputs["data:weights:propeller:mass"]
        m_mot = inputs["data:weights:motor:mass"]
        m_bat = inputs["data:weights:battery:mass"]
        # m_total = inputs["data:weights:MTOW"]

        x_cg = (x_cg_fus * m_fus
                + x_cg_w * m_wing
                + x_cg_ht * m_ht
                + x_cg_vt * m_vt
                + x_cg_prop * m_prop
                + x_cg_mot * m_mot
                + x_cg_bat * m_bat) / (m_fus + m_wing + m_ht + m_vt + m_prop + m_mot + m_bat)

        outputs["data:stability:CoG"] = x_cg