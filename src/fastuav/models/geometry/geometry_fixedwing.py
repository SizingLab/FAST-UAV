"""
Fixed Wing Airframe Geometry
"""

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from scipy.constants import g

from fastuav.constants import FW_PROPULSION, PROPULSION_ID_LIST


@oad.RegisterOpenMDAOSystem("fastuav.geometry.fixedwing")
class Geometry(om.Group):
    """
    Group containing the airframe geometries calculation
    """

    def setup(self):
        self.add_subsystem("wing", WingGeometry(), promotes=["*"])
        self.add_subsystem("horizontal_tail", HorizontalTailGeometry(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailGeometry(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageGeometry(), promotes=["*"])

        constraints = self.add_subsystem("constraints", om.Group(), promotes=["*"])
        # constraints.add_subsystem("projected_areas", ProjectedAreasConstraint(), promotes=["*"])
        constraints.add_subsystem("fuselage_volume", FuselageVolumeConstraint(), promotes=["*"])


class WingGeometry(om.ExplicitComponent):
    """
    Computes Wing geometry
    """

    def setup(self):
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("optimization:variables:geometry:wing:lambda", val=np.nan, units=None)
        self.add_input("data:geometry:wing:sweep:LE", val=np.nan, units="rad")
        self.add_input("optimization:variables:geometry:wing:MAC:LE:x:k", val=0.40, units=None)
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_output("data:geometry:wing:surface", units="m**2", lower=0.0)
        self.add_output("data:geometry:wing:span", units="m", lower=0.0)
        self.add_output("data:geometry:wing:root:chord", units="m", lower=0.0)
        self.add_output("data:geometry:wing:tip:chord", units="m", lower=0.0)
        self.add_output("data:geometry:wing:MAC:length", units="m", lower=0.0)
        self.add_output("data:geometry:wing:root:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:wing:tip:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:wing:MAC:y", units="m", lower=0.0)
        self.add_output("data:geometry:wing:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:wing:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:geometry:wing:root:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:wing:root:TE:x", units="m", lower=0.0)
        self.add_output("data:geometry:wing:sweep:TE", units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        WS = inputs["data:geometry:wing:loading"]
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        sweep_LE = inputs["data:geometry:wing:sweep:LE"]

        # design variables
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        lambda_w = inputs["optimization:variables:geometry:wing:lambda"]
        k_xw = inputs["optimization:variables:geometry:wing:MAC:LE:x:k"]

        # Wing sizing
        S_w = m_uav_guess * g / WS  # wing surface [m2]
        b_w = np.sqrt(AR_w * S_w)  # wing span [m]
        c_root = 2 * S_w / b_w / (1 + lambda_w)  # chord at root [m]
        c_tip = lambda_w * c_root  # chord at tip [m]
        c_MAC = (2 / 3) * c_root * (1 + lambda_w + lambda_w**2) / (1 + lambda_w)  # MAC = MGC [m]
        t_root = c_root * tc_ratio  # wing thickness at root [m]
        t_tip = c_tip * tc_ratio  # wing thickness at tip [m]

        # Wing location
        y_MAC = (
            (b_w / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )  # y-location of MAC (from wing root, i.e. symmetry axis of the UAV) [m]
        x_MAC_LE_loc = y_MAC * np.tan(
            sweep_LE
        )  # x-location of MAC leading edge (from leading edge of root) [m]
        x_MAC_LE = k_xw * b_w  # x-location of MAC leading edge (from nose tip) [m]
        x_MAC_c4 = x_MAC_LE + 0.25 * c_MAC  # x-location of MAC quarter chord (from nose tip) [m]
        x_root_LE = x_MAC_LE - x_MAC_LE_loc  # x-location of root leading edge (from nose tip) [m]
        x_root_TE = x_root_LE + c_root  # x-location of root trailing edge (from nose tip) [m]
        sweep_TE = np.arctan(np.tan(sweep_LE) - 4 / AR_w * (1 - lambda_w) / (1 + lambda_w))

        outputs["data:geometry:wing:surface"] = S_w
        outputs["data:geometry:wing:span"] = b_w
        outputs["data:geometry:wing:root:chord"] = c_root
        outputs["data:geometry:wing:tip:chord"] = c_tip
        outputs["data:geometry:wing:MAC:length"] = c_MAC
        outputs["data:geometry:wing:root:thickness"] = t_root
        outputs["data:geometry:wing:tip:thickness"] = t_tip
        outputs["data:geometry:wing:MAC:y"] = y_MAC
        outputs["data:geometry:wing:MAC:LE:x"] = x_MAC_LE
        outputs["data:geometry:wing:MAC:C4:x"] = x_MAC_c4
        outputs["data:geometry:wing:root:LE:x"] = x_root_LE
        outputs["data:geometry:wing:root:TE:x"] = x_root_TE
        outputs["data:geometry:wing:sweep:TE"] = sweep_TE

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        WS = inputs["data:geometry:wing:loading"]
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        sweep_LE = inputs["data:geometry:wing:sweep:LE"]

        # design variables
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        lambda_w = inputs["optimization:variables:geometry:wing:lambda"]
        k_xw = inputs["optimization:variables:geometry:wing:MAC:LE:x:k"]

        # Wing sizing
        S_w = m_uav_guess * g / WS  # wing surface [m2]
        b_w = np.sqrt(AR_w * S_w)  # wing span [m]
        c_root = 2 * S_w / b_w / (1 + lambda_w)  # chord at root [m]
        c_tip = lambda_w * c_root  # chord at tip [m]
        c_MAC = (2 / 3) * c_root * (1 + lambda_w + lambda_w**2) / (1 + lambda_w)  # MAC = MGC [m]
        c_root * tc_ratio  # wing thickness at root [m]
        c_tip * tc_ratio  # wing thickness at tip [m]

        # Wing location
        y_MAC = (
            (b_w / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )  # y-location of MAC (from wing root, i.e. symmetry axis of the UAV) [m]
        x_MAC_LE_loc = y_MAC * np.tan(
            sweep_LE
        )  # x-location of MAC leading edge (from leading edge of root) [m]
        x_MAC_LE = k_xw * b_w  # x-location of MAC leading edge (from nose tip) [m]
        x_MAC_LE + 0.25 * c_MAC  # x-location of MAC quarter chord (from nose tip) [m]
        x_root_LE = x_MAC_LE - x_MAC_LE_loc  # x-location of root leading edge (from nose tip) [m]
        x_root_LE + c_root  # x-location of root trailing edge (from nose tip) [m]
        np.arctan(np.tan(sweep_LE) - 4 / AR_w * (1 - lambda_w) / (1 + lambda_w))

        out1_ID = "data:geometry:wing:surface"
        partials[out1_ID, "optimization:variables:weight:mtow:guess"] = g / WS
        partials[out1_ID, "data:geometry:wing:loading"] = -m_uav_guess * g / WS**2

        out2_ID = "data:geometry:wing:span"
        partials[out2_ID, "optimization:variables:weight:mtow:guess"] = (
            0.5 * np.sqrt(AR_w / S_w) * g / WS
        )
        partials[out2_ID, "data:geometry:wing:loading"] = (
            0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2)
        )
        partials[out2_ID, "optimization:variables:geometry:wing:AR"] = 0.5 * np.sqrt(S_w / AR_w)

        out3_ID = "data:geometry:wing:root:chord"
        partials[out3_ID, "optimization:variables:weight:mtow:guess"] = (
            1 / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * g / WS
        )
        partials[out3_ID, "data:geometry:wing:loading"] = (
            1 / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * (-m_uav_guess * g / WS**2)
        )
        partials[out3_ID, "optimization:variables:geometry:wing:AR"] = -np.sqrt(S_w / AR_w**3) / (
            1 + lambda_w
        )
        partials[out3_ID, "optimization:variables:geometry:wing:lambda"] = (
            -2 * S_w / b_w / (1 + lambda_w) ** 2
        )

        out4_ID = "data:geometry:wing:tip:chord"
        partials[out4_ID, "optimization:variables:weight:mtow:guess"] = (
            lambda_w / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * g / WS
        )
        partials[out4_ID, "data:geometry:wing:loading"] = (
            lambda_w / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * (-m_uav_guess * g / WS**2)
        )
        partials[out4_ID, "optimization:variables:geometry:wing:AR"] = (
            -lambda_w * np.sqrt(S_w / AR_w**3) / (1 + lambda_w)
        )
        partials[out4_ID, "optimization:variables:geometry:wing:lambda"] = (
            2 * S_w / b_w / (1 + lambda_w) ** 2
        )

        out5_ID = "data:geometry:wing:MAC:length"
        partials[out5_ID, "optimization:variables:weight:mtow:guess"] = (
            (2 / 3)
            * (1 + lambda_w + lambda_w**2)
            / (1 + lambda_w)
            / (np.sqrt(S_w * AR_w) * (1 + lambda_w))
            * g
            / WS
        )
        partials[out5_ID, "data:geometry:wing:loading"] = (
            (2 / 3)
            * (1 + lambda_w + lambda_w**2)
            / (1 + lambda_w)
            / (np.sqrt(S_w * AR_w) * (1 + lambda_w))
            * (-m_uav_guess * g / WS**2)
        )
        partials[out5_ID, "optimization:variables:geometry:wing:AR"] = (
            (2 / 3)
            * (1 + lambda_w + lambda_w**2)
            / (1 + lambda_w)
            * (-np.sqrt(S_w / AR_w**3) / (1 + lambda_w))
        )
        partials[out5_ID, "optimization:variables:geometry:wing:lambda"] = (
            (2 / 3) * 2 * S_w / b_w * (lambda_w - 1) / (1 + lambda_w) ** 3
        )

        out6_ID = "data:geometry:wing:root:thickness"
        partials[out6_ID, "optimization:variables:weight:mtow:guess"] = (
            tc_ratio / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * g / WS
        )
        partials[out6_ID, "data:geometry:wing:loading"] = (
            tc_ratio / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * (-m_uav_guess * g / WS**2)
        )
        partials[out6_ID, "optimization:variables:geometry:wing:AR"] = (
            -tc_ratio * np.sqrt(S_w / AR_w**3) / (1 + lambda_w)
        )
        partials[out6_ID, "optimization:variables:geometry:wing:lambda"] = (
            -2 * tc_ratio * S_w / b_w / (1 + lambda_w) ** 2
        )
        partials[out6_ID, "data:geometry:wing:tc"] = c_root

        out7_ID = "data:geometry:wing:tip:thickness"
        partials[out7_ID, "optimization:variables:weight:mtow:guess"] = (
            tc_ratio * lambda_w / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * g / WS
        )
        partials[out7_ID, "data:geometry:wing:loading"] = (
            tc_ratio
            * lambda_w
            / (np.sqrt(S_w * AR_w) * (1 + lambda_w))
            * (-m_uav_guess * g / WS**2)
        )
        partials[out7_ID, "optimization:variables:geometry:wing:AR"] = (
            -tc_ratio * lambda_w * np.sqrt(S_w / AR_w**3) / (1 + lambda_w)
        )
        partials[out7_ID, "optimization:variables:geometry:wing:lambda"] = (
            2 * tc_ratio * S_w / b_w / (1 + lambda_w) ** 2
        )
        partials[out7_ID, "data:geometry:wing:tc"] = c_tip

        out8_ID = "data:geometry:wing:MAC:y"
        partials[out8_ID, "optimization:variables:weight:mtow:guess"] = (
            (0.5 * np.sqrt(AR_w / S_w) * g / WS / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )
        partials[out8_ID, "data:geometry:wing:loading"] = (
            (0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2) / 6)
            * (1 + 2 * lambda_w)
            / (1 + lambda_w)
        )
        partials[out8_ID, "optimization:variables:geometry:wing:AR"] = (
            (0.5 * np.sqrt(S_w / AR_w) / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )
        partials[out8_ID, "optimization:variables:geometry:wing:lambda"] = (b_w / 6) / (
            1 + lambda_w
        ) ** 2

        out9_ID = "data:geometry:wing:MAC:LE:x"
        partials[out9_ID, "optimization:variables:weight:mtow:guess"] = (
            k_xw * 0.5 * np.sqrt(AR_w / S_w) * g / WS
        )
        partials[out9_ID, "data:geometry:wing:loading"] = (
            k_xw * 0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2)
        )
        partials[out9_ID, "optimization:variables:geometry:wing:AR"] = (
            k_xw * 0.5 * np.sqrt(S_w / AR_w)
        )
        partials[out9_ID, "optimization:variables:geometry:wing:MAC:LE:x:k"] = b_w

        out10_ID = "data:geometry:wing:MAC:C4:x"
        partials[out10_ID, "optimization:variables:weight:mtow:guess"] = (
            k_xw * 0.5 * np.sqrt(AR_w / S_w) * g / WS
            + 0.25
            * (2 / 3)
            * (1 + lambda_w + lambda_w**2)
            / (1 + lambda_w)
            / (np.sqrt(S_w * AR_w) * (1 + lambda_w))
            * g
            / WS
        )
        partials[out10_ID, "data:geometry:wing:loading"] = k_xw * 0.5 * np.sqrt(AR_w / S_w) * (
            -m_uav_guess * g / WS**2
        ) + 0.25 * (2 / 3) * (1 + lambda_w + lambda_w**2) / (1 + lambda_w) / (
            np.sqrt(S_w * AR_w) * (1 + lambda_w)
        ) * (-m_uav_guess * g / WS**2)
        partials[out10_ID, "optimization:variables:geometry:wing:AR"] = k_xw * 0.5 * np.sqrt(
            S_w / AR_w
        ) + 0.25 * (2 / 3) * (1 + lambda_w + lambda_w**2) / (1 + lambda_w) * (
            -np.sqrt(S_w / AR_w**3) / (1 + lambda_w)
        )
        partials[out10_ID, "optimization:variables:geometry:wing:lambda"] = (
            0.25 * (2 / 3) * 2 * S_w / b_w * (lambda_w - 1) / (1 + lambda_w) ** 3
        )
        partials[out10_ID, "optimization:variables:geometry:wing:MAC:LE:x:k"] = b_w

        out11_ID = "data:geometry:wing:root:LE:x"
        partials[out11_ID, "optimization:variables:weight:mtow:guess"] = k_xw * 0.5 * np.sqrt(
            AR_w / S_w
        ) * g / WS - np.tan(sweep_LE) * (
            (0.5 * np.sqrt(AR_w / S_w) * g / WS / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )
        partials[out11_ID, "data:geometry:wing:loading"] = k_xw * 0.5 * np.sqrt(AR_w / S_w) * (
            -m_uav_guess * g / WS**2
        ) - np.tan(sweep_LE) * (
            (0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2) / 6)
            * (1 + 2 * lambda_w)
            / (1 + lambda_w)
        )
        partials[out11_ID, "optimization:variables:geometry:wing:AR"] = k_xw * 0.5 * np.sqrt(
            S_w / AR_w
        ) - np.tan(sweep_LE) * (
            (0.5 * np.sqrt(S_w / AR_w) / 6) * (1 + 2 * lambda_w) / (1 + lambda_w)
        )
        partials[out11_ID, "optimization:variables:geometry:wing:MAC:LE:x:k"] = b_w
        partials[out11_ID, "optimization:variables:geometry:wing:lambda"] = (
            -np.tan(sweep_LE) * (b_w / 6) / (1 + lambda_w) ** 2
        )
        partials[out11_ID, "data:geometry:wing:sweep:LE"] = -1 / np.cos(sweep_LE) ** 2 * y_MAC

        out12_ID = "data:geometry:wing:root:TE:x"
        partials[out12_ID, "optimization:variables:weight:mtow:guess"] = (
            k_xw * 0.5 * np.sqrt(AR_w / S_w) * g / WS
            - np.tan(sweep_LE)
            * ((0.5 * np.sqrt(AR_w / S_w) * g / WS / 6) * (1 + 2 * lambda_w) / (1 + lambda_w))
            + 1 / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * g / WS
        )
        partials[out12_ID, "data:geometry:wing:loading"] = (
            k_xw * 0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2)
            - np.tan(sweep_LE)
            * (
                (0.5 * np.sqrt(AR_w / S_w) * (-m_uav_guess * g / WS**2) / 6)
                * (1 + 2 * lambda_w)
                / (1 + lambda_w)
            )
            + 1 / (np.sqrt(S_w * AR_w) * (1 + lambda_w)) * (-m_uav_guess * g / WS**2)
        )
        partials[out12_ID, "optimization:variables:geometry:wing:AR"] = (
            k_xw * 0.5 * np.sqrt(S_w / AR_w)
            - np.tan(sweep_LE)
            * ((0.5 * np.sqrt(S_w / AR_w) / 6) * (1 + 2 * lambda_w) / (1 + lambda_w))
            + -np.sqrt(S_w / AR_w**3) / (1 + lambda_w)
        )
        partials[out12_ID, "optimization:variables:geometry:wing:MAC:LE:x:k"] = b_w
        partials[out12_ID, "optimization:variables:geometry:wing:lambda"] = (
            -np.tan(sweep_LE) * (b_w / 6) / (1 + lambda_w) ** 2
            - 2 * S_w / b_w / (1 + lambda_w) ** 2
        )
        partials[out12_ID, "data:geometry:wing:sweep:LE"] = -((1 / np.cos(sweep_LE)) ** 2) * y_MAC

        out13_ID = "data:geometry:wing:sweep:TE"
        partials[out13_ID, "optimization:variables:geometry:wing:AR"] = (
            4
            * (1 - lambda_w)
            / (1 + lambda_w)
            / (1 + (np.tan(sweep_LE) - 4 / AR_w * (1 - lambda_w) / (1 + lambda_w)) ** 2)
            / AR_w**2
        )
        partials[out13_ID, "optimization:variables:geometry:wing:lambda"] = (
            4
            / AR_w
            * (2)
            / (1 + lambda_w) ** 2
            / (1 + (np.tan(sweep_LE) - 4 / AR_w * (1 - lambda_w) / (1 + lambda_w)) ** 2)
        )
        partials[out13_ID, "data:geometry:wing:sweep:LE"] = (
            1
            / np.cos(sweep_LE) ** 2
            / (1 + (np.tan(sweep_LE) - 4 / AR_w * (1 - lambda_w) / (1 + lambda_w)) ** 2)
        )


class HorizontalTailGeometry(om.ExplicitComponent):
    """
    Computes Horizontal Tail geometry
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:tail:horizontal:AR", val=4.0, units=None)
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:coefficient", val=0.5, units=None)
        self.add_input("data:geometry:tail:horizontal:lambda", val=0.9, units=None)
        self.add_input(
            "optimization:variables:geometry:tail:horizontal:arm:k", val=0.75, units=None
        )
        self.add_output("data:geometry:tail:horizontal:surface", units="m**2", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:arm", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:span", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:root:chord", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:tip:chord", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:MAC:length", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:root:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:tip:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:MAC:y", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:root:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:horizontal:root:TE:x", units="m", lower=0.0)

        # self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        # self.add_output("optimization:variables:geometry:tail:horizontal:AR", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_ht = inputs["optimization:variables:geometry:tail:horizontal:AR"]
        S_w = inputs["data:geometry:wing:surface"]
        b_w = inputs["data:geometry:wing:span"]
        c_MAC_w = inputs["data:geometry:wing:MAC:length"]
        x_MAC_c4_w = inputs["data:geometry:wing:MAC:C4:x"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        lmbda_ht = inputs["data:geometry:tail:horizontal:lambda"]
        tc_ratio = inputs["data:geometry:wing:tc"]  # assume same profile as wing

        # AR_w = inputs["optimization:variables:geometry:wing:AR"]
        # AR_ht = (2 / 3) * AR_w  # horizontal tail AR [-]

        # design variable
        k_ht = inputs["optimization:variables:geometry:tail:horizontal:arm:k"]

        # Tail sizing
        l_ht = (
            k_ht * b_w
        )  # horizontal tail arm: distance between wing MAC quarter chord and HT MAC quarter chord [m]
        S_ht = V_ht * S_w * c_MAC_w / l_ht  # horizontal tail surface [m2]
        b_ht = np.sqrt(AR_ht * S_ht)  # HT span [m]
        c_root_ht = 2 * S_ht / b_ht / (1 + lmbda_ht)  # chord at root [m]
        c_tip_ht = lmbda_ht * c_root_ht  # chord at tip [m]
        c_MAC_ht = (
            (2 / 3) * c_root_ht * (1 + lmbda_ht + lmbda_ht**2) / (1 + lmbda_ht)
        )  # MAC = MGC [m]
        t_root_ht = c_root_ht * tc_ratio  # wing thickness at root [m]
        t_tip_ht = c_tip_ht * tc_ratio  # wing thickness at tip [m]

        # Tail location
        y_MAC_ht = (
            (b_ht / 6) * (1 + 2 * lmbda_ht) / (1 + lmbda_ht)
        )  # y-location of MAC (from the root) [m]
        x_MAC_LE_loc_ht = 0  # x-location of MAC leading edge (from the leading edge of the root) [m] TODO: add sweep
        x_MAC_c4_ht = x_MAC_c4_w + l_ht  # x-location of MAC quarter chord (from nose tip) [m]
        x_MAC_LE_ht = (
            x_MAC_c4_ht - 0.25 * c_MAC_ht
        )  # x-location of MAC leading edge (from nose tip) [m]
        x_root_LE_ht = (
            x_MAC_LE_ht - x_MAC_LE_loc_ht
        )  # x-location of root leading edge (from nose tip) [m]
        x_root_TE_ht = (
            x_root_LE_ht + c_root_ht
        )  # x-location of root trailing edge (from nose tip) [m]

        outputs["data:geometry:tail:horizontal:arm"] = l_ht
        outputs["data:geometry:tail:horizontal:surface"] = S_ht
        outputs["data:geometry:tail:horizontal:span"] = b_ht
        outputs["data:geometry:tail:horizontal:root:chord"] = c_root_ht
        outputs["data:geometry:tail:horizontal:tip:chord"] = c_tip_ht
        outputs["data:geometry:tail:horizontal:MAC:length"] = c_MAC_ht
        outputs["data:geometry:tail:horizontal:root:thickness"] = t_root_ht
        outputs["data:geometry:tail:horizontal:tip:thickness"] = t_tip_ht
        outputs["data:geometry:tail:horizontal:MAC:y"] = y_MAC_ht
        outputs["data:geometry:tail:horizontal:MAC:LE:x"] = x_MAC_LE_ht
        outputs["data:geometry:tail:horizontal:MAC:C4:x"] = x_MAC_c4_ht
        outputs["data:geometry:tail:horizontal:root:LE:x"] = x_root_LE_ht
        outputs["data:geometry:tail:horizontal:root:TE:x"] = x_root_TE_ht
        # outputs["optimization:variables:geometry:tail:horizontal:AR"] = AR_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_ht = inputs["optimization:variables:geometry:tail:horizontal:AR"]
        S_w = inputs["data:geometry:wing:surface"]
        b_w = inputs["data:geometry:wing:span"]
        c_MAC_w = inputs["data:geometry:wing:MAC:length"]
        x_MAC_c4_w = inputs["data:geometry:wing:MAC:C4:x"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        lmbda_ht = inputs["data:geometry:tail:horizontal:lambda"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        k_ht = inputs["optimization:variables:geometry:tail:horizontal:arm:k"]

        # Intermediate values
        l_ht = k_ht * b_w
        S_ht = V_ht * S_w * c_MAC_w / l_ht
        b_ht = np.sqrt(AR_ht * S_ht)
        c_root_ht = 2 * S_ht / b_ht / (1 + lmbda_ht)
        c_tip_ht = lmbda_ht * c_root_ht
        c_MAC_ht = (2 / 3) * c_root_ht * (1 + lmbda_ht + lmbda_ht**2) / (1 + lmbda_ht)
        x_MAC_c4_ht = x_MAC_c4_w + l_ht
        x_MAC_c4_ht - 0.25 * c_MAC_ht

        # --- Key intermediate partials (chain rule building blocks) ---
        # dl_ht/d...
        dl_ht_dk_ht = b_w
        dl_ht_db_w = k_ht

        # dS_ht/d... (note: S_ht = V_ht * S_w * c_MAC_w / l_ht)
        dS_ht_dV_ht = S_w * c_MAC_w / l_ht
        dS_ht_dS_w = V_ht * c_MAC_w / l_ht
        dS_ht_dc_MAC_w = V_ht * S_w / l_ht
        dS_ht_dl_ht = -V_ht * S_w * c_MAC_w / (l_ht**2)
        dS_ht_dk_ht = dS_ht_dl_ht * dl_ht_dk_ht
        dS_ht_db_w = dS_ht_dl_ht * dl_ht_db_w

        # db_ht/d... (note: b_ht = sqrt(AR_ht * S_ht))
        db_ht_dAR_ht = 0.5 * np.sqrt(S_ht / AR_ht)
        db_ht_dS_ht = 0.5 * np.sqrt(AR_ht / S_ht)
        db_ht_dV_ht = db_ht_dS_ht * dS_ht_dV_ht
        db_ht_dS_w = db_ht_dS_ht * dS_ht_dS_w
        db_ht_dc_MAC_w = db_ht_dS_ht * dS_ht_dc_MAC_w
        db_ht_dk_ht = db_ht_dS_ht * dS_ht_dk_ht
        db_ht_db_w = db_ht_dS_ht * dS_ht_db_w

        # dc_root_ht/d... (note: c_root_ht = 2*S_ht / b_ht / (1+lambda);
        # c_root depends on S_ht BOTH directly and through b_ht, so every
        # variable that moves S_ht must propagate through both paths.)
        dc_root_dS_ht = 2 / b_ht / (1 + lmbda_ht)
        dc_root_db_ht = -2 * S_ht / (b_ht**2) / (1 + lmbda_ht)
        dc_root_dlambda = -2 * S_ht / b_ht / (1 + lmbda_ht) ** 2
        # FIX: add the b_ht path for S_w, c_MAC_w, V_ht (was missing -> factor-2 error)
        dc_root_dV_ht = dc_root_dS_ht * dS_ht_dV_ht + dc_root_db_ht * db_ht_dV_ht
        dc_root_dS_w = dc_root_dS_ht * dS_ht_dS_w + dc_root_db_ht * db_ht_dS_w
        dc_root_dc_MAC_w = dc_root_dS_ht * dS_ht_dc_MAC_w + dc_root_db_ht * db_ht_dc_MAC_w
        dc_root_dk_ht = dc_root_dS_ht * dS_ht_dk_ht + dc_root_db_ht * db_ht_dk_ht
        dc_root_db_w = dc_root_dS_ht * dS_ht_db_w + dc_root_db_ht * db_ht_db_w
        dc_root_dAR_ht = dc_root_db_ht * db_ht_dAR_ht

        # dc_MAC_ht/d... (note: c_MAC_ht = (2/3)*c_root_ht*(1+lambda+lambda^2)/(1+lambda))
        coef_MAC = (2 / 3) * (1 + lmbda_ht + lmbda_ht**2) / (1 + lmbda_ht)
        dc_MAC_dc_root = coef_MAC
        # FIX: total derivative wrt lambda = c_root-path term + correct explicit term.
        # d/dlambda[(1+l+l^2)/(1+l)] = l*(2+l)/(1+l)^2
        dc_MAC_dlambda = (
            coef_MAC * dc_root_dlambda
            + (2 / 3) * c_root_ht * lmbda_ht * (2 + lmbda_ht) / (1 + lmbda_ht) ** 2
        )
        dc_MAC_dV_ht = dc_MAC_dc_root * dc_root_dV_ht
        dc_MAC_dS_w = dc_MAC_dc_root * dc_root_dS_w
        dc_MAC_dc_MAC_w = dc_MAC_dc_root * dc_root_dc_MAC_w
        dc_MAC_dk_ht = dc_MAC_dc_root * dc_root_dk_ht
        dc_MAC_db_w = dc_MAC_dc_root * dc_root_db_w
        dc_MAC_dAR_ht = dc_MAC_dc_root * dc_root_dAR_ht

        # --- Output 1: l_ht ---
        partials[
            "data:geometry:tail:horizontal:arm",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dl_ht_dk_ht
        partials["data:geometry:tail:horizontal:arm", "data:geometry:wing:span"] = dl_ht_db_w

        # --- Output 2: S_ht ---
        partials[
            "data:geometry:tail:horizontal:surface", "data:geometry:tail:horizontal:coefficient"
        ] = dS_ht_dV_ht
        partials["data:geometry:tail:horizontal:surface", "data:geometry:wing:surface"] = dS_ht_dS_w
        partials["data:geometry:tail:horizontal:surface", "data:geometry:wing:MAC:length"] = (
            dS_ht_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:surface",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dS_ht_dk_ht
        partials["data:geometry:tail:horizontal:surface", "data:geometry:wing:span"] = dS_ht_db_w

        # --- Output 3: b_ht ---
        partials[
            "data:geometry:tail:horizontal:span",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = db_ht_dAR_ht
        partials[
            "data:geometry:tail:horizontal:span", "data:geometry:tail:horizontal:coefficient"
        ] = db_ht_dV_ht
        partials["data:geometry:tail:horizontal:span", "data:geometry:wing:surface"] = db_ht_dS_w
        partials["data:geometry:tail:horizontal:span", "data:geometry:wing:MAC:length"] = (
            db_ht_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:span",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = db_ht_dk_ht
        partials["data:geometry:tail:horizontal:span", "data:geometry:wing:span"] = db_ht_db_w

        # --- Output 4: c_root_ht ---
        partials[
            "data:geometry:tail:horizontal:root:chord",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = dc_root_dAR_ht
        partials[
            "data:geometry:tail:horizontal:root:chord", "data:geometry:tail:horizontal:coefficient"
        ] = dc_root_dV_ht
        partials["data:geometry:tail:horizontal:root:chord", "data:geometry:wing:surface"] = (
            dc_root_dS_w
        )
        partials["data:geometry:tail:horizontal:root:chord", "data:geometry:wing:MAC:length"] = (
            dc_root_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:root:chord",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dc_root_dk_ht
        partials["data:geometry:tail:horizontal:root:chord", "data:geometry:wing:span"] = (
            dc_root_db_w
        )
        partials[
            "data:geometry:tail:horizontal:root:chord", "data:geometry:tail:horizontal:lambda"
        ] = dc_root_dlambda

        # --- Output 5: c_tip_ht = lambda * c_root_ht ---
        partials[
            "data:geometry:tail:horizontal:tip:chord",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = lmbda_ht * dc_root_dAR_ht
        partials[
            "data:geometry:tail:horizontal:tip:chord", "data:geometry:tail:horizontal:coefficient"
        ] = lmbda_ht * dc_root_dV_ht
        partials["data:geometry:tail:horizontal:tip:chord", "data:geometry:wing:surface"] = (
            lmbda_ht * dc_root_dS_w
        )
        partials["data:geometry:tail:horizontal:tip:chord", "data:geometry:wing:MAC:length"] = (
            lmbda_ht * dc_root_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:tip:chord",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = lmbda_ht * dc_root_dk_ht
        partials["data:geometry:tail:horizontal:tip:chord", "data:geometry:wing:span"] = (
            lmbda_ht * dc_root_db_w
        )
        partials[
            "data:geometry:tail:horizontal:tip:chord", "data:geometry:tail:horizontal:lambda"
        ] = c_root_ht + lmbda_ht * dc_root_dlambda

        # --- Output 6: c_MAC_ht ---
        partials[
            "data:geometry:tail:horizontal:MAC:length",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = dc_MAC_dAR_ht
        partials[
            "data:geometry:tail:horizontal:MAC:length", "data:geometry:tail:horizontal:coefficient"
        ] = dc_MAC_dV_ht
        partials["data:geometry:tail:horizontal:MAC:length", "data:geometry:wing:surface"] = (
            dc_MAC_dS_w
        )
        partials["data:geometry:tail:horizontal:MAC:length", "data:geometry:wing:MAC:length"] = (
            dc_MAC_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:MAC:length",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dc_MAC_dk_ht
        partials["data:geometry:tail:horizontal:MAC:length", "data:geometry:wing:span"] = (
            dc_MAC_db_w
        )
        partials[
            "data:geometry:tail:horizontal:MAC:length", "data:geometry:tail:horizontal:lambda"
        ] = dc_MAC_dlambda

        # --- Output 7: t_root_ht = c_root_ht * tc_ratio ---
        partials[
            "data:geometry:tail:horizontal:root:thickness",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = tc_ratio * dc_root_dAR_ht
        partials[
            "data:geometry:tail:horizontal:root:thickness",
            "data:geometry:tail:horizontal:coefficient",
        ] = tc_ratio * dc_root_dV_ht
        partials["data:geometry:tail:horizontal:root:thickness", "data:geometry:wing:surface"] = (
            tc_ratio * dc_root_dS_w
        )
        partials[
            "data:geometry:tail:horizontal:root:thickness", "data:geometry:wing:MAC:length"
        ] = tc_ratio * dc_root_dc_MAC_w
        partials[
            "data:geometry:tail:horizontal:root:thickness",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = tc_ratio * dc_root_dk_ht
        partials["data:geometry:tail:horizontal:root:thickness", "data:geometry:wing:span"] = (
            tc_ratio * dc_root_db_w
        )
        partials[
            "data:geometry:tail:horizontal:root:thickness", "data:geometry:tail:horizontal:lambda"
        ] = tc_ratio * dc_root_dlambda
        partials["data:geometry:tail:horizontal:root:thickness", "data:geometry:wing:tc"] = (
            c_root_ht
        )

        # --- Output 8: t_tip_ht = c_tip_ht * tc_ratio ---
        partials[
            "data:geometry:tail:horizontal:tip:thickness",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = tc_ratio * lmbda_ht * dc_root_dAR_ht
        partials[
            "data:geometry:tail:horizontal:tip:thickness",
            "data:geometry:tail:horizontal:coefficient",
        ] = tc_ratio * lmbda_ht * dc_root_dV_ht
        partials["data:geometry:tail:horizontal:tip:thickness", "data:geometry:wing:surface"] = (
            tc_ratio * lmbda_ht * dc_root_dS_w
        )
        partials["data:geometry:tail:horizontal:tip:thickness", "data:geometry:wing:MAC:length"] = (
            tc_ratio * lmbda_ht * dc_root_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:tip:thickness",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = tc_ratio * lmbda_ht * dc_root_dk_ht
        partials["data:geometry:tail:horizontal:tip:thickness", "data:geometry:wing:span"] = (
            tc_ratio * lmbda_ht * dc_root_db_w
        )
        partials[
            "data:geometry:tail:horizontal:tip:thickness", "data:geometry:tail:horizontal:lambda"
        ] = tc_ratio * (c_root_ht + lmbda_ht * dc_root_dlambda)
        partials["data:geometry:tail:horizontal:tip:thickness", "data:geometry:wing:tc"] = c_tip_ht

        # --- Output 9: y_MAC_ht = (b_ht/6) * (1+2*lambda) / (1+lambda) ---
        dy_MAC_db_ht = (1 + 2 * lmbda_ht) / (6 * (1 + lmbda_ht))
        dy_MAC_dlambda = (
            (b_ht / 6) * (2 * (1 + lmbda_ht) - (1 + 2 * lmbda_ht)) / (1 + lmbda_ht) ** 2
        )
        partials[
            "data:geometry:tail:horizontal:MAC:y",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = dy_MAC_db_ht * db_ht_dAR_ht
        partials[
            "data:geometry:tail:horizontal:MAC:y", "data:geometry:tail:horizontal:coefficient"
        ] = dy_MAC_db_ht * db_ht_dV_ht
        partials["data:geometry:tail:horizontal:MAC:y", "data:geometry:wing:surface"] = (
            dy_MAC_db_ht * db_ht_dS_w
        )
        partials["data:geometry:tail:horizontal:MAC:y", "data:geometry:wing:MAC:length"] = (
            dy_MAC_db_ht * db_ht_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:MAC:y",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dy_MAC_db_ht * db_ht_dk_ht
        partials["data:geometry:tail:horizontal:MAC:y", "data:geometry:wing:span"] = (
            dy_MAC_db_ht * db_ht_db_w
        )
        partials["data:geometry:tail:horizontal:MAC:y", "data:geometry:tail:horizontal:lambda"] = (
            dy_MAC_dlambda
        )

        # --- Output 10: x_MAC_c4_ht = x_MAC_c4_w + l_ht ---
        partials["data:geometry:tail:horizontal:MAC:C4:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:horizontal:MAC:C4:x",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dl_ht_dk_ht
        partials["data:geometry:tail:horizontal:MAC:C4:x", "data:geometry:wing:span"] = dl_ht_db_w

        # --- Output 11: x_MAC_LE_ht = x_MAC_c4_ht - 0.25*c_MAC_ht ---
        partials["data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:horizontal:MAC:LE:x",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = -0.25 * dc_MAC_dAR_ht
        partials[
            "data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:tail:horizontal:coefficient"
        ] = -0.25 * dc_MAC_dV_ht
        partials["data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w
        )
        partials["data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:wing:MAC:length"] = (
            -0.25 * dc_MAC_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:MAC:LE:x",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dl_ht_dk_ht - 0.25 * dc_MAC_dk_ht
        partials["data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:wing:span"] = (
            dl_ht_db_w - 0.25 * dc_MAC_db_w
        )
        partials[
            "data:geometry:tail:horizontal:MAC:LE:x", "data:geometry:tail:horizontal:lambda"
        ] = -0.25 * dc_MAC_dlambda

        # --- Output 12: x_root_LE_ht = x_MAC_LE_ht (since x_MAC_LE_loc_ht = 0) ---
        partials["data:geometry:tail:horizontal:root:LE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:horizontal:root:LE:x",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = -0.25 * dc_MAC_dAR_ht
        partials[
            "data:geometry:tail:horizontal:root:LE:x", "data:geometry:tail:horizontal:coefficient"
        ] = -0.25 * dc_MAC_dV_ht
        partials["data:geometry:tail:horizontal:root:LE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w
        )
        partials["data:geometry:tail:horizontal:root:LE:x", "data:geometry:wing:MAC:length"] = (
            -0.25 * dc_MAC_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:root:LE:x",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dl_ht_dk_ht - 0.25 * dc_MAC_dk_ht
        partials["data:geometry:tail:horizontal:root:LE:x", "data:geometry:wing:span"] = (
            dl_ht_db_w - 0.25 * dc_MAC_db_w
        )
        partials[
            "data:geometry:tail:horizontal:root:LE:x", "data:geometry:tail:horizontal:lambda"
        ] = -0.25 * dc_MAC_dlambda

        # --- Output 13: x_root_TE_ht = x_root_LE_ht + c_root_ht ---
        partials["data:geometry:tail:horizontal:root:TE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:horizontal:root:TE:x",
            "optimization:variables:geometry:tail:horizontal:AR",
        ] = -0.25 * dc_MAC_dAR_ht + dc_root_dAR_ht
        partials[
            "data:geometry:tail:horizontal:root:TE:x", "data:geometry:tail:horizontal:coefficient"
        ] = -0.25 * dc_MAC_dV_ht + dc_root_dV_ht
        partials["data:geometry:tail:horizontal:root:TE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w + dc_root_dS_w
        )
        partials["data:geometry:tail:horizontal:root:TE:x", "data:geometry:wing:MAC:length"] = (
            -0.25 * dc_MAC_dc_MAC_w + dc_root_dc_MAC_w
        )
        partials[
            "data:geometry:tail:horizontal:root:TE:x",
            "optimization:variables:geometry:tail:horizontal:arm:k",
        ] = dl_ht_dk_ht - 0.25 * dc_MAC_dk_ht + dc_root_dk_ht
        partials["data:geometry:tail:horizontal:root:TE:x", "data:geometry:wing:span"] = (
            dl_ht_db_w - 0.25 * dc_MAC_db_w + dc_root_db_w
        )
        partials[
            "data:geometry:tail:horizontal:root:TE:x", "data:geometry:tail:horizontal:lambda"
        ] = -0.25 * dc_MAC_dlambda + dc_root_dlambda


class VerticalTailGeometry(om.ExplicitComponent):
    """
    Computes Vertical Tail geometry
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:tail:vertical:AR", val=1.5, units=None)
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:tail:vertical:coefficient", val=0.04, units=None)
        self.add_input("data:geometry:tail:vertical:lambda", val=0.9, units=None)
        self.add_input("data:geometry:tail:horizontal:arm", val=np.nan, units="m")
        self.add_output("data:geometry:tail:vertical:surface", units="m**2", lower=0.0)
        self.add_output("data:geometry:tail:vertical:arm", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:span", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:root:chord", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:tip:chord", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:MAC:length", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:root:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:tip:thickness", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:MAC:z", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:root:LE:x", units="m", lower=0.0)
        self.add_output("data:geometry:tail:vertical:root:TE:x", units="m", lower=0.0)

        # self.add_input("optimization:variables:geometry:tail:horizontal:AR", val=np.nan, units=None)
        # self.add_output("optimization:variables:geometry:tail:vertical:AR", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_vt = inputs["optimization:variables:geometry:tail:vertical:AR"]
        S_w = inputs["data:geometry:wing:surface"]
        b_w = inputs["data:geometry:wing:span"]
        x_MAC_c4_w = inputs["data:geometry:wing:MAC:C4:x"]
        V_vt = inputs["data:geometry:tail:vertical:coefficient"]
        lmbda_vt = inputs["data:geometry:tail:vertical:lambda"]
        tc_ratio = inputs["data:geometry:wing:tc"]  # assume same profile as wing
        l_ht = inputs["data:geometry:tail:horizontal:arm"]

        # AR_ht = inputs["optimization:variables:geometry:tail:horizontal:AR"]
        # AR_vt = (1 / 2) * AR_ht  # vertical tail AR [-]

        # Tail sizing
        l_vt = l_ht  # vertical and horizontal tails arms [m]
        S_vt = V_vt * S_w * b_w / l_vt  # vertical tail surface [m2]
        b_vt = np.sqrt(AR_vt * S_vt)  # VT span [m]
        c_root_vt = 2 * S_vt / b_vt / (1 + lmbda_vt)  # chord at root [m]
        c_tip_vt = lmbda_vt * c_root_vt  # chord at tip [m]
        c_MAC_vt = (
            (2 / 3) * c_root_vt * (1 + lmbda_vt + lmbda_vt**2) / (1 + lmbda_vt)
        )  # MAC = MGC [m]
        t_root_vt = c_root_vt * tc_ratio  # wing thickness at root [m]
        t_tip_vt = c_tip_vt * tc_ratio  # wing thickness at tip [m]

        # Tail location
        z_MAC_vt = (
            (b_vt / 6) * (1 + 2 * lmbda_vt) / (1 + lmbda_vt)
        )  # z-location of MAC (from the root) [m]
        x_MAC_LE_loc_vt = 0  # x-location of MAC leading edge (from the leading edge of the root) [m] TODO: add sweep
        x_MAC_c4_vt = x_MAC_c4_w + l_vt  # x-location of MAC quarter chord (from nose tip) [m]
        x_MAC_LE_vt = (
            x_MAC_c4_vt - 0.25 * c_MAC_vt
        )  # x-location of MAC leading edge (from nose tip) [m]
        x_root_LE_vt = (
            x_MAC_LE_vt - x_MAC_LE_loc_vt
        )  # x-location of root leading edge (from nose tip) [m]
        x_root_TE_vt = (
            x_root_LE_vt + c_root_vt
        )  # x-location of root trailing edge (from nose tip) [m]

        outputs["data:geometry:tail:vertical:arm"] = l_vt
        outputs["data:geometry:tail:vertical:surface"] = S_vt
        outputs["data:geometry:tail:vertical:span"] = b_vt
        outputs["data:geometry:tail:vertical:root:chord"] = c_root_vt
        outputs["data:geometry:tail:vertical:tip:chord"] = c_tip_vt
        outputs["data:geometry:tail:vertical:MAC:length"] = c_MAC_vt
        outputs["data:geometry:tail:vertical:root:thickness"] = t_root_vt
        outputs["data:geometry:tail:vertical:tip:thickness"] = t_tip_vt
        outputs["data:geometry:tail:vertical:MAC:z"] = z_MAC_vt
        outputs["data:geometry:tail:vertical:MAC:LE:x"] = x_MAC_LE_vt
        outputs["data:geometry:tail:vertical:MAC:C4:x"] = x_MAC_c4_vt
        outputs["data:geometry:tail:vertical:root:LE:x"] = x_root_LE_vt
        outputs["data:geometry:tail:vertical:root:TE:x"] = x_root_TE_vt
        # outputs["optimization:variables:geometry:tail:vertical:AR"] = AR_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_vt = inputs["optimization:variables:geometry:tail:vertical:AR"]
        S_w = inputs["data:geometry:wing:surface"]
        b_w = inputs["data:geometry:wing:span"]
        x_MAC_c4_w = inputs["data:geometry:wing:MAC:C4:x"]
        V_vt = inputs["data:geometry:tail:vertical:coefficient"]
        lmbda_vt = inputs["data:geometry:tail:vertical:lambda"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        l_ht = inputs["data:geometry:tail:horizontal:arm"]

        # Intermediate values
        l_vt = l_ht
        S_vt = V_vt * S_w * b_w / l_vt
        b_vt = np.sqrt(AR_vt * S_vt)
        c_root_vt = 2 * S_vt / b_vt / (1 + lmbda_vt)
        c_tip_vt = lmbda_vt * c_root_vt
        c_MAC_vt = (2 / 3) * c_root_vt * (1 + lmbda_vt + lmbda_vt**2) / (1 + lmbda_vt)
        x_MAC_c4_vt = x_MAC_c4_w + l_vt
        x_MAC_c4_vt - 0.25 * c_MAC_vt

        # --- Key intermediate partials (chain rule building blocks) ---
        # dl_vt/d...
        dl_vt_dl_ht = 1.0

        # dS_vt/d... (note: S_vt = V_vt * S_w * b_w / l_vt)
        dS_vt_dV_vt = S_w * b_w / l_vt
        dS_vt_dS_w = V_vt * b_w / l_vt
        dS_vt_db_w = V_vt * S_w / l_vt
        dS_vt_dl_vt = -V_vt * S_w * b_w / (l_vt**2)
        dS_vt_dl_ht = dS_vt_dl_vt * dl_vt_dl_ht

        # db_vt/d... (note: b_vt = sqrt(AR_vt * S_vt))
        db_vt_dAR_vt = 0.5 * np.sqrt(S_vt / AR_vt)
        db_vt_dS_vt = 0.5 * np.sqrt(AR_vt / S_vt)
        db_vt_dV_vt = db_vt_dS_vt * dS_vt_dV_vt
        db_vt_dS_w = db_vt_dS_vt * dS_vt_dS_w
        db_vt_db_w = db_vt_dS_vt * dS_vt_db_w
        db_vt_dl_ht = db_vt_dS_vt * dS_vt_dl_ht

        # dc_root_vt/d... (note: c_root_vt = 2*S_vt / b_vt / (1+lambda);
        # c_root depends on S_vt BOTH directly and through b_vt, so every
        # variable that moves S_vt must propagate through both paths.)
        dc_root_dS_vt = 2 / b_vt / (1 + lmbda_vt)
        dc_root_db_vt = -2 * S_vt / (b_vt**2) / (1 + lmbda_vt)
        dc_root_dlambda = -2 * S_vt / b_vt / (1 + lmbda_vt) ** 2
        # FIX: add the b_vt path for V_vt and S_w (was missing -> factor-2 error)
        dc_root_dV_vt = dc_root_dS_vt * dS_vt_dV_vt + dc_root_db_vt * db_vt_dV_vt
        dc_root_dS_w = dc_root_dS_vt * dS_vt_dS_w + dc_root_db_vt * db_vt_dS_w
        dc_root_db_w = dc_root_dS_vt * dS_vt_db_w + dc_root_db_vt * db_vt_db_w
        dc_root_dl_ht = dc_root_dS_vt * dS_vt_dl_ht + dc_root_db_vt * db_vt_dl_ht
        dc_root_dAR_vt = dc_root_db_vt * db_vt_dAR_vt

        # dc_MAC_vt/d... (note: c_MAC_vt = (2/3)*c_root_vt*(1+lambda+lambda^2)/(1+lambda))
        coef_MAC = (2 / 3) * (1 + lmbda_vt + lmbda_vt**2) / (1 + lmbda_vt)
        dc_MAC_dc_root = coef_MAC
        # FIX: total derivative wrt lambda = c_root-path term + correct explicit term.
        # d/dlambda[(1+l+l^2)/(1+l)] = l*(2+l)/(1+l)^2
        dc_MAC_dlambda = (
            coef_MAC * dc_root_dlambda
            + (2 / 3) * c_root_vt * lmbda_vt * (2 + lmbda_vt) / (1 + lmbda_vt) ** 2
        )
        dc_MAC_dV_vt = dc_MAC_dc_root * dc_root_dV_vt
        dc_MAC_dS_w = dc_MAC_dc_root * dc_root_dS_w
        dc_MAC_db_w = dc_MAC_dc_root * dc_root_db_w
        dc_MAC_dl_ht = dc_MAC_dc_root * dc_root_dl_ht
        dc_MAC_dAR_vt = dc_MAC_dc_root * dc_root_dAR_vt

        # --- Output 1: l_vt ---
        partials["data:geometry:tail:vertical:arm", "data:geometry:tail:horizontal:arm"] = (
            dl_vt_dl_ht
        )

        # --- Output 2: S_vt ---
        partials[
            "data:geometry:tail:vertical:surface", "data:geometry:tail:vertical:coefficient"
        ] = dS_vt_dV_vt
        partials["data:geometry:tail:vertical:surface", "data:geometry:wing:surface"] = dS_vt_dS_w
        partials["data:geometry:tail:vertical:surface", "data:geometry:wing:span"] = dS_vt_db_w
        partials["data:geometry:tail:vertical:surface", "data:geometry:tail:horizontal:arm"] = (
            dS_vt_dl_ht
        )

        # --- Output 3: b_vt ---
        partials[
            "data:geometry:tail:vertical:span", "optimization:variables:geometry:tail:vertical:AR"
        ] = db_vt_dAR_vt
        partials["data:geometry:tail:vertical:span", "data:geometry:tail:vertical:coefficient"] = (
            db_vt_dV_vt
        )
        partials["data:geometry:tail:vertical:span", "data:geometry:wing:surface"] = db_vt_dS_w
        partials["data:geometry:tail:vertical:span", "data:geometry:wing:span"] = db_vt_db_w
        partials["data:geometry:tail:vertical:span", "data:geometry:tail:horizontal:arm"] = (
            db_vt_dl_ht
        )

        # --- Output 4: c_root_vt ---
        partials[
            "data:geometry:tail:vertical:root:chord",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = dc_root_dAR_vt
        partials[
            "data:geometry:tail:vertical:root:chord", "data:geometry:tail:vertical:coefficient"
        ] = dc_root_dV_vt
        partials["data:geometry:tail:vertical:root:chord", "data:geometry:wing:surface"] = (
            dc_root_dS_w
        )
        partials["data:geometry:tail:vertical:root:chord", "data:geometry:wing:span"] = dc_root_db_w
        partials["data:geometry:tail:vertical:root:chord", "data:geometry:tail:horizontal:arm"] = (
            dc_root_dl_ht
        )
        partials["data:geometry:tail:vertical:root:chord", "data:geometry:tail:vertical:lambda"] = (
            dc_root_dlambda
        )

        # --- Output 5: c_tip_vt = lambda * c_root_vt ---
        partials[
            "data:geometry:tail:vertical:tip:chord",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = lmbda_vt * dc_root_dAR_vt
        partials[
            "data:geometry:tail:vertical:tip:chord", "data:geometry:tail:vertical:coefficient"
        ] = lmbda_vt * dc_root_dV_vt
        partials["data:geometry:tail:vertical:tip:chord", "data:geometry:wing:surface"] = (
            lmbda_vt * dc_root_dS_w
        )
        partials["data:geometry:tail:vertical:tip:chord", "data:geometry:wing:span"] = (
            lmbda_vt * dc_root_db_w
        )
        partials["data:geometry:tail:vertical:tip:chord", "data:geometry:tail:horizontal:arm"] = (
            lmbda_vt * dc_root_dl_ht
        )
        partials["data:geometry:tail:vertical:tip:chord", "data:geometry:tail:vertical:lambda"] = (
            c_root_vt + lmbda_vt * dc_root_dlambda
        )

        # --- Output 6: c_MAC_vt ---
        partials[
            "data:geometry:tail:vertical:MAC:length",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = dc_MAC_dAR_vt
        partials[
            "data:geometry:tail:vertical:MAC:length", "data:geometry:tail:vertical:coefficient"
        ] = dc_MAC_dV_vt
        partials["data:geometry:tail:vertical:MAC:length", "data:geometry:wing:surface"] = (
            dc_MAC_dS_w
        )
        partials["data:geometry:tail:vertical:MAC:length", "data:geometry:wing:span"] = dc_MAC_db_w
        partials["data:geometry:tail:vertical:MAC:length", "data:geometry:tail:horizontal:arm"] = (
            dc_MAC_dl_ht
        )
        partials["data:geometry:tail:vertical:MAC:length", "data:geometry:tail:vertical:lambda"] = (
            dc_MAC_dlambda
        )

        # --- Output 7: t_root_vt = c_root_vt * tc_ratio ---
        partials[
            "data:geometry:tail:vertical:root:thickness",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = tc_ratio * dc_root_dAR_vt
        partials[
            "data:geometry:tail:vertical:root:thickness", "data:geometry:tail:vertical:coefficient"
        ] = tc_ratio * dc_root_dV_vt
        partials["data:geometry:tail:vertical:root:thickness", "data:geometry:wing:surface"] = (
            tc_ratio * dc_root_dS_w
        )
        partials["data:geometry:tail:vertical:root:thickness", "data:geometry:wing:span"] = (
            tc_ratio * dc_root_db_w
        )
        partials[
            "data:geometry:tail:vertical:root:thickness", "data:geometry:tail:horizontal:arm"
        ] = tc_ratio * dc_root_dl_ht
        partials[
            "data:geometry:tail:vertical:root:thickness", "data:geometry:tail:vertical:lambda"
        ] = tc_ratio * dc_root_dlambda
        partials["data:geometry:tail:vertical:root:thickness", "data:geometry:wing:tc"] = c_root_vt

        # --- Output 8: t_tip_vt = c_tip_vt * tc_ratio ---
        partials[
            "data:geometry:tail:vertical:tip:thickness",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = tc_ratio * lmbda_vt * dc_root_dAR_vt
        partials[
            "data:geometry:tail:vertical:tip:thickness", "data:geometry:tail:vertical:coefficient"
        ] = tc_ratio * lmbda_vt * dc_root_dV_vt
        partials["data:geometry:tail:vertical:tip:thickness", "data:geometry:wing:surface"] = (
            tc_ratio * lmbda_vt * dc_root_dS_w
        )
        partials["data:geometry:tail:vertical:tip:thickness", "data:geometry:wing:span"] = (
            tc_ratio * lmbda_vt * dc_root_db_w
        )
        partials[
            "data:geometry:tail:vertical:tip:thickness", "data:geometry:tail:horizontal:arm"
        ] = tc_ratio * lmbda_vt * dc_root_dl_ht
        partials[
            "data:geometry:tail:vertical:tip:thickness", "data:geometry:tail:vertical:lambda"
        ] = tc_ratio * (c_root_vt + lmbda_vt * dc_root_dlambda)
        partials["data:geometry:tail:vertical:tip:thickness", "data:geometry:wing:tc"] = c_tip_vt

        # --- Output 9: z_MAC_vt = (b_vt/6) * (1+2*lambda) / (1+lambda) ---
        dz_MAC_db_vt = (1 + 2 * lmbda_vt) / (6 * (1 + lmbda_vt))
        dz_MAC_dlambda = (
            (b_vt / 6) * (2 * (1 + lmbda_vt) - (1 + 2 * lmbda_vt)) / (1 + lmbda_vt) ** 2
        )
        partials[
            "data:geometry:tail:vertical:MAC:z", "optimization:variables:geometry:tail:vertical:AR"
        ] = dz_MAC_db_vt * db_vt_dAR_vt
        partials["data:geometry:tail:vertical:MAC:z", "data:geometry:tail:vertical:coefficient"] = (
            dz_MAC_db_vt * db_vt_dV_vt
        )
        partials["data:geometry:tail:vertical:MAC:z", "data:geometry:wing:surface"] = (
            dz_MAC_db_vt * db_vt_dS_w
        )
        partials["data:geometry:tail:vertical:MAC:z", "data:geometry:wing:span"] = (
            dz_MAC_db_vt * db_vt_db_w
        )
        partials["data:geometry:tail:vertical:MAC:z", "data:geometry:tail:horizontal:arm"] = (
            dz_MAC_db_vt * db_vt_dl_ht
        )
        partials["data:geometry:tail:vertical:MAC:z", "data:geometry:tail:vertical:lambda"] = (
            dz_MAC_dlambda
        )

        # --- Output 10: x_MAC_c4_vt = x_MAC_c4_w + l_vt ---
        partials["data:geometry:tail:vertical:MAC:C4:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials["data:geometry:tail:vertical:MAC:C4:x", "data:geometry:tail:horizontal:arm"] = (
            dl_vt_dl_ht
        )

        # --- Output 11: x_MAC_LE_vt = x_MAC_c4_vt - 0.25*c_MAC_vt ---
        partials["data:geometry:tail:vertical:MAC:LE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:vertical:MAC:LE:x",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = -0.25 * dc_MAC_dAR_vt
        partials[
            "data:geometry:tail:vertical:MAC:LE:x", "data:geometry:tail:vertical:coefficient"
        ] = -0.25 * dc_MAC_dV_vt
        partials["data:geometry:tail:vertical:MAC:LE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w
        )
        partials["data:geometry:tail:vertical:MAC:LE:x", "data:geometry:wing:span"] = (
            -0.25 * dc_MAC_db_w
        )
        partials["data:geometry:tail:vertical:MAC:LE:x", "data:geometry:tail:horizontal:arm"] = (
            dl_vt_dl_ht - 0.25 * dc_MAC_dl_ht
        )
        partials["data:geometry:tail:vertical:MAC:LE:x", "data:geometry:tail:vertical:lambda"] = (
            -0.25 * dc_MAC_dlambda
        )

        # --- Output 12: x_root_LE_vt = x_MAC_LE_vt ---
        partials["data:geometry:tail:vertical:root:LE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:vertical:root:LE:x",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = -0.25 * dc_MAC_dAR_vt
        partials[
            "data:geometry:tail:vertical:root:LE:x", "data:geometry:tail:vertical:coefficient"
        ] = -0.25 * dc_MAC_dV_vt
        partials["data:geometry:tail:vertical:root:LE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w
        )
        partials["data:geometry:tail:vertical:root:LE:x", "data:geometry:wing:span"] = (
            -0.25 * dc_MAC_db_w
        )
        partials["data:geometry:tail:vertical:root:LE:x", "data:geometry:tail:horizontal:arm"] = (
            dl_vt_dl_ht - 0.25 * dc_MAC_dl_ht
        )
        partials["data:geometry:tail:vertical:root:LE:x", "data:geometry:tail:vertical:lambda"] = (
            -0.25 * dc_MAC_dlambda
        )

        # --- Output 13: x_root_TE_vt = x_root_LE_vt + c_root_vt ---
        partials["data:geometry:tail:vertical:root:TE:x", "data:geometry:wing:MAC:C4:x"] = 1.0
        partials[
            "data:geometry:tail:vertical:root:TE:x",
            "optimization:variables:geometry:tail:vertical:AR",
        ] = -0.25 * dc_MAC_dAR_vt + dc_root_dAR_vt
        partials[
            "data:geometry:tail:vertical:root:TE:x", "data:geometry:tail:vertical:coefficient"
        ] = -0.25 * dc_MAC_dV_vt + dc_root_dV_vt
        partials["data:geometry:tail:vertical:root:TE:x", "data:geometry:wing:surface"] = (
            -0.25 * dc_MAC_dS_w + dc_root_dS_w
        )
        partials["data:geometry:tail:vertical:root:TE:x", "data:geometry:wing:span"] = (
            -0.25 * dc_MAC_db_w + dc_root_db_w
        )
        partials["data:geometry:tail:vertical:root:TE:x", "data:geometry:tail:horizontal:arm"] = (
            dl_vt_dl_ht - 0.25 * dc_MAC_dl_ht + dc_root_dl_ht
        )
        partials["data:geometry:tail:vertical:root:TE:x", "data:geometry:tail:vertical:lambda"] = (
            -0.25 * dc_MAC_dlambda + dc_root_dlambda
        )


class FuselageGeometry(om.ExplicitComponent):
    """
    Computes Fuselage geometry
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:diameter:k", val=0.2, units=None)
        self.add_input("data:geometry:fuselage:fineness", val=5.0, units=None)
        self.add_input("data:geometry:wing:root:TE:x", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:root:TE:x", val=np.nan, units="m")
        self.add_output("data:geometry:fuselage:length", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:length:nose", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:length:mid", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:length:rear", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:diameter:mid", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:diameter:tip", units="m", lower=0.0)
        self.add_output("data:geometry:fuselage:surface", units="m**2", lower=0.0)
        self.add_output("data:geometry:fuselage:surface:nose", units="m**2", lower=0.0)
        self.add_output("data:geometry:fuselage:surface:mid", units="m**2", lower=0.0)
        self.add_output("data:geometry:fuselage:surface:rear", units="m**2", lower=0.0)
        self.add_output("data:geometry:fuselage:volume:nose", units="m**3", lower=0.0)
        self.add_output("data:geometry:fuselage:volume:mid", units="m**3", lower=0.0)
        self.add_output("data:geometry:fuselage:volume:rear", units="m**3", lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        lmbda_f = inputs["data:geometry:fuselage:fineness"]  # fuselage fineness ratio [-]
        k_df = inputs["data:geometry:fuselage:diameter:k"]  # tail cone diameters ratio (<1) [-]
        x_root_TE_w = inputs["data:geometry:wing:root:TE:x"]
        x_root_TE_ht = inputs["data:geometry:tail:horizontal:root:TE:x"]

        l_fus = x_root_TE_ht  # fuselage length [m]
        d_fus_mid = l_fus / lmbda_f  # max. fuselage diameter (mid) [m]
        d_fus_tip = k_df * d_fus_mid  # min. fuselage diameter (tail tip) [m]

        l_nose = d_fus_mid / 2  # nose length (half spherical) [m]
        l_rear = (
            x_root_TE_ht - x_root_TE_w
        )  # [m] rear fuselage length: from wing trailing edge to HT trailing edge
        l_mid = l_fus - l_rear - l_nose  # [m] mid fuselage length (cylindrical part)

        S_rear = (
            np.pi * (d_fus_mid + d_fus_tip) / 2 * l_rear + np.pi * (d_fus_tip / 2) ** 2
        )  # rear part of fuselage (conical) [m2]
        S_mid = np.pi * d_fus_mid * l_mid  # mid part of fuselage (cylindrical) [m2]
        S_nose = 2 * np.pi * (d_fus_mid / 2) ** 2  # nose part of fuselage (half spherical) [m2]
        S_fus = S_rear + S_mid + S_nose  # total fuselage area [m2]

        V_nose = (
            np.pi * (4 / 6) * l_nose * (0.5 * d_fus_mid) ** 2
        )  # nose part of fuselage (half ellipsoid) [m3]
        V_mid = np.pi * (d_fus_mid / 2) ** 2 * l_mid  # mid part of fuselage (cylindrical) [m3]
        (
            np.pi
            * l_rear
            * ((0.5 * d_fus_mid) ** 2 + (0.5 * d_fus_tip) ** 2 + 0.25 * d_fus_mid * d_fus_tip)
            / 3
        )  # rear part of fuselage (truncated cone) [m3]

        outputs["data:geometry:fuselage:length"] = l_fus
        outputs["data:geometry:fuselage:length:nose"] = l_nose
        outputs["data:geometry:fuselage:length:mid"] = l_mid
        outputs["data:geometry:fuselage:length:rear"] = l_rear
        outputs["data:geometry:fuselage:diameter:mid"] = d_fus_mid
        outputs["data:geometry:fuselage:diameter:tip"] = d_fus_tip
        outputs["data:geometry:fuselage:surface"] = S_fus
        outputs["data:geometry:fuselage:surface:nose"] = S_nose
        outputs["data:geometry:fuselage:surface:mid"] = S_mid
        outputs["data:geometry:fuselage:surface:rear"] = S_rear
        outputs["data:geometry:fuselage:volume:nose"] = V_nose
        outputs["data:geometry:fuselage:volume:mid"] = V_mid

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        lmbda_f = inputs["data:geometry:fuselage:fineness"]
        k_df = inputs["data:geometry:fuselage:diameter:k"]
        x_root_TE_w = inputs["data:geometry:wing:root:TE:x"]
        x_root_TE_ht = inputs["data:geometry:tail:horizontal:root:TE:x"]

        # Intermediate values
        l_fus = x_root_TE_ht
        d_fus_mid = l_fus / lmbda_f
        d_fus_tip = k_df * d_fus_mid
        l_nose = d_fus_mid / 2
        l_rear = x_root_TE_ht - x_root_TE_w
        l_mid = l_fus - l_rear - l_nose

        # --- Key intermediate partials ---
        # l_fus = x_root_TE_ht
        dl_fus_dx_ht = 1.0

        # d_fus_mid = l_fus / lambda_f
        dd_mid_dl_fus = 1.0 / lmbda_f
        dd_mid_dlambda = -l_fus / (lmbda_f**2)
        dd_mid_dx_ht = dd_mid_dl_fus * dl_fus_dx_ht

        # d_fus_tip = k_df * d_fus_mid
        dd_tip_dk_df = d_fus_mid
        dd_tip_dd_mid = k_df
        dd_tip_dx_ht = dd_tip_dd_mid * dd_mid_dx_ht
        dd_tip_dlambda = dd_tip_dd_mid * dd_mid_dlambda

        # l_nose = d_fus_mid / 2
        dl_nose_dd_mid = 0.5
        dl_nose_dx_ht = dl_nose_dd_mid * dd_mid_dx_ht
        dl_nose_dlambda = dl_nose_dd_mid * dd_mid_dlambda

        # l_rear = x_root_TE_ht - x_root_TE_w
        dl_rear_dx_ht = 1.0
        dl_rear_dx_w = -1.0

        # l_mid = l_fus - l_rear - l_nose
        dl_mid_dx_ht = dl_fus_dx_ht - dl_rear_dx_ht - dl_nose_dx_ht
        dl_mid_dx_w = -dl_rear_dx_w
        dl_mid_dlambda = -dl_nose_dlambda

        # S_rear = pi*(d_mid + d_tip)/2 * l_rear + pi*(d_tip/2)^2
        dS_rear_dd_mid = np.pi / 2 * l_rear
        dS_rear_dd_tip = np.pi / 2 * l_rear + 2 * np.pi * (d_fus_tip / 2) * (1 / 2)
        dS_rear_dl_rear = np.pi * (d_fus_mid + d_fus_tip) / 2
        dS_rear_dx_ht = (
            dS_rear_dd_mid * dd_mid_dx_ht
            + dS_rear_dd_tip * dd_tip_dx_ht
            + dS_rear_dl_rear * dl_rear_dx_ht
        )
        dS_rear_dx_w = dS_rear_dl_rear * dl_rear_dx_w
        dS_rear_dlambda = dS_rear_dd_mid * dd_mid_dlambda + dS_rear_dd_tip * dd_tip_dlambda
        dS_rear_dk_df = dS_rear_dd_tip * dd_tip_dk_df

        # S_mid = pi * d_mid * l_mid
        dS_mid_dd_mid = np.pi * l_mid
        dS_mid_dl_mid = np.pi * d_fus_mid
        dS_mid_dx_ht = dS_mid_dd_mid * dd_mid_dx_ht + dS_mid_dl_mid * dl_mid_dx_ht
        dS_mid_dx_w = dS_mid_dl_mid * dl_mid_dx_w
        dS_mid_dlambda = dS_mid_dd_mid * dd_mid_dlambda + dS_mid_dl_mid * dl_mid_dlambda

        # S_nose = 2*pi * (d_mid/2)^2 = pi/2 * d_mid^2
        dS_nose_dd_mid = np.pi * d_fus_mid
        dS_nose_dx_ht = dS_nose_dd_mid * dd_mid_dx_ht
        dS_nose_dlambda = dS_nose_dd_mid * dd_mid_dlambda

        # S_fus = S_rear + S_mid + S_nose
        dS_fus_dx_ht = dS_rear_dx_ht + dS_mid_dx_ht + dS_nose_dx_ht
        dS_fus_dx_w = dS_rear_dx_w + dS_mid_dx_w
        dS_fus_dlambda = dS_rear_dlambda + dS_mid_dlambda + dS_nose_dlambda
        dS_fus_dk_df = dS_rear_dk_df

        # V_mid = pi/4 * d_mid^2 * l_mid
        dV_mid_dd_mid = np.pi / 2 * d_fus_mid * l_mid
        dV_mid_dl_mid = np.pi / 4 * (d_fus_mid**2)
        dV_mid_dx_ht = dV_mid_dd_mid * dd_mid_dx_ht + dV_mid_dl_mid * dl_mid_dx_ht
        dV_mid_dx_w = dV_mid_dl_mid * dl_mid_dx_w
        dV_mid_dlambda = dV_mid_dd_mid * dd_mid_dlambda + dV_mid_dl_mid * dl_mid_dlambda

        # V_nose = pi * (4/6) * l_nose * (0.5*d_mid)^2
        dV_nose_dl_nose = np.pi * (4 / 6) * (0.5 * d_fus_mid) ** 2
        dV_nose_dd_mid = np.pi * (4 / 6) * l_nose * 2 * (0.5 * d_fus_mid) * 0.5
        dV_nose_dx_ht = dV_nose_dl_nose * dl_nose_dx_ht + dV_nose_dd_mid * dd_mid_dx_ht
        dV_nose_dlambda = dV_nose_dl_nose * dl_nose_dlambda + dV_nose_dd_mid * dd_mid_dlambda

        # --- Output 1: l_fus ---
        partials["data:geometry:fuselage:length", "data:geometry:tail:horizontal:root:TE:x"] = (
            dl_fus_dx_ht
        )

        # --- Output 2: l_nose ---
        partials[
            "data:geometry:fuselage:length:nose", "data:geometry:tail:horizontal:root:TE:x"
        ] = dl_nose_dx_ht
        partials["data:geometry:fuselage:length:nose", "data:geometry:fuselage:fineness"] = (
            dl_nose_dlambda
        )

        # --- Output 3: l_mid ---
        partials["data:geometry:fuselage:length:mid", "data:geometry:tail:horizontal:root:TE:x"] = (
            dl_mid_dx_ht
        )
        partials["data:geometry:fuselage:length:mid", "data:geometry:wing:root:TE:x"] = dl_mid_dx_w
        partials["data:geometry:fuselage:length:mid", "data:geometry:fuselage:fineness"] = (
            dl_mid_dlambda
        )

        # --- Output 4: l_rear ---
        partials[
            "data:geometry:fuselage:length:rear", "data:geometry:tail:horizontal:root:TE:x"
        ] = dl_rear_dx_ht
        partials["data:geometry:fuselage:length:rear", "data:geometry:wing:root:TE:x"] = (
            dl_rear_dx_w
        )

        # --- Output 5: d_fus_mid ---
        partials[
            "data:geometry:fuselage:diameter:mid", "data:geometry:tail:horizontal:root:TE:x"
        ] = dd_mid_dx_ht
        partials["data:geometry:fuselage:diameter:mid", "data:geometry:fuselage:fineness"] = (
            dd_mid_dlambda
        )

        # --- Output 6: d_fus_tip ---
        partials[
            "data:geometry:fuselage:diameter:tip", "data:geometry:tail:horizontal:root:TE:x"
        ] = dd_tip_dx_ht
        partials["data:geometry:fuselage:diameter:tip", "data:geometry:fuselage:fineness"] = (
            dd_tip_dlambda
        )
        partials["data:geometry:fuselage:diameter:tip", "data:geometry:fuselage:diameter:k"] = (
            dd_tip_dk_df
        )

        # --- Output 7: S_fus ---
        partials["data:geometry:fuselage:surface", "data:geometry:tail:horizontal:root:TE:x"] = (
            dS_fus_dx_ht
        )
        partials["data:geometry:fuselage:surface", "data:geometry:wing:root:TE:x"] = dS_fus_dx_w
        partials["data:geometry:fuselage:surface", "data:geometry:fuselage:fineness"] = (
            dS_fus_dlambda
        )
        partials["data:geometry:fuselage:surface", "data:geometry:fuselage:diameter:k"] = (
            dS_fus_dk_df
        )

        # --- Output 8: S_nose ---
        partials[
            "data:geometry:fuselage:surface:nose", "data:geometry:tail:horizontal:root:TE:x"
        ] = dS_nose_dx_ht
        partials["data:geometry:fuselage:surface:nose", "data:geometry:fuselage:fineness"] = (
            dS_nose_dlambda
        )

        # --- Output 9: S_mid ---
        partials[
            "data:geometry:fuselage:surface:mid", "data:geometry:tail:horizontal:root:TE:x"
        ] = dS_mid_dx_ht
        partials["data:geometry:fuselage:surface:mid", "data:geometry:wing:root:TE:x"] = dS_mid_dx_w
        partials["data:geometry:fuselage:surface:mid", "data:geometry:fuselage:fineness"] = (
            dS_mid_dlambda
        )

        # --- Output 10: S_rear ---
        partials[
            "data:geometry:fuselage:surface:rear", "data:geometry:tail:horizontal:root:TE:x"
        ] = dS_rear_dx_ht
        partials["data:geometry:fuselage:surface:rear", "data:geometry:wing:root:TE:x"] = (
            dS_rear_dx_w
        )
        partials["data:geometry:fuselage:surface:rear", "data:geometry:fuselage:fineness"] = (
            dS_rear_dlambda
        )
        partials["data:geometry:fuselage:surface:rear", "data:geometry:fuselage:diameter:k"] = (
            dS_rear_dk_df
        )

        # --- Output 11: V_mid ---
        partials["data:geometry:fuselage:volume:mid", "data:geometry:tail:horizontal:root:TE:x"] = (
            dV_mid_dx_ht
        )
        partials["data:geometry:fuselage:volume:mid", "data:geometry:wing:root:TE:x"] = dV_mid_dx_w
        partials["data:geometry:fuselage:volume:mid", "data:geometry:fuselage:fineness"] = (
            dV_mid_dlambda
        )

        # --- Output 12: V_nose ---
        partials[
            "data:geometry:fuselage:volume:nose", "data:geometry:tail:horizontal:root:TE:x"
        ] = dV_nose_dx_ht
        partials["data:geometry:fuselage:volume:nose", "data:geometry:fuselage:fineness"] = (
            dV_nose_dlambda
        )


class ProjectedAreasGuess(om.ExplicitComponent):
    """
    Computes a rough estimate of the projected area(s) of the UAV from the wing loading.
    """

    def setup(self):
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input(
            "optimization:variables:geometry:projected_area:top:k", val=np.nan, units=None
        )
        self.add_output("data:geometry:projected_area:top", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        WS = inputs["data:geometry:wing:loading"]
        mtow_guess = inputs["optimization:variables:weight:mtow:guess"]
        k_top = inputs["optimization:variables:geometry:projected_area:top:k"]

        S_top = k_top * mtow_guess * g / WS  # [m**2] top area guess

        outputs["data:geometry:projected_area:top"] = S_top


class ProjectedAreasConstraint(om.ExplicitComponent):
    """
    Projected area(s) consistency constraint.
    """

    def setup(self):
        self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:tail:horizontal:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface", val=np.nan, units="m**2")
        self.add_output("optimization:constraints:geometry:projected_area:top", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        S_top_guess = inputs[
            "data:geometry:projected_area:top"
        ]  # [m**2] projected area initial guess

        S_w = inputs["data:geometry:wing:surface"]
        S_ht = inputs["data:geometry:tail:horizontal:surface"]
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_fus_proj = S_fus / np.pi  # [m**2] projected area of the fuselage
        S_top = S_w + S_ht + S_fus_proj  # [m**2] projected area

        S_constraint = (S_top_guess - S_top) / S_top  # [-] projected area consistency constraint

        outputs["optimization:constraints:geometry:projected_area:top"] = S_constraint

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        S_top_guess = inputs[
            "data:geometry:projected_area:top"
        ]  # [m**2] projected area initial guess
        S_w = inputs["data:geometry:wing:surface"]
        S_ht = inputs["data:geometry:tail:horizontal:surface"]
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_fus_proj = S_fus / np.pi  # [m**2] projected area of the fuselage
        S_top = S_w + S_ht + S_fus_proj  # [m**2] projected area

        partials[
            "optimization:constraints:geometry:projected_area:top",
            "data:geometry:projected_area:top",
        ] = 1 / S_top
        partials[
            "optimization:constraints:geometry:projected_area:top", "data:geometry:wing:surface"
        ] = -S_top_guess / S_top**2
        partials[
            "optimization:constraints:geometry:projected_area:top",
            "data:geometry:tail:horizontal:surface",
        ] = -S_top_guess / S_top**2
        partials[
            "optimization:constraints:geometry:projected_area:top", "data:geometry:fuselage:surface"
        ] = -S_top_guess / np.pi / S_top**2


class FuselageVolumeConstraint(om.ExplicitComponent):
    """
    Fuselage volume constraint definition.
    The mid fuselage part has to house the payload and the batteries.
    Therefore, a constraint is set on the volume of the mid fuselage part.
    """

    def initialize(self):
        self.options.declare(
            "propulsion_id_list",
            default=[FW_PROPULSION],
            values=[[FW_PROPULSION], PROPULSION_ID_LIST],
        )

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]
        for propulsion_id in propulsion_id_list:
            self.add_input(
                "data:propulsion:%s:battery:volume" % propulsion_id, val=np.nan, units="m**3"
            )
        self.add_input("data:geometry:fuselage:volume:mid", val=np.nan, units="m**3")
        self.add_input("mission:sizing:payload:volume", val=np.nan, units="m**3")
        self.add_output("optimization:constraints:geometry:fuselage:volume", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]
        V_bat = sum(
            inputs["data:propulsion:%s:battery:volume" % propulsion_id]
            for propulsion_id in propulsion_id_list
        )
        V_fus = inputs[
            "data:geometry:fuselage:volume:mid"
        ]  # only the mid-fuselage part is considered
        V_pay = inputs["mission:sizing:payload:volume"]
        V_req = V_pay + V_bat

        V_cnstr = (V_fus - V_req) / V_req  # mid fuselage volume constraint

        outputs["optimization:constraints:geometry:fuselage:volume"] = V_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id_list = self.options["propulsion_id_list"]
        V_bat = sum(
            inputs["data:propulsion:%s:battery:volume" % propulsion_id]
            for propulsion_id in propulsion_id_list
        )
        V_fus = inputs["data:geometry:fuselage:volume:mid"]
        V_pay = inputs["mission:sizing:payload:volume"]
        V_req = V_pay + V_bat

        partials[
            "optimization:constraints:geometry:fuselage:volume", "data:geometry:fuselage:volume:mid"
        ] = 1 / V_req
        partials[
            "optimization:constraints:geometry:fuselage:volume", "mission:sizing:payload:volume"
        ] = -V_fus / V_req**2

        for propulsion_id in propulsion_id_list:
            partials[
                "optimization:constraints:geometry:fuselage:volume",
                "data:propulsion:%s:battery:volume" % propulsion_id,
            ] = -V_fus / V_req**2
