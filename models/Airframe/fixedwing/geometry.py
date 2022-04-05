"""
Fixed Wing Airframe Geometry
"""
import openmdao.api as om
import numpy as np
from scipy.constants import g


class GeometryFW(om.Group):
    """
    Group containing the airframe geometries calculation
    """

    def setup(self):
        self.add_subsystem("wing", WingGeometry(), promotes=["*"])
        self.add_subsystem("horizontal_tail", HorizontalTailGeometry(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailGeometry(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageGeometry(), promotes=["*"])
        self.add_subsystem("constraints", GeometryConstraints(), promotes=["*"])


class WingGeometry(om.ExplicitComponent):
    """
    Computes Wing geometry
    """

    def setup(self):
        self.add_input("data:airframe:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("data:airframe:wing:AR", val=np.nan, units=None)
        self.add_input("data:airframe:wing:lambda", val=np.nan, units=None)
        self.add_input("data:airframe:wing:MAC:LE:x:k", val=0.40, units=None)
        self.add_input("data:airframe:wing:tc", val=0.15, units=None)
        self.add_input("data:system:MTOW:guess", val=np.nan, units="kg")
        self.add_output("data:airframe:wing:surface", units="m**2", lower=0.0)
        self.add_output("data:airframe:wing:span", units="m", lower=0.0)
        self.add_output("data:airframe:wing:root:chord", units="m", lower=0.0)
        self.add_output("data:airframe:wing:tip:chord", units="m", lower=0.0)
        self.add_output("data:airframe:wing:MAC:length", units="m", lower=0.0)
        self.add_output("data:airframe:wing:root:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:wing:tip:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:wing:MAC:y", units="m", lower=0.0)
        self.add_output("data:airframe:wing:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:wing:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:airframe:wing:root:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:wing:root:TE:x", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        WS = inputs["data:airframe:wing:loading"]
        Mtotal_guess = inputs["data:system:MTOW:guess"]
        tc_ratio = inputs["data:airframe:wing:tc"]

        # design variables
        AR_w = inputs["data:airframe:wing:AR"]
        lmbda_w = inputs["data:airframe:wing:lambda"]
        k_xw = inputs['data:airframe:wing:MAC:LE:x:k']

        # Wing sizing
        S_w = Mtotal_guess * g / WS  # wing surface [m2]
        b_w = np.sqrt(AR_w * S_w)  # wing span [m]
        c_root = 2 * S_w / b_w / (1 + lmbda_w)  # chord at root [m]
        c_tip = lmbda_w * c_root  # chord at tip [m]
        c_MAC = (2 / 3) * c_root * (1 + lmbda_w + lmbda_w ** 2) / (1 + lmbda_w)  # MAC = MGC [m]
        t_root = c_root * tc_ratio  # wing thickness at root [m]
        t_tip = c_tip * tc_ratio  # wing thickness at tip [m]

        # Wing location
        y_MAC = (b_w / 6) * (1 + 2 * lmbda_w) / (1 + lmbda_w)  # y-location of MAC (from the root) [m]
        x_MAC_LE_loc = 0  # x-location of MAC leading edge (from the leading edge of the root) [m] # TODO: add sweep
        x_MAC_LE = k_xw * b_w  # x-location of MAC leading edge (from nose tip) [m]
        x_MAC_c4 = x_MAC_LE + 0.25 * c_MAC  # x-location of MAC quarter chord (from nose tip) [m]
        x_root_LE = x_MAC_LE - x_MAC_LE_loc  # x-location of root leading edge (from nose tip) [m]
        x_root_TE = x_root_LE + c_root  # x-location of root trailing edge (from nose tip) [m]

        outputs["data:airframe:wing:surface"] = S_w
        outputs["data:airframe:wing:span"] = b_w
        outputs["data:airframe:wing:root:chord"] = c_root
        outputs["data:airframe:wing:tip:chord"] = c_tip
        outputs["data:airframe:wing:MAC:length"] = c_MAC
        outputs["data:airframe:wing:root:thickness"] = t_root
        outputs["data:airframe:wing:tip:thickness"] = t_tip
        outputs["data:airframe:wing:MAC:y"] = y_MAC
        outputs["data:airframe:wing:MAC:LE:x"] = x_MAC_LE
        outputs["data:airframe:wing:MAC:C4:x"] = x_MAC_c4
        outputs["data:airframe:wing:root:LE:x"] = x_root_LE
        outputs["data:airframe:wing:root:TE:x"] = x_root_TE


class HorizontalTailGeometry(om.ExplicitComponent):
    """
    Computes Horizontal Tail geometry
    """

    def setup(self):
        self.add_input("data:airframe:tail:horizontal:AR", val=4.0, units=None)
        self.add_input("data:airframe:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:airframe:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:airframe:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_input("data:airframe:wing:tc", val=0.15, units=None)
        self.add_input("data:airframe:wing:span", val=np.nan, units="m")
        self.add_input("data:airframe:tail:horizontal:coefficient", val=0.5, units=None)
        self.add_input("data:airframe:tail:horizontal:lambda", val=0.9, units=None)
        self.add_input("data:airframe:tail:horizontal:arm:k", val=0.75, units=None)
        self.add_output("data:airframe:tail:horizontal:surface", units="m**2", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:arm", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:span", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:root:chord", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:tip:chord", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:MAC:length", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:root:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:tip:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:MAC:y", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:root:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:horizontal:root:TE:x", units="m", lower=0.0)

        # self.add_input("data:airframe:wing:AR", val=np.nan, units=None)
        # self.add_output("data:airframe:tail:horizontal:AR", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        AR_ht = inputs["data:airframe:tail:horizontal:AR"]
        S_w = inputs["data:airframe:wing:surface"]
        b_w = inputs["data:airframe:wing:span"]
        c_MAC_w = inputs["data:airframe:wing:MAC:length"]
        x_MAC_c4_w = inputs["data:airframe:wing:MAC:C4:x"]
        V_ht = inputs["data:airframe:tail:horizontal:coefficient"]
        lmbda_ht = inputs["data:airframe:tail:horizontal:lambda"]
        tc_ratio = inputs["data:airframe:wing:tc"]  # assume same profile as wing

        # AR_w = inputs["data:airframe:wing:AR"]
        # AR_ht = (2 / 3) * AR_w  # horizontal tail AR [-]

        # design variable
        k_ht = inputs["data:airframe:tail:horizontal:arm:k"]

        # Tail sizing
        l_ht = k_ht * b_w  # horizontal tail arm: distance between wing MAC quarter chord and HT MAC quarter chord [m]
        S_ht = V_ht * S_w * c_MAC_w / l_ht  # horizontal tail surface [m2]
        b_ht = np.sqrt(AR_ht * S_ht)  # HT span [m]
        c_root_ht = 2 * S_ht / b_ht / (1 + lmbda_ht)  # chord at root [m]
        c_tip_ht = lmbda_ht * c_root_ht  # chord at tip [m]
        c_MAC_ht = (2 / 3) * c_root_ht * (1 + lmbda_ht + lmbda_ht ** 2) / (1 + lmbda_ht)  # MAC = MGC [m]
        t_root_ht = c_root_ht * tc_ratio  # wing thickness at root [m]
        t_tip_ht = c_tip_ht * tc_ratio  # wing thickness at tip [m]

        # Tail location
        y_MAC_ht = (b_ht / 6) * (1 + 2 * lmbda_ht) / (1 + lmbda_ht)  # y-location of MAC (from the root) [m]
        x_MAC_LE_loc_ht = 0  # x-location of MAC leading edge (from the leading edge of the root) [m] TODO: add sweep
        x_MAC_c4_ht = x_MAC_c4_w + l_ht  # x-location of MAC quarter chord (from nose tip) [m]
        x_MAC_LE_ht = x_MAC_c4_ht - 0.25 * c_MAC_ht  # x-location of MAC leading edge (from nose tip) [m]
        x_root_LE_ht = x_MAC_LE_ht - x_MAC_LE_loc_ht  # x-location of root leading edge (from nose tip) [m]
        x_root_TE_ht = x_root_LE_ht + c_root_ht  # x-location of root trailing edge (from nose tip) [m]

        outputs["data:airframe:tail:horizontal:arm"] = l_ht
        outputs["data:airframe:tail:horizontal:surface"] = S_ht
        outputs["data:airframe:tail:horizontal:span"] = b_ht
        outputs["data:airframe:tail:horizontal:root:chord"] = c_root_ht
        outputs["data:airframe:tail:horizontal:tip:chord"] = c_tip_ht
        outputs["data:airframe:tail:horizontal:MAC:length"] = c_MAC_ht
        outputs["data:airframe:tail:horizontal:root:thickness"] = t_root_ht
        outputs["data:airframe:tail:horizontal:tip:thickness"] = t_tip_ht
        outputs["data:airframe:tail:horizontal:MAC:y"] = y_MAC_ht
        outputs["data:airframe:tail:horizontal:MAC:LE:x"] = x_MAC_LE_ht
        outputs["data:airframe:tail:horizontal:MAC:C4:x"] = x_MAC_c4_ht
        outputs["data:airframe:tail:horizontal:root:LE:x"] = x_root_LE_ht
        outputs["data:airframe:tail:horizontal:root:TE:x"] = x_root_TE_ht
        # outputs["data:airframe:tail:horizontal:AR"] = AR_ht


class VerticalTailGeometry(om.ExplicitComponent):
    """
    Computes Vertical Tail geometry
    """

    def setup(self):
        self.add_input("data:airframe:tail:vertical:AR", val=1.5, units=None)
        self.add_input("data:airframe:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:airframe:wing:span", val=np.nan, units="m")
        self.add_input("data:airframe:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_input("data:airframe:wing:tc", val=0.15, units=None)
        self.add_input("data:airframe:tail:vertical:coefficient", val=0.04, units=None)
        self.add_input("data:airframe:tail:vertical:lambda", val=0.9, units=None)
        self.add_input("data:airframe:tail:horizontal:arm", val=np.nan, units="m")
        self.add_output("data:airframe:tail:vertical:surface", units="m**2", lower=0.0)
        self.add_output("data:airframe:tail:vertical:arm", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:span", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:root:chord", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:tip:chord", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:MAC:length", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:root:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:tip:thickness", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:MAC:z", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:MAC:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:MAC:C4:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:root:LE:x", units="m", lower=0.0)
        self.add_output("data:airframe:tail:vertical:root:TE:x", units="m", lower=0.0)

        # self.add_input("data:airframe:tail:horizontal:AR", val=np.nan, units=None)
        # self.add_output("data:airframe:tail:vertical:AR", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        AR_vt = inputs["data:airframe:tail:vertical:AR"]
        S_w = inputs["data:airframe:wing:surface"]
        b_w = inputs["data:airframe:wing:span"]
        x_MAC_c4_w = inputs["data:airframe:wing:MAC:C4:x"]
        V_vt = inputs["data:airframe:tail:vertical:coefficient"]
        lmbda_vt = inputs["data:airframe:tail:vertical:lambda"]
        tc_ratio = inputs["data:airframe:wing:tc"]  # assume same profile as wing
        l_ht = inputs["data:airframe:tail:horizontal:arm"]

        # AR_ht = inputs["data:airframe:tail:horizontal:AR"]
        # AR_vt = (1 / 2) * AR_ht  # vertical tail AR [-]

        # Tail sizing
        l_vt = l_ht  # vertical and horizontal tails arms [m]
        S_vt = V_vt * S_w * b_w / l_vt  # vertical tail surface [m2]
        b_vt = np.sqrt(AR_vt * S_vt)  # VT span [m]
        c_root_vt = 2 * S_vt / b_vt / (1 + lmbda_vt)  # chord at root [m]
        c_tip_vt = lmbda_vt * c_root_vt  # chord at tip [m]
        c_MAC_vt = (2 / 3) * c_root_vt * (1 + lmbda_vt + lmbda_vt ** 2) / (1 + lmbda_vt)  # MAC = MGC [m]
        t_root_vt = c_root_vt * tc_ratio  # wing thickness at root [m]
        t_tip_vt = c_tip_vt * tc_ratio  # wing thickness at tip [m]

        # Tail location
        z_MAC_vt = (b_vt / 6) * (1 + 2 * lmbda_vt) / (1 + lmbda_vt)  # z-location of MAC (from the root) [m]
        x_MAC_LE_loc_vt = 0  # x-location of MAC leading edge (from the leading edge of the root) [m] TODO: add sweep
        x_MAC_c4_vt = x_MAC_c4_w + l_vt  # x-location of MAC quarter chord (from nose tip) [m]
        x_MAC_LE_vt = x_MAC_c4_vt - 0.25 * c_MAC_vt  # x-location of MAC leading edge (from nose tip) [m]
        x_root_LE_vt = x_MAC_LE_vt - x_MAC_LE_loc_vt  # x-location of root leading edge (from nose tip) [m]
        x_root_TE_vt = x_root_LE_vt + c_root_vt  # x-location of root trailing edge (from nose tip) [m]

        outputs["data:airframe:tail:vertical:arm"] = l_vt
        outputs["data:airframe:tail:vertical:surface"] = S_vt
        outputs["data:airframe:tail:vertical:span"] = b_vt
        outputs["data:airframe:tail:vertical:root:chord"] = c_root_vt
        outputs["data:airframe:tail:vertical:tip:chord"] = c_tip_vt
        outputs["data:airframe:tail:vertical:MAC:length"] = c_MAC_vt
        outputs["data:airframe:tail:vertical:root:thickness"] = t_root_vt
        outputs["data:airframe:tail:vertical:tip:thickness"] = t_tip_vt
        outputs["data:airframe:tail:vertical:MAC:z"] = z_MAC_vt
        outputs["data:airframe:tail:vertical:MAC:LE:x"] = x_MAC_LE_vt
        outputs["data:airframe:tail:vertical:MAC:C4:x"] = x_MAC_c4_vt
        outputs["data:airframe:tail:vertical:root:LE:x"] = x_root_LE_vt
        outputs["data:airframe:tail:vertical:root:TE:x"] = x_root_TE_vt
        # outputs["data:airframe:tail:vertical:AR"] = AR_vt


class FuselageGeometry(om.ExplicitComponent):
    """
    Computes Fuselage geometry
    """

    def setup(self):
        self.add_input("data:airframe:fuselage:diameter:k", val=0.2, units=None)
        self.add_input("data:airframe:fuselage:fineness", val=5.0, units=None)
        self.add_input("data:airframe:wing:root:TE:x", val=np.nan, units="m")
        self.add_input("data:airframe:tail:horizontal:root:TE:x", val=np.nan, units="m")
        self.add_output("data:airframe:fuselage:length", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:length:nose", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:length:mid", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:length:rear", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:diameter:mid", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:diameter:tip", units="m", lower=0.0)
        self.add_output("data:airframe:fuselage:surface", units="m**2", lower=0.0)
        self.add_output("data:airframe:fuselage:surface:nose", units="m**2", lower=0.0)
        self.add_output("data:airframe:fuselage:surface:mid", units="m**2", lower=0.0)
        self.add_output("data:airframe:fuselage:surface:rear", units="m**2", lower=0.0)
        self.add_output("data:airframe:fuselage:volume:mid", units="m**3", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        lmbda_f = inputs["data:airframe:fuselage:fineness"]  # fuselage fineness ratio [-]
        k_df = inputs["data:airframe:fuselage:diameter:k"]  # tail cone diameters ratio (<1) [-]
        x_root_TE_w = inputs["data:airframe:wing:root:TE:x"]
        x_root_TE_ht = inputs["data:airframe:tail:horizontal:root:TE:x"]

        l_fus = x_root_TE_ht  # fuselage length [m]
        d_fus_mid = l_fus / lmbda_f  # max. fuselage diameter (mid) [m]
        d_fus_tip = k_df * d_fus_mid  # min. fuselage diameter (tail tip) [m]

        l_nose = d_fus_mid / 2  # nose length (half spherical) [m]
        l_rear = x_root_TE_ht - x_root_TE_w  # [m] rear fuselage length: from wing trailing edge to HT trailing edge
        l_mid = l_fus - l_rear - l_nose  # [m] mid fuselage length (cylindrical part)

        S_rear = np.pi * (d_fus_mid + d_fus_tip) / 2 * l_rear + np.pi * (
                    d_fus_tip / 2) ** 2  # rear part of fuselage (conical) [m2]
        S_mid = np.pi * d_fus_mid * l_mid  # mid part of fuselage (cylindrical) [m2]
        S_nose = 2 * np.pi * (d_fus_mid / 2) ** 2  # nose part of fuselage (half spherical) [m2]
        S_fus = S_rear + S_mid + S_nose  # total fuselage area [m2]

        V_mid = S_mid * l_mid  # mid part of fuselage (cylindrical) [m3]

        outputs["data:airframe:fuselage:length"] = l_fus
        outputs["data:airframe:fuselage:length:nose"] = l_nose
        outputs["data:airframe:fuselage:length:mid"] = l_mid
        outputs["data:airframe:fuselage:length:rear"] = l_rear
        outputs["data:airframe:fuselage:diameter:mid"] = d_fus_mid
        outputs["data:airframe:fuselage:diameter:tip"] = d_fus_tip
        outputs["data:airframe:fuselage:surface"] = S_fus
        outputs["data:airframe:fuselage:surface:nose"] = S_nose
        outputs["data:airframe:fuselage:surface:mid"] = S_mid
        outputs["data:airframe:fuselage:surface:rear"] = S_rear
        outputs["data:airframe:fuselage:volume:mid"] = V_mid


class GeometryConstraints(om.ExplicitComponent):
    """
    Geometry constraint definition.
    The mid fuselage part has to house the payload and the batteries.
    Therefore, a constraint is set on the volume of the mid fuselage part.
    """

    def setup(self):
        self.add_input("data:airframe:fuselage:volume:mid", val=np.nan, units="m**3")
        self.add_input("specifications:payload:volume", val=np.nan, units="m**3")
        self.add_input("data:battery:volume", val=np.nan, units="m**3")
        self.add_output("data:airframe:fuselage:constraints:volume", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        V_fus_mid = inputs["data:airframe:fuselage:volume:mid"]
        V_pay = inputs["specifications:payload:volume"]
        V_bat = inputs["data:battery:volume"]
        V_req = V_pay + V_bat

        V_cnstr = (V_fus_mid - V_req) / V_req  # mid fuselage volume constraint

        outputs["data:airframe:fuselage:constraints:volume"] = V_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_fus = inputs["data:airframe:fuselage:volume:mid"]
        V_pay = inputs["specifications:payload:volume"]
        V_bat = inputs["data:battery:volume"]
        V_req = V_pay + V_bat

        partials["data:airframe:fuselage:constraints:volume",
                 "data:airframe:fuselage:volume:mid"] = 1 / V_req
        partials["data:airframe:fuselage:constraints:volume",
                 "specifications:payload:volume"] = - V_fus / V_req ** 2
        partials["data:airframe:fuselage:constraints:volume",
                 "data:battery:volume"] = - V_fus / V_req ** 2
