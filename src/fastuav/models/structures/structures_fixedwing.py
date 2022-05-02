"""
Fixed Wing Structures
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from scipy.constants import g


@oad.RegisterOpenMDAOSystem("fastuav.structures.fixedwing")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem(
            "wing", WingWeight(spar_model=self.options["spar_model"]), promotes=["*"]
        )
        self.add_subsystem("horizontal_tail", HorizontalTailWeight(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailWeight(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageWeight(), promotes=["*"])
        # self.add_subsystem("constraints", Constraints(spar_model=self.options["spar_model"]), promotes=["*"])


class WingWeightModel:
    """
    Mass model for the wing.
    """

    @staticmethod
    def spar_I_detailed(F_max, y, L, k_a, sig_max, rho_spar):
        """
        Structural analysis for a cantilever I-shaped beam of length L with a load F_max applied at the position y.
        The model assumes that the bending moment is entirely reacted by the spar flanges.

        :param F_max: Maximum load applied to the spar [N]
        :param y: Position of the point of application of the load [m]
        :param L: Length of the spar [m]
        :param k_a: Depth ratio of the spar (k_a = flange depth / web depth) [-]
        :param sig_max: Maximum allowable stress [Pa]
        :param rho_spar: Material density of the spar [kg/m3]
        :return m_spar: Mass of the spar [kg]
        :return h_web: Depth of the web [kg]
        :return t_web: Thickness of the web [kg]
        :return a_flange: Depth of the flange [kg]
        :return b_flange: Thickness of the flange [kg]
        """
        k_web = 30  # depth-to-thickness ratio for the web : t_web = h_web / k_web
        k_flange = 0.1  # depth-to-thickness ratio for the flange : b_flange = a_flange / k_flange
        M_root = F_max * y  # maximum bending moment at root [N.m]
        h_web = (M_root * (1 + k_a) / (sig_max * k_a**2 * (1 + k_a**2 / 3) / k_flange)) ** (
            1 / 3
        )  # web depth [m]
        t_web = h_web / k_web  # web thickness [m]
        a_flange = k_a * h_web  # flange depth [m]
        b_flange = a_flange / k_flange  # flange thickness [m]
        m_spar = (
            rho_spar * L * (2 * a_flange * b_flange + (h_web - a_flange) * t_web)
        )  # mass of spar [kg]
        return m_spar, h_web, t_web, a_flange, b_flange

    @staticmethod
    def spar_I_simplified(F_max, y, L, h_web, sig_max, rho_spar):
        """
        Structural analysis for a cantilever I-shaped beam of length L with a load F_max applied at the position y.
        The model assumes that the bending moment is entirely reacted by the spar flanges.
        This model is an approximation valid for thin walled beams,
        i.e. the spar flanges are thin compared to the spar depth : A_flange << h^2

        :param F_max: Maximum load applied to the spar [N]
        :param y: Position of the point of application of the load [m]
        :param L: Length of the spar [m]
        :param h_web: Depth of the web [kg]
        :param sig_max: Maximum allowable stress [Pa]
        :param rho_spar: Material density of the spar [kg/m3]
        :return m_spar: Mass of the spar [kg]
        :return t_web: Thickness of the web [kg]
        :return a_flange: Depth of the flange [kg]
        :return b_flange: Thickness of the flange [kg]
        """
        k_web = 30  # depth-to-thickness ratio for the web : t_web = h_web / k_web
        k_flange = 0.1  # depth-to-thickness ratio for the flange : b_flange = a_flange / k_flange
        M_root = F_max * y  # maximum bending moment at root [N.m]
        A_flange = M_root / (h_web * sig_max)  # spar flange area [m2]
        t_web = h_web / k_web  # web thickness [m]
        a_flange = np.sqrt(k_flange * A_flange)  # flange depth [m]
        b_flange = a_flange / k_flange  # flange thickness [m]
        A_spar = 2 * A_flange + t_web * h_web  # sectional area of the spar [m2]
        m_spar = rho_spar * L * A_spar  # mass of spar [kg]
        return m_spar, t_web, a_flange, b_flange

    @staticmethod
    def spar_pipe_detailed(F_max, y, L, k_d, sig_max, rho_spar):
        """
        Structural analysis for a cantilever pipe of length L with a load F_max applied at the position y.

        :param F_max: Maximum load applied to the spar [N]
        :param y: Position of the point of application of the load [m]
        :param L: Length of the spar [m]
        :param k_d: Diameter ratio of the spar (k_d = d_in / d_out) [-]
        :param sig_max: Maximum allowable stress [Pa]
        :param rho_spar: Material density of the spar [kg/m3]
        :return m_spar: Mass of the spar [kg]
        :return d_in: Inner diameter of the spar [m]
        :return d_out: Outer diameter of the spar [m]
        """
        M_root = F_max * y  # maximum bending moment at root [N.m]
        d_out = ((32 * M_root) / (np.pi * (1 - k_d**4) * sig_max)) ** (
            1 / 3
        )  # outer diameter of the spar [m]
        d_in = k_d * d_out  # inner diameter of the spar [m]
        A_spar = np.pi / 4 * (d_out**2 - d_in**2)  # sectional area of the spar [m2]
        m_spar = rho_spar * L * A_spar  # mass of spar [kg]
        return m_spar, d_in, d_out

    @staticmethod
    def spar_pipe_simplified(F_max, y, L, d_RMS, sig_max, rho_spar):
        """
        Structural analysis for a cantilever pipe of length L with a load F_max applied at the position y.
        This model is an approximation valid for thin wall pipes,
        i.e. the thickness of the pipe is small compared to its diameter.

        :param F_max: Maximum load applied to the spar [N]
        :param y: Position of the point of application of the load [m]
        :param L: Length of the spar [m]
        :param d_RMS: RMS value of the spar's diameter [m]
        :param sig_max: Maximum allowable stress [Pa]
        :param rho_spar: Material density of the spar [kg/m3]
        :return m_spar: Mass of the spar [kg]
        :return d_in: Inner diameter of the spar [m]
        :return d_out: Outer diameter of the spar [m]
        """
        M_root = F_max * y  # maximum bending moment at root [N.m]
        A_spar = 4 * M_root / sig_max / d_RMS  # sectional area of the spar [m]
        d_out = np.sqrt(2 * A_spar / np.pi + d_RMS**2)
        d_in = np.sqrt(2 * d_RMS**2 - d_out**2)
        m_spar = rho_spar * L * np.pi / 4 * (d_out**2 - d_in**2)  # mass of spar [kg]
        return m_spar, d_in, d_out

    @staticmethod
    def ribs(L, c_MAC, c_root, c_tip, t_root, t_tip, t_rib, rho_rib):
        """
        Computes the weight of the ribs for a simple tapered planform of length L with chords c and thicknesses t.

        :param L: Length of the spar [m]
        :param c_MAC: mean aerodynamic chord of the wing [m]
        :param c_root: Chord at the wing's root [m]
        :param c_tip: Chord at the wing's tip [m]
        :param t_root: Thickness at the wing's root [m]
        :param t_tip: Thickness at the wing's tip [m]
        :param t_rib: Thickness of the ribs [m]
        :param rho_rib: Material density of the rib [kg/m3]
        :return m_ribs: Total mass of the ribs [kg]
        :return N_ribs: Number of ribs [-]
        """
        N_ribs = 2 * L / c_MAC  # np.ceil(2 * L / c_MAC)  # number of ribs for wing [-]
        S_rib_root = c_root * t_root  # surface of the rib at wing root (approximation) [m2]
        S_rib_tip = c_tip * t_tip  # surface of the rib at wing tip (approximation) [m2]
        m_ribs = N_ribs * rho_rib * t_rib * (S_rib_root + S_rib_tip) / 2  # mass of ribs [kg]
        return m_ribs, N_ribs

    @staticmethod
    def skin(S, rho_skin):
        """
        Computes the weight of the skin for a planform of surface area S.

        :param S: Surface area of the wing [m2]
        :param rho_skin: Area density of the skin [kg/m2]
        :return m_skin: Mass of the skin [kg]
        """
        S_skin = 2 * S  # surface of skin (two times the planform area) [m2]
        m_skin = S_skin * rho_skin  # mass of skin [kg]
        return m_skin


class WingWeight(om.ExplicitComponent):
    """
    Computes Wing mass
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_input("data:loads:vertical:factor", val=4.5, units=None)
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")
        self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:weights:airframe:wing:spar:density", val=np.nan, units="kg/m**3")
        self.add_input("data:structures:wing:ribs:thickness", val=np.nan, units="m")
        self.add_input("data:weights:airframe:wing:ribs:density", val=np.nan, units="kg/m**3")
        self.add_input("data:weights:airframe:wing:skin:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weights:airframe:wing:mass", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:wing:spar:mass", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:wing:skin:mass", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:wing:ribs:mass", units="kg", lower=0.0)
        self.add_output("data:structures:wing:ribs:number", units=None, lower=0.0)

        if self.options["spar_model"] == "pipe":
            self.add_input("data:structures:wing:spar:diameter:k", val=0.9, units=None)
            self.add_design_var(
                "data:structures:wing:spar:diameter:k", lower=0.01, upper=0.99, ref=0.9, units=None
            )
            self.add_output("data:structures:wing:spar:diameter:inner", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:diameter:outer", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:diameter:constraint", units=None)
            self.add_constraint("data:structures:wing:spar:diameter:constraint", lower=0.0)

        elif self.options["spar_model"] == "I_beam":
            self.add_input("data:structures:wing:spar:depth:k", val=0.1, units=None)
            self.add_design_var(
                "data:structures:wing:spar:depth:k", lower=0.01, upper=0.99, ref=0.1, units=None
            )
            self.add_output("data:structures:wing:spar:web:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:web:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:depth:constraint", units=None)
            self.add_constraint("data:structures:wing:spar:depth:constraint", lower=0.0, units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        n_ult = inputs["data:loads:vertical:factor"]
        Mtotal_guess = inputs["data:weights:mtow:guess"]
        S_w = inputs["data:geometry:wing:surface"]
        b_w = inputs["data:geometry:wing:span"]
        y_MAC = inputs["data:geometry:wing:MAC:y"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        c_root = inputs["data:geometry:wing:root:chord"]
        c_tip = inputs["data:geometry:wing:tip:chord"]
        t_root = inputs["data:geometry:wing:root:thickness"]
        t_tip = inputs["data:geometry:wing:tip:thickness"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]
        rho_spar = inputs["data:weights:airframe:wing:spar:density"]
        t_rib = inputs["data:structures:wing:ribs:thickness"]
        rho_rib = inputs["data:weights:airframe:wing:ribs:density"]
        rho_skin = inputs["data:weights:airframe:wing:skin:density"]

        F_max = (
            n_ult * Mtotal_guess * g / 2
        )  # ultimate aerodynamic load to be supported by (half) wing [N]
        m_ribs, N_ribs = WingWeightModel.ribs(
            b_w / 2, c_MAC, c_root, c_tip, t_root, t_tip, t_rib, rho_rib
        )  # ribs
        m_skin = WingWeightModel.skin(S_w / 2, rho_skin)  # skin

        m_spar = 0
        if self.options["spar_model"] == "pipe":
            k_d = inputs["data:structures:wing:spar:diameter:k"]
            m_spar, d_in, d_out = WingWeightModel.spar_pipe_detailed(
                F_max, y_MAC, b_w / 2, k_d, sig_max, rho_spar
            )
            # d_RMS = k_d * t_tip
            # m_spar, d_in, d_out = WingWeightModel.spar_pipe_simplified(F_max, y_MAC, b_w / 2, d_RMS, sig_max, rho_spar)
            spar_cnstr = (
                t_tip - d_out
            ) / d_out  # constraint on spar diameter and wing thickness [-]
            outputs["data:structures:wing:spar:diameter:constraint"] = spar_cnstr
            outputs["data:structures:wing:spar:diameter:inner"] = d_in
            outputs["data:structures:wing:spar:diameter:outer"] = d_out

        elif self.options["spar_model"] == "I_beam":
            k_a = inputs["data:structures:wing:spar:depth:k"]
            m_spar, h_web, t_web, a_flange, b_flange = WingWeightModel.spar_I_detailed(
                F_max, y_MAC, b_w / 2, k_a, sig_max, rho_spar
            )
            spar_cnstr = (t_tip - (h_web + a_flange)) / (
                h_web + a_flange
            )  # constraint on spar depth and wing thickness [-]
            outputs["data:structures:wing:spar:web:depth"] = h_web
            outputs["data:structures:wing:spar:web:thickness"] = t_web
            outputs["data:structures:wing:spar:flange:depth"] = a_flange
            outputs["data:structures:wing:spar:flange:thickness"] = b_flange
            outputs["data:structures:wing:spar:depth:constraint"] = spar_cnstr

        m_wing = 2 * (m_spar + m_ribs + m_skin)  # total mass (both sides) [kg]

        outputs["data:weights:airframe:wing:mass"] = m_wing
        outputs["data:weights:airframe:wing:spar:mass"] = 2 * m_spar
        outputs["data:weights:airframe:wing:skin:mass"] = 2 * m_skin
        outputs["data:weights:airframe:wing:ribs:mass"] = 2 * m_ribs
        outputs["data:structures:wing:ribs:number"] = N_ribs


class HorizontalTailWeight(om.ExplicitComponent):
    """
    Computes Horizontal Tail mass
    """

    def setup(self):
        self.add_input("data:geometry:tail:horizontal:surface", val=np.nan, units="m**2")
        self.add_input("data:weights:airframe:tail:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weights:airframe:tail:horizontal:mass", units="kg", lower=0.0)
        # self.add_output("data:weights:airframe:tail:horizontal:skin:mass", units="kg", lower=0.0)
        # self.add_input("data:loads:vertical:factor", val=4.5, units=None)
        # self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        # self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        # self.add_input("data:geometry:tail:horizontal:MAC:y:local", val=np.nan, units="m")
        # self.add_input("data:structures:wing:spar:diameter:k", val=np.nan, units=None)
        # self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        # self.add_input("data:weights:airframe:wing:spar:density", val=np.nan, units="kg/m**3")
        # self.add_input("data:geometry:tail:horizontal:span", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:root:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:tip:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:root:thickness", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:tip:thickness", val=np.nan, units="m")
        # self.add_input("data:structures:wing:ribs:thickness", val=np.nan, units="m")
        # self.add_input("data:weights:airframe:wing:ribs:density", val=np.nan, units="kg/m**3")
        # self.add_output("data:weights:airframe:tail:horizontal:ribs:mass", units="kg", lower=0.0)
        # self.add_output("data:structures:tail:horizontal:ribs:number", units=None, lower=0.0)
        # self.add_output("data:weights:airframe:tail:horizontal:spar:mass", units="kg", lower=0.0)
        # self.add_output("data:structures:tail:horizontal:spar:diameter:inner", units="m", lower=0.0)
        # self.add_output("data:structures:tail:horizontal:spar:diameter:outer", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_ht = inputs["data:geometry:tail:horizontal:surface"]
        rho_skin = inputs["data:weights:airframe:tail:density"]
        # n_ult = inputs["data:loads:vertical:factor"]
        # Mtotal_guess = inputs["data:weights:mtow:guess"]
        # S_w = inputs["data:geometry:wing:surface"]
        # y_MAC = inputs["data:geometry:tail:horizontal:MAC:y:local"]
        # k_spar = inputs["data:structures:wing:spar:diameter:k"]
        # sig_max = inputs["data:structures:wing:spar:stress:max"]
        # rho_spar = inputs["data:weights:airframe:wing:spar:density"]
        # b_ht = inputs["data:geometry:tail:horizontal:span"]
        # c_MAC = inputs["data:geometry:tail:horizontal:MAC:length"]
        # c_root = inputs["data:geometry:tail:horizontal:root:chord"]
        # c_tip = inputs["data:geometry:tail:horizontal:tip:chord"]
        # t_root = inputs["data:geometry:tail:horizontal:root:thickness"]
        # t_tip = inputs["data:geometry:tail:horizontal:tip:thickness"]
        # t_rib = inputs["data:structures:wing:ribs:thickness"]
        # rho_rib = inputs["data:weights:airframe:wing:ribs:density"]

        # F_max = S_ht / S_w * n_ult * Mtotal_guess * g / 2  # ultimate load to be supported by (half) HT [N]
        # m_spar, D_in, D_out = WingWeightModel.spar(F_max, y_MAC, b_ht / 2, k_spar, sig_max, rho_spar)  # spar
        # m_ribs, N_ribs = WingWeightModel.ribs(b_ht / 2, c_MAC, c_root, c_tip, t_root, t_tip, t_rib, rho_rib)  # ribs
        m_skin = WingWeightModel.skin(S_ht / 2, rho_skin)  # skin
        m_wing = 2 * m_skin  # total mass (both sides) [kg]

        outputs["data:weights:airframe:tail:horizontal:mass"] = m_wing
        # outputs["data:weights:airframe:tail:horizontal:skin:mass"] = 2 * m_skin
        # outputs["data:weights:airframe:tail:horizontal:ribs:mass"] = 2 * m_ribs
        # outputs["data:structures:tail:horizontal:ribs:number"] = N_ribs
        # outputs["data:weights:airframe:tail:horizontal:spar:mass"] = 2 * m_spar
        # outputs["data:structures:tail:horizontal:spar:diameter:inner"] = D_in
        # outputs["data:structures:tail:horizontal:spar:diameter:outer"] = D_out


class VerticalTailWeight(om.ExplicitComponent):
    """
    Computes Vertical Tail mass
    """

    def setup(self):
        self.add_input("data:geometry:tail:vertical:surface", val=np.nan, units="m**2")
        self.add_input("data:weights:airframe:tail:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weights:airframe:tail:vertical:mass", units="kg", lower=0.0)
        # self.add_output("data:weights:airframe:tail:vertical:skin:mass", units="kg", lower=0.0)
        # self.add_output("data:weights:airframe:tail:vertical:ribs:mass", units="kg", lower=0.0)
        # self.add_output("data:structures:tail:vertical:ribs:number", units=None, lower=0.0)
        # self.add_input("data:scenarios:cruise:speed", val=np.nan, units="m/s")
        # self.add_input("data:scenarios:cruise:atmosphere:density", val=np.nan, units="kg/m**3")
        # self.add_input("data:loads:lateral:sideslip", val=np.radians(20), units="rad")
        # self.add_input("data:aerodynamics:Cn_beta", val=0.3, units=None)
        # self.add_input("data:geometry:tail:vertical:MAC:y:local", val=np.nan, units="m")
        # self.add_input("data:structures:wing:spar:diameter:k", val=np.nan, units=None)
        # self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        # self.add_input("data:weights:airframe:wing:spar:density", val=np.nan, units="kg/m**3")
        # self.add_input("data:geometry:tail:vertical:span", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:root:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:tip:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:root:thickness", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:tip:thickness", val=np.nan, units="m")
        # self.add_input("data:structures:wing:ribs:thickness", val=np.nan, units="m")
        # self.add_input("data:weights:airframe:wing:ribs:density", val=np.nan, units="kg/m**3")
        # self.add_output("data:structures:tail:vertical:spar:diameter:inner", units="m", lower=0.0)
        # self.add_output("data:structures:tail:vertical:spar:diameter:outer", units="m", lower=0.0)
        # self.add_output("data:weights:airframe:tail:vertical:spar:mass", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_vt = inputs["data:geometry:tail:vertical:surface"]
        rho_skin = inputs["data:weights:airframe:tail:density"]
        # Cn_beta = inputs["data:aerodynamics:Cn_beta"]
        # beta = inputs["data:loads:lateral:sideslip"]
        # k_spar = inputs["data:structures:wing:spar:diameter:k"]
        # sig_max = inputs["data:structures:wing:spar:stress:max"]
        # rho_spar = inputs["data:weights:airframe:wing:spar:density"]
        # V_cruise = inputs["data:scenarios:cruise:speed"]
        # rho_air = inputs["data:scenarios:cruise:atmosphere:density"]
        # y_MAC = inputs["data:geometry:tail:vertical:MAC:y:local"]
        # q_cruise = 0.5 * rho_air * V_cruise ** 2
        # b_vt = inputs["data:geometry:tail:vertical:span"]
        # c_MAC = inputs["data:geometry:tail:vertical:MAC:length"]
        # c_root = inputs["data:geometry:tail:vertical:root:chord"]
        # c_tip = inputs["data:geometry:tail:vertical:tip:chord"]
        # t_root = inputs["data:geometry:tail:vertical:root:thickness"]
        # t_tip = inputs["data:geometry:tail:vertical:tip:thickness"]
        # _rib = inputs["data:structures:wing:ribs:thickness"]
        # rho_rib = inputs["data:weights:airframe:wing:ribs:density"]

        # F_max = q_cruise * S_vt * Cn_beta * beta  # ultimate aerodynamic load to be supported by vertical tail [N]
        # m_spar, d_spar_in, d_spar_out = WingWeightModel.spar(F_max, y_MAC, b_vt, k_spar, sig_max, rho_spar)  # spar
        # m_ribs, N_ribs = WingWeightModel.ribs(b_vt, c_MAC, c_root, c_tip, t_root, t_tip, t_rib, rho_rib)  # ribs
        m_skin = WingWeightModel.skin(S_vt, rho_skin)  # skin
        m_wing = m_skin  # total mass (both sides) [kg]

        outputs["data:weights:airframe:tail:vertical:mass"] = m_wing
        # outputs["data:weights:airframe:tail:vertical:skin:mass"] = m_skin
        # outputs["data:weights:airframe:tail:vertical:ribs:mass"] = m_ribs
        # outputs["data:structures:tail:vertical:ribs:number"] = N_ribs
        # outputs["data:weights:airframe:tail:vertical:spar:mass"] = m_spar
        # outputs["data:structures:tail:vertical:spar:diameter:inner"] = d_spar_in
        # outputs["data:structures:tail:vertical:spar:diameter:outer"] = d_spar_out


class FuselageWeight(om.ExplicitComponent):
    """
    Computes Fuselage mass
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:nose", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:mid", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:rear", val=np.nan, units="m**2")
        self.add_input("data:weights:airframe:fuselage:mass:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weights:airframe:fuselage:mass", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:nose", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:mid", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:rear", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_nose = inputs["data:geometry:fuselage:surface:nose"]
        S_mid = inputs["data:geometry:fuselage:surface:mid"]
        S_rear = inputs["data:geometry:fuselage:surface:rear"]
        rho_fus = inputs["data:weights:airframe:fuselage:mass:density"]

        m_nose = S_nose * rho_fus
        m_mid = S_mid * rho_fus
        m_rear = S_rear * rho_fus
        m_fus = S_fus * rho_fus  # mass of fuselage [kg]

        outputs["data:weights:airframe:fuselage:mass"] = m_fus
        outputs["data:weights:airframe:fuselage:mass:nose"] = m_nose
        outputs["data:weights:airframe:fuselage:mass:mid"] = m_mid
        outputs["data:weights:airframe:fuselage:mass:rear"] = m_rear


class Constraints(om.ExplicitComponent):
    """
    Constraints definition for the Structures and Weights discipline.
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:horizontal:tip:thickness", val=np.nan, units="m")
        # self.add_input("data:structures:tail:horizontal:spar:diameter:outer", val=np.nan, units="m")
        # self.add_input("data:geometry:tail:vertical:tip:thickness", val=np.nan, units="m")
        # self.add_input("data:structures:tail:vertical:spar:diameter:outer", val=np.nan, units="m")
        if self.options["spar_model"] == "pipe":
            self.add_input("data:structures:wing:spar:diameter:outer", val=np.nan, units="m")
        elif self.options["spar_model"] == "I_beam":
            self.add_input("data:structures:wing:spar:web:depth", val=np.nan, units="m")
        self.add_output("data:structures:wing:spar:diameter:constraint", units=None)
        # self.add_output("data:structures:tail:horizontal:spar:diameter:constraint", units=None)
        # self.add_output("data:structures:tail:vertical:spar:diameter:constraint", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        t_tip_w = inputs["data:geometry:wing:tip:thickness"]
        # t_tip_ht = inputs["data:geometry:tail:horizontal:tip:thickness"]
        # d_spar_ht = inputs["data:structures:tail:horizontal:spar:diameter:outer"]
        # t_tip_vt = inputs["data:geometry:tail:vertical:tip:thickness"]
        # d_spar_vt = inputs["data:structures:tail:vertical:spar:diameter:outer"]
        d_spar_w = t_tip_w  # default value
        if self.options["spar_model"] == "pipe":
            d_spar_w = inputs["data:structures:wing:spar:diameter:outer"]
        elif self.options["spar_model"] == "I_beam":
            d_spar_w = inputs["data:structures:wing:spar:web:depth"]

        spar_cnstr_wing = (
            t_tip_w - d_spar_w
        ) / d_spar_w  # constraint on spar diameter for the wing [-]
        # spar_cnstr_ht = (t_tip_ht - d_spar_ht) / d_spar_ht  # constraint on spar diameter for the HT [-]
        # spar_cnstr_vt = (t_tip_vt - d_spar_vt) / d_spar_vt  # constraint on spar diameter for the VT [-]

        outputs["data:structures:wing:spar:diameter:constraint"] = spar_cnstr_wing
        # outputs["data:structures:tail:horizontal:spar:diameter:constraint"] = spar_cnstr_ht
        # outputs["data:structures:tail:vertical:spar:diameter:constraint"] = spar_cnstr_vt
