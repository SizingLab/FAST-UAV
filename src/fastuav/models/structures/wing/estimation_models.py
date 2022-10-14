"""
Estimation models for the Wing Structures and Weights
"""

import openmdao.api as om
import numpy as np
from scipy.constants import g


class WingStructuresEstimationModels:
    """
    Estimation models for the wing's structures and weights.
    """

    @staticmethod
    def spar_i_beam(h_web, k_spar, L, rho_spar, k_flange=0.1, k_web=30):
        """
        Estimation model for the mass and dimensions of a I-shaped beam of length L.

        :param h_web: Depth of the web [m]
        :param k_spar: Depth ratio of the spar (k_a = flange depth / web depth) [-]
        :param L: Length of the spar [m]
        :param rho_spar: Material density of the spar [kg/m3]
        :param k_flange: Aspect ratio of the flange (b_flange = a_flange / k_flange) [-]
        :param k_web: Aspect ratio of the web (t_web = h_web / k_web)
        :return m_spar: Mass of the spar [kg]
        :return t_web: Thickness of the web [m]
        :return a_flange: Depth of the flange [m]
        :return b_flange: Thickness of the flange [m]
        """
        # GEOMETRY
        t_web = h_web / k_web  # web thickness [m]
        a_flange = k_spar * h_web  # flange depth [m]
        b_flange = a_flange / k_flange  # flange thickness [m]

        # MASS
        m_spar = (
            rho_spar * L * (2 * a_flange * b_flange + (h_web - a_flange) * t_web)
        )  # mass of spar [kg]

        return m_spar, t_web, a_flange, b_flange

    @staticmethod
    def spar_pipe(d_out, k_spar, L, rho_spar):
        """
        Estimation model for the mass and dimensions of a circular hollow beam of length L.

        :param d_out: External diameter of the spar [m]
        :param k_spar: Diameter ratio of the spar (k_d = d_in / d_out) [-]
        :param L: Length of the spar [m]
        :param rho_spar: Material density of the spar [kg/m3]
        :return m_spar: Mass of the spar [kg]
        :return d_in: Inner diameter of the spar [m]
        """
        # GEOMETRY
        d_in = k_spar * d_out  # inner diameter of the spar [m]
        A_spar = np.pi / 4 * (d_out**2 - d_in**2)  # sectional area of the spar [m2]

        # MASS
        m_spar = rho_spar * L * A_spar  # mass of spar [kg]

        return m_spar, d_in

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


class WingStructuresEstimationModelsGroup(om.Group):
    """
    Estimation models for the Wing Structures
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        spar_model = self.options["spar_model"]
        self.add_subsystem("spars", Spars(spar_model=spar_model), promotes=["*"])
        self.add_subsystem("ribs", Ribs(), promotes=["*"])
        self.add_subsystem("skin", Skin(), promotes=["*"])
        self.add_subsystem("wing", WingComponent(), promotes=["*"])


class Spars(om.ExplicitComponent):
    """
    Computes spars weight and dimensions.
    The spars are initially sized from an ultimate aerodynamic load applied at the MAC position
    of the wing.
    Constraints can be added in a second step to ensure appropriate sizing with respect
    to the wing's thickness and, for FW-VTOL UAVs, the takeoff thrust of the VTOL propellers.
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_input("data:scenarios:load_factor:vertical:ultimate", val=4.5, units=None)
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:y", val=np.nan, units="m")
        self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:weight:airframe:wing:spar:density", val=np.nan, units="kg/m**3")
        self.add_output("data:weight:airframe:wing:spar:mass", units="kg", lower=0.0)

        if self.options["spar_model"] == "pipe":
            self.add_input("data:structures:wing:spar:diameter:k", val=0.9, units=None)
            self.add_input("data:structures:wing:spar:diameter:outer:k", val=1.0, units=None)
            self.add_output("data:structures:wing:spar:diameter:inner", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:diameter:outer", units="m", lower=0.0)

        elif self.options["spar_model"] == "I_beam":
            self.add_input("data:structures:wing:spar:depth:k", val=0.1, units=None)
            self.add_input("data:structures:wing:spar:web:depth:k", val=1.0, units=None)
            self.add_output("data:structures:wing:spar:web:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:web:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:depth", units="m", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        spar_model = self.options["spar_model"]
        n_ult = inputs["data:scenarios:load_factor:vertical:ultimate"]
        m_uav_guess = inputs["data:weight:mtow:guess"]
        b_w = inputs["data:geometry:wing:span"]
        y_MAC = inputs["data:geometry:wing:MAC:y"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]
        rho_spar = inputs["data:weight:airframe:wing:spar:density"]

        # LOADS
        F_max = n_ult * m_uav_guess * g / 2  # ultimate aerodynamic load [N]
        M_root = F_max * y_MAC  # bending moment at root [N.m]

        if spar_model == "pipe":  # Circular hollow beam model
            # aspect ratio of the spar [-]:
            k_spar = inputs["data:structures:wing:spar:diameter:k"]
            # under-sizing coef. [-] on spar outer diameter (1.0 for Fixed-Wing / des. var. for Hybrid):
            k_d = inputs["data:structures:wing:spar:diameter:outer:k"]
            # Outer diameter calculation [m]:
            d_out = k_d * ((32 * M_root) / (np.pi * (1 - k_spar ** 4) * sig_max)) ** (1 / 3)
            # Mass and inner diameter calculations:
            m_spar, d_in = WingStructuresEstimationModels.spar_pipe(d_out,
                                                                    k_spar,
                                                                    b_w / 2,
                                                                    rho_spar)
            outputs["data:structures:wing:spar:diameter:inner"] = d_in
            outputs["data:structures:wing:spar:diameter:outer"] = d_out

        else:  # I-beam model
            # aspect ratio of the spar [-], i.e. flanges' thickness over distance between the two flanges:
            k_spar = inputs["data:structures:wing:spar:depth:k"]
            # under-sizing coef. [-] on spar web depth (1.0 for Fixed-Wing / des. var. for Hybrid):
            k_h = inputs["data:structures:wing:spar:web:depth:k"]
            # flange depth-to-thickness ratio [-]: b_flange = a_flange / k_flange
            k_flange = 0.1
            # web depth calculation [m]:
            h_web = k_h * (M_root * (1 + k_spar) / (sig_max * k_spar ** 2 * (1 + k_spar ** 2 / 3) / k_flange)) ** (
                    1 / 3
            )
            # Mass and secondary geometry calculations:
            m_spar, t_web, a_flange, b_flange = WingStructuresEstimationModels.spar_i_beam(h_web,
                                                                                           k_spar,
                                                                                           b_w / 2,
                                                                                           rho_spar,
                                                                                           k_flange=k_flange)
            outputs["data:structures:wing:spar:web:depth"] = h_web
            outputs["data:structures:wing:spar:web:thickness"] = t_web
            outputs["data:structures:wing:spar:flange:depth"] = a_flange
            outputs["data:structures:wing:spar:flange:thickness"] = b_flange
            outputs["data:structures:wing:spar:depth"] = h_web + a_flange

        outputs["data:weight:airframe:wing:spar:mass"] = 2 * m_spar


class Ribs(om.ExplicitComponent):
    """
    Computes number of ribs and their mass
    """

    def setup(self):
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")
        self.add_input("data:structures:wing:ribs:thickness", val=np.nan, units="m")
        self.add_input("data:weight:airframe:wing:ribs:density", val=np.nan, units="kg/m**3")
        self.add_output("data:weight:airframe:wing:ribs:mass", units="kg", lower=0.0)
        self.add_output("data:structures:wing:ribs:number", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        b_w = inputs["data:geometry:wing:span"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        c_root = inputs["data:geometry:wing:root:chord"]
        c_tip = inputs["data:geometry:wing:tip:chord"]
        t_root = inputs["data:geometry:wing:root:thickness"]
        t_tip = inputs["data:geometry:wing:tip:thickness"]
        t_rib = inputs["data:structures:wing:ribs:thickness"]
        rho_rib = inputs["data:weight:airframe:wing:ribs:density"]

        # RIBS (half wing)
        m_ribs, N_ribs = WingStructuresEstimationModels.ribs(
            b_w / 2, c_MAC, c_root, c_tip, t_root, t_tip, t_rib, rho_rib
        )

        outputs["data:weight:airframe:wing:ribs:mass"] = 2 * m_ribs
        outputs["data:structures:wing:ribs:number"] = 2 * N_ribs


class Skin(om.ExplicitComponent):
    """
    Computes mass of wing skin
    """

    def setup(self):
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:weight:airframe:wing:skin:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weight:airframe:wing:skin:mass", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_w = inputs["data:geometry:wing:surface"]
        rho_skin = inputs["data:weight:airframe:wing:skin:density"]

        # SKIN (half wing)
        m_skin = WingStructuresEstimationModels.skin(S_w / 2, rho_skin)

        outputs["data:weight:airframe:wing:skin:mass"] = 2 * m_skin


class WingComponent(om.ExplicitComponent):
    """
    Computes total wing mass
    """

    def setup(self):
        self.add_input("data:weight:airframe:wing:spar:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:ribs:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:skin:mass", val=np.nan, units="kg")
        self.add_output("data:weight:airframe:wing:mass", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        m_spar = inputs["data:weight:airframe:wing:spar:mass"]
        m_ribs = inputs["data:weight:airframe:wing:ribs:mass"]
        m_skin = inputs["data:weight:airframe:wing:skin:mass"]

        # TOTAL WING (both sides)
        m_wing = m_spar + m_ribs + m_skin  # total mass [kg]

        outputs["data:weight:airframe:wing:mass"] = m_wing