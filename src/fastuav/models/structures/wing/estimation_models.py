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
    
    @staticmethod
    def skin_dS(S, rho_skin):
        """
        Derivative of skin mass wrt surface area S.
        m_skin = 2*S*rho_skin, so dm_skin/dS = 2*rho_skin
        """
        return 2.0 * rho_skin

    @staticmethod
    def skin_drho(S, rho_skin):
        """
        Derivative of skin mass wrt area density rho_skin.
        m_skin = 2*S*rho_skin, so dm_skin/drho_skin = 2*S
        """
        return 2.0 * S


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
        self.add_input("mission:sizing:load_factor:ultimate", val=3.0, units=None)
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:y", val=np.nan, units="m")
        self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        self.add_input("data:weight:airframe:wing:spar:density", val=np.nan, units="kg/m**3")
        self.add_output("data:weight:airframe:wing:spar:mass", units="kg", lower=0.0)

        if self.options["spar_model"] == "pipe":
            self.add_input("optimization:variables:structures:wing:spar:diameter:k", val=0.9, units=None)
            self.add_input("optimization:variables:structures:wing:spar:diameter:outer:k", val=1.0, units=None)
            self.add_output("data:structures:wing:spar:diameter:inner", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:diameter:outer", units="m", lower=0.0)

        elif self.options["spar_model"] == "I_beam":
            self.add_input("optimization:variables:structures:wing:spar:depth:k", val=0.1, units=None)
            self.add_input("optimization:variables:structures:wing:spar:web:depth:k", val=1.0, units=None)
            self.add_output("data:structures:wing:spar:web:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:web:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:depth", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:flange:thickness", units="m", lower=0.0)
            self.add_output("data:structures:wing:spar:depth", units="m", lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        spar_model = self.options["spar_model"]
        n_ult = inputs["mission:sizing:load_factor:ultimate"]
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        b_w = inputs["data:geometry:wing:span"]
        y_MAC = inputs["data:geometry:wing:MAC:y"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]
        rho_spar = inputs["data:weight:airframe:wing:spar:density"]

        # LOADS
        F_max = n_ult * m_uav_guess * g / 2  # ultimate aerodynamic load [N]
        M_root = F_max * y_MAC  # bending moment at root [N.m]

        if spar_model == "pipe":  # Circular hollow beam model
            # aspect ratio of the spar [-]:
            k_spar = inputs["optimization:variables:structures:wing:spar:diameter:k"]
            # under-sizing coef. [-] on spar outer diameter (1.0 for FW (monotonicity eq.)/ des. var. for Hybrid):
            k_d = inputs["optimization:variables:structures:wing:spar:diameter:outer:k"]
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
            k_spar = inputs["optimization:variables:structures:wing:spar:depth:k"]
            # under-sizing coef. [-] on spar web depth (1.0 for FW (monotonicity eq.)/ des. var. for Hybrid):
            k_h = inputs["optimization:variables:structures:wing:spar:web:depth:k"]
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

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        spar_model = self.options["spar_model"]
        
        n_ult = inputs["mission:sizing:load_factor:ultimate"]
        m_uav = inputs["optimization:variables:weight:mtow:guess"]
        b_w = inputs["data:geometry:wing:span"]
        y_MAC = inputs["data:geometry:wing:MAC:y"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]
        rho_spar = inputs["data:weight:airframe:wing:spar:density"]
        
        g = 9.80665  # m/s^2
        
        # Common computations
        F_max = n_ult * m_uav * g / 2
        M_root = F_max * y_MAC
        L_spar = b_w / 2
        
        # dM_root / d(...)
        dM_root_dn_ult = m_uav * g / 2 * y_MAC
        dM_root_dm_uav = n_ult * g / 2 * y_MAC
        dM_root_dy_MAC = n_ult * m_uav * g / 2
        
        if spar_model == "pipe":
            # Pipe model
            k_spar = inputs["optimization:variables:structures:wing:spar:diameter:k"]
            k_d = inputs["optimization:variables:structures:wing:spar:diameter:outer:k"]
            
            # Outer diameter: d_out = k_d * (32*M_root / (π*(1-k_spar^4)*sig_max))^(1/3)
            X = 32 * M_root / (np.pi * (1 - k_spar ** 4) * sig_max)
            d_out = k_d * (X ** (1.0 / 3.0))
            
            # Derivatives of X wrt inputs
            dX_dM_root = 32 / (np.pi * (1 - k_spar ** 4) * sig_max)
            dX_dk_spar = 32 * M_root / (np.pi * sig_max) * (4 * k_spar ** 3) / ((1 - k_spar ** 4) ** 2)
            dX_dsig_max = -32 * M_root / (np.pi * (1 - k_spar ** 4) * sig_max ** 2)
            
            # Derivatives of d_out wrt X and k_d
            dd_out_dX = (k_d / 3.0) * (X ** (-2.0 / 3.0))
            dd_out_dk_d = X ** (1.0 / 3.0)
            
            # Inner diameter: d_in = k_spar * d_out
            d_in = k_spar * d_out
            
            # Cross-sectional area: A = π/4 * (d_out^2 - d_in^2) = π/4 * d_out^2 * (1 - k_spar^2)
            A_spar = np.pi / 4 * (d_out ** 2 - d_in ** 2)
            
            # Mass: m_spar = rho_spar * L_spar * A_spar
            m_spar = rho_spar * L_spar * A_spar
            
            # Derivatives of A_spar
            dA_dd_out = np.pi / 4 * (2 * d_out - 2 * d_in * (dM_root_dn_ult / dM_root_dn_ult if dM_root_dn_ult != 0 else 0))
            # Actually, d_in = k_spar * d_out, so:
            # A = π/4 * (d_out^2 - k_spar^2 * d_out^2) = π/4 * d_out^2 * (1 - k_spar^2)
            # dA/dd_out = π/2 * d_out * (1 - k_spar^2)
            # dA/dk_spar = π/4 * d_out^2 * (-2*k_spar) = -π/2 * d_out^2 * k_spar
            
            dA_dd_out = np.pi / 2 * d_out * (1 - k_spar ** 2)
            dA_dk_spar = -np.pi / 2 * d_out ** 2 * k_spar
            
            # Mass derivatives
            dm_spar_drho = L_spar * A_spar
            dm_spar_db_w = rho_spar * 0.5 * A_spar
            dm_spar_dA = rho_spar * L_spar
            
            # Chain rule: d(d_out) / d(various inputs)
            dd_out_dn_ult = dd_out_dX * dX_dM_root * dM_root_dn_ult
            dd_out_dm_uav = dd_out_dX * dX_dM_root * dM_root_dm_uav
            dd_out_dy_MAC = dd_out_dX * dX_dM_root * dM_root_dy_MAC
            dd_out_dsig_max = dd_out_dX * dX_dsig_max
            dd_out_dk_spar = dd_out_dX * dX_dk_spar
            
            # d_in derivatives: d_in = k_spar * d_out
            # dd_in/dk_spar = d_out + k_spar * dd_out/dk_spar
            # dd_in/dd_out = k_spar
            dd_in_dk_spar = d_out + k_spar * dd_out_dk_spar
            dd_in_dn_ult = k_spar * dd_out_dn_ult
            dd_in_dm_uav = k_spar * dd_out_dm_uav
            dd_in_dy_MAC = k_spar * dd_out_dy_MAC
            dd_in_dsig_max = k_spar * dd_out_dsig_max
            dd_in_dk_d = k_spar * dd_out_dk_d
            
            # m_spar derivatives (via chain rule through A and d_out)
            # dm/d(...) = dm/dA * dA/d(...) + dm/dA * dA/dd_out * dd_out/d(...)
            
            partials["data:weight:airframe:wing:spar:mass", "mission:sizing:load_factor:ultimate"] = (
                2*dm_spar_dA * (dA_dd_out * dd_out_dn_ult + dA_dk_spar * 0)  # k_spar doesn't depend on n_ult
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:weight:mtow:guess"] = (
                2*dm_spar_dA * dA_dd_out * dd_out_dm_uav
            )
            partials["data:weight:airframe:wing:spar:mass", "data:geometry:wing:span"] = (
                2*dm_spar_db_w + 2*dm_spar_dA * dA_dd_out * 0  # b_w doesn't affect d_out after M_root
            )
            partials["data:weight:airframe:wing:spar:mass", "data:geometry:wing:MAC:y"] = (
                2*dm_spar_dA * dA_dd_out * dd_out_dy_MAC
            )
            partials["data:weight:airframe:wing:spar:mass", "data:structures:wing:spar:stress:max"] = (
                2*dm_spar_dA * dA_dd_out * dd_out_dsig_max
            )
            partials["data:weight:airframe:wing:spar:mass", "data:weight:airframe:wing:spar:density"] = (
                2*dm_spar_drho
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:structures:wing:spar:diameter:k"] = (
                2*dm_spar_dA * (dA_dd_out * dd_out_dk_spar + dA_dk_spar)
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:structures:wing:spar:diameter:outer:k"] = (
                2*dm_spar_dA * dA_dd_out * dd_out_dk_d
            )
            
            # d_out partials
            partials["data:structures:wing:spar:diameter:outer", "mission:sizing:load_factor:ultimate"] = dd_out_dn_ult
            partials["data:structures:wing:spar:diameter:outer", "optimization:variables:weight:mtow:guess"] = dd_out_dm_uav
            partials["data:structures:wing:spar:diameter:outer", "data:geometry:wing:MAC:y"] = dd_out_dy_MAC
            partials["data:structures:wing:spar:diameter:outer", "data:structures:wing:spar:stress:max"] = dd_out_dsig_max
            partials["data:structures:wing:spar:diameter:outer", "optimization:variables:structures:wing:spar:diameter:k"] = dd_out_dk_spar
            partials["data:structures:wing:spar:diameter:outer", "optimization:variables:structures:wing:spar:diameter:outer:k"] = dd_out_dk_d
            
            # d_in partials
            partials["data:structures:wing:spar:diameter:inner", "mission:sizing:load_factor:ultimate"] = dd_in_dn_ult
            partials["data:structures:wing:spar:diameter:inner", "optimization:variables:weight:mtow:guess"] = dd_in_dm_uav
            partials["data:structures:wing:spar:diameter:inner", "data:geometry:wing:MAC:y"] = dd_in_dy_MAC
            partials["data:structures:wing:spar:diameter:inner", "data:structures:wing:spar:stress:max"] = dd_in_dsig_max
            partials["data:structures:wing:spar:diameter:inner", "optimization:variables:structures:wing:spar:diameter:k"] = dd_in_dk_spar
            partials["data:structures:wing:spar:diameter:inner", "optimization:variables:structures:wing:spar:diameter:outer:k"] = dd_in_dk_d
    
        else:  # I-beam model
            k_spar = inputs["optimization:variables:structures:wing:spar:depth:k"]
            k_h = inputs["optimization:variables:structures:wing:spar:web:depth:k"]
            k_flange = 0.1
            k_web = 30
            
            # Web depth: h_web = k_h * (M_root * (1+k_spar) / (sig_max * k_spar^2 * (1+k_spar^2/3) / k_flange))^(1/3)
            # Simplify: Y = M_root * (1+k_spar) / (sig_max * k_spar^2 * (1+k_spar^2/3) / k_flange)
            # Y = M_root * (1+k_spar) * k_flange / [sig_max * k_spar^2 * (1+k_spar^2/3)]
            Y = M_root * (1 + k_spar) * k_flange / (sig_max * k_spar ** 2 * (1 + k_spar ** 2 / 3))
            h_web = k_h * (Y ** (1.0 / 3.0))

            # Geometry
            t_web = h_web / k_web
            a_flange = k_spar * h_web
            b_flange = a_flange / k_flange
            cross_section = 2 * a_flange * b_flange + (h_web - a_flange) * t_web
            m_spar = rho_spar * L_spar * cross_section

            # Y derivatives — use log derivative for cleanliness
            dY_dM_root = Y / M_root
            dY_dsig_max = -Y / sig_max
            dY_dk_spar = Y * (
                1.0 / (1 + k_spar)
                - 2.0 / k_spar
                - (2 * k_spar / 3) / (1 + k_spar ** 2 / 3)
            )

            # h_web = k_h * Y^(1/3)
            # dh_web/dY = (k_h/3) * Y^(-2/3) = h_web / (3*Y)
            dh_web_dY = h_web / (3 * Y)

            dh_web_dM_root = dh_web_dY * dY_dM_root
            dh_web_dk_spar = dh_web_dY * dY_dk_spar
            dh_web_dk_h = h_web / k_h
            dh_web_dsig_max = dh_web_dY * dY_dsig_max

            dh_web_dn_ult = dh_web_dM_root * dM_root_dn_ult
            dh_web_dm_uav = dh_web_dM_root * dM_root_dm_uav
            dh_web_dy_MAC = dh_web_dM_root * dM_root_dy_MAC          
           
            
            # Cross-section derivative
            # cross_section = 2*k_spar*h_web * k_spar*h_web/k_flange + (h_web - k_spar*h_web)*h_web/k_web
            #              = 2*k_spar^2*h_web^2/k_flange + (1-k_spar)*h_web^2/k_web
            
            dcs_dh_web = 4 * k_spar ** 2 * h_web / k_flange + 2 * (1 - k_spar) * h_web / k_web
            dcs_dk_spar = 4 * k_spar * h_web ** 2 / k_flange - h_web ** 2 / k_web


            dm_spar_drho = L_spar * cross_section
            dm_spar_db_w = rho_spar * 0.5 * cross_section
            dm_spar_dcs = rho_spar * L_spar

            
            # h_web partials
            partials["data:structures:wing:spar:web:depth", "mission:sizing:load_factor:ultimate"] = dh_web_dn_ult
            partials["data:structures:wing:spar:web:depth", "optimization:variables:weight:mtow:guess"] = dh_web_dm_uav
            partials["data:structures:wing:spar:web:depth", "data:geometry:wing:MAC:y"] = dh_web_dy_MAC
            partials["data:structures:wing:spar:web:depth", "data:structures:wing:spar:stress:max"] = dh_web_dsig_max
            partials["data:structures:wing:spar:web:depth", "optimization:variables:structures:wing:spar:depth:k"] = dh_web_dk_spar
            partials["data:structures:wing:spar:web:depth", "optimization:variables:structures:wing:spar:web:depth:k"] = dh_web_dk_h
            
            # t_web partials: t_web = h_web / k_web
            partials["data:structures:wing:spar:web:thickness", "mission:sizing:load_factor:ultimate"] = dh_web_dn_ult / k_web
            partials["data:structures:wing:spar:web:thickness", "optimization:variables:weight:mtow:guess"] = dh_web_dm_uav / k_web
            partials["data:structures:wing:spar:web:thickness", "data:geometry:wing:MAC:y"] = dh_web_dy_MAC / k_web
            partials["data:structures:wing:spar:web:thickness", "data:structures:wing:spar:stress:max"] = dh_web_dsig_max / k_web
            partials["data:structures:wing:spar:web:thickness", "optimization:variables:structures:wing:spar:depth:k"] = dh_web_dk_spar / k_web
            partials["data:structures:wing:spar:web:thickness", "optimization:variables:structures:wing:spar:web:depth:k"] = dh_web_dk_h / k_web
            
            # a_flange partials: a_flange = k_spar * h_web
            partials["data:structures:wing:spar:flange:depth", "mission:sizing:load_factor:ultimate"] = k_spar * dh_web_dn_ult
            partials["data:structures:wing:spar:flange:depth", "optimization:variables:weight:mtow:guess"] = k_spar * dh_web_dm_uav
            partials["data:structures:wing:spar:flange:depth", "data:geometry:wing:MAC:y"] = k_spar * dh_web_dy_MAC
            partials["data:structures:wing:spar:flange:depth", "data:structures:wing:spar:stress:max"] = k_spar * dh_web_dsig_max
            partials["data:structures:wing:spar:flange:depth", "optimization:variables:structures:wing:spar:depth:k"] = h_web + k_spar * dh_web_dk_spar
            partials["data:structures:wing:spar:flange:depth", "optimization:variables:structures:wing:spar:web:depth:k"] = k_spar * dh_web_dk_h
            
            # b_flange partials: b_flange = a_flange / k_flange = k_spar * h_web / k_flange
            partials["data:structures:wing:spar:flange:thickness", "mission:sizing:load_factor:ultimate"] = k_spar * dh_web_dn_ult / k_flange
            partials["data:structures:wing:spar:flange:thickness", "optimization:variables:weight:mtow:guess"] = k_spar * dh_web_dm_uav / k_flange
            partials["data:structures:wing:spar:flange:thickness", "data:geometry:wing:MAC:y"] = k_spar * dh_web_dy_MAC / k_flange
            partials["data:structures:wing:spar:flange:thickness", "data:structures:wing:spar:stress:max"] = k_spar * dh_web_dsig_max / k_flange
            partials["data:structures:wing:spar:flange:thickness", "optimization:variables:structures:wing:spar:depth:k"] = (h_web + k_spar * dh_web_dk_spar) / k_flange
            partials["data:structures:wing:spar:flange:thickness", "optimization:variables:structures:wing:spar:web:depth:k"] = k_spar * dh_web_dk_h / k_flange
            
            # depth (h_web + a_flange) partials
            partials["data:structures:wing:spar:depth", "mission:sizing:load_factor:ultimate"] = dh_web_dn_ult * (1 + k_spar)
            partials["data:structures:wing:spar:depth", "optimization:variables:weight:mtow:guess"] = dh_web_dm_uav * (1 + k_spar)
            partials["data:structures:wing:spar:depth", "data:geometry:wing:MAC:y"] = dh_web_dy_MAC * (1 + k_spar)
            partials["data:structures:wing:spar:depth", "data:structures:wing:spar:stress:max"] = dh_web_dsig_max * (1 + k_spar)
            partials["data:structures:wing:spar:depth", "optimization:variables:structures:wing:spar:depth:k"] = dh_web_dk_spar * (1 + k_spar) + h_web
            partials["data:structures:wing:spar:depth", "optimization:variables:structures:wing:spar:web:depth:k"] = dh_web_dk_h * (1 + k_spar)
            
            # m_spar partials (chain rule via cross_section and h_web)
            dm_spar_dh_web = rho_spar * L_spar * dcs_dh_web
            
            partials["data:weight:airframe:wing:spar:mass", "mission:sizing:load_factor:ultimate"] = (
                2*dm_spar_db_w * 0 + 2*dm_spar_dcs * (dcs_dh_web * dh_web_dn_ult + dcs_dk_spar * 0)
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:weight:mtow:guess"] = (
                2*dm_spar_dcs * dcs_dh_web * dh_web_dm_uav
            )
            partials["data:weight:airframe:wing:spar:mass", "data:geometry:wing:span"] = (
                2*dm_spar_db_w
            )
            partials["data:weight:airframe:wing:spar:mass", "data:geometry:wing:MAC:y"] = (
                2*dm_spar_dcs * dcs_dh_web * dh_web_dy_MAC
            )
            partials["data:weight:airframe:wing:spar:mass", "data:structures:wing:spar:stress:max"] = (
                2*dm_spar_dcs * dcs_dh_web * dh_web_dsig_max
            )
            partials["data:weight:airframe:wing:spar:mass", "data:weight:airframe:wing:spar:density"] = (
                2*dm_spar_drho
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:structures:wing:spar:depth:k"] = (
                2*dm_spar_dcs * (dcs_dh_web * dh_web_dk_spar + dcs_dk_spar)
            )
            partials["data:weight:airframe:wing:spar:mass", "optimization:variables:structures:wing:spar:web:depth:k"] = (
                2*dm_spar_dcs * dcs_dh_web * dh_web_dk_h
            )



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
