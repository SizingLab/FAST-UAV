"""
Propeller performances
"""
import openmdao.api as om
import numpy as np
from models.Propulsion.Propeller.estimation.models import PropellerAerodynamicsModel


class PropellerPerfoModel:
    """
    Propeller model for performances calculation
    """

    @staticmethod
    def speed(F_pro, D_pro, C_t, rho_air):
        n_pro = (F_pro / (C_t * rho_air * D_pro**4)) ** 0.5 \
            if (C_t and rho_air and D_pro) \
            else 0  # [Hz] Propeller speed
        W_pro = n_pro * 2 * np.pi  # [rad/s] Propeller speed
        return W_pro

    @staticmethod
    def power(W_pro, D_pro, C_p, rho_air):
        P_pro = (
            C_p * rho_air * (W_pro / (2 * np.pi)) ** 3 * D_pro**5
        )  # [W] Propeller power
        return P_pro

    @staticmethod
    def torque(P_pro, W_pro):
        Q_pro = P_pro / W_pro if W_pro else 0  # [N.m] Propeller torque
        return Q_pro

    @staticmethod
    def performances(F_pro, D_pro, C_t, C_p, rho_air):
        W_pro = PropellerPerfoModel.speed(
            F_pro, D_pro, C_t, rho_air
        )  # [rad/s] Propeller speed
        P_pro = PropellerPerfoModel.power(W_pro, D_pro, C_p, rho_air)  # [W] Propeller power
        Q_pro = PropellerPerfoModel.torque(P_pro, W_pro)  # [N.m] Propeller torque
        return W_pro, P_pro, Q_pro


class PropellerPerfos(om.Group):
    """
    Group containing the performance functions of the propeller
    """

    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("cruise", Cruise(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes performances of the propeller for takeoff
    """

    def setup(self):
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CT:static", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CP:static", val=np.nan, units=None)
        self.add_input("data:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("mission:design_mission:takeoff:atmosphere:density", val=np.nan, units="kg/m**3")
        self.add_output("data:propeller:speed:takeoff", units="rad/s")
        self.add_output("data:propeller:torque:takeoff", units="N*m")
        self.add_output("data:propeller:power:takeoff", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        F_pro_to = inputs["data:propeller:thrust:takeoff"]
        rho_air = inputs["mission:design_mission:takeoff:atmosphere:density"]

        # C_t_sta = inputs["data:propeller:aerodynamics:CT:static"]
        # C_p_sta = inputs["data:propeller:aerodynamics:CP:static"]
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta)

        Wpro_to = PropellerPerfoModel.speed(F_pro_to, Dpro, C_t, rho_air)
        Ppro_to = PropellerPerfoModel.power(Wpro_to, Dpro, C_p, rho_air)
        Qpro_to = PropellerPerfoModel.torque(Ppro_to, Wpro_to)

        outputs["data:propeller:speed:takeoff"] = Wpro_to
        outputs["data:propeller:torque:takeoff"] = Qpro_to
        outputs["data:propeller:power:takeoff"] = Ppro_to


class Hover(om.ExplicitComponent):
    """
    Computes performances of the propeller for hover
    """

    def setup(self):
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CT:static", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CP:static", val=np.nan, units=None)
        self.add_input("data:propeller:thrust:hover", val=np.nan, units="N")
        self.add_input("mission:design_mission:hover:atmosphere:density", val=np.nan, units="kg/m**3")
        self.add_output("data:propeller:speed:hover", units="rad/s")
        self.add_output("data:propeller:torque:hover", units="N*m")
        self.add_output("data:propeller:power:hover", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        F_pro_hov = inputs["data:propeller:thrust:hover"]
        rho_air = inputs["mission:design_mission:hover:atmosphere:density"]

        # C_t_sta = inputs["data:propeller:aerodynamics:CT:static"]
        # C_p_sta = inputs["data:propeller:aerodynamics:CP:static"]
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta)

        Wpro_hover = PropellerPerfoModel.speed(F_pro_hov, Dpro, C_t, rho_air)
        Ppro_hover = PropellerPerfoModel.power(Wpro_hover, Dpro, C_p, rho_air)
        Qpro_hover = PropellerPerfoModel.torque(Ppro_hover, Wpro_hover)

        outputs["data:propeller:speed:hover"] = Wpro_hover
        outputs["data:propeller:torque:hover"] = Qpro_hover
        outputs["data:propeller:power:hover"] = Ppro_hover


class Climb(om.ExplicitComponent):
    """
    Computes performances of the propeller for climb
    """

    def setup(self):
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:climb", val=np.nan, units=None)
        self.add_input("mission:design_mission:climb:AoA", val=np.nan, units="rad")
        # self.add_input("data:propeller:aerodynamics:CT:axial", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CP:axial", val=np.nan, units=None)
        self.add_input("data:propeller:thrust:climb", val=np.nan, units="N")
        self.add_input("mission:design_mission:climb:atmosphere:density", val=np.nan, units="kg/m**3")
        self.add_output("data:propeller:speed:climb", units="rad/s")
        self.add_output("data:propeller:torque:climb", units="N*m")
        self.add_output("data:propeller:power:climb", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_cl = inputs["data:propeller:advance_ratio:climb"]
        alpha = inputs["mission:design_mission:climb:AoA"]
        F_pro_cl = inputs["data:propeller:thrust:climb"]
        rho_air = inputs["mission:design_mission:climb:atmosphere:density"]

        # C_t_axial = inputs["data:propeller:aerodynamics:CT:axial"]
        # C_p_axial = inputs["data:propeller:aerodynamics:CP:axial"]
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J_cl, alpha)

        Wpro_cl = PropellerPerfoModel.speed(F_pro_cl, Dpro, C_t, rho_air)
        Ppro_cl = PropellerPerfoModel.power(Wpro_cl, Dpro, C_p, rho_air)
        Qpro_cl = PropellerPerfoModel.torque(Ppro_cl, Wpro_cl)

        outputs["data:propeller:speed:climb"] = Wpro_cl
        outputs["data:propeller:torque:climb"] = Qpro_cl
        outputs["data:propeller:power:climb"] = Ppro_cl


class Cruise(om.ExplicitComponent):
    """
    Computes performances of the propeller for cruise
    """

    def setup(self):
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:cruise", val=np.nan, units=None)
        self.add_input("mission:design_mission:cruise:AoA", val=np.nan, units="rad")
        # self.add_input("data:propeller:aerodynamics:CT:incidence", val=np.nan, units=None)
        # self.add_input("data:propeller:aerodynamics:CP:incidence", val=np.nan, units=None)
        self.add_input("data:propeller:thrust:cruise", val=np.nan, units="N")
        self.add_input("mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3")
        self.add_output("data:propeller:speed:cruise", units="rad/s")
        self.add_output("data:propeller:torque:cruise", units="N*m")
        self.add_output("data:propeller:power:cruise", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_cr = inputs["data:propeller:advance_ratio:cruise"]
        alpha = inputs["mission:design_mission:cruise:AoA"]
        F_pro_cr = inputs["data:propeller:thrust:cruise"]
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]

        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J_cr, alpha)
        # C_t_inc = inputs["data:propeller:aerodynamics:CT:incidence"]
        # C_p_inc = inputs["data:propeller:aerodynamics:CP:incidence"]

        Wpro_cr = PropellerPerfoModel.speed(F_pro_cr, Dpro, C_t, rho_air)
        Ppro_cr = PropellerPerfoModel.power(Wpro_cr, Dpro, C_p, rho_air)
        Qpro_cr = PropellerPerfoModel.torque(Ppro_cr, Wpro_cr)

        outputs["data:propeller:speed:cruise"] = Wpro_cr
        outputs["data:propeller:torque:cruise"] = Qpro_cr
        outputs["data:propeller:power:cruise"] = Ppro_cr
