"""
Propeller performances
"""
import openmdao.api as om
import numpy as np
from scipy.optimize import fsolve
from fastuav.models.propulsion.propeller.aerodynamics.surrogate_models import PropellerAerodynamicsModel
from stdatm import AtmosphereSI


class PropellerPerformanceModel:
    """
    Propeller model for performances calculation
    """

    @staticmethod
    def speed(F_pro, D_pro, c_t, rho_air):
        n_pro = (
            (F_pro / (c_t * rho_air * D_pro**4)) ** 0.5 if (c_t and rho_air and D_pro) else 0.0
        )  # [Hz] Propeller speed
        W_pro = n_pro * 2 * np.pi  # [rad/s] Propeller speed
        return W_pro

    @staticmethod
    def power(W_pro, D_pro, c_p, rho_air):
        P_pro = c_p * rho_air * (W_pro / (2 * np.pi)) ** 3 * D_pro**5  # [W] Propeller power
        return P_pro

    @staticmethod
    def torque(P_pro, W_pro):
        Q_pro = P_pro / W_pro if W_pro else 0.0  # [N.m] Propeller torque
        return Q_pro

    @staticmethod
    def induced_velocity(F_pro, D_pro, V_inf, alpha, rho_air):
        """
        Computes the induced velocity from Glauert's model
        """
        func = lambda x: x - F_pro / (2 * rho_air * np.pi * (D_pro / 2) ** 2) / (
                    (V_inf * np.cos(alpha)) ** 2 + (V_inf * np.sin(alpha) + x) ** 2) ** (1 / 2)
        v_i = fsolve(func, x0=1)[0]
        return v_i

    @staticmethod
    def efficiency(F_pro, W_pro, D_pro, c_p, c_t, V_inf, alpha, rho_air):
        """
        Computes the propeller efficiency.
        Valid in any condition (hover and forward flight).
        """
        v_i = PropellerPerformanceModel.induced_velocity(F_pro, D_pro, V_inf, alpha, rho_air)
        try:
            eta = (V_inf * np.sin(alpha) + v_i) / (W_pro / (2 * np.pi) * D_pro) * c_t / c_p
        except (ZeroDivisionError, ValueError):
            eta = 0.0
        return eta

    @staticmethod
    def efficiency_high_speed(c_p, c_t, J):
        """
        Computes the propeller efficiency with the high speed (or no incidence angle) approximation.
        Valid for axial flight at non-zero flight speed.
        """
        eta_pro = c_t / c_p * J  # efficiency
        return eta_pro

    @staticmethod
    def figure_of_merit(c_p, c_t):
        """
        Computes the figure of merit, that is, the efficiency in hover flight.
        """
        fom = c_t ** (3 / 2) / c_p  # figure of merit
        return fom


class PropellerPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the propeller
    """

    def setup(self):
        self.add_subsystem("takeoff",
                           PropellerPerformance(scenario="takeoff"),
                           promotes=["*"])
        self.add_subsystem("hover",
                           PropellerPerformance(scenario="hover"),
                           promotes=["*"])
        self.add_subsystem("climb",
                           PropellerPerformance(scenario="climb"),
                           promotes=["*"])
        self.add_subsystem("cruise",
                           PropellerPerformance(scenario="cruise"),
                           promotes=["*"])


class PropellerPerformance(om.ExplicitComponent):
    """
    Computes performances of the propeller for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default="cruise", values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:beta", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:static:polynomial", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:static:polynomial", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:dynamic:polynomial", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:dynamic:polynomial", shape_by_conn=True, val=np.nan, units=None)
        if scenario == "takeoff":
            self.add_input("mission:sizing:main_route:takeoff:altitude", val=0.0, units="m")
        elif scenario == "hover":
            self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")  # conservative assumption
        else:
            self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")  # conservative assumption
            self.add_input("mission:sizing:main_route:%s:speed" % scenario, val=np.nan, units="m/s")
            self.add_input("optimization:variables:propulsion:propeller:advance_ratio:%s" % scenario, val=np.nan, units=None)
            self.add_input("data:propulsion:propeller:AoA:%s" % scenario, val=np.nan, units="rad")
            self.add_output("data:propulsion:propeller:efficiency:%s" % scenario, units=None)
        self.add_input("mission:sizing:dISA", val=np.nan, units="K")
        self.add_input("data:propulsion:propeller:thrust:%s" % scenario, val=np.nan, units="N")
        self.add_output("data:propulsion:propeller:speed:%s" % scenario, units="rad/s")
        self.add_output("data:propulsion:propeller:torque:%s" % scenario, units="N*m")
        self.add_output("data:propulsion:propeller:power:%s" % scenario, units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        D_pro = inputs["data:propulsion:propeller:diameter"]
        beta = inputs["data:propulsion:propeller:beta"]
        ct_model_sta = inputs["data:propulsion:propeller:Ct:static:polynomial"]
        cp_model_sta = inputs["data:propulsion:propeller:Cp:static:polynomial"]
        ct_model_dyn = inputs["data:propulsion:propeller:Ct:dynamic:polynomial"]
        cp_model_dyn = inputs["data:propulsion:propeller:Cp:dynamic:polynomial"]
        F_pro = inputs["data:propulsion:propeller:thrust:%s" % scenario]
        dISA = inputs["mission:sizing:dISA"]
        alpha = 0.0

        if scenario == "takeoff":
            altitude = inputs["mission:sizing:main_route:takeoff:altitude"]
            c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_static(beta,
                                                                           ct_model=ct_model_sta,
                                                                           cp_model=cp_model_sta)

        elif scenario == "hover":
            altitude = inputs["mission:sizing:main_route:cruise:altitude"]
            c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_static(beta,
                                                                           ct_model=ct_model_sta,
                                                                           cp_model=cp_model_sta)

        else:
            altitude = inputs["mission:sizing:main_route:cruise:altitude"]
            J = inputs["optimization:variables:propulsion:propeller:advance_ratio:%s" % scenario]
            alpha = inputs["data:propulsion:propeller:AoA:%s" % scenario]
            c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta,
                                                                              J,
                                                                              alpha,
                                                                              ct_model=ct_model_dyn,
                                                                              cp_model=cp_model_dyn)

        rho_air = AtmosphereSI(altitude, dISA).density  # [kg/m3] Air density
        W_pro = PropellerPerformanceModel.speed(F_pro, D_pro, c_t, rho_air)
        P_pro = PropellerPerformanceModel.power(W_pro, D_pro, c_p, rho_air)
        Q_pro = PropellerPerformanceModel.torque(P_pro, W_pro)

        if scenario in ["climb", "cruise"]:
            V_inf = inputs["mission:sizing:main_route:%s:speed" % scenario]
            eta = PropellerPerformanceModel.efficiency(F_pro, W_pro, D_pro, c_p, c_t, V_inf, alpha, rho_air)
            outputs["data:propulsion:propeller:efficiency:%s" % scenario] = eta

        outputs["data:propulsion:propeller:speed:%s" % scenario] = W_pro
        outputs["data:propulsion:propeller:torque:%s" % scenario] = Q_pro
        outputs["data:propulsion:propeller:power:%s" % scenario] = P_pro
