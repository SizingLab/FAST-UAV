"""
Propeller performances
"""
import openmdao.api as om
import numpy as np
from fastuav.models.propulsion.propeller.aerodynamics.surrogate_models import PropellerAerodynamicsModel
from stdatm import AtmosphereSI
from fastuav.utils.constants import PROPULSION_ID_LIST


class PropellerPerformanceModel:
    """
    Propeller model for performances calculation
    """

    @staticmethod
    def speed(F_pro, D_pro, C_t, rho_air):
        n_pro = (
            (F_pro / (C_t * rho_air * D_pro**4)) ** 0.5 if (C_t and rho_air and D_pro) else 0.0
        )  # [Hz] Propeller speed
        W_pro = n_pro * 2 * np.pi  # [rad/s] Propeller speed
        return W_pro

    @staticmethod
    def power(W_pro, D_pro, C_p, rho_air):
        P_pro = C_p * rho_air * (W_pro / (2 * np.pi)) ** 3 * D_pro**5  # [W] Propeller power
        return P_pro

    @staticmethod
    def torque(P_pro, W_pro):
        Q_pro = P_pro / W_pro if W_pro else 0.0  # [N.m] Propeller torque
        return Q_pro


class PropellerPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the propeller
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_subsystem("takeoff",
                           PropellerPerformance(scenario="takeoff", propulsion_id=propulsion_id),
                           promotes=["*"])
        self.add_subsystem("hover",
                           PropellerPerformance(scenario="hover", propulsion_id=propulsion_id),
                           promotes=["*"])
        self.add_subsystem("climb",
                           PropellerPerformance(scenario="climb", propulsion_id=propulsion_id),
                           promotes=["*"])
        self.add_subsystem("cruise",
                           PropellerPerformance(scenario="cruise", propulsion_id=propulsion_id),
                           promotes=["*"])


class PropellerPerformance(om.ExplicitComponent):
    """
    Computes performances of the propeller for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default="cruise", values=["takeoff", "climb", "hover", "cruise"])
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:beta", val=np.nan, units=None)
        if scenario == "takeoff":
            self.add_input("data:scenarios:takeoff:altitude", val=np.nan, units="m")
        elif scenario == "hover":
            self.add_input("data:scenarios:cruise:altitude", val=np.nan, units="m")  # conservative assumption
        else:
            self.add_input("data:scenarios:cruise:altitude", val=np.nan, units="m")  # conservative assumption
            self.add_input("data:propulsion:propeller:advance_ratio:%s" % scenario, val=np.nan, units=None)
            self.add_input("data:propulsion:propeller:AoA:%s" % scenario, val=np.nan, units="rad")
        self.add_input("data:scenarios:dISA", val=np.nan, units="K")
        self.add_input("data:propulsion:propeller:thrust:%s" % scenario, val=np.nan, units="N")
        self.add_output("data:propulsion:propeller:speed:%s" % scenario, units="rad/s")
        self.add_output("data:propulsion:propeller:torque:%s" % scenario, units="N*m")
        self.add_output("data:propulsion:propeller:power:%s" % scenario, units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        propulsion_id = self.options["propulsion_id"]
        Dpro = inputs["data:propulsion:propeller:diameter"]
        beta = inputs["data:propulsion:propeller:beta"]
        F_pro = inputs["data:propulsion:propeller:thrust:%s" % scenario]
        dISA = inputs["data:scenarios:dISA"]

        if scenario == "takeoff":
            altitude = inputs["data:scenarios:takeoff:altitude"]
            C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta, propulsion_id=propulsion_id)
        elif scenario == "hover":
            altitude = inputs["data:scenarios:cruise:altitude"]
            C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta, propulsion_id=propulsion_id)
        else:
            altitude = inputs["data:scenarios:cruise:altitude"]
            J = inputs["data:propulsion:propeller:advance_ratio:%s" % scenario]
            alpha = inputs["data:propulsion:propeller:AoA:%s" % scenario]
            C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J, alpha,
                                                                              propulsion_id=propulsion_id)

        rho_air = AtmosphereSI(altitude, dISA).density  # [kg/m3] Air density
        W_pro = PropellerPerformanceModel.speed(F_pro, Dpro, C_t, rho_air)
        P_pro = PropellerPerformanceModel.power(W_pro, Dpro, C_p, rho_air)
        Q_pro = PropellerPerformanceModel.torque(P_pro, W_pro)

        outputs["data:propulsion:propeller:speed:%s" % scenario] = W_pro
        outputs["data:propulsion:propeller:torque:%s" % scenario] = Q_pro
        outputs["data:propulsion:propeller:power:%s" % scenario] = P_pro
