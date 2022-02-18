"""
Estimation models for the propeller
"""
import openmdao.api as om
import numpy as np
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from models.Uncertainty.uncertainty import (
    add_subsystem_with_deviation,
    add_model_deviation,
)


class PropellerEstimationModels(om.Group):
    """
    Group containing the estimation models for the propeller.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_model_deviation(
            self,
            "aerodynamics_model_deviation",
            PropellerAerodynamicsModel,
            uncertain_parameters=[
                "propeller:aerodynamics:CT:static",
                "propeller:aerodynamics:CP:static",
                "propeller:aerodynamics:CT:axial",
                "propeller:aerodynamics:CP:axial",
                "propeller:aerodynamics:CT:incidence",
                "propeller:aerodynamics:CP:incidence",
            ],
        )
        self.add_subsystem("aerodynamics", Aerodynamics(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "diameter",
            Diameter(),
            uncertain_outputs={"data:propeller:geometry:diameter:estimated": "m"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:propeller:mass:estimated": "kg"},
        )


class PropellerAerodynamicsModel:
    """
    Aerodynamics model for the propeller
    """

    # Model deviations (for uncertainty purpose)
    init_uncertain_parameters = {
        "uncertainty:propeller:aerodynamics:CT:static:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CP:static:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CT:axial:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CP:axial:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CT:incidence:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CP:incidence:abs": 0.0,
        "uncertainty:propeller:aerodynamics:CT:static:rel": 0.0,
        "uncertainty:propeller:aerodynamics:CP:static:rel": 0.0,
        "uncertainty:propeller:aerodynamics:CT:axial:rel": 0.0,
        "uncertainty:propeller:aerodynamics:CP:axial:rel": 0.0,
        "uncertainty:propeller:aerodynamics:CT:incidence:rel": 0.0,
        "uncertainty:propeller:aerodynamics:CP:incidence:rel": 0.0,
    }
    uncertain_parameters = init_uncertain_parameters.copy()

    @staticmethod
    def aero_coefficients_static(beta):
        C_t_sta = 4.27e-02 + 1.44e-01 * beta  # Thrust coef with T=C_T.rho.n^2.D^4
        C_p_sta = -1.48e-03 + 9.72e-02 * beta  # Power coef with P=C_p.rho.n^3.D^5
        # Deviations (for uncertainty purpose)
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:static:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:static:rel"
        ]
        C_t_sta = C_t_sta * (1 + eps) + delta
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:static:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:static:rel"
        ]
        C_p_sta = C_p_sta * (1 + eps) + delta
        return C_t_sta, C_p_sta

    @staticmethod
    def aero_coefficients_axial(beta, J):
        C_t_axial = (
            0.02791
            - 0.06543 * J
            + 0.11867 * beta
            + 0.27334 * beta**2
            - 0.28852 * beta**3
            + 0.02104 * J**3
            - 0.23504 * J**2
            + 0.18677 * beta * J**2
        )  # thrust coef for APC props in dynamics
        C_p_axial = (
            0.01813
            - 0.06218 * beta
            + 0.00343 * J
            + 0.35712 * beta**2
            - 0.23774 * beta**3
            + 0.07549 * beta * J
            - 0.1235 * J**2
        )  # power coef for APC props in dynamics
        # Deviations (for uncertainty purpose)
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:axial:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:axial:rel"
        ]
        C_t_axial = C_t_axial * (1 + eps) + delta
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:axial:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:axial:rel"
        ]
        C_p_axial = C_p_axial * (1 + eps) + delta
        return C_t_axial, C_p_axial

    @staticmethod
    def aero_coefficients_incidence(
        beta, J, alpha, N_blades=2, chord_to_radius=0.15, r_norm=0.75
    ):
        """
        Incidence power coefficient (Y. Leng et al. model)
        """

        # Parameters at zero incidence propeller angle (vertical flight)
        J_axial = J * np.sin(alpha)
        C_t_axial, C_p_axial = PropellerAerodynamicsModel.aero_coefficients_axial(
            beta, J_axial
        )
        J_0t_axial = 0.197 + 1.094 * beta
        J_0p_axial = 0.286 + 0.993 * beta

        # solidity correction factor
        sigma = N_blades * chord_to_radius / np.pi
        delta = (
            3
            / 2
            * np.cos(beta)
            * (
                1
                + sigma
                / np.tan(beta)
                * (1 + np.sqrt(1 + 2 * np.tan(beta) / sigma))
                * (1 - np.sin(alpha))
            )
        )

        # incidence ratios
        eta_t = (
            1
            + (J * np.cos(alpha) / np.pi / r_norm) ** 2
            / 2
            / (1 - J / J_0t_axial * np.sin(alpha))
            * delta
        )
        eta_p = (
            1
            + (J * np.cos(alpha) / np.pi / r_norm) ** 2
            / 2
            / (1 - J / J_0p_axial * np.sin(alpha))
            * delta
        )

        # thrust and power coefficients
        C_t_inc = C_t_axial * eta_t
        C_p_inc = C_p_axial * eta_p

        # Deviations (for uncertainty purpose)
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:incidence:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:incidence:rel"
        ]
        C_t_inc = C_t_inc * (1 + eps) + delta
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:incidence:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:incidence:rel"
        ]
        C_p_inc = C_p_inc * (1 + eps) + delta

        return max(1e-10, C_t_inc), max(
            1e-10, C_p_inc
        )  # set minimum value to avoid negative thrust or power


@ValidityDomainChecker(
    {
        "data:propeller:aerodynamics:CT:incidence:estimated": (1e-9, None),
        "data:propeller:aerodynamics:CP:incidence:estimated": (1e-9, None),
    },
)
class Aerodynamics(om.ExplicitComponent):
    """
    Computes aerodynamics coefficients of the propeller
    """

    def setup(self):
        self.add_input("data:propeller:geometry:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:climb", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:forward", val=np.nan, units=None)
        self.add_input("mission:design_mission:forward:angle", val=np.nan, units="rad")
        self.add_output("data:propeller:aerodynamics:CT:static:estimated", units=None)
        self.add_output("data:propeller:aerodynamics:CP:static:estimated", units=None)
        self.add_output("data:propeller:aerodynamics:CT:axial:estimated", units=None)
        self.add_output("data:propeller:aerodynamics:CP:axial:estimated", units=None)
        self.add_output(
            "data:propeller:aerodynamics:CT:incidence:estimated", units=None
        )
        self.add_output(
            "data:propeller:aerodynamics:CP:incidence:estimated", units=None
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        beta = inputs["data:propeller:geometry:beta:estimated"]
        J_cl = inputs["data:propeller:advance_ratio:climb"]
        J_ff = inputs["data:propeller:advance_ratio:forward"]
        alpha = inputs["mission:design_mission:forward:angle"]

        C_t_sta, C_p_sta = PropellerAerodynamicsModel.aero_coefficients_static(beta)
        C_t_axial, C_p_axial = PropellerAerodynamicsModel.aero_coefficients_axial(
            beta, J_cl
        )
        C_t_inc, C_p_inc = PropellerAerodynamicsModel.aero_coefficients_incidence(
            beta, J_ff, alpha
        )

        outputs["data:propeller:aerodynamics:CT:static:estimated"] = C_t_sta
        outputs["data:propeller:aerodynamics:CP:static:estimated"] = C_p_sta
        outputs["data:propeller:aerodynamics:CT:axial:estimated"] = C_t_axial
        outputs["data:propeller:aerodynamics:CP:axial:estimated"] = C_p_axial
        outputs["data:propeller:aerodynamics:CT:incidence:estimated"] = C_t_inc
        outputs["data:propeller:aerodynamics:CP:incidence:estimated"] = C_p_inc


class Diameter(om.ExplicitComponent):
    """
    Computes propeller diameter
    """

    def setup(self):
        self.add_input("data:propeller:thrust:max", val=np.nan, units="N")
        self.add_input(
            "mission:design_mission:air_density", val=np.nan, units="kg/m**3"
        )
        self.add_input(
            "data:propeller:aerodynamics:CT:static:estimated", val=np.nan, units=None
        )
        self.add_input("data:propeller:ND:takeoff", val=np.nan, units="m/s")
        self.add_output("data:propeller:geometry:diameter:estimated", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_to = inputs["data:propeller:thrust:max"]
        C_t_sta = inputs["data:propeller:aerodynamics:CT:static:estimated"]
        rho_air = inputs["mission:design_mission:air_density"]
        ND = inputs["data:propeller:ND:takeoff"]

        Dpro = (
            F_pro_to / (C_t_sta * rho_air * ND**2)
        ) ** 0.5  # [m] Propeller diameter

        outputs["data:propeller:geometry:diameter:estimated"] = Dpro


class Weight(om.ExplicitComponent):
    """
    Computes propeller weight
    """

    def setup(self):
        self.add_input(
            "data:propeller:geometry:diameter:estimated", val=np.nan, units="m"
        )
        self.add_input("data:propeller:reference:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:reference:mass", val=np.nan, units="kg")
        self.add_output("data:propeller:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter:estimated"]
        Dpro_ref = inputs["data:propeller:reference:diameter"]
        Mpro_ref = inputs["data:propeller:reference:mass"]

        Mpro = Mpro_ref * (Dpro / Dpro_ref) ** 3  # [kg] Propeller mass

        outputs["data:propeller:mass:estimated"] = Mpro
