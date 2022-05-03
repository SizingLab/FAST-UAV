"""
Propeller surrogate models describing the propeller's aerodynamics performance.
"""

import numpy as np
from fastuav.utils.constants import MR_PROPULSION, FW_PROPULSION


class PropellerAerodynamicsModel:
    """
    Surrogate models for the propeller aerodynamics performance.

    The thrust coefficient (Ct) and the power coefficient (Cp) are calculated from
        - the propeller's geometry (pitch-to-diameter ratio)
        - and the operating conditions (advance ratio and rotor disk angle of attack).
    These coefficients are defined as follows:
        Ct = Thrust / (air_density * rotation_speed ^ 2 * diameter ^ 4)
        Cp = Power / (air_density * rotation_speed ^ 3 * diameter ^ 5)

    The functions are defined as static method that can be called from anywhere without instantiating the class:
    >> C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J, alpha)

    It is also possible to add a deviation to the model's outputs.
    To do so, simply setup this model in an OpenMDAO group using the "add_model_deviation" function defined
    in "fastuav.utils.uncertainty".
    """

    # Model deviations (for uncertainty purpose only)
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
    def aero_coefficients_static(beta, propulsion_id: str = None):
        """
        Compute the thrust and power coefficient in static conditions (zero advance ratio)
        """
        C_t_static, C_p_static = .0, .0
        if propulsion_id == MR_PROPULSION:  # Multi-rotor propellers (APC)
            C_t_static = 4.27e-02 + 1.44e-01 * beta  # thrust coefficient in static
            C_p_static = -1.48e-03 + 9.72e-02 * beta  # power coefficient in static

        elif propulsion_id == FW_PROPULSION:  # TODO: add static performances for thin electric APC propellers
            C_t_static = 4.27e-02 + 1.44e-01 * beta  # thrust coefficient in static
            C_p_static = -1.48e-03 + 9.72e-02 * beta  # power coefficient in static

        # Deviations (for uncertainty purpose)
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:static:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:static:rel"
        ]
        C_t_static = C_t_static * (1 + eps) + delta
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:static:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:static:rel"
        ]
        C_p_static = C_p_static * (1 + eps) + delta

        return C_t_static, C_p_static

    @staticmethod
    def aero_coefficients_axial(beta, J, propulsion_id: str = None):
        """
        Computes the thrust and power coefficients in axial flight conditions (non-zero advance ratio, axial flow).
        """
        C_t_axial, C_p_axial = .0, .0

        if propulsion_id == MR_PROPULSION:  # Multi-rotor propellers (APC)
            C_t_axial = (
                0.02791
               - 0.06543 * J
               + 0.11867 * beta
               + 0.27334 * beta**2
               - 0.28852 * beta**3
               + 0.02104 * J**3
               - 0.23504 * J**2
               + 0.18677 * beta * J**2
            )
            C_p_axial = (
               0.01813
               - 0.06218 * beta
               + 0.00343 * J
               + 0.35712 * beta**2
               - 0.23774 * beta**3
               + 0.07549 * beta * J
               - 0.1235 * J**2
            )

        elif propulsion_id == FW_PROPULSION:  # Thin Electric propellers (APC pareto high thrust, high efficiency)
            C_t_axial = 0.09613 - 0.26688 * J + 0.37102 * J * beta - 0.15240 * beta * J**2
            C_p_axial = (
                0.00440
                - 0.03854 * beta**2
                - 0.08185 * J**3
                + 0.12568 * beta**2 * J
                - 0.03864 * J
                + 0.08432 * beta
            )

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

        return max(1e-10, C_t_axial), max(1e-10, C_p_axial)

    @staticmethod
    def aero_coefficients_incidence(beta, J, alpha, N_blades: int = 2, chord_to_radius=0.15, r_norm=0.75,
                                    propulsion_id: str = None):
        """
        Generalized model to compute the thrust and power coefficient in any operating conditions, i.e. non-zero
        advance ratio and non-axial flow. The non-axial flow correction is based on the analytical model provided by
        Y. Leng et al, "An Analytical Model For Propeller Aerodynamic Efforts At High Incidence", 2019.
        Parameter alpha is the rotor disk angle of attack (equals pi/2 if fully axial flow).
        """

        # Parameters at zero incidence propeller angle (axial flight)
        J_axial = J * np.sin(alpha)
        C_t_axial, C_p_axial = PropellerAerodynamicsModel.aero_coefficients_axial(beta,
                                                                                  J_axial,
                                                                                  propulsion_id=propulsion_id)
        # Zero_thrust advance ratios in axial flight
        J_0t_axial, J_0p_axial = .0, .0
        if propulsion_id == MR_PROPULSION:  # Multi-rotor propellers (APC)
            J_0t_axial = 0.197 + 1.094 * beta  # zero-thrust advance ratio
            J_0p_axial = 0.286 + 0.993 * beta  # zero-power advance ratio
        elif propulsion_id == FW_PROPULSION:  # APC thin electric propellers (APC pareto high thrust, high efficiency)
            J_0t_axial = 0.21272 + 1.00040 * beta  # zero-thrust advance ratio
            J_0p_axial = 0.20710 + 1.03642 * beta  # zero-power advance ratio

        # Solidity correction factor
        sigma = N_blades * chord_to_radius / np.pi
        delta_t = delta_p = (
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
            * delta_t
        )
        eta_p = (
            1
            + (J * np.cos(alpha) / np.pi / r_norm) ** 2
            / 2
            / (1 - J / J_0p_axial * np.sin(alpha))
            * delta_p
        )

        # thrust and power coefficients
        C_t = C_t_axial * eta_t
        C_p = C_p_axial * eta_p

        # Deviations (for uncertainty purpose)
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:incidence:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CT:incidence:rel"
        ]
        C_t = C_t * (1 + eps) + delta
        delta = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:incidence:abs"
        ]
        eps = PropellerAerodynamicsModel.uncertain_parameters[
            "uncertainty:propeller:aerodynamics:CP:incidence:rel"
        ]
        C_p = C_p * (1 + eps) + delta

        return max(1e-10, C_t), max(
            1e-10, C_p
        )  # set minimum value to avoid negative thrust or power