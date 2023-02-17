"""
Propeller surrogate models describing the propeller's aerodynamics performance.
"""

import numpy as np
# from scipy.optimize import fsolve


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
    >> c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J, alpha)
    """

    @staticmethod
    def aero_coefficients_static(beta,
                                 ct_model: np.array = np.array([4.27e-02, 1.44e-01]),
                                 cp_model: np.array = np.array([-1.48e-03, 9.72e-02])):
        """
        Compute the thrust and power coefficient in static conditions (zero advance ratio)

        Parameters
        ----------
        beta: pitch-to-diameter ratio (-)
        ct_model: np.array
                Array of model parameters for the static thrust coefficient.
                ct = ct_model[0] + ct_model[1] * beta
        cp_model: np.array
                Array of model parameters for the static power coefficient.
                cp = cp_model[0] + cp_model[1] * beta

        Returns
        -------
        c_t_static: static thrust coefficient (-)
        c_p_static: static power coefficient (-)
        """
        c_t_static = ct_model[0] + ct_model[1] * beta  # thrust coefficient in static
        c_p_static = cp_model[0] + cp_model[1] * beta  # power coefficient in static

        return c_t_static, c_p_static

    @staticmethod
    def aero_coefficients_axial(beta,
                                J,
                                ct_model: np.array = np.array(
                                    [0.02791, 0.11867, 0.27334, - 0.28852, - 0.06543, - 0.23504, 0.02104, 0.0, 0.0, 0.18677]),
                                cp_model: np.array = np.array(
                                    [0.01813, - 0.06218, 0.35712, - 0.23774, 0.00343, - 0.1235, 0.0, 0.07549, 0.0, 0.0])):
        """
        Compute the thrust and power coefficients in axial flight conditions (non-zero advance ratio, axial flow).

        Parameters
        ----------
        beta: pitch-to-diameter ratio (-)
        J: advance ratio V/nD (-)
        ct_model: np.array
                Array of model parameters for the axial thrust coefficient.
                ct = ct_model[0] + ct_model[1] * beta + ct_model[1] * beta**2 + ...
        cp_model: np.array
                Array of model parameters for the axial power coefficient.
                cp = cp_model[0] + cp_model[1] * beta + cp_model[1] * beta**2 + ...

        Returns
        -------
        c_t_axial: axial thrust coefficient (-)
        c_p_axial: axial power coefficient (-)
        """
        c_t_axial = (
            ct_model[0]
            + ct_model[1] * beta
            + ct_model[2] * beta**2
            + ct_model[3] * beta**3
            + ct_model[4] * J
            + ct_model[5] * J**2
            + ct_model[6] * J**3
            + ct_model[7] * beta * J
            + ct_model[8] * beta**2 * J
            + ct_model[9] * beta * J**2
        )
        c_p_axial = (
            cp_model[0]
            + cp_model[1] * beta
            + cp_model[2] * beta**2
            + cp_model[3] * beta**3
            + cp_model[4] * J
            + cp_model[5] * J**2
            + cp_model[6] * J**3
            + cp_model[7] * beta * J
            + cp_model[8] * beta**2 * J
            + cp_model[9] * beta * J**2
        )
        return max(1e-10, c_t_axial), max(1e-10, c_p_axial)

    @staticmethod
    def aero_coefficients_incidence(beta,
                                    J,
                                    alpha,
                                    n_blades: int = 2,
                                    chord_to_radius: float = 0.15,
                                    r_norm: float = 0.75,
                                    ct_model: np.array = np.array(
                                        [0.02791, 0.11867, 0.27334, - 0.28852, - 0.06543, - 0.23504, 0.02104, 0.0, 0.0, 0.18677,
                                         0.197, 1.094]),
                                    cp_model: np.array = np.array([
                                        0.01813, - 0.06218, 0.35712, - 0.23774, 0.00343, - 0.1235, 0.0, 0.07549, 0.0, 0.0,
                                        0.286, 0.993]),
                                    ):
        """
        Generalized model to compute the thrust and power coefficient in any operating conditions, i.e. non-zero
        advance ratio and non-axial flow. The non-axial flow correction is based on the analytical model provided by
        Y. Leng et al., "An Analytical Model For Propeller Aerodynamic Efforts At High Incidence", 2019.

        Parameters
        ----------
        beta: pitch-to-diameter ratio (-)
        J: advance ratio V/nD (-)
        alpha: rotor disk angle of attack (equals pi/2 if fully axial flow).
        n_blades: number of blades of the propeller
        chord_to_radius: chord to radius ratio at r_norm
        r_norm: position of representative section in percentage radius (usually 0.75)
        ct_model: np.array
                The first n-2 scalars are the model parameters for the thrust coefficient:
                ct = ct_model[0] + ct_model[1] * beta + ct_model[1] * beta**2 + ...
                The n-1 and n scalars are the model parameters for the calculation of the zero-thrust advance ratio:
                J0t = ct_model[-2] + ct_model[-1] * beta
        cp_model: np.array
                The first n-2 scalars are the model parameters for the power coefficient:
                cp = cp_model[0] + cp_model[1] * beta + cp_model[1] * beta**2 + ...
                The n-1 and n scalars are the model parameters for the calculation of the zero-power advance ratio:
                J0p = cp_model[-2] + cp_model[-1] * beta

        Returns
        -------
        c_t: thrust coefficient (-)
        c_p: power coefficient (-)
        """

        # Parameters at zero incidence propeller angle (axial flight)
        J_axial = J * np.sin(alpha)
        c_t_axial, c_p_axial = PropellerAerodynamicsModel.aero_coefficients_axial(beta,
                                                                                  J_axial,
                                                                                  ct_model=ct_model[:-2],
                                                                                  cp_model=cp_model[:-2])
        # Zero thrust advance ratios in axial flight

        # 1) With solver
        # func = lambda x: PropellerAerodynamicsModel.aero_coefficients_axial(beta,
        #                                                                     x,
        #                                                                     ct_model=ct_model,
        #                                                                     cp_model=cp_model)[0]
        # J_0t_axial = fsolve(func, [0.5])  # [-] solving for zero thrust advance ratio
        # func = lambda x: PropellerAerodynamicsModel.aero_coefficients_axial(beta,
        #                                                                     x,
        #                                                                     ct_model=ct_model,
        #                                                                     cp_model=cp_model)[1]
        # J_0p_axial = fsolve(func, [0.5])  # [-] solving for zero power advance ratio

        # 2) With provided model parameters
        J_0t_axial = ct_model[-2] + ct_model[-1] * beta
        J_0p_axial = cp_model[-2] + cp_model[-1] * beta

        # Solidity correction factor
        sigma = n_blades * chord_to_radius / np.pi
        pitch_angle = np.arctan(beta / 0.7 / np.pi)
        delta_t = delta_p = (
            3
            / 2
            * np.cos(pitch_angle)
            * (
                1
                + sigma
                / np.tan(pitch_angle)
                * (1 + np.sqrt(1 + 2 * np.tan(pitch_angle) / sigma))
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
        c_t = c_t_axial * eta_t
        c_p = c_p_axial * eta_p

        return max(1e-10, c_t), max(
            1e-10, c_p
        )  # set minimum value to avoid negative thrust or power
