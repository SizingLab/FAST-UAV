"""
UAV flight model - static methods definition.
"""

import numpy as np
from scipy.constants import g
from scipy.optimize import brentq


class MultirotorFlightModel:
    """
    Flight model (aerodynamics and forces)
    """

    @staticmethod
    def get_drag(V, alpha, S_front, S_top, C_D, rho_air):
        """
        Computes body drag from drag coefficient, reference surfaces and velocity
        """
        S_ref = S_top * np.sin(alpha) + S_front * np.cos(alpha)  # [m2] reference area
        drag = 0.5 * rho_air * C_D * S_ref * V**2  # [N] drag
        return drag

    @staticmethod
    def get_lift(V, alpha, S_top, C_L0, rho_air):
        """
        Computes body lift from lift coefficient, reference surface and velocity.
        Derived from a flat plate model.
        """
        S_ref = S_top * np.sin(alpha)  # [m2] reference area
        C_L = C_L0 * np.sin(
            -2 * alpha
        )  # [-] flat plate model (minus in front of alpha because of angle definition)
        lift = 0.5 * rho_air * C_L * S_ref * V**2  # [N] lift
        return lift

    @staticmethod
    def get_AoA(Mtotal, V, theta, S_front, S_top, C_D, C_L0, rho_air):
        """
        Computes required angle of attack to maintain flight path
        """

        def func(x):
            drag = MultirotorFlightModel.get_drag(V, x, S_front, S_top, C_D, rho_air)  # [N] drag
            lift = MultirotorFlightModel.get_lift(V, x, S_top, C_L0, rho_air)  # [N] lift
            weight = Mtotal * g  # [N] weight
            res = np.tan(abs(x - theta)) - (drag * np.cos(theta) + lift * np.sin(theta)) / (
                weight + drag * np.sin(theta) - lift * np.cos(theta)
            )  # [-] equilibrium residual
            return res

        alpha = brentq(func, 0, np.pi / 2)  # [rad] angle of attack
        return alpha

    @staticmethod
    def get_thrust(Mtotal, V, theta, S_front, S_top, C_D, C_L0, rho_air):
        """
        Computes required thrust to maintain flight path.
        Generic function for any case.
        """
        alpha = MultirotorFlightModel.get_AoA(
            Mtotal, V, theta, S_front, S_top, C_D, C_L0, rho_air
        )  # [rad] angle of attack
        weight = Mtotal * g  # [N] weight
        lift = MultirotorFlightModel.get_lift(V, alpha, S_top, C_L0, rho_air)  # [N] lift
        drag = MultirotorFlightModel.get_drag(V, alpha, S_front, S_top, C_D, rho_air)  # [N] drag
        thrust = (
            (weight + drag * np.sin(theta) - lift * np.cos(theta)) ** 2
            + (drag * np.cos(theta) + lift * np.sin(theta)) ** 2
        ) ** (
            1 / 2
        )  # [N] thrust requirement
        return thrust, alpha
