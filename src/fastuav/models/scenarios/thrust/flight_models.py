"""
UAV flight models for thrust calculations - static methods definition.
(Unused in current version of FAST-UAV).
"""

import numpy as np
from scipy.constants import g
from scipy.optimize import brentq


class MultirotorFlightModel:
    """
    Flight model for multirotor.
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
    def get_angle_of_attack(m_uav, V, RoC, S_front, S_top, C_D, C_L0, rho_air):
        """
        Computes angle of attack to maintain flight path
        """
        theta = np.arcsin(RoC / V)  # [rad] flight path angle

        def func(x):
            drag = MultirotorFlightModel.get_drag(V, x, S_front, S_top, C_D, rho_air)  # [N] drag
            lift = MultirotorFlightModel.get_lift(V, x, S_top, C_L0, rho_air)  # [N] lift
            weight = m_uav * g  # [N] weight
            res = np.tan(abs(x - theta)) - (drag * np.cos(theta) + lift * np.sin(theta)) / (
                weight + drag * np.sin(theta) - lift * np.cos(theta)
            )  # [-] equilibrium residual
            return res

        alpha = brentq(func, 0, np.pi / 2)  # [rad] angle of attack
        return alpha

    @staticmethod
    def get_thrust(m_uav, V, RoC, alpha, S_front, S_top, C_D, C_L0, rho_air):
        """
        Computes thrust to maintain flight path
        """
        theta = np.arcsin(RoC / V)  # [rad] flight path angle
        weight = m_uav * g  # [N] weight
        lift = MultirotorFlightModel.get_lift(V, alpha, S_top, C_L0, rho_air)  # [N] lift
        drag = MultirotorFlightModel.get_drag(V, alpha, S_front, S_top, C_D, rho_air)  # [N] drag
        thrust = (
            (weight + drag * np.sin(theta) - lift * np.cos(theta)) ** 2
            + (drag * np.cos(theta) + lift * np.sin(theta)) ** 2
        ) ** (
            1 / 2
        )  # [N] total thrust requirement
        return thrust


class FixedwingFlightModel:
    """
    Flight model for fixed wings.
    """

    @staticmethod
    def get_angle_of_attack():
        """
        Computes angle of attack to maintain flight path
        """
        alpha = np.pi / 2  # [rad] Rotor disk Angle of Attack (assumption: axial flight TODO: estimate trim?)
        return alpha

    @staticmethod
    def get_thrust(m_uav, V, RoC, WS, K, CD0, rho_air):
        """
        Computes thrust to maintain flight path
        """
        q = 0.5 * rho_air * V ** 2  # [Pa] dynamic pressure
        TW = (
                RoC / V + q * CD0 / WS + K / q * WS
        )  # thrust-to-weight ratio in climb conditions [-]
        thrust = TW * m_uav * g  # [N] total thrust requirement
        return thrust
