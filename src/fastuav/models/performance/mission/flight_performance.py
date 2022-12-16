"""
This module provides the flight performance model for the UAV,
based on the individual models defined in each discipline.
"""

import numpy as np
from scipy.constants import g
from scipy.optimize import brentq
from stdatm import AtmosphereSI
from fastuav.models.scenarios.thrust.flight_models import MultirotorFlightModel, FixedwingFlightModel
from fastuav.models.propulsion.energy.battery.performance_analysis import BatteryPerformanceModel
from fastuav.models.propulsion.propeller.performance_analysis import PropellerPerformanceModel
from fastuav.models.propulsion.propeller.aerodynamics.surrogate_models import PropellerAerodynamicsModel
from fastuav.models.propulsion.motor.performance_analysis import MotorPerformanceModel
from fastuav.models.propulsion.esc.performance_analysis import ESCPerformanceModel
from fastuav.utils.constants import MR_PROPULSION, FW_PROPULSION


class FlightPerformanceModel:
    """
    Flight performance model of UAV.
    Calculates and returns the flight parameters for a given flight scenario.
    """

    def __init__(self,
                 uav_model: str,
                 uav_mass: float,
                 airspeed: float,
                 climb_rate: float = 0.0,
                 altitude: float = 0.0,
                 delta_isa: float = 0.0
                 ):
        """
        :param uav_model: fixed wing or multirotor or hybrid
        :param uav_mass: mass of the uav (kg)
        :param altitude: altitude (m) of the flight point
        :param airspeed: true airspeed at the flight point
        """

        # inputs
        self.uav_model = uav_model
        self.uav_mass = uav_mass
        self.airspeed = airspeed
        self.climb_rate = climb_rate
        self.altitude = altitude
        self.delta_isa = delta_isa

        # aerodynamics parameters
        self.fw_induced_drag_constant = None
        self.fw_parasitic_drag_coef = None
        self.mr_parasitic_drag_coef = None
        self.mr_lift_coef = 0.0

        # geometry parameters
        self.mr_area_front = None
        self.mr_area_top = None
        self.wing_area = None

        # propulsion system parameters
        self.battery_voltage = None
        self.esc_efficiency = None
        self.gearbox_ratio = 1.0
        self.motor_speed_constant = None
        self.motor_torque_friction = None
        self.motor_resistance = None
        self.propeller_number = None
        self.propeller_diameter = None
        self.propeller_beta = None
        self.propeller_ct_model = None
        self.propeller_cp_model = None
        self.payload_power = 0.0

        # performance outputs
        self._wing_loading = None
        self._thrust_per_propeller = None
        self._propeller_angle_of_attack = None
        self._battery_power = None
        self._esc_power = None
        self._motor_torque = None
        self._motor_speed = None
        self._motor_current = None
        self._motor_voltage = None
        self._motor_power = None
        self._advance_ratio = None
        self._propeller_ct = None
        self._propeller_cp = None
        self._propeller_speed = None
        self._propeller_power = None
        self._propeller_torque = None
        self._air_density = None

    @property
    def wing_loading(self) -> float:
        """Wing load in Pa."""
        if self._wing_loading is None:
            self._wing_loading = self.uav_mass * g / self.wing_area
        return self._wing_loading

    @property
    def thrust_per_propeller(self) -> float:
        """Thrust per propeller in N."""
        if self._thrust_per_propeller is None:
            if self.uav_model == MR_PROPULSION:
                if self.mr_area_front is not None and self.mr_area_top is not None and self.mr_parasitic_drag_coef is not None:
                    thrust = MultirotorFlightModel.get_thrust(
                        self.uav_mass,
                        self.airspeed,
                        self.climb_rate,
                        self.propeller_angle_of_attack,
                        self.mr_area_front,
                        self.mr_area_top,
                        self.mr_parasitic_drag_coef,
                        self.mr_lift_coef,
                        self.air_density,
                    )
                    self._thrust_per_propeller = thrust / self.propeller_number
            elif self.uav_model == FW_PROPULSION:
                if self.fw_induced_drag_constant is not None and self.fw_parasitic_drag_coef is not None:
                    thrust = FixedwingFlightModel.get_thrust(
                        self.uav_mass,
                        self.airspeed,
                        self.climb_rate,
                        self.wing_loading,
                        self.fw_induced_drag_constant,
                        self.fw_parasitic_drag_coef,
                        self.air_density,
                    )
                    self._thrust_per_propeller = thrust / self.propeller_number
        return self._thrust_per_propeller

    @property
    def propeller_angle_of_attack(self) -> float:
        """Propeller disk angle of attack in rad."""
        if self._propeller_angle_of_attack is None:
            if self.uav_model == MR_PROPULSION:
                if self.mr_area_front is not None and self.mr_area_top is not None and self.mr_parasitic_drag_coef is not None:
                    self._propeller_angle_of_attack = MultirotorFlightModel.get_angle_of_attack(
                        self.uav_mass,
                        self.airspeed,
                        self.climb_rate,
                        self.mr_area_front,
                        self.mr_area_top,
                        self.mr_parasitic_drag_coef,
                        self.mr_lift_coef,
                        self.air_density,
                    )
            elif self.uav_model == FW_PROPULSION:
                self._propeller_angle_of_attack = FixedwingFlightModel.get_angle_of_attack()
        return self._propeller_angle_of_attack

    @property
    def battery_power(self) -> float:
        """Battery power in W."""
        if self._battery_power is None and self.propeller_number is not None:
            self._battery_power = BatteryPerformanceModel.power(self.motor_power,
                                                                self.propeller_number,
                                                                self.esc_efficiency,
                                                                self.payload_power)
        return self._battery_power

    @property
    def motor_torque(self) -> float:
        """Motor torque in N*m."""
        if self._motor_torque is None:
            self._motor_torque = MotorPerformanceModel.torque(self.propeller_torque,
                                                              self.gearbox_ratio)
        return self._motor_torque

    @property
    def motor_speed(self) -> float:
        """Motor rotation speed in rad/s."""
        if self._motor_speed is None:
            self._motor_speed = MotorPerformanceModel.speed(self.propeller_speed,
                                                            self.gearbox_ratio)
        return self._motor_speed

    @property
    def motor_current(self) -> float:
        """Motor current in A."""
        if self._motor_current is None and self.motor_torque_friction is not None and self.motor_speed_constant is not None:
            self._motor_current = MotorPerformanceModel.current(self.motor_torque,
                                                                self.motor_torque_friction,
                                                                self.motor_speed_constant)
        return self._motor_current

    @property
    def motor_voltage(self) -> float:
        """Motor voltage in V."""
        if self._motor_voltage is None and self.motor_resistance is not None and self.motor_speed_constant is not None:
            self._motor_voltage = MotorPerformanceModel.voltage(self.motor_current,
                                                                self.motor_speed,
                                                                self.motor_resistance,
                                                                self.motor_speed_constant)
        return self._motor_voltage

    @property
    def motor_power(self) -> float:
        """Motor power in W."""
        if self._motor_power is None:
            self._motor_power = MotorPerformanceModel.power(self.motor_voltage,
                                                            self.motor_current)
        return self._motor_power

    @property
    def advance_ratio(self) -> float:
        """
        Advance ratio (J) of the propeller.
        It is obtained by solving J = V/nD with
            nD = (thrust / (air_density * propeller_diameter**2 * propeller_ct)) ** (1/2)
        with the thrust coefficient of the propeller being dependent on the advance ratio.
        """
        if self._advance_ratio is None and self.propeller_diameter is not None and self.propeller_beta is not None \
                and self.thrust_per_propeller > 0:

            def func(x):
                propeller_ct, _ = PropellerAerodynamicsModel.aero_coefficients_incidence(self.propeller_beta,
                                                                                         x,
                                                                                         self.propeller_angle_of_attack,
                                                                                         ct_model=self.propeller_ct_model,
                                                                                         cp_model=self.propeller_cp_model)
                res = x - self.airspeed * np.sqrt(
                    self.air_density * self.propeller_diameter ** 2 * propeller_ct / self.thrust_per_propeller)
                return res

            self._advance_ratio = brentq(func, 0.0, 3.0) if self.airspeed > 0 else 0.0  # [-] solving for advance ratio
        return self._advance_ratio

    @property
    def propeller_ct(self) -> float:
        """Thrust coefficient of the propeller, under the given flight conditions."""
        if self._propeller_ct is None and self.propeller_beta is not None:
            self._propeller_ct, _ = PropellerAerodynamicsModel.aero_coefficients_incidence(self.propeller_beta,
                                                                                           self.advance_ratio,
                                                                                           self.propeller_angle_of_attack,
                                                                                           ct_model=self.propeller_ct_model,
                                                                                           cp_model=self.propeller_cp_model)
        return self._propeller_ct

    @property
    def propeller_cp(self) -> float:
        """Power coefficient of the propeller, under the given flight conditions."""
        if self._propeller_cp is None and self.propeller_beta is not None:
            _, self._propeller_cp = PropellerAerodynamicsModel.aero_coefficients_incidence(self.propeller_beta,
                                                                                           self.advance_ratio,
                                                                                           self.propeller_angle_of_attack,
                                                                                           ct_model=self.propeller_ct_model,
                                                                                           cp_model=self.propeller_cp_model)
        return self._propeller_cp

    @property
    def propeller_speed(self) -> float:
        """Propeller rotation speed in rad/s."""
        if self._propeller_speed is None and self.propeller_diameter is not None:
            self._propeller_speed = PropellerPerformanceModel.speed(self.thrust_per_propeller,
                                                                    self.propeller_diameter,
                                                                    self.propeller_ct,
                                                                    self.air_density)
        return self._propeller_speed

    @property
    def propeller_power(self) -> float:
        """Propeller power in W."""
        if self._propeller_power is None and self.propeller_diameter is not None:
            self._propeller_power = PropellerPerformanceModel.power(self.propeller_speed,
                                                                    self.propeller_diameter,
                                                                    self.propeller_cp,
                                                                    self.air_density)
        return self._propeller_power

    @property
    def propeller_torque(self) -> float:
        """Propeller torque in N*m."""
        if self._propeller_torque is None:
            self._propeller_torque = PropellerPerformanceModel.torque(self.propeller_power,
                                                                      self.propeller_speed)
        return self._propeller_torque

    @property
    def esc_power(self) -> float:
        """ESC power in W."""
        if self._esc_power is None and self.battery_voltage is not None:
            self._esc_power = ESCPerformanceModel.power(self.motor_power,
                                                        self.motor_voltage,
                                                        self.battery_voltage)
        return self._esc_power

    @property
    def air_density(self) -> float:
        """Air density in kg/m**3."""
        if self._air_density is None:
            self._air_density = AtmosphereSI(self.altitude,
                                             self.delta_isa).density
        return self._air_density

