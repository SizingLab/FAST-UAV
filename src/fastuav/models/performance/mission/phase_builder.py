"""
Flight Phase generator.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import MR_PROPULSION, FW_PROPULSION, PROPULSION_ID_LIST, HOVER_TAG, CLIMB_TAG, CRUISE_TAG, PHASE_TAGS_LIST
from fastuav.models.performance.mission.flight_performance import FlightPerformanceModel


class PhaseBuilder(om.Group):
    """
    This class builds a flight phase from a provided definition.
    It calculates the energy consumption and flight duration of the flight phase.
    If the mission is a sizing mission, the flight phase parameters are retrieved from the sizing scenarios definition.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("is_sizing", default=False, types=bool)
        self.options.declare("route_name", default=None, types=str)
        self.options.declare("phase_name", default=None, types=str)
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        mission_name = self.options["mission_name"]
        is_sizing = self.options["is_sizing"]
        route_name = self.options["route_name"]
        phase_name = self.options["phase_name"]
        propulsion_id = self.options["propulsion_id"]

        if is_sizing:
            self.add_subsystem("set_flight_parameters",
                               SetFlightParameters(mission_name=mission_name,
                                                   route_name=route_name,
                                                   phase_name=phase_name,
                                                   propulsion_id=propulsion_id),
                               promotes=["*"])

        self.add_subsystem("compute_performance",
                           PhaseComponent(mission_name=mission_name,
                                          is_sizing=is_sizing,
                                          route_name=route_name,
                                          phase_name=phase_name,
                                          propulsion_id=propulsion_id),
                           promotes=["*"])


class SetFlightParameters(om.ExplicitComponent):
    """
    This class sets the flight parameters of the flight phase from the sizing scenarios.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("route_name", default=None, types=str)
        self.options.declare("phase_name", default=None, types=str)
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        phase_name = self.options["phase_name"]
        propulsion_id = self.options["propulsion_id"]

        if phase_name == HOVER_TAG:
            self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=np.nan, units="m")
            self.add_output("mission:%s:%s:hover:altitude" % (mission_name, route_name), units="m")

        elif phase_name == CLIMB_TAG:
            self.add_input("data:scenarios:%s:takeoff:altitude" % propulsion_id, val=np.nan, units="m")
            self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=np.nan, units="m")
            self.add_input("data:scenarios:%s:climb:speed" % propulsion_id, val=np.nan, units="m/s")
            self.add_output("mission:%s:%s:climb:altitude" % (mission_name, route_name), units="m")
            self.add_output("mission:%s:%s:climb:speed" % (mission_name, route_name), units="m/s")
            self.add_output("mission:%s:%s:climb:distance" % (mission_name, route_name), units="m")
        elif phase_name == CRUISE_TAG:
            self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=np.nan, units="m")
            self.add_input("data:scenarios:%s:cruise:speed" % propulsion_id, val=np.nan, units="m/s")
            self.add_output("mission:%s:%s:cruise:altitude" % (mission_name, route_name), units="m")
            self.add_output("mission:%s:%s:cruise:speed" % (mission_name, route_name), val=np.nan, units="m/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        phase_name = self.options["phase_name"]
        propulsion_id = self.options["propulsion_id"]

        if phase_name == HOVER_TAG:
            outputs["mission:%s:%s:hover:altitude" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:cruise:altitude" % propulsion_id]
        elif phase_name == CLIMB_TAG:
            outputs["mission:%s:%s:climb:altitude" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:cruise:altitude" % propulsion_id]
            outputs["mission:%s:%s:climb:speed" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:climb:speed" % propulsion_id]  # [m/s]
            outputs["mission:%s:%s:climb:distance" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:cruise:altitude" % propulsion_id] - inputs[
                "data:scenarios:%s:takeoff:altitude" % propulsion_id]
        elif phase_name == CRUISE_TAG:
            outputs["mission:%s:%s:cruise:altitude" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:cruise:altitude" % propulsion_id]
            outputs["mission:%s:%s:cruise:speed" % (mission_name, route_name)] = inputs[
                "data:scenarios:%s:cruise:speed" % propulsion_id]  # [m/s]


class PhaseComponent(om.ExplicitComponent):
    """
    This class builds and computes the UAV performance for a flight phase (energy consumption and phase duration).
        - If the mission is a sizing mission, then the parameters required for calculation are directly retrieved
        from the sizing process.
        - Else, the parameters are recalculated for the new flight conditions.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("is_sizing", default=False, types=bool)
        self.options.declare("route_name", default=None, types=str)
        self.options.declare("phase_name", default=None, values=PHASE_TAGS_LIST)
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        mission_name = self.options["mission_name"]
        is_sizing = self.options["is_sizing"]
        route_name = self.options["route_name"]
        phase_name = self.options["phase_name"]
        propulsion_id = self.options["propulsion_id"]

        if phase_name == HOVER_TAG:
            self.add_input("mission:%s:%s:hover:altitude" % (mission_name, route_name), val=np.nan, units="m")
            self.add_input("mission:%s:%s:hover:duration" % (mission_name, route_name), val=np.nan, units="min")
        elif phase_name == CLIMB_TAG:
            self.add_input("mission:%s:%s:climb:altitude" % (mission_name, route_name), val=np.nan, units="m")
            self.add_input("mission:%s:%s:climb:distance" % (mission_name, route_name), val=np.nan, units="m")
            self.add_input("mission:%s:%s:climb:speed" % (mission_name, route_name), val=np.nan, units="m/s")
            self.add_output("mission:%s:%s:climb:duration" % (mission_name, route_name), units="min")
        elif phase_name == CRUISE_TAG:
            self.add_input("mission:%s:%s:cruise:altitude" % (mission_name, route_name), val=np.nan, units="m")
            self.add_input("mission:%s:%s:cruise:distance" % (mission_name, route_name), val=np.nan, units="m")
            self.add_input("mission:%s:%s:cruise:speed" % (mission_name, route_name), val=np.nan, units="m/s")
            self.add_output("mission:%s:%s:cruise:duration" % (mission_name, route_name), units="min")

        if is_sizing:
            self.add_input("data:propulsion:%s:battery:power:%s" % (propulsion_id, phase_name), val=np.nan, units="W")
        else:
            self.add_input("mission:%s:%s:tow" % (mission_name, route_name), val=np.nan, units="kg")
            self.add_input("mission:%s:dISA" % mission_name, val=np.nan, units="K")
            self.add_input("data:propulsion:%s:battery:voltage" % propulsion_id, val=np.nan, units="V")
            self.add_input("data:propulsion:%s:esc:efficiency" % propulsion_id, val=np.nan, units=None)
            self.add_input("data:propulsion:%s:gearbox:N_red" % propulsion_id, val=np.nan, units=None)
            self.add_input("data:propulsion:%s:motor:torque:coefficient" % propulsion_id, val=np.nan, units="N*m/A")
            self.add_input("data:propulsion:%s:motor:torque:friction" % propulsion_id, val=np.nan, units="N*m")
            self.add_input("data:propulsion:%s:motor:resistance" % propulsion_id, val=np.nan, units="V/A")
            self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
            self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_id, val=np.nan, units="m")
            self.add_input("data:propulsion:%s:propeller:beta" % propulsion_id, val=np.nan, units=None)
            self.add_input("mission:%s:%s:%s:payload:power" % (mission_name, route_name, phase_name), val=np.nan, units="W")
            if phase_name == CLIMB_TAG:
                self.add_input("mission:%s:%s:climb:rate" % (mission_name, route_name), val=np.nan, units="m/s")
            if propulsion_id == MR_PROPULSION:
                self.add_input("data:aerodynamics:%s:CD" % propulsion_id, val=np.nan, units=None)
                self.add_input("data:aerodynamics:%s:CL" % propulsion_id, val=np.nan, units=None)
                self.add_input("data:geometry:projected_area:front", val=np.nan, units="m**2")
                self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
            elif propulsion_id == FW_PROPULSION:
                self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
                self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
                self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")

        self.add_output("mission:%s:%s:%s:energy" % (mission_name, route_name, phase_name), units="kJ")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        is_sizing = self.options["is_sizing"]
        route_name = self.options["route_name"]
        phase_name = self.options["phase_name"]
        propulsion_id = self.options["propulsion_id"]
        t = .0  # phase duration [s]
        V = .0  # airspeed [m/s]
        altitude = inputs["mission:%s:%s:%s:altitude" % (mission_name, route_name, phase_name)]  # altitude [m]

        # PHASE DURATION
        if phase_name == HOVER_TAG:
            t = inputs["mission:%s:%s:%s:duration" % (mission_name, route_name, phase_name)] * 60  # [s]
        elif phase_name == CLIMB_TAG:
            d = inputs["mission:%s:%s:climb:distance" % (mission_name, route_name)]  # [m]
            V = inputs["mission:%s:%s:climb:speed" % (mission_name, route_name)]
            t = d / V if V > 0 else 0.0  # [s]
            outputs["mission:%s:%s:climb:duration" % (mission_name, route_name)] = t / 60  # [min]
        elif phase_name == CRUISE_TAG:
            d = inputs["mission:%s:%s:cruise:distance" % (mission_name, route_name)]  # [m]
            V = inputs["mission:%s:%s:cruise:speed" % (mission_name, route_name)]
            t = d / V if V > 0 else 0.0  # [s]
            outputs["mission:%s:%s:cruise:duration" % (mission_name, route_name)] = t / 60  # [min]

        # POWER CONSUMPTION
        if is_sizing:
            power = inputs["data:propulsion:%s:battery:power:%s" % (propulsion_id, phase_name)]
        else:
            # flight parameters
            tow = inputs["mission:%s:%s:tow" % (mission_name, route_name)]
            dISA = inputs["mission:%s:dISA" % mission_name]
            RoC = inputs["mission:%s:%s:climb:rate" % (mission_name, route_name)] if phase_name == CLIMB_TAG else 0.0

            # setup flight model
            flight_model = FlightPerformanceModel(propulsion_id, tow, V, RoC, altitude, dISA)
            flight_model.battery_voltage = inputs["data:propulsion:%s:battery:voltage" % propulsion_id]
            flight_model.esc_efficiency = inputs["data:propulsion:%s:esc:efficiency" % propulsion_id]
            flight_model.gearbox_ratio = inputs["data:propulsion:%s:gearbox:N_red" % propulsion_id]
            flight_model.motor_torque_coef = inputs["data:propulsion:%s:motor:torque:coefficient" % propulsion_id]
            flight_model.motor_torque_friction = inputs["data:propulsion:%s:motor:torque:friction" % propulsion_id]
            flight_model.motor_resistance = inputs["data:propulsion:%s:motor:resistance" % propulsion_id]
            flight_model.propeller_number = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
            flight_model.propeller_diameter = inputs["data:propulsion:%s:propeller:diameter" % propulsion_id]
            flight_model.propeller_beta = inputs["data:propulsion:%s:propeller:beta" % propulsion_id]
            flight_model.payload_power = inputs["mission:%s:%s:%s:payload:power" % (mission_name, route_name, phase_name)]

            if propulsion_id == MR_PROPULSION:
                flight_model.mr_parasitic_drag_coef = inputs["data:aerodynamics:%s:CD" % propulsion_id]
                flight_model.mr_lift_coef = inputs["data:aerodynamics:%s:CL" % propulsion_id]
                flight_model.mr_area_front = inputs["data:geometry:projected_area:front"]
                flight_model.mr_area_top = inputs["data:geometry:projected_area:top"]

            elif propulsion_id == FW_PROPULSION:
                flight_model.fw_induced_drag_constant = inputs["data:aerodynamics:CDi:K"]
                flight_model.fw_parasitic_drag_coef = inputs["data:aerodynamics:CD0"]
                flight_model.wing_area = inputs["data:geometry:wing:surface"]

            power = flight_model.battery_power

        energy = power * t  # [J] required energy to complete the flight phase_name

        outputs["mission:%s:%s:%s:energy" % (mission_name, route_name, phase_name)] = energy / 1000  # [kJ]