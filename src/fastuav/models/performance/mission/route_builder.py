"""
Route generator.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import *
from fastuav.models.performance.mission.phase_builder import PhaseBuilder


class RouteBuilder(om.Group):
    """
    This class builds a route from a provided definition.
    If the mission is not a sizing one, then the takeoff weight is recalculated from the route payload.
    Else, the takeoff weight is taken as the maximum takeoff weight.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("is_sizing", default=False, types=bool)
        self.options.declare("route_name", default=None, types=str)
        self.options.declare("route_definition", default=None, types=dict)

    def setup(self):
        mission_name = self.options["mission_name"]
        is_sizing = self.options["is_sizing"]
        route_name = self.options["route_name"]
        route_definition = self.options["route_definition"]
        phases_dict = {}

        if not is_sizing:
            self.add_subsystem("compute_tow",
                               ComputeTOW(mission_name=mission_name,
                                          route_name=route_name),
                               promotes=["*"])

        for phase_definition in route_definition.values():
            phase_id = phase_definition[PHASE_ID_TAG]
            phase_name, propulsion_id = self.get_part_attributes(phase_id)
            if phase_name is not None:
                phases_dict[phase_name] = propulsion_id
                self.add_subsystem(phase_id,
                                   PhaseBuilder(mission_name=mission_name,
                                                is_sizing=is_sizing,
                                                route_name=route_name,
                                                phase_name=phase_name,
                                                propulsion_id=propulsion_id),
                                   promotes=["*"])
        self.add_subsystem("route",
                           RouteComponent(mission_name=mission_name,
                                          route_name=route_name,
                                          phases_dict=phases_dict,
                                          propulsion_id_list=RouteBuilder.get_propulsion_id_list(route_definition)),
                           promotes=["*"])

    @staticmethod
    def get_part_attributes(part_id):
        """
        Gets the phase name (e.g. "climb") corresponding the to phase identifier (e.g. "vertical_climb"),
        and also returns the propulsion identifier (e.g. "multirotor" or "fixedwing") used to complete this phase.
        Note that take-off is not taken into account when building the mission profile, as its duration is neglected.
        """
        if part_id == VERTICAL_CLIMB_TAG:
            part_name = CLIMB_TAG
            propulsion_id = MR_PROPULSION
        elif part_id == FW_CLIMB_TAG:
            part_name = CLIMB_TAG
            propulsion_id = FW_PROPULSION
        elif part_id == MR_CRUISE_TAG:
            part_name = CRUISE_TAG
            propulsion_id = MR_PROPULSION
        elif part_id == FW_CRUISE_TAG:
            part_name = CRUISE_TAG
            propulsion_id = FW_PROPULSION
        elif part_id == HOVER_TAG:
            part_name = HOVER_TAG
            propulsion_id = MR_PROPULSION
        else:
            part_name = None
            propulsion_id = None
        return part_name, propulsion_id

    @staticmethod
    def get_propulsion_id_list(route_definition):
        """
        Gets the list of propulsion identifiers (e.g. "multirotor", "fixedwing") used to complete the route.
        """
        propulsion_id_list = []
        for phase_definition in route_definition.values():
            phase_id = phase_definition[PHASE_ID_TAG]
            _, propulsion_id = RouteBuilder.get_part_attributes(phase_id)
            if propulsion_id is not None:
                propulsion_id_list.append(propulsion_id)
        return set(propulsion_id_list)


class ComputeTOW(om.ExplicitComponent):
    """
    Computes Take Off Weight (TOW) from Maximum Take Off Weight (MTOW), design paylaod and route payload
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("route_name", default=None, types=str)

    def setup(self):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        self.add_input("data:weight:mtow", val=np.nan, units="kg")
        self.add_input("mission:sizing:payload:mass", val=np.nan, units="kg")
        self.add_input("mission:%s:%s:payload:mass" % (mission_name, route_name), val=np.nan, units="kg")
        self.add_output("mission:%s:%s:tow" % (mission_name, route_name), units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        mtow = inputs["data:weight:mtow"]
        m_pay_design = inputs["mission:sizing:payload:mass"]
        m_pay_mission = inputs["mission:%s:%s:payload:mass" % (mission_name, route_name)]

        tow = mtow - m_pay_design + m_pay_mission

        outputs["mission:%s:%s:tow" % (mission_name, route_name)] = tow


class RouteComponent(om.ExplicitComponent):
    """
    This component computes the route parameters (energy and duration) from the phases its made of.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("route_name", default=None, types=str)
        self.options.declare("phases_dict", default={}, types=dict)
        self.options.declare("propulsion_id_list", default=None, types=set)

    def setup(self):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        phases_dict = self.options["phases_dict"]
        propulsion_id_list = self.options["propulsion_id_list"]

        for phase_name in phases_dict.keys():
            self.add_input("mission:%s:%s:%s:energy" % (mission_name, route_name, phase_name),
                           val=np.nan,
                           units="kJ")
            self.add_input("mission:%s:%s:%s:duration" % (mission_name, route_name, phase_name),
                           val=np.nan,
                           units="min")

        for propulsion_id in propulsion_id_list:
            self.add_output("mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id), units="kJ")
            self.add_output("mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id), units="min")

        self.add_output("mission:%s:%s:energy" % (mission_name, route_name), units="kJ")
        self.add_output("mission:%s:%s:duration" % (mission_name, route_name), units="min")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        route_name = self.options["route_name"]
        phases_dict = self.options["phases_dict"]
        propulsion_id_list = self.options["propulsion_id_list"]

        # Initialize route parameters
        E_route = .0  # [kJ] energy consumption on whole route
        t_route = .0  # [min] duration of whole route

        # Loop through propulsion systems to associate energy consumption to corresponding battery
        for propulsion_id in propulsion_id_list:
            E_route_prop_id = sum(
                inputs["mission:%s:%s:%s:energy" % (mission_name, route_name, phase_name)] for phase_name in phases_dict
                if phases_dict[phase_name] == propulsion_id)
            t_route_prop_id = sum(
                inputs["mission:%s:%s:%s:duration" % (mission_name, route_name, phase_name)] for phase_name in phases_dict
                if phases_dict[phase_name] == propulsion_id)
            outputs["mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id)] = E_route_prop_id
            outputs["mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id)] = t_route_prop_id

            # Add to route
            E_route += E_route_prop_id
            t_route += t_route_prop_id

        outputs["mission:%s:%s:energy" % (mission_name, route_name)] = E_route
        outputs["mission:%s:%s:duration" % (mission_name, route_name)] = t_route
























