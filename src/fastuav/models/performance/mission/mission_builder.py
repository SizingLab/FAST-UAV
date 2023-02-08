"""
Mission generator.
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from itertools import chain
from fastuav.utils.constants import *
from fastuav.models.performance.mission.mission_definition.schema import MissionDefinition
from fastuav.models.performance.mission.route_builder import RouteBuilder


@oad.RegisterOpenMDAOSystem("fastuav.performance.mission")
class MissionBuilder(om.Group):
    """
    This class builds a mission from a provided definition.
    """

    def initialize(self):
        self.options.declare("file_path", default=None, types=str)

    def setup(self):
        file_path = self.options["file_path"]
        mission_dict = MissionDefinition(file_path)

        for mission_name, mission_definition in mission_dict[MISSION_DEFINITION_TAG].items():
            routes_list = []  # list of routes names
            propulsion_id_dict = {}  # list of propulsion systems used to complete the mission
            is_sizing = True if mission_name == SIZING_MISSION_TAG else False  # sizing mission flag

            # Create mission group
            mission_group = self.add_subsystem(mission_name, om.Group(), promotes=["*"])

            # Add routes to the mission group
            for route in mission_definition[PARTS_TAG]:
                _, route_name = tuple(*route.items())  # get route name
                route_definition = mission_dict[ROUTE_DEFINITION_TAG][route_name]  # get route definition
                routes_list.append(route_name)
                propulsion_id_dict[route_name] = RouteBuilder.get_propulsion_id_list(route_definition)
                # Add OpenMDAO subgroup to mission group
                mission_group.add_subsystem(route_name,
                                            RouteBuilder(mission_name=mission_name,
                                                         is_sizing=is_sizing,
                                                         route_name=route_name,
                                                         route_definition=route_definition),
                                            promotes=["*"])

            # Add mission component to sum up the routes calculations outputs
            mission_group.add_subsystem("mission",
                                        MissionComponent(mission_name=mission_name,
                                                         routes_list=routes_list,
                                                         propulsion_id_dict=propulsion_id_dict),
                                        promotes=["*"])

            # Add constraint for sizing the battery capacity / energy to complete the mission
            mission_group.add_subsystem("constraints",
                                        MissionConstraints(mission_name=mission_name,
                                                           propulsion_id_dict=propulsion_id_dict),
                                        promotes=["*"])


class MissionComponent(om.ExplicitComponent):
    """
    This component computes the mission parameters (energy and duration) from the routes its made of.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("routes_list", default=[], types=list)
        self.options.declare("propulsion_id_dict", default=None, types=dict)

    def setup(self):
        mission_name = self.options["mission_name"]
        routes_list = self.options["routes_list"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for route_name in routes_list:
            for propulsion_id in propulsion_id_dict[route_name]:
                self.add_input("mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id),
                               val=np.nan,
                               units="kJ")
                self.add_input("mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id),
                               val=np.nan,
                               units="min")
            self.add_input("mission:%s:%s:energy" % (mission_name, route_name),
                           val=np.nan,
                           units="kJ")
            self.add_input("mission:%s:%s:duration" % (mission_name, route_name),
                           val=np.nan,
                           units="min")

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            self.add_output("mission:%s:energy:%s" % (mission_name, propulsion_id), units="kJ")
            self.add_output("mission:%s:duration:%s" % (mission_name, propulsion_id), units="min")
        self.add_output("mission:%s:energy" % mission_name, units="kJ")
        self.add_output("mission:%s:duration" % mission_name, units="min")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        routes_list = self.options["routes_list"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            outputs["mission:%s:energy:%s" % (mission_name, propulsion_id)] = sum(
                inputs["mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id)] for route_name in
                routes_list if propulsion_id in propulsion_id_dict[route_name])
            outputs["mission:%s:duration:%s" % (mission_name, propulsion_id)] = sum(
                inputs["mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id)] for route_name in
                routes_list if propulsion_id in propulsion_id_dict[route_name])

        outputs["mission:%s:energy" % mission_name] = sum(
            inputs["mission:%s:%s:energy" % (mission_name, route_name)] for route_name in routes_list)

        outputs["mission:%s:duration" % mission_name] = sum(
            inputs["mission:%s:%s:duration" % (mission_name, route_name)] for route_name in routes_list)


class MissionConstraints(om.ExplicitComponent):
    """
    This component computes the constraints associated with the battery energy required to perform a mission.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("propulsion_id_dict", default=None, types=dict)

    def setup(self):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]
        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            self.add_input("mission:%s:energy:%s" % (mission_name, propulsion_id), val=np.nan, units="kJ")
            self.add_input("data:propulsion:%s:battery:energy" % propulsion_id, val=0.0, units="kJ")
            self.add_input("data:propulsion:%s:battery:DoD:max" % propulsion_id, val=0.8, units=None)
            self.add_output("mission:%s:energy:%s:constraint" % (mission_name, propulsion_id), units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            E_mission = inputs["mission:%s:energy:%s" % (mission_name, propulsion_id)]
            E_bat = inputs["data:propulsion:%s:battery:energy" % propulsion_id]
            C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
            energy_con = (E_bat * C_ratio - E_mission) / (E_bat * C_ratio) if E_bat > 0 else -1e6
            outputs["mission:%s:energy:%s:constraint" % (mission_name, propulsion_id)] = energy_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            E_mission = inputs["mission:%s:energy:%s" % (mission_name, propulsion_id)]
            E_bat = inputs["data:propulsion:%s:battery:energy" % propulsion_id]
            C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
            partials[
                "mission:%s:energy:%s:constraint" % (mission_name, propulsion_id),
                "mission:%s:energy:%s" % (mission_name, propulsion_id),
            ] = -1.0 / (E_bat * C_ratio) if E_bat > 0 else 0.0
            partials[
                "mission:%s:energy:%s:constraint" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:energy" % propulsion_id,
            ] = E_mission / (E_bat**2 * C_ratio) if E_bat > 0 else 0.0
            partials[
                "mission:%s:energy:%s:constraint" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:DoD:max" % propulsion_id,
            ] = E_mission / (E_bat * C_ratio**2) if E_bat > 0 else 0.0