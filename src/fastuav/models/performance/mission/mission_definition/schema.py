"""
Schema for mission definition files.
"""
#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2021 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from importlib.resources import open_text
from os import PathLike
from typing import Union

from ensure import Ensure
from jsonschema import validate
from ruamel.yaml import YAML

from . import resources

JSON_SCHEMA_NAME = "mission_schema.json"

# Tags
PHASE_ID_TAG = "phase_id"
ROUTE_TAG = "route"
PARTS_TAG = "parts"
TAKEOFF_PART_TAG = "takeoff_part"
CLIMB_PART_TAG = "climb_part"
CRUISE_PART_TAG = "cruise_part"
DESCENT_PART_TAG = "descent_part"
HOVER_PART_TAG = "hover_part"
MISSION_DEFINITION_TAG = "missions"
ROUTE_DEFINITION_TAG = "routes"
SIZING_MISSION_TAG = "sizing"
MAIN_ROUTE_TAG = "main_route"


class MissionDefinition(dict):
    def __init__(self, file_path: Union[str, PathLike] = None):
        """
        Class for reading a mission definition from a YAML file.

        Path of YAML file should be provided at instantiation, or in
        :meth:`load`.

        :param file_path: path of YAML file to read.
        """
        super().__init__()
        if file_path:
            self.load(file_path)

    def load(self, file_path: Union[str, PathLike]):
        """
        Loads a mission definition from provided file path.

        Any existing definition will be overwritten.

        :param file_path: path of YAML file to read.
        """
        self.clear()
        yaml = YAML()

        with open(file_path) as yaml_file:
            data = yaml.load(yaml_file)

        with open_text(resources, JSON_SCHEMA_NAME) as json_file:
            json_schema = json.loads(json_file.read())
        validate(data, json_schema)

        self._validate(data)
        self.update(data)

    @classmethod
    def _validate(cls, content: dict):
        """
        Does a second pass validation of file content.
        Errors are raised if file content is incorrect.

        :param content:
        """

        # Ensure sizing mission is defined
        Ensure(SIZING_MISSION_TAG).is_in(content[MISSION_DEFINITION_TAG].keys())

        # Ensure main_route is defined
        Ensure(MAIN_ROUTE_TAG).is_in(content[ROUTE_DEFINITION_TAG].keys())

        for mission_definition in content[MISSION_DEFINITION_TAG].values():
            for part in mission_definition[PARTS_TAG]:
                part_type, value = tuple(*part.items())
                Ensure(part_type).equals(ROUTE_TAG)
                Ensure(value).is_in(content[ROUTE_DEFINITION_TAG].keys())

            # Disable multiple routes definition for sizing mission,
            # as it would require processing to get sizing values from multiple routes.
            if mission_definition == SIZING_MISSION_TAG:
                routes_count = 0
