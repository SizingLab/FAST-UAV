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

import os.path as pth
from collections import OrderedDict

from ..schema import MissionDefinition

MISSIONS_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_schema():
    obtained_dict = MissionDefinition(pth.join(MISSIONS_FOLDER_PATH, "mission.yaml"))

    # As we use Python 3.7+, Python dictionaries are ordered, but they are still
    # considered equal when the order of key differs.
    # To check order, we need to convert both dictionaries to OrderedDict (recursively!)
    obtained_dict = _to_ordered_dict(obtained_dict)
    expected_dict = _to_ordered_dict(_get_expected_dict())

    assert obtained_dict == expected_dict


def _to_ordered_dict(item):
    """Returns the item with all dictionaries inside transformed to OrderedDict."""
    if isinstance(item, dict):
        ordered_dict = OrderedDict(item)
        for key, value in ordered_dict.items():
            ordered_dict[key] = _to_ordered_dict(value)
        return ordered_dict
    elif isinstance(item, list):
        for i, value in enumerate(item):
            item[i] = _to_ordered_dict(value)
    else:
        return item


def _get_expected_dict():
    return {
        "routes": {
            "main_route": {
                "takeoff_part": {
                    "phase_id": "vertical_takeoff"
                },
                "climb_part": {
                    "phase_id": "multirotor_climb"
                },
                "cruise_part": {
                    "phase_id": "fixedwing_cruise"
                },
            },
            "diversion": {
                "climb_part": {
                    "phase_id": "fixedwing_climb"
                },
                "cruise_part": {
                    "phase_id": "fixedwing_cruise"
                },
            }
        },
        "missions": {
            "sizing": {
                "parts": [
                    {"route": "main_route"},
                    {"route": "diversion"},
                ]
            }
        }
    }