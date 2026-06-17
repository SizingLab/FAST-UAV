"""
Airfoil reshape function.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import logging
import os.path as pth
import warnings

import numpy as np
import pandas as pd

from fastuav.models.aerodynamics import airfoil_folder

from .profile import Profile

_LOGGER = logging.getLogger(__name__)


def get_profile(
    airfoil_folder_path: str = None,
    file_name: str = None,
    thickness_ratio=None,
    chord_length=None,
) -> Profile:
    """
    Reads profile from indicated resource file and returns it after resize

    :param file_name: name of resource (ex: "naca23012.af")
    :param thickness_ratio:
    :param chord_length:
    :return: Profile object.
    """

    profile = Profile()
    if airfoil_folder_path is None:
        x_z = genfromtxt(pth.join(airfoil_folder.__path__[0], file_name))
    else:
        x_z = genfromtxt(pth.join(airfoil_folder_path, file_name))
    profile.set_points(x_z["x"], x_z["z"])

    if thickness_ratio:
        if abs(profile.thickness_ratio - thickness_ratio) / thickness_ratio > 0.01:
            warnings.warn(
                "The airfoil thickness ratio from file "
                + pth.join(airfoil_folder.__path__[0], file_name)
                # + " differs from user defined input data:geometry:wing:thickness_ratio!"
                + " differs from user defined input data:geometry:wing:tc!"
            )
        profile.thickness_ratio = thickness_ratio

    if chord_length:
        profile.chord_length = chord_length

    return profile


def genfromtxt(file_name: str = None) -> pd.DataFrame:
    with open(file_name, "r") as file:
        data = file.readlines()
        # Extract data
        x_data = []
        z_data = []
        for i in range(len(data)):
            line = data[i].split()
            if len(line) == 2:
                # noinspection PyBroadException
                try:
                    float(line[0])
                    float(line[1])
                    if 0.0 <= np.ceil(float(line[0])) <= 1.0:
                        x_data.append(float(line[0]))
                        z_data.append(float(line[1]))
                except ValueError:
                    if line[0] != "NACA" and line[0] != "AIRFOIL":
                        _LOGGER.info(
                            "Problem occurred while reading %s file!",
                            pth.join(airfoil_folder.__path__[0], file_name),
                        )
                    else:
                        # Skipping to next line
                        pass

    return pd.DataFrame(data={"x": x_data, "z": z_data})
