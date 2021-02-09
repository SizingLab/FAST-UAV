"""
This module is for registering all internal OpenMDAO modules that we want
available through OpenMDAOSystemRegistry
"""
#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from .Propeller.propeller import PropellerMR
from .Motor.motor import Motor
from .Energy_source.energy_source import Battery
from .ESC.esc import ESC
from .Structure.structure import StructureMR
from .Missions.missions import MissionsMR
from .Objectives.objectives import Objective
from fastoad.module_management import OpenMDAOSystemRegistry


def register_openmdao_systems():
    """
        The place where to register FAST-OAD internal models.

        Warning: this function is effective only if called from a Python module that
        is a started bundle for iPOPO
        """
    OpenMDAOSystemRegistry.register_system(MissionsMR, "multirotor.missions")
    OpenMDAOSystemRegistry.register_system(PropellerMR, "multirotor.propeller")
    OpenMDAOSystemRegistry.register_system(Motor, "multirotor.motor")
    OpenMDAOSystemRegistry.register_system(Battery, "energy.battery")
    OpenMDAOSystemRegistry.register_system(ESC, "multirotor.esc")
    OpenMDAOSystemRegistry.register_system(StructureMR, "multirotor.structure")
    OpenMDAOSystemRegistry.register_system(Objective, "multirotor.objective")



