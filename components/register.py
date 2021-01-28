from Propeller.propeller import PropellerMR
from Motor.motor import MotorMR
from Energy_source.energy_source import Battery
from ESC.esc import ESC
from Structure.structure import StructureMR
from Missions.missions import MissionsMR
from Objectives.objectives import Objective
from fastoad.module_management import OpenMDAOSystemRegistry

OpenMDAOSystemRegistry.register_system(MissionsMR, "missions.multirotor")
OpenMDAOSystemRegistry.register_system(PropellerMR, "propeller.multirotor")
OpenMDAOSystemRegistry.register_system(MotorMR, "motor.multirotor")
OpenMDAOSystemRegistry.register_system(Battery, "energy.battery")
OpenMDAOSystemRegistry.register_system(ESC, "esc")
OpenMDAOSystemRegistry.register_system(StructureMR, "structure.multirotor")
OpenMDAOSystemRegistry.register_system(Objective, "objective")



