from models.Propeller.propeller import PropellerMR
from models.Motor.motor import Motor
from models.Energy_source.energy_source import Battery
from models.ESC.esc import ESC
from models.Structure.structure import StructureMR
from models.Missions.missions import MissionsMR
from models.Objectives.objectives import Objective
#from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.module_management import OpenMDAOSystemRegistry

OpenMDAOSystemRegistry.register_system(MissionsMR, "missions.multirotor")
OpenMDAOSystemRegistry.register_system(PropellerMR, "propeller.multirotor")
OpenMDAOSystemRegistry.register_system(Motor, "motor")
OpenMDAOSystemRegistry.register_system(Battery, "energy.battery")
OpenMDAOSystemRegistry.register_system(ESC, "esc")
OpenMDAOSystemRegistry.register_system(StructureMR, "structure.multirotor")
OpenMDAOSystemRegistry.register_system(Objective, "objective")
#OpenMDAOSystemRegistry.register_system(DecisionTrees, "decision_trees")



