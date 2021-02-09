from .Propeller.propeller import PropellerMR
from .Motor.motor import Motor
from .Energy_source.energy_source import Battery
from .ESC.esc import ESC
from .Structure.structure import StructureMR
from .Missions.missions import MissionsMR
from .Objectives.objectives import Objective
#from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.module_management import OpenMDAOSystemRegistry


def register_openmdao_systems():
    """
        The place where to register FAST-OAD internal models.

        Warning: this function is effective only if called from a Python module that
        is a started bundle for iPOPO
        """
    OpenMDAOSystemRegistry.register_system(MissionsMR, "missions.multirotor")
    OpenMDAOSystemRegistry.register_system(PropellerMR, "propeller.multirotor")
    OpenMDAOSystemRegistry.register_system(Motor, "motor")
    OpenMDAOSystemRegistry.register_system(Battery, "energy.battery")
    OpenMDAOSystemRegistry.register_system(ESC, "esc")
    OpenMDAOSystemRegistry.register_system(StructureMR, "structure.multirotor")
    OpenMDAOSystemRegistry.register_system(Objective, "objective")
    #OpenMDAOSystemRegistry.register_system(DecisionTrees, "decision_trees")



