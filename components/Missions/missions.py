"""
Missions definition : sizing scenarios
"""
import openmdao.api as om
from components.Missions.Scenarios.scenarios import SizingScenarios



class MissionsMR(om.Group):
    """
    Group containing the sizings scenarios for Multi-Rotor drones
    """

    def setup(self):
        self.add_subsystem("sizing_scenarios", SizingScenarios(), promotes=["*"])
        #self.add_subsystem("objective", WeightObjective(), promotes=["*"])
        #self.add_subsystem("constraints", Constraints(), promotes=["*"])
