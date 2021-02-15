"""
Missions definition : sizing scenarios
"""
import openmdao.api as om
from models.Missions.Scenarios.scenarios import SizingScenarios



class MissionsMR(om.Group):
    """
    Group containing the sizings scenarios for Multi-Rotor drones
    """

    def setup(self):
        self.add_subsystem("sizing_scenarios", SizingScenarios(), promotes=["*"])
