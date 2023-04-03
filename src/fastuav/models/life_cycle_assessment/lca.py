"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
from fastuav.constants import LCA_DEFAULT_PROJECT, LCA_DEFAULT_ECOINVENT, LCA_DEFAULT_METHOD
from fastuav.models.life_cycle_assessment.lca_core import LCAcore
from fastuav.models.life_cycle_assessment.lca_postprocessing import SpecificComponentContributions


@oad.RegisterOpenMDAOSystem("fastuav.plugin.lca.multirotor")
class LCAmultirotor(om.Group):
    """
    LCA group for multirotor designs
    """

    def initialize(self):
        # Generic options for LCA module
        self.options.declare("project", default=LCA_DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=LCA_DEFAULT_ECOINVENT, types=str)
        self.options.declare("model", default="kg.km", types=str)
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # FAST-UAV model specific parameters
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

        # FAST-UAV specific option for selecting mission to evaluate
        self.options.declare("mission", default="sizing", types=str)

    def setup(self):
        self.add_subsystem("core",
                           LCAcore(project=self.options["project"],
                                   database=self.options["database"],
                                   model=self.options["model"],
                                   methods=self.options["methods"],
                                   parameters=self.options["parameters"],
                                   mission=self.options["mission"]
                                   ),
                           promotes=["*"]
                           )

        self.add_subsystem("postprocessing",
                           SpecificComponentContributions(methods=self.options["methods"],
                                                          model=self.options["model"]),
                           promotes=["*"]
                           )
