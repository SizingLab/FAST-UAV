"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
from fastuav.constants import LCA_DEFAULT_PROJECT, LCA_DEFAULT_ECOINVENT, LCA_DEFAULT_METHOD, SIZING_MISSION_TAG, \
    LCA_DEFAULT_FUNCTIONAL_UNIT
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
        self.options.declare("functional_unit", default=LCA_DEFAULT_FUNCTIONAL_UNIT, types=str)
        self.options.declare("methods", default=LCA_DEFAULT_METHOD, types=list)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)
        self.options.declare("max_level_processes", default=10, types=int)

        # Computation options for optimization
        self.options.declare("analytical_derivatives", default=True, types=bool)

        # FAST-UAV model specific parameters
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

        # FAST-UAV specific option for selecting mission to evaluate
        self.options.declare("mission", default=SIZING_MISSION_TAG, types=str)

        # Postprocessing modules
        self.options.declare("postprocessing_multirotor", default=False, types=bool)

    def setup(self):
        self.add_subsystem("core",
                           LCAcore(project=self.options["project"],
                                   database=self.options["database"],
                                   functional_unit=self.options["functional_unit"],
                                   methods=self.options["methods"],
                                   normalization=self.options["normalization"],
                                   weighting=self.options["weighting"],
                                   max_level_processes=self.options["max_level_processes"],
                                   analytical_derivatives=self.options["analytical_derivatives"],
                                   parameters=self.options["parameters"],
                                   mission=self.options["mission"]
                                   ),
                           promotes=["*"]
                           )

        if self.options["postprocessing_multirotor"]:
            self.add_subsystem("postprocessing",
                               SpecificComponentContributions(methods=self.options["methods"],
                                                              functional_unit=self.options["functional_unit"],
                                                              normalization=self.options["normalization"],
                                                              weighting=self.options["weighting"],),
                               promotes=["*"]
                               )
