"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
from fastuav.models.life_cycle_assessment.lca_core import LifeCycleAssessment
# from fastuav.models.life_cycle_assessment.lca_postprocessing import SpecificComponentContributions


@oad.RegisterOpenMDAOSystem("fastuav.plugin.lca.multirotor")
class LCAmultirotor(om.Group):
    """
    LCA group for multirotor designs
    """

    def initialize(self):
        # Generic options for LCA module
        self.options.declare("configuration_file", default=None, types=str)
        self.options.declare("axis", default=None, types=str)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)

        # Postprocessing modules
        self.options.declare("postprocessing_multirotor", default=False, types=bool)

    def setup(self):
        self.add_subsystem("lca",
                           LifeCycleAssessment(
                               configuration_file=self.options["configuration_file"],
                               axis=self.options["axis"],
                               normalization=self.options["normalization"],
                               weighting=self.options["weighting"],
                                   ),
                           promotes=["*"]
                           )

        #if self.options["postprocessing_multirotor"]:  # TO BE UPDATED WITH NEW LCA MODULE
        #    self.add_subsystem("postprocessing",
        #                       SpecificComponentContributions(methods=self.options["methods"],
        #                                                      functional_unit=self.options["functional_unit"],
        #                                                      normalization=self.options["normalization"],
        #                                                      weighting=self.options["weighting"],),
        #                       promotes=["*"]
        #                       )
