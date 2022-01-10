"""
ESC component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Components.ESC.definition_parameters import ESCDefinitionParameters
from models.Components.ESC.estimation.models import ESCEstimationModels
from models.Components.ESC.estimation.catalogue import ESCCatalogueSelection
from models.Components.ESC.performances import ESCPerfos
from models.Components.ESC.constraints import ESCConstraints


@oad.RegisterOpenMDAOSystem("esc")
class ESC(om.Group):
    """
    Group containing the Electronic Speed Controller analysis.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem('definition_parameters', ESCDefinitionParameters(), promotes=["*"])
        estimation = self.add_subsystem('estimation', om.Group(), promotes=["*"])
        estimation.add_subsystem('models', ESCEstimationModels(), promotes=["*"])
        estimation.add_subsystem('catalogue' if self.options["use_catalogue"] else 'no_catalogue',
                             ESCCatalogueSelection(use_catalogue=self.options['use_catalogue']), promotes=["*"])
        self.add_subsystem('performances', ESCPerfos(), promotes=["*"])
        self.add_subsystem('constraints', ESCConstraints(), promotes=["*"])
