"""
ESC component
"""
import fastoad.api as oad
import openmdao.api as om
from models.ESC.Scaling.esc_scaling import ESCScaling
from models.ESC.Performances.esc_performance import ESCPerfos
from models.ESC.Constraints.esc_constraints import ESCConstraints
from models.ESC.DecisionTree.ESC_catalog import ESCCatalogueSelection


@oad.RegisterOpenMDAOSystem("esc")
class ESC(om.Group):
    """
    Group containing the ESC MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("scaling", ESCScaling(), promotes=["*"])
        self.add_subsystem("catalogue", ESCCatalogueSelection(use_catalogue=self.options['use_catalogue']),
                           promotes=["*"])
        self.add_subsystem("performances", ESCPerfos(), promotes=["*"])
        self.add_subsystem("constraints", ESCConstraints(), promotes=["*"])
