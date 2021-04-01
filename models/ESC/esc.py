"""
ESC component
"""
import openmdao.api as om
from models.ESC.Characteristics.esc_characteristics import ComputeESCCharacteristics
from models.ESC.Performances.esc_performance import ComputeESCPerfo
from models.ESC.Weight.esc_weight import ComputeESCWeight
from models.ESC.Constraints.esc_constraints import ESCConstraints
from models.ESC.DecisionTree.ESC_catalog import ESCCatalogueSelection

class ESC(om.Group):
    """
    Group containing the ESC MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("characteristics", ComputeESCCharacteristics(), promotes=["*"])
        self.add_subsystem("weight", ComputeESCWeight(), promotes=["*"])

        # Choose between estimated parameters and catalogue components
        self.add_subsystem("catalogue", ESCCatalogueSelection(use_catalogue=self.options['use_catalogue']),
                           promotes=["*"])

        self.add_subsystem("performances", ComputeESCPerfo(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", ESCConstraints(), promotes=["*"])