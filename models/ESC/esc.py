"""
ESC component
"""
import openmdao.api as om
from models.ESC.Characteristics.esc_characteristics import ComputeESCCharacteristics
from models.ESC.Performances.esc_performance import ComputeESCPerfo
from models.ESC.Weight.esc_weight import ComputeESCWeight
from models.ESC.Constraints.esc_constraints import ESCConstraints
from models.ESC.DecisionTree.ESC_catalog import ESCDecisionTree

class ESC(om.Group):
    """
    Group containing the ESC MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        self.add_subsystem("characteristics", ComputeESCCharacteristics(), promotes=["*"])
        self.add_subsystem("weight", ComputeESCWeight(), promotes=["*"])
        self.add_subsystem("performances", ComputeESCPerfo(), promotes=["*"])

        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogues' = true)
        if self.options["use_catalogues"]:
            self.add_subsystem("catalogue_selection", ESCDecisionTree(), promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", ESCConstraints(use_catalogues=self.options['use_catalogues']), promotes=["*"])