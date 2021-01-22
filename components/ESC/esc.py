"""
ESC component
"""
import openmdao.api as om
from ESC.Performances.esc_performance import ComputeESCPerfo
from ESC.Weight.esc_weight import ComputeESCWeight
from ESC.Constraints.esc_constraints import ESCConstraints

class ESC(om.Group):
    """
    Group containing the ESC MDA.
    """

    def setup(self):
        self.add_subsystem("compute_perfo", ComputeESCPerfo(), promotes=["*"])
        self.add_subsystem("compute_weight", ComputeESCWeight(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", ESCConstraints(), promotes=["*"])