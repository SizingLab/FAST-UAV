"""
Propeller component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Propeller.Scaling.prop_scaling import PropellerScaling, Weight
from models.Propeller.Performances.propeller_performance import PropellerPerfos
from models.Propeller.Constraints.propeller_constraints import PropellerConstraints
from models.Propeller.DecisionTree.propeller_catalog import PropellerCatalogueSelection


@oad.RegisterOpenMDAOSystem("propeller")
class Propeller(om.Group):
    """
    Group containing the Propeller MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("scaling", PropellerScaling(), promotes=["*"])
        self.add_subsystem("catalogue", PropellerCatalogueSelection(use_catalogue=self.options['use_catalogue']), promotes=["*"])
        self.add_subsystem("weight", Weight(), promotes=["*"])
        self.add_subsystem("performances", PropellerPerfos(), promotes=["*"])
        self.add_subsystem("constraints", PropellerConstraints(), promotes=["*"])

