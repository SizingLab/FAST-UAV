"""
Propeller component
"""
import openmdao.api as om
from fastoad.models.options import OpenMdaoOptionDispatcherGroup
from models.Propeller.Aerodynamics.propeller_aero import ComputePropellerAeroMR
from models.Propeller.Geometry.propeller_geometry import ComputePropellerGeometryMR
from models.Propeller.Performances.propeller_performance import ComputePropellerPerfoMR
from models.Propeller.Weight.propeller_weight import ComputePropellerWeightMR
from models.Propeller.Constraints.propeller_constraints import PropellerConstraintsMR
from models.Propeller.DecisionTree.propeller_catalog import PropellerDecisionTree

class PropellerMR(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        self.add_subsystem("aerodynamics", ComputePropellerAeroMR(), promotes=["*"])
        self.add_subsystem("geometry", ComputePropellerGeometryMR(), promotes=["*"])

        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogues' = true)
        if self.options["use_catalogues"]:
            self.add_subsystem("catalogue_selection", PropellerDecisionTree(), promotes=["*"])

        self.add_subsystem("performances", ComputePropellerPerfoMR(use_catalogues=self.options['use_catalogues']),
                           promotes=["*"])
        self.add_subsystem("weight", ComputePropellerWeightMR(use_catalogues=self.options['use_catalogues']),
                           promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", PropellerConstraintsMR(use_catalogues=self.options['use_catalogues']),
                           promotes=["*"])

