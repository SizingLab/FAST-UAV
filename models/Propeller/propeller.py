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
from models.Propeller.DecisionTree.propeller_catalog import PropellerCatalogueSelection


class PropellerMR(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("aerodynamics", ComputePropellerAeroMR(), promotes=["*"])
        self.add_subsystem("geometry", ComputePropellerGeometryMR(), promotes=["*"])
        self.add_subsystem("catalogue", PropellerCatalogueSelection(use_catalogue=self.options['use_catalogue']),
                                                                              promotes=["*"])
        self.add_subsystem("performances", ComputePropellerPerfoMR(), promotes=["*"])
        self.add_subsystem("weight", ComputePropellerWeightMR(), promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", PropellerConstraintsMR(), promotes=["*"])

