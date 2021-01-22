"""
Propeller component
"""
import openmdao.api as om
from Propeller.Aerodynamics.propeller_aero import ComputePropellerAeroMR
from Propeller.Geometry.propeller_geometry import ComputePropellerGeometryMR
from Propeller.Performances.propeller_performance import ComputePropellerPerfoMR
from Propeller.Weight.propeller_weight import ComputePropellerWeightMR
from Propeller.Constraints.propeller_constraints import PropellerConstraintsMR

class PropellerMR(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def setup(self):
        self.add_subsystem("compute_aero", ComputePropellerAeroMR(), promotes=["*"])
        self.add_subsystem("compute_geom", ComputePropellerGeometryMR(), promotes=["*"])
        self.add_subsystem("compute_perfo", ComputePropellerPerfoMR(), promotes=["*"])
        self.add_subsystem("compute_weight", ComputePropellerWeightMR(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", PropellerConstraintsMR(), promotes=["*"])

