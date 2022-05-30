"""
Wing Structures and Weights
"""
import openmdao.api as om
from fastuav.models.structures.wing.estimation_models import WingStructuresEstimationModelsGroup
from fastuav.models.structures.wing.structural_analysis import SparsStressVTOL
from fastuav.models.structures.wing.constraints import SparsGeometricalConstraint, SparsStressVTOLConstraint


class WingStructuresFW(om.Group):
    """
    Computes Wing Structures and Masses for fixed-wing UAVs.
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        spar_model = self.options["spar_model"]
        self.add_subsystem("estimation_models",
                           WingStructuresEstimationModelsGroup(spar_model=spar_model),
                           promotes=["*"])
        self.add_subsystem("constraints",
                           SparsGeometricalConstraint(spar_model=spar_model),
                           promotes=["*"])


class WingStructuresHybrid(om.Group):
    """
    Computes Wing Structures and Masses for FW-VTOL UAVs.
    The difference with fixed-wing UAVs consists of the additional loads resulting
    from the vertical takeoff scenario (maximum thrust of VTOL propellers).
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        spar_model = self.options["spar_model"]

        self.add_subsystem("estimation_models",
                           WingStructuresEstimationModelsGroup(spar_model=spar_model),
                           promotes=["*"])

        self.add_subsystem("structural_analysis",
                           SparsStressVTOL(spar_model=spar_model),
                           promotes=["*"])

        constraints = self.add_subsystem("constraints", om.Group(), promotes=["*"])
        constraints.add_subsystem("spar_height",
                                  SparsGeometricalConstraint(spar_model=spar_model),
                                  promotes=["*"])
        constraints.add_subsystem("vtol_stress",
                                  SparsStressVTOLConstraint(),
                                  promotes=["*"])


