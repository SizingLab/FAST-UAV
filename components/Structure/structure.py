"""
Structure component
"""
import openmdao.api as om
from Structure.Geometry.structure_geometry import ComputeStructureGeometryMR
from Structure.Weight.structure_weight import ComputeStructureWeightMR


class StructureMR(om.Group):
    """
    Group containing the Structure MDA of a Multi-Rotor.
    """

    def setup(self):
        self.add_subsystem("compute_geom", ComputeStructureGeometryMR(), promotes=["*"])
        self.add_subsystem("compute_weight", ComputeStructureWeightMR(), promotes=["*"])


