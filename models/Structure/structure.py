"""
Structure component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Structure.Geometry.structure_geometry import ComputeStructureGeometryMR
from models.Structure.Weight.structure_weight import ComputeStructureWeightMR


@oad.RegisterOpenMDAOSystem("multirotor.structure")
class StructureMR(om.Group):
    """
    Group containing the Structure MDA of a Multi-Rotor.
    """

    def setup(self):
        self.add_subsystem("geometry", ComputeStructureGeometryMR(), promotes=["*"])
        self.add_subsystem("weight", ComputeStructureWeightMR(), promotes=["*"])


