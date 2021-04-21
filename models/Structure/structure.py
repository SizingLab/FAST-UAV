"""
Structure component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Structure.Scaling.structure_scaling import StructureScaling


@oad.RegisterOpenMDAOSystem("multirotor.structure")
class StructureMR(om.Group):
    """
    Group containing the Structure MDA of a Multi-Rotor.
    """

    def setup(self):
        self.add_subsystem("scaling", StructureScaling(), promotes=["*"])


