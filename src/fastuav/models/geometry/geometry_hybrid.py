"""
Hybrid VTOL Airframe Geometry
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.geometry.geometry_fixedwing import WingGeometry, HorizontalTailGeometry, VerticalTailGeometry, FuselageGeometry, ProjectedAreasConstraint, FuselageVolumeConstraint


@oad.RegisterOpenMDAOSystem("fastuav.geometry.hybrid")
class Geometry(om.Group):
    """
    Group containing the airframe geometries calculation
    """

    def setup(self):
        self.add_subsystem("wing", WingGeometry(), promotes=["*"])
        self.add_subsystem("horizontal_tail", HorizontalTailGeometry(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailGeometry(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageGeometry(), promotes=["*"])

        constraints = self.add_subsystem("constraints", om.Group(), promotes=["*"])
        constraints.add_subsystem("projected_areas", ProjectedAreasConstraint(), promotes=["*"])
        constraints.add_subsystem("fuselage_volume", FuselageVolumeConstraint(), promotes=["*"])