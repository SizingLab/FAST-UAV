"""
Hybrid VTOL Airframe Geometry
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.models.geometry.geometry_fixedwing import WingGeometry, HorizontalTailGeometry, VerticalTailGeometry, FuselageGeometry, ProjectedAreasConstraint, FuselageVolumeConstraint
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION


@oad.RegisterOpenMDAOSystem("fastuav.geometry.hybrid")
class Geometry(om.Group):
    """
    Group containing the geometries calculation
    """

    def setup(self):
        self.add_subsystem("wing", WingGeometry(), promotes=["*"])
        self.add_subsystem("horizontal_tail", HorizontalTailGeometry(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailGeometry(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageGeometry(), promotes=["*"])
        self.add_subsystem("vtol_propellers", PropellersVTOL(), promotes=["*"])
        self.add_subsystem("vtol_arms", ArmsVTOL(), promotes=["*"])

        constraints = self.add_subsystem("constraints", om.Group(), promotes=["*"])
        constraints.add_subsystem("projected_areas", ProjectedAreasConstraint(), promotes=["*"])
        constraints.add_subsystem("fuselage_volume", FuselageVolumeConstraint(), promotes=["*"])


class PropellersVTOL(om.ExplicitComponent):
    """
    Computes the positions of the VTOL propellers
    """

    def initialize(self):
        self.options.declare("propulsion_fw", default=FW_PROPULSION, values=[FW_PROPULSION])
        self.options.declare("propulsion_mr", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_fw = self.options["propulsion_fw"]
        propulsion_mr = self.options["propulsion_mr"]

        self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_fw, val=np.nan, units="m")
        self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_mr, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:clearance" % propulsion_mr, val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:LE:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:TE:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep:LE", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep:TE", val=np.nan, units="rad")

        self.add_output("data:geometry:%s:propeller:front:x" % propulsion_mr, units="m")
        self.add_output("data:geometry:%s:propeller:front:y" % propulsion_mr, units="m")
        self.add_output("data:geometry:%s:propeller:rear:x" % propulsion_mr, units="m")
        self.add_output("data:geometry:%s:propeller:rear:y" % propulsion_mr, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_fw = self.options["propulsion_fw"]
        propulsion_mr = self.options["propulsion_mr"]

        D_pro_FW = inputs["data:propulsion:%s:propeller:diameter" % propulsion_fw]
        D_pro_MR = inputs["data:propulsion:%s:propeller:diameter" % propulsion_mr]
        c_pro_MR = inputs["data:geometry:%s:propeller:clearance" % propulsion_mr]
        x_root_LE = inputs["data:geometry:wing:root:LE:x"]
        x_root_TE = inputs["data:geometry:wing:root:TE:x"]
        sweep_LE = inputs["data:geometry:wing:sweep:LE"]
        sweep_TE = inputs["data:geometry:wing:sweep:TE"]

        # front-left propeller location (front-right propeller is symmetrical along the x-axis)
        y_front = D_pro_FW / 2 + c_pro_MR + D_pro_MR / 2  # [m] y-location
        x_front = x_root_LE + y_front * np.tan(sweep_LE) - (D_pro_MR / 2 + c_pro_MR) / np.cos(sweep_LE)

        # rear-left propeller location (rear-right propeller is symmetrical along the x-axis)
        y_rear = D_pro_FW / 2 + c_pro_MR + D_pro_MR / 2  # [m] y-location
        x_rear = x_root_TE + y_rear * np.tan(sweep_TE) + (D_pro_MR / 2 + c_pro_MR) / np.cos(sweep_LE)

        outputs["data:geometry:%s:propeller:front:x" % propulsion_mr] = x_front
        outputs["data:geometry:%s:propeller:front:y" % propulsion_mr] = y_front
        outputs["data:geometry:%s:propeller:rear:x" % propulsion_mr] = x_rear
        outputs["data:geometry:%s:propeller:rear:y" % propulsion_mr] = y_rear


class ArmsVTOL(om.ExplicitComponent):
    """
    Computes the geometry of the arms supporting the VTOL propellers
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:propulsion:%s:propeller:is_coaxial" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:geometry:%s:propeller:front:x" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:front:y" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:rear:x" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:%s:propeller:rear:y" % propulsion_id, val=np.nan, units="m")
        self.add_output("data:geometry:arms:number", units=None)
        self.add_output("data:geometry:arms:prop_per_arm", units=None)
        self.add_output("data:geometry:arms:length", units="m", lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        is_coaxial = inputs["data:propulsion:%s:propeller:is_coaxial" % propulsion_id]
        x_front = inputs["data:geometry:%s:propeller:front:x" % propulsion_id]
        y_front = inputs["data:geometry:%s:propeller:front:y" % propulsion_id]
        x_rear = inputs["data:geometry:%s:propeller:rear:x" % propulsion_id]
        y_rear = inputs["data:geometry:%s:propeller:rear:y" % propulsion_id]

        # Number of arms: fixed value for quad-planes
        N_arms = 4

        # Number of propellers per arm (coaxial or single configuration)
        Npro_arm = 1 + is_coaxial

        # Arms length
        L = np.sqrt((x_front - x_rear) ** 2 + (y_front - y_rear) ** 2)  # [m] distance between front and rear props
        L_arm = L / 2  # [m] length of one arm

        outputs["data:geometry:arms:number"] = N_arms
        outputs["data:geometry:arms:prop_per_arm"] = Npro_arm
        outputs["data:geometry:arms:length"] = L_arm