"""
Center of gravity for hybrid VTOL UAVs.
"""
import openmdao.api as om
from fastuav.models.stability.static_longitudinal.center_of_gravity.cog_components import *
from fastuav.utils.constants import FW_PROPULSION


class CenterOfGravity(om.Group):
    """
    Group containing the fixed wing center of gravity calculations
    """

    def setup(self):
        self.add_subsystem("fuselage", CoG_fuselage(), promotes=["*"])
        self.add_subsystem("wing", CoG_wing(), promotes=["*"])
        self.add_subsystem("horizontal_tail", CoG_ht(), promotes=["*"])
        self.add_subsystem("vertical_tail", CoG_vt(), promotes=["*"])
        self.add_subsystem("propeller", CoG_propeller_tractor(), promotes=["*"])
        self.add_subsystem("motor", CoG_motor_tractor(), promotes=["*"])
        self.add_subsystem("battery", CoG_battery(), promotes=["*"])
        self.add_subsystem("UAV", CoG_UAV(), promotes=["*"])


class CoG_UAV(om.ExplicitComponent):
    """
    Computes the center of gravity of a hybrid VTOL UAV.
    """
    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:stability:CoG:fuselage", val=np.nan, units="m")
        self.add_input("data:stability:CoG:wing", val=np.nan, units="m")
        self.add_input("data:stability:CoG:tail:horizontal", val=np.nan, units="m")
        self.add_input("data:stability:CoG:tail:vertical", val=np.nan, units="m")
        self.add_input("data:stability:CoG:propeller", val=np.nan, units="m")
        self.add_input("data:stability:CoG:motor", val=np.nan, units="m")
        self.add_input("data:stability:CoG:battery", val=np.nan, units="m")
        self.add_input("data:weights:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:tail:horizontal:mass", val=np.nan, units="kg")
        self.add_input("data:weights:airframe:tail:vertical:mass", val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:propeller:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:motor:mass" % propulsion_id, val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:%s:battery:mass" % propulsion_id, val=np.nan, units="kg")
        # self.add_input("data:weights:mtow", val=np.nan, units="kg")
        self.add_output("data:stability:CoG", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        x_cg_fus = inputs["data:stability:CoG:fuselage"]
        x_cg_w = inputs["data:stability:CoG:wing"]
        x_cg_ht = inputs["data:stability:CoG:tail:horizontal"]
        x_cg_vt = inputs["data:stability:CoG:tail:vertical"]
        x_cg_prop = inputs["data:stability:CoG:propeller"]
        x_cg_mot = inputs["data:stability:CoG:motor"]
        x_cg_bat = inputs["data:stability:CoG:battery"]
        m_fus = inputs["data:weights:airframe:fuselage:mass"]
        m_wing = inputs["data:weights:airframe:wing:mass"]
        m_ht = inputs["data:weights:airframe:tail:horizontal:mass"]
        m_vt = inputs["data:weights:airframe:tail:vertical:mass"]
        m_prop = inputs["data:weights:propulsion:%s:propeller:mass" % propulsion_id]
        m_mot = inputs["data:weights:propulsion:%s:motor:mass" % propulsion_id]
        m_bat = inputs["data:weights:propulsion:%s:battery:mass" % propulsion_id]
        # m_total = inputs["data:weights:mtow"]

        x_cg = (
            x_cg_fus * m_fus
            + x_cg_w * m_wing
            + x_cg_ht * m_ht
            + x_cg_vt * m_vt
            + x_cg_prop * m_prop
            + x_cg_mot * m_mot
            + x_cg_bat * m_bat
        ) / (m_fus + m_wing + m_ht + m_vt + m_prop + m_mot + m_bat)

        outputs["data:stability:CoG"] = x_cg