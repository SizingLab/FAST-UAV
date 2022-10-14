"""
MTOW calculations
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.utils.constants import MR_PROPULSION, FW_PROPULSION


@oad.RegisterOpenMDAOSystem("fastuav.mtow")
class MTOW(om.Group):
    """
    Group containing the mtow calculation and associated constraints
    """

    def initialize(self):
        self.options.declare("propulsion_id_list",
                             default=[MR_PROPULSION],
                             values=[[MR_PROPULSION], [FW_PROPULSION], [MR_PROPULSION, FW_PROPULSION]])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]
        self.add_subsystem("mtow_calculation", MtowCalculation(propulsion_id_list=propulsion_id_list), promotes=["*"])
        self.add_subsystem("constraints", MtowConstraints(), promotes=["*"])


class MtowGuess(om.ExplicitComponent):
    """
    Computes an initial guess for the MTOW. This module is used as a preliminary calculation for sizing scenarios.
    """

    def setup(self):
        self.add_input("data:weight:mtow:k", val=np.nan, units=None)
        self.add_input("data:scenarios:payload:mass", val=np.nan, units="kg")
        self.add_output("data:weight:mtow:guess", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_M = inputs["data:weight:mtow:k"]
        m_load = inputs["data:scenarios:payload:mass"]

        m_uav_guess = (
            k_M * m_load
        )  # [kg] Estimate of the total mass (or equivalent weight of dynamic scenario)

        outputs["data:weight:mtow:guess"] = m_uav_guess

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_M = inputs["data:weight:mtow:k"]
        m_load = inputs["data:scenarios:payload:mass"]
        partials["data:weight:mtow:guess", "data:weight:mtow:k"] = m_load
        partials["data:weight:mtow:guess", "data:scenarios:payload:mass"] = k_M


class MtowCalculation(om.ExplicitComponent):
    """
    Maximum Take-Off Weight calculation
    """

    def initialize(self):
        self.options.declare("propulsion_id_list",
                             default=[MR_PROPULSION],
                             values=[[MR_PROPULSION], [FW_PROPULSION], [MR_PROPULSION, FW_PROPULSION]])

    def setup(self):
        propulsion_id_list = self.options["propulsion_id_list"]
        self.add_input("data:scenarios:payload:mass", val=0.0, units="kg")

        for propulsion_id in propulsion_id_list:
            self.add_input("data:weight:propulsion:%s:gearbox:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weight:propulsion:%s:esc:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weight:propulsion:%s:wires:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weight:propulsion:%s:motor:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weight:propulsion:%s:battery:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weight:propulsion:%s:propeller:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)

        if MR_PROPULSION in propulsion_id_list:
            self.add_input("data:weight:airframe:body:mass", val=0.0, units="kg")
            self.add_input("data:weight:airframe:arms:mass", val=0.0, units="kg")
        if FW_PROPULSION in propulsion_id_list:
            self.add_input("data:weight:airframe:wing:mass", val=0.0, units="kg")
            self.add_input("data:weight:airframe:tail:horizontal:mass", val=0.0, units="kg")
            self.add_input("data:weight:airframe:tail:vertical:mass", val=0.0, units="kg")
            self.add_input("data:weight:airframe:fuselage:mass", val=0.0, units="kg")

        self.add_output("data:weight:mtow", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]
        mtow = inputs["data:scenarios:payload:mass"]

        for propulsion_id in propulsion_id_list:
            mtow += ((inputs["data:weight:propulsion:%s:gearbox:mass" % propulsion_id]
                     + inputs["data:weight:propulsion:%s:motor:mass" % propulsion_id]
                     + inputs["data:weight:propulsion:%s:esc:mass" % propulsion_id]
                     + inputs["data:weight:propulsion:%s:propeller:mass" % propulsion_id]
                      ) * inputs["data:propulsion:%s:propeller:number" % propulsion_id]
                     + inputs["data:weight:propulsion:%s:wires:mass" % propulsion_id]
                     + inputs["data:weight:propulsion:%s:battery:mass" % propulsion_id])

        if MR_PROPULSION in propulsion_id_list:
            mtow += (inputs["data:weight:airframe:body:mass"]
                     + inputs["data:weight:airframe:arms:mass"])

        if FW_PROPULSION in propulsion_id_list:
            mtow += (inputs["data:weight:airframe:wing:mass"]
                     + inputs["data:weight:airframe:fuselage:mass"]
                     + inputs["data:weight:airframe:tail:horizontal:mass"]
                     + inputs["data:weight:airframe:tail:vertical:mass"])

        outputs["data:weight:mtow"] = mtow

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id_list = self.options["propulsion_id_list"]
        partials["data:weight:mtow",
                 "data:scenarios:payload:mass"] = 1.0

        for propulsion_id in propulsion_id_list:
            N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
            m_gearbox = inputs["data:weight:propulsion:%s:gearbox:mass" % propulsion_id]
            m_motor = inputs["data:weight:propulsion:%s:motor:mass" % propulsion_id]
            m_esc = inputs["data:weight:propulsion:%s:esc:mass" % propulsion_id]
            m_propeller = inputs["data:weight:propulsion:%s:propeller:mass" % propulsion_id]
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:gearbox:mass" % propulsion_id] = N_pro
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:motor:mass" % propulsion_id] = N_pro
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:esc:mass" % propulsion_id] = N_pro
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:propeller:mass" % propulsion_id] = N_pro
            partials["data:weight:mtow",
                     "data:propulsion:%s:propeller:number" % propulsion_id] = m_gearbox + m_motor + m_esc + m_propeller
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:wires:mass" % propulsion_id] = 1.0
            partials["data:weight:mtow",
                     "data:weight:propulsion:%s:battery:mass" % propulsion_id] = 1.0

        if MR_PROPULSION in propulsion_id_list:
            partials["data:weight:mtow",
                     "data:weight:airframe:body:mass"] = 1.0
            partials["data:weight:mtow",
                     "data:weight:airframe:arms:mass"] = 1.0

        if FW_PROPULSION in propulsion_id_list:
            partials["data:weight:mtow",
                     "data:weight:airframe:wing:mass"] = 1.0
            partials["data:weight:mtow",
                     "data:weight:airframe:fuselage:mass"] = 1.0
            partials["data:weight:mtow",
                     "data:weight:airframe:tail:horizontal:mass"] = 1.0
            partials["data:weight:mtow",
                     "data:weight:airframe:tail:vertical:mass"] = 1.0


class MtowConstraints(om.ExplicitComponent):
    """
    MTOW constraints
    """

    def setup(self):
        self.add_input("data:weight:mtow:requirement", val=np.nan, units="kg")
        self.add_input("data:weight:mtow", val=np.nan, units="kg")
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_output("data:weight:mtow:guess:constraint", units=None)
        self.add_output("data:weight:mtow:requirement:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        MTOW = inputs["data:weight:mtow:requirement"]
        m_uav = inputs["data:weight:mtow"]
        m_uav_guess = inputs["data:weight:mtow:guess"]

        mass_con = (m_uav_guess - m_uav) / m_uav  # mass convergence
        MTOW_con = (
            MTOW - m_uav
        ) / m_uav  # Max. takeoff weight specification, e.g. for endurance maximization

        outputs["data:weight:mtow:guess:constraint"] = mass_con
        outputs["data:weight:mtow:requirement:constraint"] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs["data:weight:mtow:requirement"]
        m_uav = inputs["data:weight:mtow"]
        m_uav_guess = inputs["data:weight:mtow:guess"]

        partials["data:weight:mtow:guess:constraint", "data:weight:mtow:guess",] = (
            1.0 / m_uav
        )
        partials["data:weight:mtow:guess:constraint", "data:weight:mtow"] = (
            -m_uav_guess / m_uav**2
        )

        partials["data:weight:mtow:requirement:constraint", "data:weight:mtow:requirement"] = 1.0 / m_uav
        partials["data:weight:mtow:requirement:constraint", "data:weight:mtow"] = -MTOW / m_uav**2
