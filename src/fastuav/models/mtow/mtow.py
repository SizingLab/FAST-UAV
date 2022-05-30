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
        self.add_input("data:weights:mtow:k", val=np.nan, units=None)
        self.add_input("data:scenarios:payload:mass", val=np.nan, units="kg")
        self.add_output("data:weights:mtow:guess", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_M = inputs["data:weights:mtow:k"]
        M_load = inputs["data:scenarios:payload:mass"]

        Mtotal_guess = (
            k_M * M_load
        )  # [kg] Estimate of the total mass (or equivalent weight of dynamic scenario)

        outputs["data:weights:mtow:guess"] = Mtotal_guess

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_M = inputs["data:weights:mtow:k"]
        M_load = inputs["data:scenarios:payload:mass"]
        partials["data:weights:mtow:guess", "data:weights:mtow:k"] = M_load
        partials["data:weights:mtow:guess", "data:scenarios:payload:mass"] = k_M


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
            self.add_input("data:weights:propulsion:%s:gearbox:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weights:propulsion:%s:esc:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weights:propulsion:%s:wires:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weights:propulsion:%s:motor:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weights:propulsion:%s:battery:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:weights:propulsion:%s:propeller:mass" % propulsion_id, val=0.0, units="kg")
            self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)

        if MR_PROPULSION in propulsion_id_list:
            self.add_input("data:weights:airframe:body:mass", val=0.0, units="kg")
            self.add_input("data:weights:airframe:arms:mass", val=0.0, units="kg")
        if FW_PROPULSION in propulsion_id_list:
            self.add_input("data:weights:airframe:wing:mass", val=0.0, units="kg")
            self.add_input("data:weights:airframe:tail:horizontal:mass", val=0.0, units="kg")
            self.add_input("data:weights:airframe:tail:vertical:mass", val=0.0, units="kg")
            self.add_input("data:weights:airframe:fuselage:mass", val=0.0, units="kg")

        self.add_output("data:weights:mtow", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id_list = self.options["propulsion_id_list"]
        mtow = inputs["data:scenarios:payload:mass"]

        for propulsion_id in propulsion_id_list:
            mtow += ((inputs["data:weights:propulsion:%s:gearbox:mass" % propulsion_id]
                     + inputs["data:weights:propulsion:%s:motor:mass" % propulsion_id]
                     + inputs["data:weights:propulsion:%s:esc:mass" % propulsion_id]
                     + inputs["data:weights:propulsion:%s:propeller:mass" % propulsion_id]
                      ) * inputs["data:propulsion:%s:propeller:number" % propulsion_id]
                     + inputs["data:weights:propulsion:%s:wires:mass" % propulsion_id]
                     + inputs["data:weights:propulsion:%s:battery:mass" % propulsion_id])

        if MR_PROPULSION in propulsion_id_list:
            mtow += (inputs["data:weights:airframe:body:mass"]
                     + inputs["data:weights:airframe:arms:mass"])

        if FW_PROPULSION in propulsion_id_list:
            mtow += (inputs["data:weights:airframe:wing:mass"]
                     + inputs["data:weights:airframe:fuselage:mass"]
                     + inputs["data:weights:airframe:tail:horizontal:mass"]
                     + inputs["data:weights:airframe:tail:vertical:mass"])

        outputs["data:weights:mtow"] = mtow

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id_list = self.options["propulsion_id_list"]
        partials["data:weights:mtow",
                 "data:scenarios:payload:mass"] = 1.0

        for propulsion_id in propulsion_id_list:
            N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
            m_gearbox = inputs["data:weights:propulsion:%s:gearbox:mass" % propulsion_id]
            m_motor = inputs["data:weights:propulsion:%s:motor:mass" % propulsion_id]
            m_esc = inputs["data:weights:propulsion:%s:esc:mass" % propulsion_id]
            m_propeller = inputs["data:weights:propulsion:%s:propeller:mass" % propulsion_id]
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:gearbox:mass" % propulsion_id] = N_pro
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:motor:mass" % propulsion_id] = N_pro
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:esc:mass" % propulsion_id] = N_pro
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:propeller:mass" % propulsion_id] = N_pro
            partials["data:weights:mtow",
                     "data:propulsion:%s:propeller:number" % propulsion_id] = m_gearbox + m_motor + m_esc + m_propeller
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:wires:mass" % propulsion_id] = 1.0
            partials["data:weights:mtow",
                     "data:weights:propulsion:%s:battery:mass" % propulsion_id] = 1.0

        if MR_PROPULSION in propulsion_id_list:
            partials["data:weights:mtow",
                     "data:weights:airframe:body:mass"] = 1.0
            partials["data:weights:mtow",
                     "data:weights:airframe:arms:mass"] = 1.0

        if FW_PROPULSION in propulsion_id_list:
            partials["data:weights:mtow",
                     "data:weights:airframe:wing:mass"] = 1.0
            partials["data:weights:mtow",
                     "data:weights:airframe:fuselage:mass"] = 1.0
            partials["data:weights:mtow",
                     "data:weights:airframe:tail:horizontal:mass"] = 1.0
            partials["data:weights:mtow",
                     "data:weights:airframe:tail:vertical:mass"] = 1.0


class MtowConstraints(om.ExplicitComponent):
    """
    MTOW constraints
    """

    def setup(self):
        self.add_input("data:weights:mtow:requirement", val=np.nan, units="kg")
        self.add_input("data:weights:mtow", val=np.nan, units="kg")
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_output("data:weights:mtow:guess:constraint", units=None)
        self.add_output("data:weights:mtow:requirement:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        MTOW = inputs["data:weights:mtow:requirement"]
        Mtotal = inputs["data:weights:mtow"]
        Mtotal_guess = inputs["data:weights:mtow:guess"]

        mass_con = (Mtotal_guess - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (
            MTOW - Mtotal
        ) / Mtotal  # Max. takeoff weight specification, e.g. for endurance maximization

        outputs["data:weights:mtow:guess:constraint"] = mass_con
        outputs["data:weights:mtow:requirement:constraint"] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs["data:weights:mtow:requirement"]
        Mtotal = inputs["data:weights:mtow"]
        Mtotal_guess = inputs["data:weights:mtow:guess"]

        partials["data:weights:mtow:guess:constraint", "data:weights:mtow:guess",] = (
            1.0 / Mtotal
        )
        partials["data:weights:mtow:guess:constraint", "data:weights:mtow"] = (
            -Mtotal_guess / Mtotal**2
        )

        partials["data:weights:mtow:requirement:constraint", "data:weights:mtow:requirement"] = 1.0 / Mtotal
        partials["data:weights:mtow:requirement:constraint", "data:weights:mtow"] = -MTOW / Mtotal**2
