"""
System parameters
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("system.multirotor")
class SystemMultirotor(om.Group):
    """
    Group containing the system parameters for Multirotors
    """

    def setup(self):
        self.add_subsystem("MTOW", MTOW_MR(), promotes=["*"])
        self.add_subsystem("hover_autonomy", MaxHoverAutonomy(), promotes=["*"])
        self.add_subsystem("max_range", MaxRange(), promotes=["*"])
        self.add_subsystem("constraints", SystemConstraints(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("system.fixedwing")
class SystemFixedWing(om.Group):
    """
    Group containing the system parameters for Fixed Wings
    """

    def setup(self):
        self.add_subsystem("MTOW", MTOW_FW(), promotes=["*"])
        self.add_subsystem("max_range", MaxRange(), promotes=["*"])
        self.add_subsystem("constraints", SystemConstraints(), promotes=["*"])


class MTOW_MR(om.ExplicitComponent):
    """
    MTOW calculation for Multirotor configurations
    """

    def setup(self):
        self.add_input("data:gearbox:mass", val=0.0, units="kg")
        self.add_input("data:ESC:mass", val=0.0, units="kg")
        self.add_input("data:cables:mass", val=0.0, units="kg")
        self.add_input("data:motor:mass", val=0.0, units="kg")
        self.add_input("data:battery:mass", val=0.0, units="kg")
        self.add_input("data:propeller:mass", val=0.0, units="kg")
        self.add_input("data:airframe:body:mass", val=0.0, units="kg")
        self.add_input("data:airframe:arms:mass", val=0.0, units="kg")
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_output("data:system:MTOW", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mgear = inputs["data:gearbox:mass"]
        Mcables = inputs["data:cables:mass"]
        Mmot = inputs["data:motor:mass"]
        Mesc = inputs["data:ESC:mass"]
        Mbat = inputs["data:battery:mass"]
        Mpro = inputs["data:propeller:mass"]
        Npro = inputs["data:propeller:number"]
        M_load = inputs["specifications:payload:mass"]
        Mbody = inputs["data:airframe:body:mass"]
        Marm = inputs["data:airframe:arms:mass"]

        Mtotal = (
            (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mbody + Marm + Mcables
        )  # total mass

        outputs["data:system:MTOW"] = Mtotal


class MTOW_FW(om.ExplicitComponent):
    """
    MTOW calculation for Fixed Wing configurations
    """

    def setup(self):
        self.add_input("data:gearbox:mass", val=0.0, units="kg")
        self.add_input("data:ESC:mass", val=0.0, units="kg")
        self.add_input("data:cables:mass", val=0.0, units="kg")
        self.add_input("data:motor:mass", val=0.0, units="kg")
        self.add_input("data:battery:mass", val=0.0, units="kg")
        self.add_input("data:propeller:mass", val=0.0, units="kg")
        self.add_input("data:airframe:wing:mass", val=0.0, units="kg")
        self.add_input("data:airframe:tail:horizontal:mass", val=0.0, units="kg")
        self.add_input("data:airframe:tail:vertical:mass", val=0.0, units="kg")
        self.add_input("data:airframe:fuselage:mass", val=0.0, units="kg")
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_output("data:system:MTOW", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mgear = inputs["data:gearbox:mass"]
        Mcables = inputs["data:cables:mass"]
        Mmot = inputs["data:motor:mass"]
        Mesc = inputs["data:ESC:mass"]
        Mbat = inputs["data:battery:mass"]
        Mpro = inputs["data:propeller:mass"]
        Npro = inputs["data:propeller:number"]
        M_load = inputs["specifications:payload:mass"]
        Mwing = inputs["data:airframe:wing:mass"]
        Mfus = inputs["data:airframe:fuselage:mass"]
        M_HT = inputs["data:airframe:tail:horizontal:mass"]
        M_VT = inputs["data:airframe:tail:vertical:mass"]

        Mtotal = (
            (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mwing + Mfus + M_HT + M_VT + Mcables
        )  # total mass

        outputs["data:system:MTOW"] = Mtotal


class MaxHoverAutonomy(om.ExplicitComponent):
    """
    Max. Hover autonomy calculation.
    Payload and avionics power consumption are taken into account.
    """

    def setup(self):
        self.add_input("data:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:battery:DoD:max", val=0.8, units=None)
        self.add_input("data:battery:current:hover", val=np.nan, units="A")
        self.add_output("data:system:endurance:hover:max", units="min")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_ratio = inputs["data:battery:DoD:max"]
        C_bat = inputs["data:battery:capacity"]
        I_bat_hov = inputs["data:battery:current:hover"]

        t_hov_max = C_ratio * C_bat / I_bat_hov  # [s] Max. hover flight time

        outputs["data:system:endurance:hover:max"] = t_hov_max / 60  # [min]


class MaxRange(om.ExplicitComponent):
    """
    Max. Range calculation at given cruise speed (without hover and climb requirements)
    Payload and avionics power consumption are also taken into account.
    """

    def setup(self):
        self.add_input("mission:design_mission:cruise:speed", val=np.nan, units="m/s")
        self.add_input("data:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:battery:DoD:max", val=0.8, units=None)
        self.add_input("data:battery:current:cruise", val=np.nan, units="A")
        self.add_output("data:system:endurance:cruise:max", units="min")
        self.add_output("data:system:range:max", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        V_cr = inputs["mission:design_mission:cruise:speed"]
        C_ratio = inputs["data:battery:DoD:max"]
        C_bat = inputs["data:battery:capacity"]
        I_bat_cr = inputs["data:battery:current:cruise"]

        t_cr_max = C_ratio * C_bat / I_bat_cr  # [s] Max. cruise flight time
        D_cr_max = V_cr * t_cr_max  # [m] Max. Range at given cruise speed

        outputs["data:system:endurance:cruise:max"] = t_cr_max / 60.0  # [min]
        outputs["data:system:range:max"] = D_cr_max  # [m]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cr = inputs["mission:design_mission:cruise:speed"]
        C_ratio = inputs["data:battery:DoD:max"]
        C_bat = inputs["data:battery:capacity"]
        I_bat_cr = inputs["data:battery:current:cruise"]
        t_cr_max = C_ratio * C_bat / I_bat_cr

        partials["data:system:endurance:cruise:max", "data:battery:DoD:max"] = C_bat / I_bat_cr / 60.0
        partials["data:system:endurance:cruise:max", "data:battery:capacity"] = C_ratio / I_bat_cr / 60.0
        partials["data:system:endurance:cruise:max", "data:battery:current:cruise"] = - C_ratio * C_bat / I_bat_cr**2 / 60.0

        partials["data:system:range:max", "mission:design_mission:cruise:speed"] = t_cr_max
        partials["data:system:range:max", "data:battery:DoD:max"] = C_bat / I_bat_cr * V_cr
        partials["data:system:range:max", "data:battery:capacity"] = C_ratio / I_bat_cr * V_cr
        partials["data:system:range:max", "data:battery:current:cruise"] = - V_cr * C_ratio * C_bat / I_bat_cr**2


class SystemConstraints(om.ExplicitComponent):
    """
    System constraints
    """

    def setup(self):
        self.add_input("specifications:MTOW", val=np.nan, units="kg")
        self.add_input("data:system:MTOW", val=np.nan, units="kg")
        self.add_input("data:system:MTOW:guess", val=np.nan, units="kg")
        self.add_output("data:system:constraints:mass:consistency", units=None)
        self.add_output("data:system:constraints:mass:MTOW", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:system:MTOW"]
        Mtotal_guess = inputs["data:system:MTOW:guess"]

        mass_con = (Mtotal_guess - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, e.g. for endurance maximization

        outputs["data:system:constraints:mass:consistency"] = mass_con
        outputs["data:system:constraints:mass:MTOW"] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:system:MTOW"]
        Mtotal_guess = inputs["data:system:MTOW:guess"]

        partials[
            "data:system:constraints:mass:consistency",
            "data:system:MTOW:guess",
        ] = (
            1.0 / Mtotal
        )
        partials["data:system:constraints:mass:consistency", "data:system:MTOW"] = (
            - Mtotal_guess / Mtotal**2
        )

        partials["data:system:constraints:mass:MTOW", "specifications:MTOW"] = (
            1.0 / Mtotal
        )
        partials["data:system:constraints:mass:MTOW", "data:system:MTOW"] = (
            - MTOW / Mtotal**2
        )
