"""
Global performance
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("performance.multirotor")
class PerformanceMultirotor(om.Group):
    """
    Group containing the global performance parameters for Multirotors
    """

    def setup(self):
        self.add_subsystem("MTOW", MTOW_MR(), promotes=["*"])
        self.add_subsystem("hover_autonomy", MaxHoverAutonomy(), promotes=["*"])
        self.add_subsystem("max_range", MaxRange(), promotes=["*"])
        self.add_subsystem("constraints", SystemConstraints(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("performance.fixedwing")
class PerformanceFixedWing(om.Group):
    """
    Group containing the global performance parameters for Fixed Wings
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
        self.add_input("data:weights:gearbox:mass", val=0.0, units="kg")
        self.add_input("data:weights:esc:mass", val=0.0, units="kg")
        self.add_input("data:weights:cables:mass", val=0.0, units="kg")
        self.add_input("data:weights:motor:mass", val=0.0, units="kg")
        self.add_input("data:weights:battery:mass", val=0.0, units="kg")
        self.add_input("data:weights:propeller:mass", val=0.0, units="kg")
        self.add_input("data:weights:body:mass", val=0.0, units="kg")
        self.add_input("data:weights:arms:mass", val=0.0, units="kg")
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_output("data:weights:MTOW", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mgear = inputs["data:weights:gearbox:mass"]
        Mcables = inputs["data:weights:cables:mass"]
        Mmot = inputs["data:weights:motor:mass"]
        Mesc = inputs["data:weights:esc:mass"]
        Mbat = inputs["data:weights:battery:mass"]
        Mpro = inputs["data:weights:propeller:mass"]
        Npro = inputs["data:propulsion:propeller:number"]
        M_load = inputs["specifications:payload:mass"]
        Mbody = inputs["data:weights:body:mass"]
        Marm = inputs["data:weights:arms:mass"]

        Mtotal = (
            (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mbody + Marm + Mcables
        )  # total mass

        outputs["data:weights:MTOW"] = Mtotal


class MTOW_FW(om.ExplicitComponent):
    """
    MTOW calculation for Fixed Wing configurations
    """

    def setup(self):
        self.add_input("data:weights:gearbox:mass", val=0.0, units="kg")
        self.add_input("data:weights:esc:mass", val=0.0, units="kg")
        self.add_input("data:weights:cables:mass", val=0.0, units="kg")
        self.add_input("data:weights:motor:mass", val=0.0, units="kg")
        self.add_input("data:weights:battery:mass", val=0.0, units="kg")
        self.add_input("data:weights:propeller:mass", val=0.0, units="kg")
        self.add_input("data:weights:wing:mass", val=0.0, units="kg")
        self.add_input("data:weights:tail:horizontal:mass", val=0.0, units="kg")
        self.add_input("data:weights:tail:vertical:mass", val=0.0, units="kg")
        self.add_input("data:weights:fuselage:mass", val=0.0, units="kg")
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_output("data:weights:MTOW", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mgear = inputs["data:weights:gearbox:mass"]
        Mcables = inputs["data:weights:cables:mass"]
        Mmot = inputs["data:weights:motor:mass"]
        Mesc = inputs["data:weights:esc:mass"]
        Mbat = inputs["data:weights:battery:mass"]
        Mpro = inputs["data:weights:propeller:mass"]
        Npro = inputs["data:propulsion:propeller:number"]
        M_load = inputs["specifications:payload:mass"]
        Mwing = inputs["data:weights:wing:mass"]
        Mfus = inputs["data:weights:fuselage:mass"]
        M_HT = inputs["data:weights:tail:horizontal:mass"]
        M_VT = inputs["data:weights:tail:vertical:mass"]

        Mtotal = (
            (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mwing + Mfus + M_HT + M_VT + Mcables
        )  # total mass

        outputs["data:weights:MTOW"] = Mtotal


class MaxHoverAutonomy(om.ExplicitComponent):
    """
    Max. Hover autonomy calculation.
    Payload and avionics power consumption are taken into account.
    """

    def setup(self):
        self.add_input("data:propulsion:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:propulsion:battery:DoD:max", val=0.8, units=None)
        self.add_input("data:propulsion:battery:current:hover", val=np.nan, units="A")
        self.add_output("data:performance:endurance:hover:max", units="min")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_ratio = inputs["data:propulsion:battery:DoD:max"]
        C_bat = inputs["data:propulsion:battery:capacity"]
        I_bat_hov = inputs["data:propulsion:battery:current:hover"]

        t_hov_max = C_ratio * C_bat / I_bat_hov  # [s] Max. hover flight time

        outputs["data:performance:endurance:hover:max"] = t_hov_max / 60  # [min]


class MaxRange(om.ExplicitComponent):
    """
    Max. Range calculation at given cruise speed (without hover and climb requirements)
    Payload and avionics power consumption are also taken into account.
    """

    def setup(self):
        self.add_input("mission:design_mission:cruise:speed", val=np.nan, units="m/s")
        self.add_input("data:propulsion:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:propulsion:battery:DoD:max", val=0.8, units=None)
        self.add_input("data:propulsion:battery:current:cruise", val=np.nan, units="A")
        self.add_output("data:performance:endurance:cruise:max", units="min")
        self.add_output("data:performance:range:max", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        V_cr = inputs["mission:design_mission:cruise:speed"]
        C_ratio = inputs["data:propulsion:battery:DoD:max"]
        C_bat = inputs["data:propulsion:battery:capacity"]
        I_bat_cr = inputs["data:propulsion:battery:current:cruise"]

        t_cr_max = C_ratio * C_bat / I_bat_cr  # [s] Max. cruise flight time
        D_cr_max = V_cr * t_cr_max  # [m] Max. Range at given cruise speed

        outputs["data:performance:endurance:cruise:max"] = t_cr_max / 60.0  # [min]
        outputs["data:performance:range:max"] = D_cr_max  # [m]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cr = inputs["mission:design_mission:cruise:speed"]
        C_ratio = inputs["data:propulsion:battery:DoD:max"]
        C_bat = inputs["data:propulsion:battery:capacity"]
        I_bat_cr = inputs["data:propulsion:battery:current:cruise"]
        t_cr_max = C_ratio * C_bat / I_bat_cr

        partials["data:performance:endurance:cruise:max", "data:propulsion:battery:DoD:max"] = C_bat / I_bat_cr / 60.0
        partials["data:performance:endurance:cruise:max", "data:propulsion:battery:capacity"] = C_ratio / I_bat_cr / 60.0
        partials["data:performance:endurance:cruise:max", "data:propulsion:battery:current:cruise"] = - C_ratio * C_bat / I_bat_cr**2 / 60.0

        partials["data:performance:range:max", "mission:design_mission:cruise:speed"] = t_cr_max
        partials["data:performance:range:max", "data:propulsion:battery:DoD:max"] = C_bat / I_bat_cr * V_cr
        partials["data:performance:range:max", "data:propulsion:battery:capacity"] = C_ratio / I_bat_cr * V_cr
        partials["data:performance:range:max", "data:propulsion:battery:current:cruise"] = - V_cr * C_ratio * C_bat / I_bat_cr**2


class SystemConstraints(om.ExplicitComponent):
    """
    Performance constraints
    """

    def setup(self):
        self.add_input("specifications:MTOW", val=np.nan, units="kg")
        self.add_input("data:weights:MTOW", val=np.nan, units="kg")
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_output("data:weights:MTOW:guess:constraint", units=None)
        self.add_output("data:weights:MTOW:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:weights:MTOW"]
        Mtotal_guess = inputs["data:weights:MTOW:guess"]

        mass_con = (Mtotal_guess - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, e.g. for endurance maximization

        outputs["data:weights:MTOW:guess:constraint"] = mass_con
        outputs["data:weights:MTOW:constraint"] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:weights:MTOW"]
        Mtotal_guess = inputs["data:weights:MTOW:guess"]

        partials[
            "data:weights:MTOW:guess:constraint",
            "data:weights:MTOW:guess",
        ] = (
            1.0 / Mtotal
        )
        partials["data:weights:MTOW:guess:constraint", "data:weights:MTOW"] = (
            - Mtotal_guess / Mtotal**2
        )

        partials["data:weights:MTOW:constraint", "specifications:MTOW"] = (
            1.0 / Mtotal
        )
        partials["data:weights:MTOW:constraint", "data:weights:MTOW"] = (
            - MTOW / Mtotal**2
        )
