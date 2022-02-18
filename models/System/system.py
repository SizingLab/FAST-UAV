"""
System parameters
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("system.multirotor")
class System(om.Group):
    """
    Group containing the system parameters
    """

    def setup(self):
        self.add_subsystem("MTOW", MTOW(), promotes=["*"])
        self.add_subsystem("max_autonomy", MaxHoverAutonomy(), promotes=["*"])
        self.add_subsystem("max_range", MaxRange(), promotes=["*"])
        self.add_subsystem("constraints", SystemConstraints(), promotes=["*"])


class MTOW(om.ExplicitComponent):
    """
    MTOW calculation
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
        self.add_input("specifications:payload:mass:max", val=np.nan, units="kg")
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
        M_load = inputs["specifications:payload:mass:max"]
        Mfra = inputs["data:airframe:body:mass"]
        Marm = inputs["data:airframe:arms:mass"]

        Mtotal = (
            (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm + Mcables
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
        self.add_output("mission:design_mission:hover:duration:max", units="min")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_ratio = inputs["data:battery:DoD:max"]
        C_bat = inputs["data:battery:capacity"]
        I_bat_hov = inputs["data:battery:current:hover"]

        t_hov_max = C_ratio * C_bat / I_bat_hov  # [s] Max. hover flight time

        outputs["mission:design_mission:hover:duration:max"] = t_hov_max / 60  # [min]


class MaxRange(om.ExplicitComponent):
    """
    Max. Range calculation at given V_ff speed (without hover and climb requirements)
    Payload and avionics power consumption are also taken into account.
    """

    def setup(self):
        self.add_input("mission:design_mission:forward:speed", val=np.nan, units="m/s")
        self.add_input("data:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:battery:DoD:max", val=0.8, units=None)
        self.add_input("data:battery:current:forward", val=np.nan, units="A")
        self.add_output("mission:design_mission:forward:distance:max", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        V_ff = inputs["mission:design_mission:forward:speed"]
        C_ratio = inputs["data:battery:DoD:max"]
        C_bat = inputs["data:battery:capacity"]
        I_bat_ff = inputs["data:battery:current:forward"]

        D_ff_max = V_ff * (C_ratio * C_bat) / I_bat_ff  # [m] Max. Range

        outputs["mission:design_mission:forward:distance:max"] = D_ff_max  # [m]


class SystemConstraints(om.ExplicitComponent):
    """
    System constraints
    """

    def setup(self):
        self.add_input("specifications:MTOW", val=np.nan, units="kg")
        self.add_input("data:system:MTOW", val=np.nan, units="kg")
        self.add_input("data:system:MTOW:guess", val=np.nan, units="kg")
        self.add_output("data:system:constraints:mass:convergence", units=None)
        self.add_output("data:system:constraints:mass:MTOW", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:system:MTOW"]
        Mtotal_guess = inputs[
            "data:system:MTOW:guess"
        ]  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)

        mass_con = (Mtotal_guess - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (
            MTOW - Mtotal
        ) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        outputs["data:system:constraints:mass:convergence"] = mass_con
        outputs["data:system:constraints:mass:MTOW"] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs["specifications:MTOW"]
        Mtotal = inputs["data:system:MTOW"]
        Mtotal_guess = inputs["data:system:MTOW:guess"]

        partials[
            "data:system:constraints:mass:convergence",
            "data:system:MTOW:guess",
        ] = (
            1 / Mtotal
        )
        partials["data:system:constraints:mass:convergence", "data:system:MTOW",] = (
            -Mtotal_guess / Mtotal**2
        )

        partials["data:system:constraints:mass:MTOW", "specifications:MTOW",] = (
            1.0 / Mtotal
        )
        partials["data:system:constraints:mass:MTOW", "data:system:MTOW",] = (
            -MTOW / Mtotal**2
        )
