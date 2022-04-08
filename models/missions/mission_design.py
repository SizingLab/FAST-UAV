"""
Design mission
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("mission.design")
class Mission(om.Group):
    """
    Group containing the design mission parameters
    """

    def setup(self):
        self.add_subsystem("climb_segment", ClimbSegment(), promotes=["*"])
        self.add_subsystem("hover_segment", HoverSegment(), promotes=["*"])
        self.add_subsystem("cruise_segment", CruiseSegment(), promotes=["*"])
        self.add_subsystem("mission", MissionComponent(), promotes=["*"])
        self.add_subsystem("constraints", MissionConstraints(), promotes=["*"])


class ClimbSegment(om.ExplicitComponent):
    """
    Climb segment
    """

    def setup(self):
        self.add_input("mission:design_mission:climb:height", val=np.nan, units="m")
        self.add_input("mission:design_mission:climb:speed", val=np.nan, units="m/s")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("mission:design_mission:climb:duration", units="min")
        self.add_output("mission:design_mission:climb:energy:propulsion", units="kJ")
        self.add_output("mission:design_mission:climb:energy:payload", units="kJ")
        self.add_output("mission:design_mission:climb:energy:avionics", units="kJ")
        self.add_output("mission:design_mission:climb:energy", units="kJ")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        D_cl = inputs["mission:design_mission:climb:height"]
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_cl = inputs["data:propulsion:motor:power:climb"]
        eta_ESC = inputs["data:propulsion:esc:efficiency"]
        V_cl = inputs["mission:design_mission:climb:speed"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        t_cl = D_cl / V_cl  # [s]
        E_cl_pro = (
            (P_el_cl * Npro) / eta_ESC * t_cl
        )  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_cl  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_cl  # [J] consumed energy for avionics

        outputs["mission:design_mission:climb:duration"] = t_cl / 60  # [min]
        outputs["mission:design_mission:climb:energy:propulsion"] = (
            E_cl_pro / 1000
        )  # [kJ]
        outputs["mission:design_mission:climb:energy:payload"] = (
            E_payload / 1000
        )  # [kJ]
        outputs["mission:design_mission:climb:energy:avionics"] = (
            E_avionics / 1000
        )  # [kJ]
        outputs["mission:design_mission:climb:energy"] = (
            E_cl_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class HoverSegment(om.ExplicitComponent):
    """
    Hover segment
    """

    def setup(self):
        self.add_input("mission:design_mission:hover:duration", val=np.nan, units="min")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:hover", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("mission:design_mission:hover:energy:propulsion", units="kJ")
        self.add_output("mission:design_mission:hover:energy:payload", units="kJ")
        self.add_output("mission:design_mission:hover:energy:avionics", units="kJ")
        self.add_output("mission:design_mission:hover:energy", units="kJ")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        t_hov = inputs["mission:design_mission:hover:duration"] * 60  # [s]
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_hover = inputs["data:propulsion:motor:power:hover"]
        eta_ESC = inputs["data:propulsion:esc:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        E_hover_pro = (
            (P_el_hover * Npro) / eta_ESC * t_hov
        )  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_hov  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_hov  # [J] consumed energy for avionics

        outputs["mission:design_mission:hover:energy:propulsion"] = (
            E_hover_pro / 1000
        )  # [kJ]
        outputs["mission:design_mission:hover:energy:payload"] = (
            E_payload / 1000
        )  # [kJ]
        outputs["mission:design_mission:hover:energy:avionics"] = (
            E_avionics / 1000
        )  # [kJ]
        outputs["mission:design_mission:hover:energy"] = (
            E_hover_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class CruiseSegment(om.ExplicitComponent):
    """
    Cruise segment
    """

    def setup(self):
        self.add_input("mission:design_mission:cruise:distance", val=np.nan, units="m")
        self.add_input("mission:design_mission:cruise:speed", val=np.nan, units="m/s")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:cruise", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("mission:design_mission:cruise:duration", units="min")
        self.add_output("mission:design_mission:cruise:energy:propulsion", units="kJ")
        self.add_output("mission:design_mission:cruise:energy:payload", units="kJ")
        self.add_output("mission:design_mission:cruise:energy:avionics", units="kJ")
        self.add_output("mission:design_mission:cruise:energy", units="kJ")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        D_cr = inputs["mission:design_mission:cruise:distance"]
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_cr = inputs["data:propulsion:motor:power:cruise"]
        eta_ESC = inputs["data:propulsion:esc:efficiency"]
        V_cr = inputs["mission:design_mission:cruise:speed"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        t_cr = D_cr / V_cr  # [s]
        E_cr_pro = (
            (P_el_cr * Npro) / eta_ESC * t_cr
        )  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_cr  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_cr  # [J] consumed energy for avionics

        outputs["mission:design_mission:cruise:duration"] = t_cr / 60  # [min]
        outputs["mission:design_mission:cruise:energy:propulsion"] = (
            E_cr_pro / 1000
        )  # [kJ]
        outputs["mission:design_mission:cruise:energy:payload"] = (
            E_payload / 1000
        )  # [kJ]
        outputs["mission:design_mission:cruise:energy:avionics"] = (
            E_avionics / 1000
        )  # [kJ]
        outputs["mission:design_mission:cruise:energy"] = (
            E_cr_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class MissionComponent(om.ExplicitComponent):
    """
    Overall nominal mission - energy and duration
    """

    def setup(self):
        self.add_input("mission:design_mission:hover:energy", val=np.nan, units="kJ")
        self.add_input("mission:design_mission:climb:energy", val=np.nan, units="kJ")
        self.add_input("mission:design_mission:cruise:energy", val=np.nan, units="kJ")
        self.add_input("mission:design_mission:hover:duration", val=np.nan, units="min")
        self.add_input("mission:design_mission:climb:duration", val=np.nan, units="min")
        self.add_input(
            "mission:design_mission:cruise:duration", val=np.nan, units="min"
        )
        self.add_output("mission:design_mission:energy", units="kJ")
        self.add_output("mission:design_mission:duration", units="min")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        E_hov = inputs["mission:design_mission:hover:energy"]
        E_cl = inputs["mission:design_mission:climb:energy"]
        E_cr = inputs["mission:design_mission:cruise:energy"]
        t_hov = inputs["mission:design_mission:hover:duration"]
        t_cl = inputs["mission:design_mission:climb:duration"]
        t_cr = inputs["mission:design_mission:cruise:duration"]

        t_mission = t_hov + t_cl + t_cr
        E_mission = E_hov + E_cl + E_cr

        outputs["mission:design_mission:energy"] = E_mission
        outputs["mission:design_mission:duration"] = t_mission


class MissionConstraints(om.ExplicitComponent):
    """
    Nominal mission constraints
    """

    def setup(self):
        self.add_input("mission:design_mission:energy", val=np.nan, units="kJ")
        self.add_input("data:propulsion:battery:energy", val=np.nan, units="kJ")
        self.add_input("data:propulsion:battery:DoD:max", val=0.8, units=None)
        self.add_output("mission:design_mission:constraints:energy", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        E_mission = inputs["mission:design_mission:energy"]
        E_bat = inputs["data:propulsion:battery:energy"]
        C_ratio = inputs["data:propulsion:battery:DoD:max"]

        energy_con = (E_bat * C_ratio - E_mission) / (E_bat * C_ratio)

        outputs["mission:design_mission:constraints:energy"] = energy_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_mission = inputs["mission:design_mission:energy"]
        E_bat = inputs["data:propulsion:battery:energy"]
        C_ratio = inputs["data:propulsion:battery:DoD:max"]

        partials[
            "mission:design_mission:constraints:energy",
            "mission:design_mission:energy",
        ] = -1.0 / (E_bat * C_ratio)
        partials[
            "mission:design_mission:constraints:energy",
            "data:propulsion:battery:energy",
        ] = E_mission / (E_bat**2 * C_ratio)
        partials[
            "mission:design_mission:constraints:energy",
            "data:propulsion:battery:DoD:max",
        ] = E_mission / (E_bat * C_ratio**2)
