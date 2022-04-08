"""
Battery performances
"""
import openmdao.api as om
import numpy as np


class BatteryPerfoModel:
    """
    Battery model for performances calculation
    """

    @staticmethod
    def current(P_req, eta_ESC, V_bat):
        I_bat = P_req / eta_ESC / V_bat  # [I] Current of the battery
        return I_bat


class BatteryPerfos(om.Group):
    """
    Group containing the performance functions of the battery
    """

    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("cruise", Cruise(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes performances of the battery for takeoff
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("data:propulsion:battery:current:takeoff", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_to = inputs["data:propulsion:motor:power:takeoff"]
        eta_ESC = inputs[
            "data:propulsion:esc:efficiency:estimated"
        ]  # TODO: replace by 'real' efficiency (ESC catalogue output, but be careful to algebraic loops...)
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # I_bat_to = (P_el_to * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery
        P_req = P_el_to * Npro + P_payload + P_avionics
        I_bat_to = BatteryPerfoModel.current(P_req, eta_ESC, V_bat)

        outputs["data:propulsion:battery:current:takeoff"] = I_bat_to


class Hover(om.ExplicitComponent):
    """
    Computes performances of the battery for hover
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:hover", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("data:propulsion:battery:current:hover", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_hover = inputs["data:propulsion:motor:power:hover"]
        eta_ESC = inputs["data:propulsion:esc:efficiency:estimated"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # I_bat_hov = (P_el_hover * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery
        P_req = P_el_hover * Npro + P_payload + P_avionics
        I_bat_hov = BatteryPerfoModel.current(P_req, eta_ESC, V_bat)

        outputs["data:propulsion:battery:current:hover"] = I_bat_hov


class Climb(om.ExplicitComponent):
    """
    Computes performances of the battery for climb
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("data:propulsion:battery:current:climb", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_cl = inputs["data:propulsion:motor:power:climb"]
        eta_ESC = inputs["data:propulsion:esc:efficiency:estimated"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # I_bat_cl = (P_el_cl * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery
        P_req = P_el_cl * Npro + P_payload + P_avionics
        I_bat_cl = BatteryPerfoModel.current(P_req, eta_ESC, V_bat)

        outputs["data:propulsion:battery:current:climb"] = I_bat_cl


class Cruise(om.ExplicitComponent):
    """
    Computes performances of the battery for cruise
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:cruise", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        self.add_output("data:propulsion:battery:current:cruise", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro = inputs["data:propulsion:propeller:number"]
        P_el_cr = inputs["data:propulsion:motor:power:cruise"]
        eta_ESC = inputs["data:propulsion:esc:efficiency:estimated"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # I_bat_cr = (P_el_cr * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery
        P_req = P_el_cr * Npro + P_payload + P_avionics
        I_bat_cr = BatteryPerfoModel.current(P_req, eta_ESC, V_bat)

        outputs["data:propulsion:battery:current:cruise"] = I_bat_cr
