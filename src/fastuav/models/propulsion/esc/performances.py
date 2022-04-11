"""
ESC performances
"""
import openmdao.api as om
import numpy as np


class ESCModel:
    """
    ESC model for performances calculation
    """

    @staticmethod
    def power(P_mot, U_mot, V_bat):
        P_esc = P_mot * V_bat / U_mot  # [W] electronic power
        return P_esc


class ESCPerfos(om.Group):
    """
    Group containing the performance functions of the ESC
    """

    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("cruise", Cruise(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Performances calculation of ESC for takeoff
    """

    def setup(self):
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:takeoff", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        V_bat = inputs["data:propulsion:battery:voltage"]

        # P_esc_to = P_mot_to * V_bat / U_mot_to  # [W] electronic power takeoff
        P_esc_to = ESCModel.power(P_mot_to, U_mot_to, V_bat)  # [W] electronic power takeoff

        outputs["data:propulsion:esc:power:takeoff"] = P_esc_to


class Hover(om.ExplicitComponent):
    """
    Performances calculation of ESC for hover
    """

    def setup(self):
        self.add_input("data:propulsion:motor:power:hover", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:hover", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:hover", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_hover = inputs["data:propulsion:motor:power:hover"]
        U_mot_hover = inputs["data:propulsion:motor:voltage:hover"]
        V_bat = inputs["data:propulsion:battery:voltage"]

        # P_esc_hover = P_el_hover * V_bat / Umot_hover  # [W] electronic power hover
        P_esc_hover = ESCModel.power(P_mot_hover, U_mot_hover, V_bat)  # [W] electronic power hover

        outputs["data:propulsion:esc:power:hover"] = P_esc_hover


class Climb(om.ExplicitComponent):
    """
    Performances calculation of ESC for climb
    """

    def setup(self):
        self.add_input("data:propulsion:motor:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:climb", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:climb", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_cl = inputs["data:propulsion:motor:power:climb"]
        U_mot_cl = inputs["data:propulsion:motor:voltage:climb"]
        V_bat = inputs["data:propulsion:battery:voltage"]

        # P_esc_cl = P_el_cl * V_bat / Umot_cl  # [W] electronic power max climb
        P_esc_cl = ESCModel.power(P_mot_cl, U_mot_cl, V_bat)  # [W] electronic power max climb

        outputs["data:propulsion:esc:power:climb"] = P_esc_cl


class Cruise(om.ExplicitComponent):
    """
    Performances calculation of ESC for cruise
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:power:cruise", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:cruise", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:cruise", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_mot_cr = inputs["data:propulsion:motor:power:cruise"]
        U_mot_cr = inputs["data:propulsion:motor:voltage:cruise"]

        # P_esc_cr = P_el_cr * V_bat / Umot_cr # [W] electronic power max cruise
        P_esc_cr = ESCModel.power(P_mot_cr, U_mot_cr, V_bat)  # [W] electronic power max cruise

        outputs["data:propulsion:esc:power:cruise"] = P_esc_cr
