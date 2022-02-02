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
        self.add_subsystem("forward", Forward(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Performances calculation of ESC for takeoff
    """

    def setup(self):
        self.add_input("data:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_output("data:ESC:power:takeoff", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_to = inputs["data:motor:power:takeoff"]
        U_mot_to = inputs["data:motor:voltage:takeoff"]
        V_bat = inputs["data:battery:voltage"]

        # P_esc_to = P_mot_to * V_bat / U_mot_to  # [W] electronic power takeoff
        P_esc_to = ESCModel.power(
            P_mot_to, U_mot_to, V_bat
        )  # [W] electronic power takeoff

        outputs["data:ESC:power:takeoff"] = P_esc_to


class Hover(om.ExplicitComponent):
    """
    Performances calculation of ESC for hover
    """

    def setup(self):
        self.add_input("data:motor:power:hover", val=np.nan, units="W")
        self.add_input("data:motor:voltage:hover", val=np.nan, units="V")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_output("data:ESC:power:hover", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_hover = inputs["data:motor:power:hover"]
        U_mot_hover = inputs["data:motor:voltage:hover"]
        V_bat = inputs["data:battery:voltage"]

        # P_esc_hover = P_el_hover * V_bat / Umot_hover  # [W] electronic power hover
        P_esc_hover = ESCModel.power(
            P_mot_hover, U_mot_hover, V_bat
        )  # [W] electronic power hover

        outputs["data:ESC:power:hover"] = P_esc_hover


class Climb(om.ExplicitComponent):
    """
    Performances calculation of ESC for climb
    """

    def setup(self):
        self.add_input("data:motor:power:climb", val=np.nan, units="W")
        self.add_input("data:motor:voltage:climb", val=np.nan, units="V")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_output("data:ESC:power:climb", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_mot_cl = inputs["data:motor:power:climb"]
        U_mot_cl = inputs["data:motor:voltage:climb"]
        V_bat = inputs["data:battery:voltage"]

        # P_esc_cl = P_el_cl * V_bat / Umot_cl  # [W] electronic power max climb
        P_esc_cl = ESCModel.power(
            P_mot_cl, U_mot_cl, V_bat
        )  # [W] electronic power max climb

        outputs["data:ESC:power:climb"] = P_esc_cl


class Forward(om.ExplicitComponent):
    """
    Performances calculation of ESC for forward flight
    """

    def setup(self):
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_input("data:motor:power:forward", val=np.nan, units="W")
        self.add_input("data:motor:voltage:forward", val=np.nan, units="V")
        self.add_output("data:ESC:power:forward", units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        V_bat = inputs["data:battery:voltage"]
        P_mot_ff = inputs["data:motor:power:forward"]
        U_mot_ff = inputs["data:motor:voltage:forward"]

        # P_esc_ff = P_el_ff * V_bat / Umot_ff # [W] electronic power max forward
        P_esc_ff = ESCModel.power(
            P_mot_ff, U_mot_ff, V_bat
        )  # [W] electronic power max forward

        outputs["data:ESC:power:forward"] = P_esc_ff
