"""
ESC performances
"""
import openmdao.api as om
import numpy as np


class ESCPerformanceModel:
    """
    ESC model for performances calculation
    """

    @staticmethod
    def power(P_mot, U_mot, V_bat):
        P_esc = P_mot * V_bat / U_mot if U_mot > 0 else 0.0  # [W] electronic power
        return P_esc


class ESCPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the ESC
    """

    def setup(self):
        self.add_subsystem("takeoff", ESCPerformance(scenario="takeoff"), promotes=["*"])
        self.add_subsystem("hover", ESCPerformance(scenario="hover"), promotes=["*"])
        self.add_subsystem("climb", ESCPerformance(scenario="climb"), promotes=["*"])
        self.add_subsystem("cruise", ESCPerformance(scenario="cruise"), promotes=["*"])


class ESCPerformance(om.ExplicitComponent):
    """
    Performances calculation of ESC for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default="cruise", values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:motor:power:%s" % scenario, val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:%s" % scenario, val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:%s" % scenario, units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        P_mot = inputs["data:propulsion:motor:power:%s" % scenario]
        U_mot = inputs["data:propulsion:motor:voltage:%s" % scenario]
        V_bat = inputs["data:propulsion:battery:voltage"]

        P_esc = ESCPerformanceModel.power(P_mot, U_mot, V_bat)  # [W] electronic power

        outputs["data:propulsion:esc:power:%s" % scenario] = P_esc

