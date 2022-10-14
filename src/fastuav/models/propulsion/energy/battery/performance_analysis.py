"""
Battery performance analysis
"""
import openmdao.api as om
import numpy as np


class BatteryPerformanceModel:
    """
    Battery model for performances calculation
    """

    @staticmethod
    def power(P_mot, N_pro, eta_esc, P_payload):
        P_bat = P_mot * N_pro / eta_esc + P_payload  # [W] Power of the battery
        return P_bat

    @staticmethod
    def current(P_bat, U_bat):
        I_bat = P_bat / U_bat if U_bat > 0 else 0.0  # [I] Current of the battery
        return I_bat


class BatteryPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the battery
    """

    def setup(self):
        self.add_subsystem("takeoff", BatteryPerformance(scenario="takeoff"), promotes=["*"])
        self.add_subsystem("hover", BatteryPerformance(scenario="hover"), promotes=["*"])
        self.add_subsystem("climb", BatteryPerformance(scenario="climb"), promotes=["*"])
        self.add_subsystem("cruise", BatteryPerformance(scenario="cruise"), promotes=["*"])


class BatteryPerformance(om.ExplicitComponent):
    """
    Computes performances of the battery for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default="cruise", values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:%s" % scenario, val=np.nan, units="W")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("data:scenarios:payload:power", val=np.nan, units="W")
        self.add_output("data:propulsion:battery:current:%s" % scenario, units="A")
        self.add_output("data:propulsion:battery:power:%s" % scenario, units="W")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        N_pro = inputs["data:propulsion:propeller:number"]
        P_mot = inputs["data:propulsion:motor:power:%s" % scenario]
        eta_ESC = inputs[
            "data:propulsion:esc:efficiency:estimated"
        ]  #TODO: replace by 'real' efficiency (ESC catalogue output, but be careful to algebraic loops...)
        U_bat = inputs["data:propulsion:battery:voltage"]
        P_payload = inputs["data:scenarios:payload:power"]

        P_bat = BatteryPerformanceModel.power(P_mot, N_pro,eta_ESC, P_payload)
        I_bat = BatteryPerformanceModel.current(P_bat, U_bat)

        outputs["data:propulsion:battery:power:%s" % scenario] = P_bat
        outputs["data:propulsion:battery:current:%s" % scenario] = I_bat