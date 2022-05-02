"""
Motor performance analysis
"""
import openmdao.api as om
import numpy as np


class MotorPerformanceModel:
    """
    Motor model for performances calculation
    """

    @staticmethod
    def torque(Q_pro, N_red):
        T_mot = Q_pro / N_red if N_red > 0 else 0.0  # [N.m] motor torque with reduction
        return T_mot

    @staticmethod
    def speed(W_pro, N_red):
        W_mot = W_pro * N_red  # [rad/s] Motor speed with reduction
        return W_mot

    @staticmethod
    def current(T_mot, Tf_mot, Kt_mot):
        I_mot = (T_mot + Tf_mot) / Kt_mot if T_mot > 0 else 0.0  # [I] Current of the motor per propeller
        return I_mot

    @staticmethod
    def voltage(I_mot, W_mot, R_mot, Kt_mot):
        U_mot = R_mot * I_mot + W_mot * Kt_mot  # [V] Voltage of the motor per propeller
        return U_mot

    @staticmethod
    def power(U_mot, I_mot):
        P_mot_el = U_mot * I_mot  # [W] electrical power
        return P_mot_el


class MotorPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the motor
    """

    def setup(self):
        self.add_subsystem("takeoff", MotorPerformance(scenario="takeoff"), promotes=["*"])
        self.add_subsystem("hover", MotorPerformance(scenario="hover"), promotes=["*"])
        self.add_subsystem("climb", MotorPerformance(scenario="climb"), promotes=["*"])
        self.add_subsystem("cruise", MotorPerformance(scenario="cruise"), promotes=["*"])


class MotorPerformance(om.ExplicitComponent):
    """
    Computes motor performances for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default="cruise", values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:propeller:speed:%s" % scenario, val=np.nan, units="rad/s")
        self.add_input("data:propulsion:propeller:torque:%s" % scenario, val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:power:%s" % scenario, units="W")
        self.add_output("data:propulsion:motor:voltage:%s" % scenario, units="V")
        self.add_output("data:propulsion:motor:current:%s" % scenario, units="A")
        self.add_output("data:propulsion:motor:torque:%s" % scenario, units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        N_red = inputs["data:propulsion:gearbox:N_red"]
        Tf_mot = inputs["data:propulsion:motor:torque:friction"]
        R_mot = inputs["data:propulsion:motor:resistance"]
        Kt_mot = inputs["data:propulsion:motor:torque:coefficient"]
        W_pro = inputs["data:propulsion:propeller:speed:%s" % scenario]
        Q_pro = inputs["data:propulsion:propeller:torque:%s" % scenario]

        T_mot = MotorPerformanceModel.torque(Q_pro, N_red)  # [N.m] motor torque with reduction
        W_mot = MotorPerformanceModel.speed(W_pro, N_red)  # [rad/s] Motor speed with reduction
        I_mot = MotorPerformanceModel.current(T_mot, Tf_mot, Kt_mot)  # [I] Current of the motor per propeller
        U_mot = MotorPerformanceModel.voltage(I_mot, W_mot, R_mot, Kt_mot)  # [V] Voltage of the motor per propeller
        P_mot_el = MotorPerformanceModel.power(U_mot, I_mot)  # [W] electrical power

        outputs["data:propulsion:motor:power:%s" % scenario] = P_mot_el
        outputs["data:propulsion:motor:voltage:%s" % scenario] = U_mot
        outputs["data:propulsion:motor:current:%s" % scenario] = I_mot
        outputs["data:propulsion:motor:torque:%s" % scenario] = T_mot
