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
        T_mot = Q_pro / N_red  # [N.m] motor torque with reduction
        return T_mot

    @staticmethod
    def speed(W_pro, N_red):
        W_mot = W_pro * N_red  # [rad/s] Motor speed with reduction
        return W_mot

    @staticmethod
    def current(T_mot, Tf_mot, Kv):
        I_mot = (T_mot + Tf_mot) * Kv  # [I] Current of the motor
        return I_mot

    @staticmethod
    def voltage(I_mot, W_mot, R, Kv):
        U_mot = R * I_mot + W_mot / Kv  # [V] Voltage of the motor
        return U_mot

    @staticmethod
    def power(U_mot, I_mot):
        P_mot_el = U_mot * I_mot  # [W] electrical power
        return P_mot_el

    @staticmethod
    def efficiency(U_mot, I_mot, T_mot, W_mot):
        P_mot_el = U_mot * I_mot
        P_mot_mech = T_mot * W_mot
        eta_mot = P_mot_mech / P_mot_el
        return eta_mot


class MotorPerformanceGroup(om.Group):
    """
    Group containing the performance functions of the motor
    """

    def setup(self):
        self.add_subsystem("takeoff", MotorPerformance(scenario="takeoff"), promotes=["*"])
        self.add_subsystem("hover", MotorPerformance(scenario="hover"), promotes=["*"])
        self.add_subsystem("climb", MotorPerformance(scenario="climb"), promotes=["*"])
        self.add_subsystem("cruise", MotorPerformance(scenario="cruise"), promotes=["*"])


class MotorPerformance(om.Group):
    """
    Computes motor performances for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_subsystem("torque", MotorTorque(scenario=scenario), promotes=["*"])
        self.add_subsystem("speed", MotorSpeed(scenario=scenario), promotes=["*"])
        self.add_subsystem("current", MotorCurrent(scenario=scenario), promotes=["*"])
        self.add_subsystem("voltage", MotorVoltage(scenario=scenario), promotes=["*"])
        self.add_subsystem("power", MotorPower(scenario=scenario), promotes=["*"])
        self.add_subsystem("efficiency", MotorEfficiency(scenario=scenario), promotes=["*"])


class MotorTorque(om.ExplicitComponent):
    """
    Computes motor torque for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:propeller:torque:%s" % scenario, val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:%s" % scenario, units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        N_red = inputs["data:propulsion:gearbox:N_red"]
        Q_pro = inputs["data:propulsion:propeller:torque:%s" % scenario]

        T_mot = MotorPerformanceModel.torque(Q_pro, N_red)  # [N.m] motor torque with reduction

        outputs["data:propulsion:motor:torque:%s" % scenario] = T_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        N_red = inputs["data:propulsion:gearbox:N_red"]
        Q_pro = inputs["data:propulsion:propeller:torque:%s" % scenario]

        partials["data:propulsion:motor:torque:%s" % scenario,
                 "data:propulsion:gearbox:N_red"
        ] = - Q_pro / N_red ** 2

        partials["data:propulsion:motor:torque:%s" % scenario,
                 "data:propulsion:propeller:torque:%s" % scenario
        ] = 1 / N_red


class MotorSpeed(om.ExplicitComponent):
    """
    Computes motor speed for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:propeller:speed:%s" % scenario, val=np.nan, units="rad/s")
        self.add_output("data:propulsion:motor:speed:%s" % scenario, units="rad/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        N_red = inputs["data:propulsion:gearbox:N_red"]
        W_pro = inputs["data:propulsion:propeller:speed:%s" % scenario]

        W_mot = MotorPerformanceModel.speed(W_pro, N_red)  # [rad/s] Motor speed with reduction

        outputs["data:propulsion:motor:speed:%s" % scenario] = W_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        N_red = inputs["data:propulsion:gearbox:N_red"]
        W_pro = inputs["data:propulsion:propeller:speed:%s" % scenario]

        partials["data:propulsion:motor:speed:%s" % scenario,
                 "data:propulsion:gearbox:N_red"
        ] = W_pro

        partials["data:propulsion:motor:speed:%s" % scenario,
                 "data:propulsion:propeller:speed:%s" % scenario
        ] = N_red


class MotorCurrent(om.ExplicitComponent):
    """
    Computes motor current for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:motor:torque:%s" % scenario, val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:speed:constant", val=np.nan, units="rad/V/s")
        self.add_output("data:propulsion:motor:current:%s" % scenario, units="A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        T_mot = inputs["data:propulsion:motor:torque:%s" % scenario]
        Tf_mot = inputs["data:propulsion:motor:torque:friction"]
        Kv = inputs["data:propulsion:motor:speed:constant"]

        I_mot = MotorPerformanceModel.current(T_mot, Tf_mot, Kv)  # [I] Current of the motor

        outputs["data:propulsion:motor:current:%s" % scenario] = I_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        T_mot = inputs["data:propulsion:motor:torque:%s" % scenario]
        Tf_mot = inputs["data:propulsion:motor:torque:friction"]
        Kv = inputs["data:propulsion:motor:speed:constant"]

        partials["data:propulsion:motor:current:%s" % scenario,
                 "data:propulsion:motor:torque:%s" % scenario
        ] = Kv

        partials["data:propulsion:motor:current:%s" % scenario,
                 "data:propulsion:motor:torque:friction"
        ] = Kv

        partials["data:propulsion:motor:current:%s" % scenario,
                 "data:propulsion:motor:speed:constant"
        ] = T_mot + Tf_mot


class MotorVoltage(om.ExplicitComponent):
    """
    Computes motor voltage for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:speed:constant", val=np.nan, units="rad/V/s")
        self.add_input("data:propulsion:motor:speed:%s" % scenario, val=np.nan, units="rad/s")
        self.add_input("data:propulsion:motor:current:%s" % scenario, val=np.nan, units="A")
        self.add_output("data:propulsion:motor:voltage:%s" % scenario, units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        R = inputs["data:propulsion:motor:resistance"]
        Kv = inputs["data:propulsion:motor:speed:constant"]
        W_mot = inputs["data:propulsion:motor:speed:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]

        U_mot = MotorPerformanceModel.voltage(I_mot, W_mot, R, Kv)  # [V] Voltage of the motor

        outputs["data:propulsion:motor:voltage:%s" % scenario] = U_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        R = inputs["data:propulsion:motor:resistance"]
        Kv = inputs["data:propulsion:motor:speed:constant"]
        W_mot = inputs["data:propulsion:motor:speed:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]

        partials["data:propulsion:motor:voltage:%s" % scenario,
                 "data:propulsion:motor:resistance"
        ] = I_mot

        partials["data:propulsion:motor:voltage:%s" % scenario,
                 "data:propulsion:motor:speed:constant"
        ] = - W_mot / Kv ** 2

        partials["data:propulsion:motor:voltage:%s" % scenario,
                 "data:propulsion:motor:speed:%s" % scenario
        ] = 1 / Kv

        partials["data:propulsion:motor:voltage:%s" % scenario,
                 "data:propulsion:motor:current:%s" % scenario
        ] = R


class MotorPower(om.ExplicitComponent):
    """
    Computes motor electrical power for given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:motor:voltage:%s" % scenario, val=np.nan, units="V")
        self.add_input("data:propulsion:motor:current:%s" % scenario, val=np.nan, units="A")
        self.add_output("data:propulsion:motor:power:%s" % scenario, units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        U_mot = inputs["data:propulsion:motor:voltage:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]

        P_mot_el = MotorPerformanceModel.power(U_mot, I_mot)  # [W] electrical power

        outputs["data:propulsion:motor:power:%s" % scenario] = P_mot_el

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        U_mot = inputs["data:propulsion:motor:voltage:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]

        partials["data:propulsion:motor:power:%s" % scenario,
                 "data:propulsion:motor:voltage:%s" % scenario
        ] = I_mot

        partials["data:propulsion:motor:power:%s" % scenario,
                 "data:propulsion:motor:current:%s" % scenario
        ] = U_mot


class MotorEfficiency(om.ExplicitComponent):
    """
    Computes motor efficiency under given flight scenario
    """

    def initialize(self):
        self.options.declare("scenario", default=None, values=["takeoff", "climb", "hover", "cruise"])

    def setup(self):
        scenario = self.options["scenario"]
        self.add_input("data:propulsion:motor:voltage:%s" % scenario, val=np.nan, units="V")
        self.add_input("data:propulsion:motor:current:%s" % scenario, val=np.nan, units="A")
        self.add_input("data:propulsion:motor:torque:%s" % scenario, val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:speed:%s" % scenario, val=np.nan, units="rad/s")
        self.add_output("data:propulsion:motor:efficiency:%s" % scenario, units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        scenario = self.options["scenario"]
        U_mot = inputs["data:propulsion:motor:voltage:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]
        T_mot = inputs["data:propulsion:motor:torque:%s" % scenario]
        W_mot = inputs["data:propulsion:motor:speed:%s" % scenario]

        eta_mot = MotorPerformanceModel.efficiency(U_mot, I_mot, T_mot, W_mot)  # [-] efficiency

        outputs["data:propulsion:motor:efficiency:%s" % scenario] = eta_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        scenario = self.options["scenario"]
        U_mot = inputs["data:propulsion:motor:voltage:%s" % scenario]
        I_mot = inputs["data:propulsion:motor:current:%s" % scenario]
        T_mot = inputs["data:propulsion:motor:torque:%s" % scenario]
        W_mot = inputs["data:propulsion:motor:speed:%s" % scenario]

        partials["data:propulsion:motor:efficiency:%s" % scenario,
                 "data:propulsion:motor:voltage:%s" % scenario
        ] = - T_mot * W_mot / (U_mot**2 * I_mot)

        partials["data:propulsion:motor:efficiency:%s" % scenario,
                 "data:propulsion:motor:voltage:%s" % scenario
        ] = - T_mot * W_mot / (U_mot * I_mot**2)

        partials["data:propulsion:motor:efficiency:%s" % scenario,
                 "data:propulsion:motor:torque:%s" % scenario
        ] = W_mot / (U_mot * I_mot)

        partials["data:propulsion:motor:efficiency:%s" % scenario,
                 "data:propulsion:motor:speed:%s" % scenario
        ] = T_mot / (U_mot * I_mot)