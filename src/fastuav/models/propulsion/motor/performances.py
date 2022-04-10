"""
Motor performances
"""
import openmdao.api as om
import numpy as np


class MotorPerfoModel:
    """
    Motor model for performances calculation
    """

    @staticmethod
    def performances(Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot):
        T_mot = Q_pro / N_red  # [N.m] motor torque with reduction
        W_mot = W_pro * N_red  # [rad/s] Motor speed with reduction
        I_mot = (T_mot + Tf_mot) / Kt_mot  # [I] Current of the motor per propeller
        U_mot = R_mot * I_mot + W_mot * Kt_mot  # [V] Voltage of the motor per propeller
        P_el = U_mot * I_mot  # [W] electrical power
        return T_mot, W_mot, I_mot, U_mot, P_el


class MotorPerfos(om.Group):
    """
    Group containing the performance functions of the motor
    """

    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("cruise", Cruise(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes motor performances for takeoff
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:propeller:speed:takeoff", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:propeller:torque:takeoff", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:power:takeoff", units="W")
        self.add_output("data:propulsion:motor:voltage:takeoff", units="V")
        self.add_output("data:propulsion:motor:current:takeoff", units="A")
        self.add_output("data:propulsion:motor:torque:takeoff", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Tfmot = inputs["data:propulsion:motor:torque:friction"]
        Rmot = inputs["data:propulsion:motor:resistance"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient"]
        Wpro_to = inputs["data:propulsion:propeller:speed:takeoff"]
        Qpro_to = inputs["data:propulsion:propeller:torque:takeoff"]

        # Tmot_to = Qpro_to / Nred  # [N.m] motor take-off torque with reduction
        # W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed with reduction
        # Imot_to = (Tmot_to + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        # Umot_to = Rmot * Imot_to + W_to_motor * Ktmot  # [V] Voltage of the motor per propeller
        # P_el_to = Umot_to * Imot_to  # [W] Takeoff : output electrical power

        Tmot_to, Wmot_to, Imot_to, Umot_to, P_el_to = MotorPerfoModel.performances(
            Qpro_to, Wpro_to, Nred, Tfmot, Ktmot, Rmot
        )

        outputs["data:propulsion:motor:power:takeoff"] = P_el_to
        outputs["data:propulsion:motor:voltage:takeoff"] = Umot_to
        outputs["data:propulsion:motor:current:takeoff"] = Imot_to
        outputs["data:propulsion:motor:torque:takeoff"] = Tmot_to


class Hover(om.ExplicitComponent):
    """
    Computes motor performances for hover
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:propeller:speed:hover", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:propeller:torque:hover", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:power:hover", units="W")
        self.add_output("data:propulsion:motor:voltage:hover", units="V")
        self.add_output("data:propulsion:motor:current:hover", units="A")
        self.add_output("data:propulsion:motor:torque:hover", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Tfmot = inputs["data:propulsion:motor:torque:friction"]
        Rmot = inputs["data:propulsion:motor:resistance"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient"]
        Wpro_hover = inputs["data:propulsion:propeller:speed:hover"]
        Qpro_hover = inputs["data:propulsion:propeller:torque:hover"]

        # Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction
        # W_hover_motor = Wpro_hover * Nred  # [rad/s] Nominal motor speed with reduction
        # Imot_hover = (Tmot_hover + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        # Umot_hover = Rmot * Imot_hover + W_hover_motor * Ktmot  # [V] Voltage of the motor per propeller
        # P_el_hover = Umot_hover * Imot_hover  # [W] Hover : output electrical power

        (
            Tmot_hover,
            Wmot_hover,
            Imot_hover,
            Umot_hover,
            P_el_hover,
        ) = MotorPerfoModel.performances(Qpro_hover, Wpro_hover, Nred, Tfmot, Ktmot, Rmot)

        outputs["data:propulsion:motor:power:hover"] = P_el_hover
        outputs["data:propulsion:motor:voltage:hover"] = Umot_hover
        outputs["data:propulsion:motor:current:hover"] = Imot_hover
        outputs["data:propulsion:motor:torque:hover"] = Tmot_hover


class Climb(om.ExplicitComponent):
    """
    Computes motor performances for climb
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:propeller:speed:climb", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:propeller:torque:climb", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:power:climb", units="W")
        self.add_output("data:propulsion:motor:voltage:climb", units="V")
        self.add_output("data:propulsion:motor:current:climb", units="A")
        self.add_output("data:propulsion:motor:torque:climb", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Tfmot = inputs["data:propulsion:motor:torque:friction"]
        Rmot = inputs["data:propulsion:motor:resistance"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient"]
        Wpro_cl = inputs["data:propulsion:propeller:speed:climb"]
        Qpro_cl = inputs["data:propulsion:propeller:torque:climb"]

        # Tmot_cl = Qpro_cl / Nred  # [N.m] motor climbing torque with reduction
        # W_cl_motor = Wpro_cl * Nred  # [rad/s] Motor Climb speed with reduction
        # Imot_cl = (Tmot_cl + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        # Umot_cl = Rmot * Imot_cl + W_cl_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        # P_el_cl = Umot_cl * Imot_cl  # [W] Power : output electrical power for climbing

        Tmot_cl, Wmot_cl, Imot_cl, Umot_cl, P_el_cl = MotorPerfoModel.performances(
            Qpro_cl, Wpro_cl, Nred, Tfmot, Ktmot, Rmot
        )
        outputs["data:propulsion:motor:power:climb"] = P_el_cl
        outputs["data:propulsion:motor:voltage:climb"] = Umot_cl
        outputs["data:propulsion:motor:current:climb"] = Imot_cl
        outputs["data:propulsion:motor:torque:climb"] = Tmot_cl


class Cruise(om.ExplicitComponent):
    """
    Computes motor performances for cruise
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:propeller:speed:cruise", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:propeller:torque:cruise", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:power:cruise", units="W")
        self.add_output("data:propulsion:motor:voltage:cruise", units="V")
        self.add_output("data:propulsion:motor:current:cruise", units="A")
        self.add_output("data:propulsion:motor:torque:cruise", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Tfmot = inputs["data:propulsion:motor:torque:friction"]
        Rmot = inputs["data:propulsion:motor:resistance"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient"]
        Wpro_cr = inputs["data:propulsion:propeller:speed:cruise"]
        Qpro_cr = inputs["data:propulsion:propeller:torque:cruise"]

        # Tmot_cr = Qpro_cr / Nred  # [N.m] motor cruise torque with reduction
        # W_cr_motor = Wpro_cr * Nred  # [rad/s] Motor cruise speed with reduction
        # Imot_cr = (Tmot_cr + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        # Umot_cr = Rmot * Imot_cr + W_cr_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        # P_el_cr = Umot_cr * Imot_cr  # [W] Power : output electrical power for climbing

        Tmot_cr, Wmot_cr, Imot_cr, Umot_cr, P_el_cr = MotorPerfoModel.performances(
            Qpro_cr, Wpro_cr, Nred, Tfmot, Ktmot, Rmot
        )

        outputs["data:propulsion:motor:power:cruise"] = P_el_cr
        outputs["data:propulsion:motor:voltage:cruise"] = Umot_cr
        outputs["data:propulsion:motor:current:cruise"] = Imot_cr
        outputs["data:propulsion:motor:torque:cruise"] = Tmot_cr
