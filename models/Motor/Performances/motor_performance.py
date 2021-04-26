"""
Motor performances
"""
import openmdao.api as om
import numpy as np


class MotorModel:
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

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("takeoff", TakeOff(use_gearbox=self.options["use_gearbox"]),promotes=["*"])
        self.add_subsystem("hover", Hover(use_gearbox=self.options["use_gearbox"]), promotes=["*"])
        self.add_subsystem("climb", Climb(use_gearbox=self.options["use_gearbox"]), promotes=["*"])
        self.add_subsystem("forward", Forward(use_gearbox=self.options["use_gearbox"]), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes motor performances for takeoff
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:motor:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:torque:coefficient', val=np.nan, units='N*m/A')
        self.add_input('data:propeller:speed:takeoff', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:takeoff', val=np.nan, units='N*m')
        self.add_output('data:motor:power:takeoff', units='W')
        self.add_output('data:motor:voltage:takeoff', units='V')
        self.add_output('data:motor:current:takeoff', units='A')
        self.add_output('data:motor:torque:takeoff', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Tfmot = inputs['data:motor:torque:friction']
        Rmot = inputs['data:motor:resistance']
        Ktmot = inputs['data:motor:torque:coefficient']
        Wpro_to = inputs['data:propeller:speed:takeoff']
        Qpro_to = inputs['data:propeller:torque:takeoff']

        # Tmot_to = Qpro_to / Nred  # [N.m] motor take-off torque with reduction
        # W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed with reduction
        # Imot_to = (Tmot_to + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        # Umot_to = Rmot * Imot_to + W_to_motor * Ktmot  # [V] Voltage of the motor per propeller
        # P_el_to = Umot_to * Imot_to  # [W] Takeoff : output electrical power

        Tmot_to, Wmot_to, Imot_to, Umot_to, P_el_to = MotorModel.performances(Qpro_to, Wpro_to, Nred, Tfmot, Ktmot, Rmot)

        outputs['data:motor:power:takeoff'] = P_el_to
        outputs['data:motor:voltage:takeoff'] = Umot_to
        outputs['data:motor:current:takeoff'] = Imot_to
        outputs['data:motor:torque:takeoff'] = Tmot_to


class Hover(om.ExplicitComponent):
    """
    Computes motor performances for hover
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:motor:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:torque:coefficient', val=np.nan, units='N*m/A')
        self.add_input('data:propeller:speed:hover', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:hover', val=np.nan, units='N*m')
        self.add_output('data:motor:power:hover', units='W')
        self.add_output('data:motor:voltage:hover', units='V')
        self.add_output('data:motor:current:hover', units='A')
        self.add_output('data:motor:torque:hover', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Tfmot = inputs['data:motor:torque:friction']
        Rmot = inputs['data:motor:resistance']
        Ktmot = inputs['data:motor:torque:coefficient']
        Wpro_hover = inputs['data:propeller:speed:hover']
        Qpro_hover = inputs['data:propeller:torque:hover']

        # Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction
        # W_hover_motor = Wpro_hover * Nred  # [rad/s] Nominal motor speed with reduction
        # Imot_hover = (Tmot_hover + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        # Umot_hover = Rmot * Imot_hover + W_hover_motor * Ktmot  # [V] Voltage of the motor per propeller
        # P_el_hover = Umot_hover * Imot_hover  # [W] Hover : output electrical power

        Tmot_hover, Wmot_hover, Imot_hover, Umot_hover, P_el_hover = MotorModel.performances(
                                                                    Qpro_hover, Wpro_hover, Nred, Tfmot, Ktmot, Rmot)

        outputs['data:motor:power:hover'] = P_el_hover
        outputs['data:motor:voltage:hover'] = Umot_hover
        outputs['data:motor:current:hover'] = Imot_hover
        outputs['data:motor:torque:hover'] = Tmot_hover


class Climb(om.ExplicitComponent):
    """
    Computes motor performances for climb
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:motor:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:torque:coefficient', val=np.nan, units='N*m/A')
        self.add_input('data:propeller:speed:climb', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:climb', val=np.nan, units='N*m')
        self.add_output('data:motor:power:climb', units='W')
        self.add_output('data:motor:voltage:climb', units='V')
        self.add_output('data:motor:current:climb', units='A')
        self.add_output('data:motor:torque:climb', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Tfmot = inputs['data:motor:torque:friction']
        Rmot = inputs['data:motor:resistance']
        Ktmot = inputs['data:motor:torque:coefficient']
        Wpro_cl = inputs['data:propeller:speed:climb']
        Qpro_cl = inputs['data:propeller:torque:climb']

        # Tmot_cl = Qpro_cl / Nred  # [N.m] motor climbing torque with reduction
        # W_cl_motor = Wpro_cl * Nred  # [rad/s] Motor Climb speed with reduction
        # Imot_cl = (Tmot_cl + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        # Umot_cl = Rmot * Imot_cl + W_cl_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        # P_el_cl = Umot_cl * Imot_cl  # [W] Power : output electrical power for climbing

        Tmot_cl, Wmot_cl, Imot_cl, Umot_cl, P_el_cl = MotorModel.performances(Qpro_cl, Wpro_cl, Nred, Tfmot, Ktmot,
                                                                              Rmot)
        outputs['data:motor:power:climb'] = P_el_cl
        outputs['data:motor:voltage:climb'] = Umot_cl
        outputs['data:motor:current:climb'] = Imot_cl
        outputs['data:motor:torque:climb'] = Tmot_cl


class Forward(om.ExplicitComponent):
    """
    Computes motor performances for forward flight
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:motor:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:torque:coefficient', val=np.nan, units='N*m/A')
        self.add_input('data:propeller:speed:forward', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:forward', val=np.nan, units='N*m')
        self.add_output('data:motor:power:forward', units='W')
        self.add_output('data:motor:voltage:forward', units='V')
        self.add_output('data:motor:current:forward', units='A')
        self.add_output('data:motor:torque:forward', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Tfmot = inputs['data:motor:torque:friction']
        Rmot = inputs['data:motor:resistance']
        Ktmot = inputs['data:motor:torque:coefficient']
        Wpro_ff = inputs['data:propeller:speed:forward']
        Qpro_ff = inputs['data:propeller:torque:forward']

        # Tmot_ff = Qpro_ff / Nred  # [N.m] motor forward flight torque with reduction
        # W_ff_motor = Wpro_ff * Nred  # [rad/s] Motor forward flight speed with reduction
        # Imot_ff = (Tmot_ff + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        # Umot_ff = Rmot * Imot_ff + W_ff_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        # P_el_ff = Umot_ff * Imot_ff  # [W] Power : output electrical power for climbing

        Tmot_ff, Wmot_ff, Imot_ff, Umot_ff, P_el_ff = MotorModel.performances(Qpro_ff, Wpro_ff, Nred, Tfmot, Ktmot,
                                                                              Rmot)

        outputs['data:motor:power:forward'] = P_el_ff
        outputs['data:motor:voltage:forward'] = Umot_ff
        outputs['data:motor:current:forward'] = Imot_ff
        outputs['data:motor:torque:forward'] = Tmot_ff

