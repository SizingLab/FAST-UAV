"""
Motor performances
"""
import openmdao.api as om
import numpy as np

class ComputeMotorPerfo(om.ExplicitComponent):
    """
    Characteristics calculation of an electrical Motor
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)

        self.add_input('data:motor:torque:friction', units='N*m')
        self.add_input('data:motor:resistance', units='V/A')
        self.add_input('data:motor:torque_coefficient', units='N*m/A')
        self.add_input('data:propeller:speed:takeoff', val=np.nan, units='rad/s')
        self.add_input('data:propeller:speed:hover', val=np.nan, units='rad/s')
        self.add_input('data:propeller:speed:climb', val=np.nan, units='rad/s')
        self.add_input('data:propeller:speed:forward', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:takeoff', val=np.nan, units='N*m')
        self.add_input('data:propeller:torque:hover', val=np.nan, units='N*m')
        self.add_input('data:propeller:torque:climb', val=np.nan, units='N*m')
        self.add_input('data:propeller:torque:forward', val=np.nan, units='N*m')
        self.add_output('data:motor:torque:hover', units='N*m')
        self.add_output('data:motor:torque:takeoff', units='N*m')
        self.add_output('data:motor:torque:climb', units='N*m')
        self.add_output('data:motor:torque:forward', units='N*m')
        self.add_output('data:motor:power:takeoff', units='W')
        self.add_output('data:motor:power:hover', units='W')
        self.add_output('data:motor:power:climb', units='W')
        self.add_output('data:motor:power:forward', units='W')
        self.add_output('data:motor:voltage:takeoff', units='V')
        self.add_output('data:motor:voltage:hover', units='V')
        self.add_output('data:motor:voltage:climb', units='V')
        self.add_output('data:motor:voltage:forward', units='V')
        self.add_output('data:motor:current:takeoff', units='A')
        self.add_output('data:motor:current:hover', units='A')
        self.add_output('data:motor:current:climb', units='A')
        self.add_output('data:motor:current:forward', units='A')

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
        Ktmot = inputs['data:motor:torque_coefficient']
        Wpro_to = inputs['data:propeller:speed:takeoff']
        Wpro_hover = inputs['data:propeller:speed:hover']
        Wpro_cl = inputs['data:propeller:speed:climb']
        Wpro_ff = inputs['data:propeller:speed:forward']
        Qpro_to = inputs['data:propeller:torque:takeoff']
        Qpro_hover = inputs['data:propeller:torque:hover']
        Qpro_cl = inputs['data:propeller:torque:climb']
        Qpro_ff = inputs['data:propeller:torque:forward']

        # Motor speeds and torques:
        W_hover_motor = Wpro_hover * Nred  # [rad/s] Nominal motor speed with reduction
        W_cl_motor = Wpro_cl * Nred  # [rad/s] Motor Climb speed with reduction
        W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed with reduction
        W_ff_motor = Wpro_ff * Nred  # [rad/s] Motor forward flight speed with reduction
        Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction
        Tmot_to = Qpro_to / Nred  # [N.m] motor take-off torque with reduction
        Tmot_cl = Qpro_cl / Nred  # [N.m] motor climbing torque with reduction
        Tmot_ff = Qpro_ff / Nred  # [N.m] motor forward flight torque with reduction

        # Hover current and voltage
        Imot_hover = (Tmot_hover + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        Umot_hover = Rmot * Imot_hover + W_hover_motor * Ktmot  # [V] Voltage of the motor per propeller
        P_el_hover = Umot_hover * Imot_hover  # [W] Hover : output electrical power

        # Take-Off current and voltage
        Imot_to = (Tmot_to + Tfmot) / Ktmot  # [I] Current of the motor per propeller
        Umot_to = Rmot * Imot_to + W_to_motor * Ktmot  # [V] Voltage of the motor per propeller
        P_el_to = Umot_to * Imot_to  # [W] Takeoff : output electrical power

        # Climbing current and voltage
        Imot_cl = (Tmot_cl + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        Umot_cl = Rmot * Imot_cl + W_cl_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        P_el_cl = Umot_cl * Imot_cl  # [W] Power : output electrical power for climbing

        # Forward flight current and voltage
        Imot_ff = (Tmot_ff + Tfmot) / Ktmot  # [I] Current of the motor per propeller for climbing
        Umot_ff = Rmot * Imot_ff + W_ff_motor * Ktmot  # [V] Voltage of the motor per propeller for climbing
        P_el_ff = Umot_ff * Imot_ff  # [W] Power : output electrical power for climbing

        outputs['data:motor:torque:hover'] = Tmot_hover
        outputs['data:motor:torque:takeoff'] = Tmot_to
        outputs['data:motor:torque:climb'] = Tmot_cl
        outputs['data:motor:torque:forward'] = Tmot_ff
        outputs['data:motor:power:hover'] = P_el_hover
        outputs['data:motor:power:takeoff'] = P_el_to
        outputs['data:motor:power:climb'] = P_el_cl
        outputs['data:motor:power:forward'] = P_el_ff
        outputs['data:motor:voltage:hover'] = Umot_hover
        outputs['data:motor:voltage:takeoff'] = Umot_to
        outputs['data:motor:voltage:climb'] = Umot_cl
        outputs['data:motor:voltage:forward'] = Umot_ff
        outputs['data:motor:current:hover'] = Imot_hover
        outputs['data:motor:current:takeoff'] = Imot_to
        outputs['data:motor:current:climb'] = Imot_cl
        outputs['data:motor:current:forward'] = Imot_ff
