"""
Motor performances
"""
import openmdao.api as om
import numpy as np

class ComputeMotorPerfoMR(om.ExplicitComponent):
    """
    Performances calculation of a Multi-Rotor Motor
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_input('data:propeller:performances:rot_speed_takeoff', val=np.nan, units='rad/s')
        self.add_input('data:propeller:performances:rot_speed_hover', val=np.nan, units='rad/s')
        self.add_input('data:propeller:performances:rot_speed_climb', val=np.nan, units='rad/s')
        self.add_input('data:propeller:performances:torque_takeoff', val=np.nan, units='N*m')
        self.add_input('data:propeller:performances:torque_hover', val=np.nan, units='N*m')
        self.add_input('data:propeller:performances:torque_climb', val=np.nan, units='N*m')
        self.add_input('data:propeller:performances:power_takeoff', units='W')
        self.add_input('optimization:settings:k_mot', val=np.nan)
        self.add_input('optimization:settings:k_speed_mot', val=np.nan)
        self.add_input('optimization:settings:k_VB', val=np.nan)
        #self.add_input('specifications:options:gearbox_mode', val=np.nan)
        self.add_input('optimization:settings:gearbox_reduction_ratio', val=np.nan)
        self.add_input('data:motor:reference:torque_nominal_ref', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque_max_ref', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque_friction_ref', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:resistance_ref', val=np.nan, units='V/A')
        self.add_input('data:motor:reference:torque_coefficient_ref', val=np.nan, units='N*m/A')
        self.add_output('data:motor:performances:torque_hover', units='N*m')
        self.add_output('data:motor:performances:torque_takeoff', units='N*m')
        self.add_output('data:motor:performances:torque_climb', units='N*m')
        self.add_output('data:motor:performances:torque_nominal', units='N*m')
        self.add_output('data:motor:performances:torque_max', units='N*m')
        self.add_output('data:motor:performances:torque_friction', units='N*m')
        self.add_output('data:motor:performances:resistance', units='V/A')
        self.add_output('data:motor:performances:torque_coefficient', units='N*m/A')
        self.add_output('data:motor:performances:elec_power_takeoff', units='W')
        self.add_output('data:motor:performances:elec_power_hover', units='W')
        self.add_output('data:motor:performances:elec_power_climb', units='W')
        self.add_output('data:motor:performances:voltage_takeoff', units='V')
        self.add_output('data:motor:performances:voltage_hover', units='V')
        self.add_output('data:motor:performances:voltage_climb', units='V')
        self.add_output('data:motor:performances:current_takeoff', units='A')
        self.add_output('data:motor:performances:current_hover', units='A')
        self.add_output('data:motor:performances:current_climb', units='A')
        self.add_output('data:battery:performances:voltage_estimation', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Wpro_to = inputs['data:propeller:performances:rot_speed_takeoff']
        Wpro_hover = inputs['data:propeller:performances:rot_speed_hover']
        Wpro_cl = inputs['data:propeller:performances:rot_speed_climb']
        Qpro_to = inputs['data:propeller:performances:torque_takeoff']
        Qpro_hover = inputs['data:propeller:performances:torque_hover']
        Qpro_cl = inputs['data:propeller:performances:torque_climb']
        Ppro_to = inputs['data:propeller:performances:power_takeoff']
        k_mot = inputs['optimization:settings:k_mot']
        k_speed_mot = inputs['optimization:settings:k_speed_mot']
        k_vb = inputs['optimization:settings:k_VB']
        #Mod = inputs['specifications:options:gearbox_mode']
        Nred = inputs['optimization:settings:gearbox_reduction_ratio']
        Tmot_ref = inputs['data:motor:reference:torque_nominal_ref']
        Tmot_max_ref = inputs['data:motor:reference:torque_max_ref']
        Tfmot_ref = inputs['data:motor:reference:torque_friction_ref']
        Rmot_ref = inputs['data:motor:reference:resistance_ref']
        Ktmot_ref = inputs['data:motor:reference:torque_coefficient_ref']
        
        
        # Motor speeds:
        if self.options["use_gearbox"]:
            W_hover_motor = Wpro_hover * Nred  # [rad/s] Nominal motor speed with reduction
            W_cl_motor = Wpro_cl * Nred  # [rad/s] Motor Climb speed with reduction
            W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed with reduction
        else:
            W_hover_motor = Wpro_hover  # [rad/s] Nominal motor speed
            W_cl_motor = Wpro_cl  # [rad/s] Motor Climb speed
            W_to_motor = Wpro_to  # [rad/s] Motor take-off speed

        # Motor torque:
        if self.options["use_gearbox"]:
            Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction
            Tmot_to = Qpro_to / Nred  # [N.m] motor take-off torque with reduction
            Tmot_cl = Qpro_cl / Nred  # [N.m] motor climbing torque with reduction
        else:
            Tmot_hover = Qpro_hover  # [N.m] motor take-off torque
            Tmot_to = Qpro_to  # [N.m] motor take-off torque
            Tmot_cl = Qpro_cl  # [N.m] motor climbing torque

        Tmot = k_mot * Tmot_hover  # [N.m] required motor nominal torque for reductor
        Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref) ** (1)  # [N.m] max torque

        # Selection with take-off speed
        V_bat_est = k_vb * 1.84 * (Ppro_to) ** (0.36)  # [V] battery voltage estimation
        Ktmot = V_bat_est / (k_speed_mot * W_to_motor)  # [N.m/A] or [V/(rad/s)] Kt motor (RI term is missing)
        Rmot = Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** (2)  # [Ohm] motor resistance
        Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque

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

        outputs['data:motor:performances:torque_hover'] = Tmot_hover
        outputs['data:motor:performances:torque_takeoff'] = Tmot_to
        outputs['data:motor:performances:torque_climb'] = Tmot_cl
        outputs['data:motor:performances:torque_nominal'] = Tmot
        outputs['data:motor:performances:torque_max'] = Tmot_max
        outputs['data:motor:performances:torque_friction'] = Tfmot
        outputs['data:motor:performances:resistance'] = Rmot
        outputs['data:motor:performances:torque_coefficient'] = Ktmot
        outputs['data:motor:performances:elec_power_takeoff'] = P_el_to
        outputs['data:motor:performances:elec_power_hover'] = P_el_hover
        outputs['data:motor:performances:elec_power_climb'] = P_el_cl
        outputs['data:motor:performances:voltage_takeoff'] = Umot_to
        outputs['data:motor:performances:voltage_hover'] = Umot_hover
        outputs['data:motor:performances:voltage_climb'] = Umot_cl
        outputs['data:motor:performances:current_takeoff'] = Imot_to
        outputs['data:motor:performances:current_hover'] = Imot_hover
        outputs['data:motor:performances:current_climb'] = Imot_cl
        outputs['data:battery:performances:voltage_estimation'] = V_bat_est
