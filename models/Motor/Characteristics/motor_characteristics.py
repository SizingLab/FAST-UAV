"""
Motor Characteristics
"""
import openmdao.api as om
import numpy as np



class ComputeMotorCharacteristics(om.ExplicitComponent):
    """
    Characteristics calculation of an electrical Motor
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('optimization:settings:gearbox_reduction_ratio', val=np.nan, units=None)

        self.add_input('data:propeller:speed:takeoff', val=np.nan, units='rad/s')
        self.add_input('data:propeller:torque:hover', val=np.nan, units='N*m')
        self.add_input('data:propeller:power:takeoff', units='W')
        self.add_input('optimization:settings:k_mot', val=np.nan, units=None)
        self.add_input('optimization:settings:k_speed_mot', val=np.nan, units=None)
        self.add_input('optimization:settings:k_VB', val=np.nan, units=None)
        self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:max', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:reference:torque_coefficient', val=np.nan, units='N*m/A')
        self.add_output('data:motor:torque:nominal', units='N*m')
        self.add_output('data:motor:torque:max', units='N*m')
        self.add_output('data:motor:torque:friction', units='N*m')
        self.add_output('data:motor:resistance', units='V/A')
        self.add_output('data:motor:torque_coefficient', units='N*m/A')
        self.add_output('data:battery:voltage:estimation', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['optimization:settings:gearbox_reduction_ratio']

        Wpro_to = inputs['data:propeller:speed:takeoff']
        Qpro_hover = inputs['data:propeller:torque:hover']
        Ppro_to = inputs['data:propeller:power:takeoff']
        k_mot = inputs['optimization:settings:k_mot']
        k_speed_mot = inputs['optimization:settings:k_speed_mot']
        k_vb = inputs['optimization:settings:k_VB']
        Tmot_ref = inputs['data:motor:reference:torque:nominal']
        Tmot_max_ref = inputs['data:motor:reference:torque:max']
        Tfmot_ref = inputs['data:motor:reference:torque:friction']
        Rmot_ref = inputs['data:motor:reference:resistance']
        Ktmot_ref = inputs['data:motor:reference:torque_coefficient']

        # Motor speed and torque for sizing
        if self.options["use_gearbox"]:
            W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed with reduction
            Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction
        else:
            W_to_motor = Wpro_to  # [rad/s] Motor take-off speed
            Tmot_hover = Qpro_hover  # [N.m] motor take-off torque

        Tmot = k_mot * Tmot_hover  # [N.m] required motor nominal torque for reductor
        Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref) ** (1)  # [N.m] max torque

        # Selection with take-off speed
        V_bat_est = k_vb * 1.84 * (Ppro_to) ** (0.36)  # [V] battery voltage estimation
        Ktmot = V_bat_est / (k_speed_mot * W_to_motor)  # [N.m/A] or [V/(rad/s)] Kt motor (RI term is missing)
        Rmot = Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** (2)  # [Ohm] motor resistance
        Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs['data:motor:torque:nominal'] = Tmot
        outputs['data:motor:torque:max'] = Tmot_max
        outputs['data:motor:torque:friction'] = Tfmot
        outputs['data:motor:resistance'] = Rmot
        outputs['data:motor:torque_coefficient'] = Ktmot
        outputs['data:battery:voltage:estimation'] = V_bat_est


