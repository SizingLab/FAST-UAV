"""
Motor Scaling
"""
import openmdao.api as om
import numpy as np


class MotorScaling(om.Group):
    """
    Group containing the scaling functions of the motor
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("nominal_torque", NominalTorque(use_gearbox=self.options["use_gearbox"]),promotes=["*"])
        self.add_subsystem("max_torque", MaxTorque(), promotes=["*"])
        self.add_subsystem("battery_voltage_guess", BatteryVoltageEstimation(), promotes=["*"])
        self.add_subsystem("constants", MotorConstants(use_gearbox=self.options["use_gearbox"]), promotes=["*"])
        self.add_subsystem("weight", Weight(), promotes=["*"])
        self.add_subsystem("geometry", Geometry(), promotes=["*"])


class NominalTorque(om.ExplicitComponent):
    """
    Computes nominal torque
    """
    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:propeller:torque:hover', val=np.nan, units='N*m')
        self.add_input('data:motor:settings:torque:k', val=np.nan, units=None)
        self.add_output('data:motor:torque:nominal:estimated', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Qpro_hover = inputs['data:propeller:torque:hover']
        k_mot = inputs['data:motor:settings:torque:k']

        Tmot_hover = Qpro_hover / Nred  # [N.m] hover torque
        Tmot = k_mot * Tmot_hover  # [N.m] required motor nominal torque

        outputs['data:motor:torque:nominal:estimated'] = Tmot


class MaxTorque(om.ExplicitComponent):
    """
    Compute maximum torque
    """
    def setup(self):
        self.add_input('data:motor:torque:nominal:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:max', val=np.nan, units='N*m')
        self.add_output('data:motor:torque:max:estimated', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmot_ref = inputs['data:motor:reference:torque:nominal']
        Tmot_max_ref = inputs['data:motor:reference:torque:max']
        Tmot = inputs['data:motor:torque:nominal:estimated']

        Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref)  # [N.m] max torque

        outputs['data:motor:torque:max:estimated'] = Tmot_max


class BatteryVoltageEstimation(om.ExplicitComponent):
    """
    Computes an estimation of battery voltage
    """
    def setup(self):
        self.add_input('data:propeller:power:takeoff', val=np.nan, units='W')
        self.add_input('data:battery:settings:voltage:k', val=np.nan, units=None)
        self.add_output('data:battery:voltage:guess', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Ppro_to = inputs['data:propeller:power:takeoff']
        k_vb = inputs['data:battery:settings:voltage:k']

        V_bat_guess = k_vb * 1.84 * (Ppro_to) ** (0.36)  # [V] battery voltage estimation

        outputs['data:battery:voltage:guess'] = V_bat_guess


class MotorConstants(om.ExplicitComponent):
    """
    Computes motor constants
    """
    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:N_red', val=1.0, units=None)
        self.add_input('data:battery:voltage:guess', val=np.nan, units='V')
        self.add_input('data:propeller:speed:takeoff', val=np.nan, units='rad/s')
        self.add_input('data:motor:torque:nominal:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:settings:speed:k', val=np.nan, units=None)
        self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:friction', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:resistance', val=np.nan, units='V/A')
        self.add_input('data:motor:reference:torque:coefficient', val=np.nan, units='N*m/A')
        self.add_output('data:motor:torque:friction:estimated', units='N*m')
        self.add_output('data:motor:resistance:estimated', units='V/A')
        self.add_output('data:motor:torque:coefficient:estimated', units='N*m/A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_gearbox"]:
            Nred = inputs['data:gearbox:N_red']
        else:
            Nred = 1.0
        Wpro_to = inputs['data:propeller:speed:takeoff']
        k_speed_mot = inputs['data:motor:settings:speed:k']
        Tmot_ref = inputs['data:motor:reference:torque:nominal']
        Tfmot_ref = inputs['data:motor:reference:torque:friction']
        Rmot_ref = inputs['data:motor:reference:resistance']
        Ktmot_ref = inputs['data:motor:reference:torque:coefficient']
        V_bat_guess = inputs['data:battery:voltage:guess']
        Tmot = inputs['data:motor:torque:nominal:estimated']

        W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed
        Ktmot = V_bat_guess / (k_speed_mot * W_to_motor)  # [N.m/A] or [V/(rad/s)] Kt motor (RI term is missing)
        Rmot = Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2  # [Ohm] motor resistance
        Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs['data:motor:torque:friction:estimated'] = Tfmot
        outputs['data:motor:resistance:estimated'] = Rmot
        outputs['data:motor:torque:coefficient:estimated'] = Ktmot


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input('data:motor:torque:nominal:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:mass', val=np.nan, units='kg')
        self.add_output('data:motor:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmot = inputs['data:motor:torque:nominal:estimated']
        Tmot_ref = inputs['data:motor:reference:torque:nominal']
        Mmot_ref = inputs['data:motor:reference:mass']

        Mmot = Mmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs['data:motor:mass:estimated'] = Mmot


class Geometry(om.ExplicitComponent):
    """
    Computes motor geometry
    """

    def setup(self):
        self.add_input('data:motor:reference:length', val=np.nan, units='m')
        self.add_input('data:motor:reference:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass:estimated', val=np.nan, units='kg')
        self.add_output('data:motor:length:estimated', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Lmot_ref = inputs['data:motor:reference:length']
        Mmot_ref = inputs['data:motor:reference:mass']
        Mmot = inputs['data:motor:mass:estimated']

        Lmot = Lmot_ref * (Mmot / Mmot_ref)**(1/3) # [m] Motor length (estimated)

        outputs['data:motor:length:estimated'] = Lmot


# class ComputeMotorCharacteristics(om.ExplicitComponent):
#     """
#     Scaling calculation of an electrical Motor
#     """
#
#     def initialize(self):
#         self.options.declare("use_gearbox", default=True, types=bool)
#
#     def setup(self):
#         if self.options["use_gearbox"]:
#             self.add_input('data:gearbox:N_red', val=1.0, units=None)
#
#         self.add_input('data:propeller:speed:takeoff', val=np.nan, units='rad/s')
#         self.add_input('data:propeller:torque:hover', val=np.nan, units='N*m')
#         self.add_input('data:propeller:power:takeoff', units='W')
#         self.add_input('data:motor:settings:torque:k', val=np.nan, units=None)
#         self.add_input('data:motor:settings:speed:k', val=np.nan, units=None)
#         self.add_input('data:battery:settings:voltage:k', val=np.nan, units=None)
#         self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
#         self.add_input('data:motor:reference:torque:max', val=np.nan, units='N*m')
#         self.add_input('data:motor:reference:torque:friction', val=np.nan, units='N*m')
#         self.add_input('data:motor:reference:resistance', val=np.nan, units='V/A')
#         self.add_input('data:motor:reference:torque:coefficient', val=np.nan, units='N*m/A')
#         self.add_output('data:motor:torque:nominal:estimated', units='N*m')
#         self.add_output('data:motor:torque:max:estimated', units='N*m')
#         self.add_output('data:motor:torque:friction:estimated', units='N*m')
#         self.add_output('data:motor:resistance:estimated', units='V/A')
#         self.add_output('data:motor:torque:coefficient:estimated', units='N*m/A')
#         self.add_output('data:battery:voltage:guess', units='V')
#
#     def setup_partials(self):
#         # Finite difference all partials.
#         self.declare_partials('*', '*', method='fd')
#
#     def compute(self, inputs, outputs):
#         if self.options["use_gearbox"]:
#             Nred = inputs['data:gearbox:N_red']
#         else:
#             Nred = 1.0
#
#         Wpro_to = inputs['data:propeller:speed:takeoff']
#         Qpro_hover = inputs['data:propeller:torque:hover']
#         Ppro_to = inputs['data:propeller:power:takeoff']
#         k_mot = inputs['data:motor:settings:torque:k']
#         k_speed_mot = inputs['data:motor:settings:speed:k']
#         k_vb = inputs['data:battery:settings:voltage:k']
#         Tmot_ref = inputs['data:motor:reference:torque:nominal']
#         Tmot_max_ref = inputs['data:motor:reference:torque:max']
#         Tfmot_ref = inputs['data:motor:reference:torque:friction']
#         Rmot_ref = inputs['data:motor:reference:resistance']
#         Ktmot_ref = inputs['data:motor:reference:torque:coefficient']
#
#         # Motor speed and torque for sizing
#         W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed
#         Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque
#
#         Tmot = k_mot * Tmot_hover  # [N.m] required motor nominal torque
#         Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref)  # [N.m] max torque
#
#         # Selection with take-off speed
#         V_bat_guess = k_vb * 1.84 * (Ppro_to) ** (0.36)  # [V] battery voltage estimation
#         Ktmot = V_bat_guess / (k_speed_mot * W_to_motor)  # [N.m/A] or [V/(rad/s)] Kt motor (RI term is missing)
#         Rmot = Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2  # [Ohm] motor resistance
#         Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque
#
#         outputs['data:motor:torque:nominal:estimated'] = Tmot
#         outputs['data:motor:torque:max:estimated'] = Tmot_max
#         outputs['data:motor:torque:friction:estimated'] = Tfmot
#         outputs['data:motor:resistance:estimated'] = Rmot
#         outputs['data:motor:torque:coefficient:estimated'] = Ktmot
#         outputs['data:battery:voltage:guess'] = V_bat_guess