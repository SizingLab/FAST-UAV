"""
Motor constraints
"""
import openmdao.api as om
import numpy as np

class MotorConstraints(om.ExplicitComponent):
    """
    Constraints definition of the motor component
    """

    def setup(self):
        self.add_input('data:motor:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:max', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:takeoff', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:climb', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:forward', val=np.nan, units='N*m')
        self.add_output('data:motor:constraints:torque:takeoff', units=None)
        self.add_output('data:motor:constraints:torque:climb', units=None)
        self.add_output('data:motor:constraints:torque:forward', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        Tmot_max = inputs['data:motor:torque:max']
        Tmot_nom = inputs['data:motor:torque:nominal']
        Tmot_to = inputs['data:motor:torque:takeoff']
        Tmot_cl = inputs['data:motor:torque:climb']
        Tmot_ff = inputs['data:motor:torque:forward']

        motor_con1 = (Tmot_max - Tmot_to) / Tmot_max  # transient torque
        motor_con2 = (Tmot_max - Tmot_cl) / Tmot_max  # transient torque
        motor_con3 = (Tmot_nom - Tmot_ff) / Tmot_max  # steady torque

        outputs['data:motor:constraints:torque:takeoff'] = motor_con1
        outputs['data:motor:constraints:torque:climb'] = motor_con2
        outputs['data:motor:constraints:torque:forward'] = motor_con3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_max = inputs['data:motor:torque:max']
        Tmot_nom = inputs['data:motor:torque:nominal']
        Tmot_to = inputs['data:motor:torque:takeoff']
        Tmot_cl = inputs['data:motor:torque:climb']
        Tmot_ff = inputs['data:motor:torque:forward']

        partials[
            'data:motor:constraints:torque:takeoff',
            'data:motor:torque:max',
        ] = Tmot_to / Tmot_max**2
        partials[
            'data:motor:constraints:torque:takeoff',
            'data:motor:torque:takeoff',
        ] = -1.0 / Tmot_max

        partials[
            'data:motor:constraints:torque:climb',
            'data:motor:torque:max',
        ] = Tmot_cl / Tmot_max**2
        partials[
            'data:motor:constraints:torque:climb',
            'data:motor:torque:climb',
        ] = -1.0 / Tmot_max

        partials[
            'data:motor:constraints:torque:forward',
            'data:motor:torque:nominal',
        ] = Tmot_ff / Tmot_nom ** 2
        partials[
            'data:motor:constraints:torque:forward',
            'data:motor:torque:forward',
        ] = -1.0 / Tmot_nom