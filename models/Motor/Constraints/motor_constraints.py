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
        self.add_input('data:motor:torque:max', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:takeoff', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:climb', val=np.nan, units='N*m')
        self.add_output('optimization:constraints:motor:torque:takeoff', units=None)
        self.add_output('optimization:constraints:motor:torque:climb', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmot_max = inputs['data:motor:torque:max']
        Tmot_to = inputs['data:motor:torque:takeoff']
        Tmot_cl = inputs['data:motor:torque:climb']

        motor_con1 = (Tmot_max - Tmot_to) / Tmot_max
        motor_con2 = (Tmot_max - Tmot_cl) / Tmot_max

        outputs['optimization:constraints:motor:torque:takeoff'] = motor_con1
        outputs['optimization:constraints:motor:torque:climb'] = motor_con2
