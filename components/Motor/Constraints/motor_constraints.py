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
        self.add_input('data:motor:performances:torque_max', val=np.nan, units='N*m')
        self.add_input('data:motor:performances:torque_takeoff', val=np.nan, units='N*m')
        self.add_input('data:motor:performances:torque_climb', val=np.nan, units='N*m')
        self.add_output('optimization:constraints:motor:cons_takeoff_torque', units=None)
        self.add_output('optimization:constraints:motor:cons_climb_torque', units=None)


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmot_max = inputs['data:motor:performances:torque_max']
        Tmot_to = inputs['data:motor:performances:torque_takeoff']
        Tmot_cl = inputs['data:motor:performances:torque_climb']

        motor_con1 = (Tmot_max - Tmot_to) / Tmot_max
        motor_con2 = (Tmot_max - Tmot_cl) / Tmot_max

        outputs['optimization:constraints:motor:cons_takeoff_torque'] = motor_con1
        outputs['optimization:constraints:motor:cons_climb_torque'] = motor_con2
