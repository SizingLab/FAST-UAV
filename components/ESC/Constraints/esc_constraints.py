"""
ESC constraints
"""
import openmdao.api as om
import numpy as np

class ESCConstraints(om.ExplicitComponent):
    """
    Constraints definition of the ESC component
    """

    def setup(self):
        self.add_input('data:ESC:performances:power_max_thrust', val=np.nan, units='W')
        self.add_input('data:ESC:performances:power_max_climb', val=np.nan, units='W')
        self.add_output('optimization:constraints:ESC:cons_climb_power', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        P_esc = inputs['data:ESC:performances:power_max_thrust']
        P_esc_cl = inputs['data:ESC:performances:power_max_climb']

        ESC_con1 = (P_esc - P_esc_cl) / P_esc

        outputs['optimization:constraints:ESC:cons_climb_power'] = ESC_con1
