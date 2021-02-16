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
        self.add_input('data:ESC:voltage', val=np.nan, units='V')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:ESC:power:max', val=np.nan, units='W')

        self.add_input('data:ESC:power:climb', val=np.nan, units='W')
        self.add_output('constraints:ESC:power:climb', units=None)
        self.add_output('constraints:ESC:voltage', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        P_esc = inputs['data:ESC:power:max']
        Vesc = inputs['data:ESC:voltage']
        V_bat = inputs['data:battery:voltage']
        P_esc_cl = inputs['data:ESC:power:climb']

        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (V_bat - Vesc) / V_bat

        outputs['constraints:ESC:power:climb'] = ESC_con1
        outputs['constraints:ESC:voltage'] = ESC_con2