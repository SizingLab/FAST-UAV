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
        self.add_input('data:ESC:power:forward', val=np.nan, units='W')
        self.add_output('data:ESC:constraints:power:climb', units=None)
        self.add_output('data:ESC:constraints:power:forward', units=None)
        self.add_output('data:ESC:constraints:voltage', units=None)

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        P_esc = inputs['data:ESC:power:max']
        Vesc = inputs['data:ESC:voltage']
        V_bat = inputs['data:battery:voltage']
        P_esc_cl = inputs['data:ESC:power:climb']
        P_esc_ff = inputs['data:ESC:power:forward']

        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (P_esc - P_esc_ff) / P_esc
        ESC_con3 = (V_bat - Vesc) / V_bat

        outputs['data:ESC:constraints:power:climb'] = ESC_con1
        outputs['data:ESC:constraints:power:forward'] = ESC_con2
        outputs['data:ESC:constraints:voltage'] = ESC_con3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        P_esc = inputs['data:ESC:power:max']
        Vesc = inputs['data:ESC:voltage']
        V_bat = inputs['data:battery:voltage']
        P_esc_cl = inputs['data:ESC:power:climb']
        P_esc_ff = inputs['data:ESC:power:forward']

        partials[
            'data:ESC:constraints:power:climb',
            'data:ESC:power:max',
        ] = P_esc_cl / P_esc**2
        partials[
            'data:ESC:constraints:power:climb',
            'data:ESC:power:climb',
        ] = - 1.0 / P_esc

        partials[
            'data:ESC:constraints:power:forward',
            'data:ESC:power:max',
        ] = P_esc_ff / P_esc ** 2
        partials[
            'data:ESC:constraints:power:forward',
            'data:ESC:power:forward',
        ] = - 1.0 / P_esc

        partials[
            'data:ESC:constraints:voltage',
            'data:battery:voltage',
        ] = Vesc / V_bat**2
        partials[
            'data:ESC:constraints:voltage',
            'data:ESC:voltage',
        ] = - 1.0 / V_bat