"""
ESC constraints
"""
import openmdao.api as om
import numpy as np

class ESCConstraints(om.ExplicitComponent):
    """
    Constraints definition of the ESC component
    """
    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        if self.options["use_catalogues"]:
            self.add_input('data:ESC:catalogue:voltage', val=np.nan, units='V')
            self.add_input('data:battery:catalogue:voltage', val=np.nan, units='V')
            self.add_input('data:ESC:catalogue:power:max', val=np.nan, units='W')
        else:
            self.add_input('data:ESC:voltage', val=np.nan, units='V')
            self.add_input('data:battery:voltage', val=np.nan, units='V')
            self.add_input('data:ESC:power:max', val=np.nan, units='W')

        self.add_input('data:ESC:power:climb', val=np.nan, units='W')
        self.add_output('optimization:constraints:ESC:power:climb', units=None)
        self.add_output('optimization:constraints:ESC:voltage', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_catalogues"]:
            P_esc = inputs['data:ESC:catalogue:power:max']
            Vesc = inputs['data:ESC:catalogue:voltage']
            V_bat = inputs['data:battery:catalogue:voltage']
        else:
            P_esc = inputs['data:ESC:power:max']
            Vesc = inputs['data:ESC:voltage']
            V_bat = inputs['data:battery:voltage']
        P_esc_cl = inputs['data:ESC:power:climb']

        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (V_bat - Vesc) / V_bat

        outputs['optimization:constraints:ESC:power:climb'] = ESC_con1
        outputs['optimization:constraints:ESC:voltage'] = ESC_con2
