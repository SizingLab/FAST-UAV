"""
ESC performances
"""
import openmdao.api as om
import numpy as np

class ComputeESCPerfo(om.ExplicitComponent):
    """
    Performances calculation of ESC
    """

    def setup(self):
        self.add_input('data:motor:power:climb', val=np.nan, units='W')
        self.add_input('data:motor:voltage:climb', val=np.nan, units='V')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_output('data:ESC:power:climb', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        P_el_cl = inputs['data:motor:power:climb']
        V_bat = inputs['data:battery:voltage']
        Umot_cl = inputs['data:motor:voltage:climb']

        P_esc_cl = P_el_cl * V_bat / Umot_cl  # [W] power electronic power max climb

        outputs['data:ESC:power:climb'] = P_esc_cl