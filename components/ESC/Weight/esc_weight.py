"""
ESC weight
"""
import openmdao.api as om
import numpy as np

class ComputeESCWeight(om.ExplicitComponent):
    """
    Weight calculation of ESC
    """

    def setup(self):
        self.add_input('data:ESC:reference:mass_ref', val=np.nan, units='kg')
        self.add_input('data:ESC:reference:power_ref', val=np.nan, units='W')
        self.add_input('data:ESC:performances:power_max_thrust', val=np.nan, units='W')
        self.add_output('data:ESC:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mesc_ref = inputs['data:ESC:reference:mass_ref']
        Pesc_ref = inputs['data:ESC:reference:power_ref']
        P_esc = inputs['data:ESC:performances:power_max_thrust']

        Mesc = Mesc_ref * (P_esc / Pesc_ref)  # [kg] Mass ESC

        outputs['data:ESC:mass'] = Mesc