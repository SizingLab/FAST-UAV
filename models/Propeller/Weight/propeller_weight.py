"""
Propeller weight
"""
import openmdao.api as om
import numpy as np

class ComputePropellerWeightMR(om.ExplicitComponent):
    """
    Weight calculation of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:reference:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:reference:mass', val=np.nan, units='kg')
        self.add_output('data:propeller:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        Dpro_ref = inputs['data:propeller:reference:diameter']
        Mpro_ref = inputs['data:propeller:reference:mass']

        Mpro = Mpro_ref * (Dpro / Dpro_ref) ** 3  # [kg] Propeller mass

        outputs['data:propeller:mass'] = Mpro
