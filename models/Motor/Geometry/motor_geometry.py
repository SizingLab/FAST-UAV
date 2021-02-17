"""
Motor geometry
"""
import openmdao.api as om
import numpy as np

class ComputeMotorGeometry(om.ExplicitComponent):
    """
    Geometry calculation of Motor
    """

    def setup(self):
        self.add_input('data:motor:reference:length', val=np.nan, units='m')
        self.add_input('data:motor:reference:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass:estimated', val=np.nan, units='kg')
        self.add_output('data:motor:length:estimated', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Lmot_ref = inputs['data:motor:reference:length']
        Mmot_ref = inputs['data:motor:reference:mass']
        Mmot = inputs['data:motor:mass:estimated']

        Lmot = Lmot_ref * (Mmot / Mmot_ref)**(1/3) # [m] Motor length (estimated)

        outputs['data:motor:length:estimated'] = Lmot