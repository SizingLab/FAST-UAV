"""
Motor weight
"""
import openmdao.api as om
import numpy as np

class ComputeMotorWeight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input('data:motor:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:torque:nominal', val=np.nan, units='N*m')
        self.add_input('data:motor:reference:mass', val=np.nan, units='kg')
        self.add_output('data:motor:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmot = inputs['data:motor:torque:nominal']
        Tmot_ref = inputs['data:motor:reference:torque:nominal']
        Mmot_ref = inputs['data:motor:reference:mass']

        Mmot = Mmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [kg] Motor mass

        outputs['data:motor:mass'] = Mmot
