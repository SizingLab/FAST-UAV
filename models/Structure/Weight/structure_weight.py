"""
Structure weight
"""
import openmdao.api as om
import numpy as np
from math import *

class ComputeStructureWeightMR(om.ExplicitComponent):
    """
    Weight of a Multi-Rotor structure
    """

    def setup(self):
        self.add_input('data:structure:arms:D_ratio_arms', val=np.nan, units=None)
        self.add_input('data:structure:arms:arm_number', val=np.nan, units=None)
        self.add_input('data:structure:arms:length', val=np.nan, units='m')
        self.add_input('data:structure:arms:diameter:outer', val=np.nan, units='m')
        self.add_input('data:structure:reference:arms:mass', val=np.nan, units='kg')
        self.add_input('data:structure:reference:frame:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:material:density', val=1700, units='kg/m**3')
        self.add_output('data:structure:arms:mass', units='kg')
        self.add_output('data:structure:frame:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Mass
        D_ratio = inputs['data:structure:arms:D_ratio_arms']
        Narm = inputs['data:structure:arms:arm_number']
        Lbra = inputs['data:structure:arms:length']
        Dout = inputs['data:structure:arms:diameter:outer']
        Marm_ref = inputs['data:structure:reference:arms:mass']
        Mfra_ref = inputs['data:structure:reference:frame:mass']
        rho = inputs['data:structure:arms:material:density']

        Marm = pi / 4 * (Dout ** 2 - (D_ratio * Dout) ** 2) * Lbra * rho * Narm  # [kg] mass of the arms
        Mfra = Mfra_ref * (Marm / Marm_ref)  # [kg] mass of the frame

        outputs['data:structure:arms:mass'] = Marm
        outputs['data:structure:frame:mass'] = Mfra
