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
        self.add_input('optimization:settings:D_ratio_arms', val=np.nan, units=None)
        self.add_input('data:structure:geometry:arms:arm_number', val=np.nan, units=None)
        self.add_input('data:structure:geometry:arms:arm_length', val=np.nan, units='m')
        self.add_input('data:structure:geometry:arms:outer_diameter', val=np.nan, units='m')
        self.add_input('data:structure:reference:mass_arms_ref', val=np.nan, units='kg')
        self.add_input('data:structure:reference:mass_frame_ref', val=np.nan, units='kg')
        self.add_input('data:structure:arms:material:density', val=1700, units='kg/m**3')
        self.add_output('data:structure:mass:arms', units='kg')
        self.add_output('data:structure:mass:frame', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Mass
        D_ratio = inputs['optimization:settings:D_ratio_arms']
        Narm = inputs['data:structure:geometry:arms:arm_number']
        Lbra = inputs['data:structure:geometry:arms:arm_length']
        Dout = inputs['data:structure:geometry:arms:outer_diameter']
        Marm_ref = inputs['data:structure:reference:mass_arms_ref']
        Mfra_ref = inputs['data:structure:reference:mass_frame_ref']
        rho = inputs['data:structure:arms:material:density']

        Marm = pi / 4 * (Dout ** 2 - (D_ratio * Dout) ** 2) * Lbra * rho * Narm  # [kg] mass of the arms
        Mfra = Mfra_ref * (Marm / Marm_ref)  # [kg] mass of the frame

        outputs['data:structure:mass:arms'] = Marm
        outputs['data:structure:mass:frame'] = Mfra
