"""
Structure geometry
"""
import openmdao.api as om
import numpy as np
import math
from math import pi
from math import sqrt

class ComputeStructureGeometryMR(om.ExplicitComponent):
    """
    Geometry of a Multi-Rotor structure
    """

    def setup(self):
        self.add_input('data:structure:arms:material:sigma_max', val=np.nan, units='N/m**2')
        self.add_input('data:structure:arms:D_ratio_arms', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:structure:arms:arm_number', val=np.nan, units=None)
        self.add_input('data:propeller:prop_number_per_arm', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:max', val=np.nan, units='N')
        self.add_output('data:structure:arms:length', units='m')
        self.add_output('data:structure:arms:diameter:outer', units='m')
        self.add_output('data:structure:arms:diameter:inner', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Frame sized from max thrust
        Sigma_max = inputs['data:structure:arms:material:sigma_max']
        D_ratio = inputs['data:structure:arms:D_ratio_arms']
        Dpro = inputs['data:propeller:geometry:diameter']
        Narm = inputs['data:structure:arms:arm_number']
        F_pro_to = inputs['data:propeller:thrust:max']
        Npro_arm = inputs['data:propeller:prop_number_per_arm']

        # Length calculation
        #    sep= 2*pi/Narm #[rad] interior angle separation between propellers
        Lbra = Dpro / 2 / (math.sin(pi / Narm))  # [m] length of the arm
        # Tube diameter & thickness
        Dout = (F_pro_to * Npro_arm * Lbra * 32 / (pi * Sigma_max * (1 - D_ratio ** 4))) ** (1 / 3)  # [m] outer diameter of the beam
        Din = D_ratio * Dout # [m] inner diameter of the beam

        outputs['data:structure:arms:length'] = Lbra
        outputs['data:structure:arms:diameter:outer'] = Dout
        outputs['data:structure:arms:diameter:inner'] = Din