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
        self.add_input('data:structure:arms:material:stress:max', val=np.nan, units='N/m**2')
        self.add_input('data:structure:arms:settings:diameter:k', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:structure:arms:number', val=np.nan, units=None)
        self.add_input('data:structure:arms:prop_per_arm', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:max', val=np.nan, units='N')
        self.add_output('data:structure:arms:length', units='m')
        self.add_output('data:structure:arms:diameter:outer', units='m')
        self.add_output('data:structure:arms:diameter:inner', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Frame sized from max thrust
        Sigma_max = inputs['data:structure:arms:material:stress:max']
        D_ratio = inputs['data:structure:arms:settings:diameter:k']
        Dpro = inputs['data:propeller:geometry:diameter']
        Narm = inputs['data:structure:arms:number']
        F_pro_to = inputs['data:propeller:thrust:max']
        Npro_arm = inputs['data:structure:arms:prop_per_arm']

        # Length calculation
        #    sep= 2*pi/Narm #[rad] interior angle separation between propellers
        Larm = Dpro / 2 / (np.sin(pi / Narm))  # [m] length of the arm
        # Tube diameter & thickness
        Dout = (F_pro_to * Npro_arm * Larm * 32 / (pi * Sigma_max * (1 - D_ratio ** 4))) ** (1 / 3)  # [m] outer diameter of the beam
        Din = D_ratio * Dout # [m] inner diameter of the beam

        outputs['data:structure:arms:length'] = Larm
        outputs['data:structure:arms:diameter:outer'] = Dout
        outputs['data:structure:arms:diameter:inner'] = Din