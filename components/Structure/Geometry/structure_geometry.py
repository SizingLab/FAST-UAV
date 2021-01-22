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
        self.add_input('data:structure:arms:material:sigma_max', val=np.nan, units='Pa')
        self.add_input('optimization:structure:arms:D_ratio', val=np.nan)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:structure:geometry:arms:arm_number', val=np.nan)
        self.add_input('data:mission:thrust:max_thrust_prop', val=np.nan, units='N')
        self.add_output('data:structure:geometry:arms:arm_length', units='m')
        self.add_output('data:structure:geometry:arms:outer_diameter', units='m')
        self.add_output('data:structure:geometry:arms:inner_diameter', units='m')


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Frame sized from max thrust
        Sigma_max = inputs['data:structure:arms:material:sigma_max']
        D_ratio = inputs['optimization:structure:arms:D_ratio']
        Dpro = inputs['data:propeller:geometry:diameter']
        Narm = inputs['data:structure:geometry:arms:arm_number']
        F_pro_to = inputs['data:mission:thrust:max_thrust_prop']

        # Length calculation
        #    sep= 2*pi/Narm #[rad] interior angle separation between propellers
        Lbra = Dpro / 2 / (math.sin(pi / Narm))  # [m] length of the arm
        # Tube diameter & thickness
        Dout = (F_pro_to * Lbra * 32 / (pi * Sigma_max * (1 - D_ratio ** 4))) ** (1 / 3)  # [m] outer diameter of the beam
        Din = D_ratio * Dout # [m] inner diameter of the beam

        outputs['data:structure:geometry:arms:arm_length'] = Lbra
        outputs['data:structure:geometry:arms:outer_diameter'] = Dout
        outputs['data:structure:geometry:arms:inner_diameter'] = Din