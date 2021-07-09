"""
Structure Scaling
"""
import openmdao.api as om
import numpy as np
from math import *


class StructureScaling(om.Group):
    """
    Group containing the scaling functions of the structure
    """
    def setup(self):
        self.add_subsystem("geometry", Geometry(), promotes=["*"])
        self.add_subsystem("weight_arms", WeightArms(), promotes=["*"])
        self.add_subsystem("weight_body", WeightBody(), promotes=["*"])


class Geometry(om.ExplicitComponent):
    """
    Computes Multi-Rotor geometry
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

        Larm = Dpro / 2 / (np.sin(pi / Narm))  # [m] length of the arm
        Dout = (F_pro_to * Npro_arm * Larm * 32 / (pi * Sigma_max * (1 - D_ratio ** 4))) ** (1 / 3)  # [m] outer diameter of the beam
        Din = D_ratio * Dout # [m] inner diameter of the beam

        outputs['data:structure:arms:length'] = Larm
        outputs['data:structure:arms:diameter:outer'] = Dout
        outputs['data:structure:arms:diameter:inner'] = Din


class WeightArms(om.ExplicitComponent):
    """
    Computes arms weight
    """
    def setup(self):
        self.add_input('data:structure:arms:settings:diameter:k', val=np.nan, units=None)
        self.add_input('data:structure:arms:number', val=np.nan, units=None)
        self.add_input('data:structure:arms:length', val=np.nan, units='m')
        self.add_input('data:structure:arms:diameter:outer', val=np.nan, units='m')
        self.add_input('data:structure:arms:material:density', val=1700, units='kg/m**3')
        self.add_output('data:structure:arms:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_ratio = inputs['data:structure:arms:settings:diameter:k']
        Narm = inputs['data:structure:arms:number']
        Larm = inputs['data:structure:arms:length']
        Dout = inputs['data:structure:arms:diameter:outer']
        rho = inputs['data:structure:arms:material:density']

        Marms = pi / 4 * (Dout ** 2 - (D_ratio * Dout) ** 2) * Larm * rho * Narm  # [kg] mass of the arms

        outputs['data:structure:arms:mass'] = Marms


class WeightBody(om.ExplicitComponent):
    """
    Computes body weight
    """
    def setup(self):
        self.add_input('data:structure:reference:arms:mass', val=np.nan, units='kg')
        self.add_input('data:structure:reference:body:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_output('data:structure:body:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Marm_ref = inputs['data:structure:reference:arms:mass']
        Mbody_ref = inputs['data:structure:reference:body:mass']
        Marms = inputs['data:structure:arms:mass']

        Mbody = Mbody_ref * (Marms / Marm_ref)  # [kg] mass of the frame

        outputs['data:structure:body:mass'] = Mbody