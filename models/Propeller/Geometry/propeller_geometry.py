"""
Propeller geometry
"""
import openmdao.api as om
import numpy as np
from fastoad import BundleLoader
from models.Propeller.DecisionTree.obsolete_propeller_dt import PropellerDecisionTree

class ComputePropellerGeometryMR(om.ExplicitComponent):
    """
    Geometry of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('data:propeller:geometry:beta', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:max', val=np.nan, units='N')
        self.add_input('mission:rho_air', val=np.nan, units='kg/m**3')
        self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('data:propeller:settings:k_ND', val=np.nan, units=None)
        self.add_output('data:propeller:geometry:diameter', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        F_pro_to = inputs['data:propeller:thrust:max']
        C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
        rho_air = inputs['mission:rho_air']
        NDmax = inputs['data:propeller:reference:nD_max']
        k_ND = inputs['data:propeller:settings:k_ND']

        Dpro = (F_pro_to / (C_t_sta * rho_air * (NDmax * k_ND) ** 2)) ** 0.5  # [m] Propeller diameter

        outputs['data:propeller:geometry:diameter'] = Dpro

