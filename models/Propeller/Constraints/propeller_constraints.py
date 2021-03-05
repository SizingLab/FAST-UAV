"""
Propeller constraints
"""
import openmdao.api as om
import numpy as np

class PropellerConstraintsMR(om.ExplicitComponent):
    """
    Constraints definition of the propeller component, for the Multi-rotor case
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('data:propeller:advance_ratio', val=np.nan, units=None)
        self.add_input('data:propeller:speed:climb', val=np.nan, units='rad/s')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_output('data:propeller:constraints:speed:max', units=None)
        self.add_output('data:propeller:constraints:speed:climb', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        NDmax = inputs['data:propeller:reference:nD_max']
        J = inputs['data:propeller:advance_ratio']
        n_pro_cl = inputs['data:propeller:speed:climb'] / 2 / 3.14
        V_cl = inputs['specifications:climb_speed']

        prop_con1 = (NDmax - n_pro_cl * Dpro) / NDmax
        prop_con2 = (-J * n_pro_cl * Dpro + V_cl)

        outputs['data:propeller:constraints:speed:max'] = prop_con1
        outputs['data:propeller:constraints:speed:climb'] = prop_con2

