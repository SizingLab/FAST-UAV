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
        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('optimization:settings:advance_ratio', val=np.nan)
        self.add_input('data:propeller:performances:rot_speed_climb', val=np.nan, units='rad/s')
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_output('optimization:constraints:propeller:cons_max_speed')
        self.add_output('optimization:constraints:propeller:cons_climb_speed')


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        NDmax = inputs['data:propeller:reference:nD_max']
        J = inputs['optimization:settings:advance_ratio']
        n_pro_cl = inputs['data:propeller:performances:rot_speed_climb'] / 2 / 3.14
        Dpro = inputs['data:propeller:geometry:diameter']
        V_cl = inputs['specifications:climb_speed']

        prop_con1 = (NDmax - n_pro_cl * Dpro) / NDmax
        prop_con2 = (-J * n_pro_cl * Dpro + V_cl)

        outputs['optimization:constraints:propeller:cons_max_speed'] = prop_con1
        outputs['optimization:constraints:propeller:cons_climb_speed'] = prop_con2
