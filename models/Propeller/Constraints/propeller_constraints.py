"""
Propeller constraints
"""
import openmdao.api as om
import numpy as np

class PropellerConstraintsMR(om.ExplicitComponent):
    """
    Constraints definition of the propeller component, for the Multi-rotor case
    """

    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        if self.options["use_catalogues"]:
            self.add_input('data:propeller:geometry:diameter:catalogue', val=np.nan, units='m')
        else:
            self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')

        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('data:propeller:settings:advance_ratio', val=np.nan, units=None)
        self.add_input('data:propeller:speed:climb', val=np.nan, units='rad/s')
        self.add_input('mission:climb_speed', val=np.nan, units='m/s')
        self.add_output('optimization:constraints:propeller:speed:max', units=None)
        self.add_output('optimization:constraints:propeller:speed:climb', units=None)


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_catalogues"]:
            Dpro = inputs['data:propeller:geometry:diameter:catalogue']
        else:
            Dpro = inputs['data:propeller:geometry:diameter']

        NDmax = inputs['data:propeller:reference:nD_max']
        J = inputs['data:propeller:settings:advance_ratio']
        n_pro_cl = inputs['data:propeller:speed:climb'] / 2 / 3.14
        V_cl = inputs['mission:climb_speed']

        prop_con1 = (NDmax - n_pro_cl * Dpro) / NDmax
        prop_con2 = (-J * n_pro_cl * Dpro + V_cl)

        outputs['optimization:constraints:propeller:speed:max'] = prop_con1
        outputs['optimization:constraints:propeller:speed:climb'] = prop_con2
