"""
Propeller constraints
"""
import openmdao.api as om
import numpy as np

class PropellerConstraints(om.ExplicitComponent):
    """
    Constraints definition of the propeller component
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:reference:ND:max', val=np.nan, units='m/s')
        self.add_input('data:propeller:advance_ratio:climb', val=np.nan, units=None)
        self.add_input('data:propeller:advance_ratio:forward', val=np.nan, units=None)
        self.add_input('data:propeller:speed:climb', val=np.nan, units='rad/s')
        self.add_input('data:propeller:speed:forward', val=np.nan, units='rad/s')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_input('data:mission_nominal:forward:speed', val=np.nan, units='m/s')
        self.add_output('data:propeller:constraints:speed:max', units=None)
        self.add_output('data:propeller:constraints:speed:climb', units=None)
        self.add_output('data:propeller:constraints:speed:forward', units=None)

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        NDmax = inputs['data:propeller:reference:ND:max']
        J_climb = inputs['data:propeller:advance_ratio:climb']
        J_forward = inputs['data:propeller:advance_ratio:forward']
        W_pro_cl = inputs['data:propeller:speed:climb']
        W_pro_ff = inputs['data:propeller:speed:forward']
        V_cl = inputs['specifications:climb_speed']
        V_ff = inputs['data:mission_nominal:forward:speed']

        prop_con1 = (NDmax - W_pro_cl * Dpro / 2 / np.pi) / NDmax
        prop_con2 = (V_cl - J_climb * W_pro_cl * Dpro / 2 / np.pi) / V_cl
        prop_con3 = (V_ff - J_forward * W_pro_ff * Dpro / 2 / np.pi) / V_ff

        outputs['data:propeller:constraints:speed:max'] = prop_con1
        outputs['data:propeller:constraints:speed:climb'] = prop_con2
        outputs['data:propeller:constraints:speed:forward'] = prop_con3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Dpro = inputs['data:propeller:geometry:diameter']
        NDmax = inputs['data:propeller:reference:ND:max']
        J_climb = inputs['data:propeller:advance_ratio:climb']
        J_forward = inputs['data:propeller:advance_ratio:forward']
        W_pro_cl = inputs['data:propeller:speed:climb']
        W_pro_ff = inputs['data:propeller:speed:forward']
        V_cl = inputs['specifications:climb_speed']
        V_ff = inputs['data:mission_nominal:forward:speed']

        partials[
            'data:propeller:constraints:speed:max',
            'data:propeller:speed:climb',
        ] = - Dpro / NDmax / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:max',
            'data:propeller:reference:ND:max',
        ] = W_pro_cl * Dpro / NDmax**2 / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:max',
            'data:propeller:geometry:diameter',
        ] = - W_pro_cl / NDmax / 2 / np.pi

        partials[
            'data:propeller:constraints:speed:climb',
            'specifications:climb_speed',
        ] = J_climb * W_pro_cl * Dpro / V_cl**2 / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:climb',
            'data:propeller:advance_ratio:climb',
        ] = - W_pro_cl * Dpro / V_cl / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:climb',
            'data:propeller:speed:climb',
        ] = - J_climb * Dpro / V_cl / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:climb',
            'data:propeller:geometry:diameter',
        ] = - J_climb * W_pro_cl / V_cl / 2 / np.pi

        partials[
            'data:propeller:constraints:speed:forward',
            'data:mission_nominal:forward:speed',
        ] = J_forward * W_pro_ff * Dpro / V_ff ** 2 / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:forward',
            'data:propeller:advance_ratio:forward',
        ] = - W_pro_ff * Dpro / V_ff / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:forward',
            'data:propeller:speed:forward',
        ] = - J_forward * Dpro / V_ff / 2 / np.pi
        partials[
            'data:propeller:constraints:speed:forward',
            'data:propeller:geometry:diameter',
        ] = - J_forward * W_pro_ff / V_ff / 2 / np.pi


