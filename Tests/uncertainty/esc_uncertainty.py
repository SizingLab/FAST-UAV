"""
Uncertainty package for the ESC.
"""

import openmdao.api as om
import numpy as np


class VoltageUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:ESC:voltage:mean', val=np.nan, units='V')
        self.add_input('uncertainty:ESC:voltage:var', val=0., units=None)
        self.add_output('data:ESC:voltage:estimated', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_esc_mean = inputs['uncertainty:ESC:voltage:mean']
        V_esc_var = inputs['uncertainty:ESC:voltage:var']

        V_esc = V_esc_mean * (1 + V_esc_var)

        outputs['data:ESC:voltage:estimated'] = V_esc


class PowerUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:ESC:power:max:mean', val=np.nan, units='W')
        self.add_input('uncertainty:ESC:power:max:var', val=0., units=None)
        self.add_output('data:ESC:power:max:estimated', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        P_esc_mean = inputs['uncertainty:ESC:power:max:mean']
        P_esc_var = inputs['uncertainty:ESC:power:max:var']

        P_esc = P_esc_mean * (1 + P_esc_var)

        outputs['data:ESC:power:max:estimated'] = P_esc


class WeightUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:ESC:mass:mean', val=np.nan, units='kg')
        self.add_input('uncertainty:ESC:mass:var', val=0., units=None)
        self.add_output('data:ESC:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mesc_mean = inputs['uncertainty:ESC:mass:mean']
        Mesc_var = inputs['uncertainty:ESC:mass:var']

        Mesc = Mesc_mean * (1 + Mesc_var)

        outputs['data:ESC:mass:estimated'] = Mesc


class EfficiencyUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:ESC:efficiency:mean', val=np.nan, units=None)
        self.add_input('uncertainty:ESC:efficiency:var', val=0., units=None)
        self.add_output('data:ESC:efficiency', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        eta_mean = inputs['uncertainty:ESC:efficiency:mean']
        eta_var = inputs['uncertainty:ESC:efficiency:var']

        eta = eta_mean * (1 + eta_var)

        outputs['data:ESC:efficiency'] = eta