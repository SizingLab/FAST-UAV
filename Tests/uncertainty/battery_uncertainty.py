"""
Uncertainty package for the battery.
"""

import openmdao.api as om
import numpy as np


class VoltageUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:battery:voltage:mean', val=np.nan, units='V')
        self.add_input('uncertainty:battery:voltage:var', val=0., units=None)
        self.add_output('data:battery:voltage:estimated', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_bat_mean = inputs['uncertainty:battery:voltage:mean']
        V_bat_var = inputs['uncertainty:battery:voltage:var']

        V_bat = V_bat_mean * (1 + V_bat_var)

        outputs['data:battery:voltage:estimated'] = V_bat


class CapacityUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:battery:capacity:mean', val=np.nan, units='A*s')
        self.add_input('uncertainty:battery:capacity:var', val=0., units=None)
        self.add_output('data:battery:capacity:estimated', units='A*s')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_bat_mean = inputs['uncertainty:battery:capacity:mean']
        C_bat_var = inputs['uncertainty:battery:capacity:var']

        C_bat = C_bat_mean * (1 + C_bat_var)

        outputs['data:battery:capacity:estimated'] = C_bat


class CurrentUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:battery:current:max:mean', val=np.nan, units='A')
        self.add_input('uncertainty:battery:current:max:var', val=0., units=None)
        self.add_output('data:battery:current:max:estimated', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Imax_mean = inputs['uncertainty:battery:current:max:mean']
        Imax_var = inputs['uncertainty:battery:current:max:var']

        Imax = Imax_mean * (1 + Imax_var)

        outputs['data:battery:current:max:estimated'] = Imax


class MaxDepthOfDischargeUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:battery:DoD:max:mean', val=np.nan, units=None)
        self.add_input('uncertainty:battery:DoD:max:var', val=0., units=None)
        self.add_output('data:battery:DoD:max', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_ratio_mean = inputs['uncertainty:battery:DoD:max:mean']
        C_ratio_var = inputs['uncertainty:battery:DoD:max:var']

        C_ratio = C_ratio_mean * (1 + C_ratio_var)

        outputs['data:battery:DoD:max'] = C_ratio