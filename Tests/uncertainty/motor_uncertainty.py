"""
Uncertainty package for the motor.
"""

import openmdao.api as om
import numpy as np


class NominalTorqueUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:torque:nominal:mean', val=np.nan, units='N*m')
        self.add_input('uncertainty:motor:torque:nominal:var', val=0., units=None)
        self.add_output('data:motor:torque:nominal:estimated', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tnom_mean = inputs['uncertainty:motor:torque:nominal:mean']
        Tnom_var = inputs['uncertainty:motor:torque:nominal:var']

        Tnom = Tnom_mean * (1 + Tnom_var)

        outputs['data:motor:torque:nominal:estimated'] = Tnom


class TorqueCoefficientUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:torque:coefficient:mean', val=np.nan, units='N*m/A')
        self.add_input('uncertainty:motor:torque:coefficient:var', val=0., units=None)
        self.add_output('data:motor:torque:coefficient:estimated', units='N*m/A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Kt_mean = inputs['uncertainty:motor:torque:coefficient:mean']
        Kt_var = inputs['uncertainty:motor:torque:coefficient:var']

        Kt = Kt_mean * (1 + Kt_var)

        outputs['data:motor:torque:coefficient:estimated'] = Kt


class MaxTorqueUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:torque:max:mean', val=np.nan, units='N*m')
        self.add_input('uncertainty:motor:torque:max:var', val=0., units=None)
        self.add_output('data:motor:torque:max:estimated', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tmax_mean = inputs['uncertainty:motor:torque:max:mean']
        Tmax_var = inputs['uncertainty:motor:torque:max:var']

        Tmax = Tmax_mean * (1 + Tmax_var)

        outputs['data:motor:torque:max:estimated'] = Tmax


class FrictionTorqueUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:torque:friction:mean', val=np.nan, units='N*m')
        self.add_input('uncertainty:motor:torque:friction:var', val=0., units=None)
        self.add_output('data:motor:torque:friction:estimated', units='N*m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Tf_mean = inputs['uncertainty:motor:torque:friction:mean']
        Tf_var = inputs['uncertainty:motor:torque:friction:var']

        Tf = Tf_mean * (1 + Tf_var)

        outputs['data:motor:torque:friction:estimated'] = Tf


class ResistanceUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:resistance:mean', val=np.nan, units='V/A')
        self.add_input('uncertainty:motor:resistance:var', val=0., units=None)
        self.add_output('data:motor:resistance:estimated', units='V/A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        R_mean = inputs['uncertainty:motor:resistance:mean']
        R_var = inputs['uncertainty:motor:resistance:var']

        R = R_mean * (1 + R_var)

        outputs['data:motor:resistance:estimated'] = R


class WeightUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:motor:mass:mean', val=np.nan, units='kg')
        self.add_input('uncertainty:motor:mass:var', val=0., units=None)
        self.add_output('data:motor:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mmot_mean = inputs['uncertainty:motor:mass:mean']
        Mmot_var = inputs['uncertainty:motor:mass:var']

        Mmot = Mmot_mean * (1 + Mmot_var)

        outputs['data:motor:mass:estimated'] = Mmot