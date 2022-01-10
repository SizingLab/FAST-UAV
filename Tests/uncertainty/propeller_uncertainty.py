"""
Uncertainty package for the propeller.
"""

import openmdao.api as om
import numpy as np


class DiameterUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:propeller:geometry:diameter:mean', val=np.nan, units='m')
        self.add_input('uncertainty:propeller:geometry:diameter:var', val=0., units=None)
        self.add_output('data:propeller:geometry:diameter:estimated', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_mean = inputs['uncertainty:propeller:geometry:diameter:mean']
        D_var = inputs['uncertainty:propeller:geometry:diameter:var']

        D = D_mean * (1 + D_var)

        outputs['data:propeller:geometry:diameter:estimated'] = D


class BetaUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:propeller:geometry:beta:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:geometry:beta:var', val=0., units=None)
        self.add_output('data:propeller:geometry:beta:estimated', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        beta_mean = inputs['uncertainty:propeller:geometry:beta:mean']
        beta_var = inputs['uncertainty:propeller:geometry:beta:var']

        beta = beta_mean * (1 + beta_var)

        outputs['data:propeller:geometry:beta:estimated'] = beta


class AerodynamicsUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:propeller:aerodynamics:CT:static:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CP:static:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CT:axial:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CP:axial:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CT:incidence:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CP:incidence:mean', val=np.nan, units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CT:var', val=0., units=None)
        self.add_input('uncertainty:propeller:aerodynamics:CP:var', val=0., units=None)
        self.add_output('data:propeller:aerodynamics:CT:static:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:static:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CT:axial:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:axial:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CT:incidence:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:incidence:estimated', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_t_sta = inputs['uncertainty:propeller:aerodynamics:CT:static:mean']
        C_p_sta = inputs['uncertainty:propeller:aerodynamics:CP:static:mean']
        C_t_axial = inputs['uncertainty:propeller:aerodynamics:CT:axial:mean']
        C_p_axial = inputs['uncertainty:propeller:aerodynamics:CP:axial:mean']
        C_t_inc = inputs['uncertainty:propeller:aerodynamics:CT:incidence:mean']
        C_p_inc = inputs['uncertainty:propeller:aerodynamics:CP:incidence:mean']

        C_t_var = inputs['uncertainty:propeller:aerodynamics:CT:var']
        C_p_var = inputs['uncertainty:propeller:aerodynamics:CP:var']

        outputs['data:propeller:aerodynamics:CT:static:estimated'] = C_t_sta * (1 + C_t_var)
        outputs['data:propeller:aerodynamics:CP:static:estimated'] = C_p_sta * (1 + C_p_var)
        outputs['data:propeller:aerodynamics:CT:axial:estimated'] = C_t_axial * (1 + C_t_var)
        outputs['data:propeller:aerodynamics:CP:axial:estimated'] = C_p_axial * (1 + C_p_var)
        outputs['data:propeller:aerodynamics:CT:incidence:estimated'] = C_t_inc * (1 + C_t_var)
        outputs['data:propeller:aerodynamics:CP:incidence:estimated'] = C_p_inc * (1 + C_p_var)


class WeightUncertainty(om.ExplicitComponent):
    """
    Adds variations to the input parameters. For uncertainty or sensitivity analysis purposes.
    """

    def setup(self):
        self.add_input('uncertainty:propeller:mass:mean', val=np.nan, units='kg')
        self.add_input('uncertainty:propeller:mass:var', val=0., units=None)
        self.add_output('data:propeller:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mpro_mean = inputs['uncertainty:propeller:mass:mean']
        Mpro_var = inputs['uncertainty:propeller:mass:var']

        Mpro = Mpro_mean * (1 + Mpro_var)

        outputs['data:propeller:mass:estimated'] = Mpro