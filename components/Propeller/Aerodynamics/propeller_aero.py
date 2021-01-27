"""
Propeller aerodynamics
"""
import openmdao.api as om
import numpy as np

class ComputePropellerAeroMR(om.ExplicitComponent):
    """
    Aerodynamics of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('optimization:settings:beta_pro', val=np.nan, units=None)
        self.add_input('optimization:settings:advance_ratio', val=np.nan, units=None)
        self.add_output('data:propeller:aerodynamics:CT_static', units=None)
        self.add_output('data:propeller:aerodynamics:CP_static', units=None)
        self.add_output('data:propeller:aerodynamics:CT_dynamic', units=None)
        self.add_output('data:propeller:aerodynamics:CP_dynamic', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        beta = inputs['optimization:settings:beta_pro']
        J = inputs['optimization:settings:advance_ratio']

        C_t_sta = 4.27e-02 + 1.44e-01 * beta  # Thrust coef with T=C_T.rho.n^2.D^4
        C_p_sta = -1.48e-03 + 9.72e-02 * beta  # Power coef with P=C_p.rho.n^3.D^5
        C_t_dyn = 0.02791 - 0.06543 * J + 0.11867 * beta + 0.27334 * beta ** 2 - 0.28852 * beta ** 3 + 0.02104 * J ** 3 \
                  - 0.23504 * J ** 2 + 0.18677 * beta * J ** 2  # thrust coef for APC props in dynamics
        C_p_dyn = 0.01813 - 0.06218 * beta + 0.00343 * J + 0.35712 * beta ** 2 - 0.23774 * beta ** 3 + 0.07549 * beta \
                  * J - 0.1235 * J ** 2  # power coef for APC props in dynamics

        outputs['data:propeller:aerodynamics:CT_static'] = C_t_sta
        outputs['data:propeller:aerodynamics:CP_static'] = C_p_sta
        outputs['data:propeller:aerodynamics:CT_dynamic'] = C_t_dyn
        outputs['data:propeller:aerodynamics:CP_dynamic'] = C_p_dyn
