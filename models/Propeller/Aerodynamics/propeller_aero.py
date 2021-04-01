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
        self.add_input('data:propeller:geometry:beta:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:advance_ratio:climb', val=np.nan, units=None)
        self.add_output('data:propeller:aerodynamics:CT:static:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:static:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CT:axial:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:axial:estimated', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        beta = inputs['data:propeller:geometry:beta:estimated']
        J = inputs['data:propeller:advance_ratio:climb']

        C_t_sta, C_p_sta = self.aero_coefficients_static(beta)
        C_t_axial, C_p_axial = self.aero_coefficients_axial(beta, J)

        outputs['data:propeller:aerodynamics:CT:static:estimated'] = C_t_sta
        outputs['data:propeller:aerodynamics:CP:static:estimated'] = C_p_sta
        outputs['data:propeller:aerodynamics:CT:axial:estimated'] = C_t_axial
        outputs['data:propeller:aerodynamics:CP:axial:estimated'] = C_p_axial

    @staticmethod
    def aero_coefficients_static(beta):
        C_t_sta = 4.27e-02 + 1.44e-01 * beta  # Thrust coef with T=C_T.rho.n^2.D^4
        C_p_sta = -1.48e-03 + 9.72e-02 * beta  # Power coef with P=C_p.rho.n^3.D^5
        return C_t_sta, C_p_sta

    @staticmethod
    def aero_coefficients_axial(beta, J):
        C_t_axial = 0.02791 - 0.06543 * J + 0.11867 * beta + 0.27334 * beta ** 2 - 0.28852 * beta ** 3 + 0.02104 * J ** 3 \
                  - 0.23504 * J ** 2 + 0.18677 * beta * J ** 2  # thrust coef for APC props in dynamics
        C_p_axial = 0.01813 - 0.06218 * beta + 0.00343 * J + 0.35712 * beta ** 2 - 0.23774 * beta ** 3 + 0.07549 * beta \
                  * J - 0.1235 * J ** 2  # power coef for APC props in dynamics
        return C_t_axial, C_p_axial

