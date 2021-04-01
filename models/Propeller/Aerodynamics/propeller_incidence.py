"""
Propeller aerodynamics
"""
from models.Propeller.Aerodynamics.propeller_aero import ComputePropellerAeroMR
import openmdao.api as om
import numpy as np

class ComputePropellerAeroIncidenceMR(om.ExplicitComponent):
    """
    Aerodynamics of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('data:propeller:geometry:beta:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:advance_ratio:forward', val=np.nan, units=None)
        self.add_input('data:mission:angle:forward', val=np.nan, units='rad')
        self.add_output('data:propeller:aerodynamics:CT:incidence:estimated', units=None)
        self.add_output('data:propeller:aerodynamics:CP:incidence:estimated', units=None)


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        beta = inputs['data:propeller:geometry:beta:estimated']
        J = inputs['data:propeller:advance_ratio:forward']
        alpha = inputs['data:mission:angle:forward']

        C_t_inc, C_p_inc = self.aero_coefficients_incidence(beta, J, alpha)

        outputs['data:propeller:aerodynamics:CT:incidence:estimated'] = C_t_inc
        outputs['data:propeller:aerodynamics:CP:incidence:estimated'] = C_p_inc


    @staticmethod
    def aero_coefficients_incidence(beta, J, alpha, N_blades=2, chord_to_radius=0.15, r_norm=0.75):
        """
        Incidence power coefficient (Y. Leng et al. model)
        """

        # Parameters at zero incidence propeller angle (vertical flight)
        J_axial = J * np.sin(alpha)
        C_t_axial, C_p_axial = ComputePropellerAeroMR.aero_coefficients_axial(beta, J_axial)
        J_0t_axial= 0.197 + 1.094 * beta
        J_0p_axial = 0.286 + 0.993 * beta

        # solidity correction factor
        sigma = N_blades * chord_to_radius / np.pi
        delta = 3 / 2 * np.cos(beta) * (
                    1 + sigma / np.tan(beta) * (1 + np.sqrt(1 + 2 * np.tan(beta) / sigma)) * (1 - np.sin(alpha)))

        # incidence ratios
        eta_t = 1 + (J * np.cos(alpha) / np.pi / r_norm) ** 2 / 2 / (1 - J / J_0t_axial * np.sin(alpha)) * delta
        eta_p = 1 + (J * np.cos(alpha) / np.pi / r_norm) ** 2 / 2 / (1 - J / J_0p_axial * np.sin(alpha)) * delta

        # thrust and power coefficients
        C_t_inc = C_t_axial * eta_t
        C_p_inc = C_p_axial * eta_p

        return C_t_inc, C_p_inc

