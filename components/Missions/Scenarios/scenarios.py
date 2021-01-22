"""
Sizing scenarios definition
"""
import openmdao.api as om
import numpy as np


class SizingScenarios(om.ExplicitComponent):
    """
    Sizing scenarios definition
    """

    def setup(self):
        self.add_input('settings:mission:k_M', val=np.nan)
        self.add_input('data:mission:load:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan)
        self.add_input('data:mission:rho_air', val=np.nan, units='kg/m**3')
        self.add_input('data:structure:aerodynamics:C_D', val=np.nan)
        self.add_input('data:structure:geometry:top_surface', val=np.nan, units='m**2')
        self.add_input('data:mission:climb_speed', val=np.nan, units='m/s')
        self.add_input('settings:mission:k_maxthrust', val=np.nan)
        self.add_output('data:mission:thrust:hover_thrust_prop', units='N')
        self.add_output('data:mission:thrust:climb_thrust_prop', units='N')
        self.add_output('data:mission:thrust:max_thrust_prop', units='N')
        self.add_output('data:mission:mass_total_estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_M = inputs['settings:mission:k_M']
        M_load = inputs['data:mission:load:mass']
        Npro = inputs['data:propeller:prop_number']
        rho_air = inputs['data:mission:rho_air']
        C_D = inputs['data:structure:aerodynamics:C_D']
        A_top = inputs['data:structure:geometry:top_surface']
        V_cl = inputs['data:mission:climb_speed']
        k_maxthrust = inputs['settings:mission:k_maxthrust']

        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        F_pro_hov = Mtotal_estimated * (9.81) / Npro  # [N] Thrust per propeller for hover
        F_pro_cl = (Mtotal_estimated * 9.81 + 0.5 * (rho_air) * (C_D) * (A_top) * (V_cl) ** 2) / Npro  # [N] Thrust per propeller for climbing
        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        outputs['data:mission:mass_total_estimated'] = Mtotal_estimated
        outputs['data:mission:thrust:hover_thrust_prop'] = F_pro_hov
        outputs['data:mission:thrust:climb_thrust_prop'] = F_pro_cl
        outputs['data:mission:thrust:max_thrust_prop'] = F_pro_to