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
        self.add_input('optimization:settings:k_M', val=np.nan, units=None)
        self.add_input('specifications:load:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number_per_arm', val=np.nan, units=None)
        self.add_input('data:structure:geometry:arms:arm_number', val=np.nan, units=None)
        self.add_input('specifications:rho_air', val=np.nan, units='kg/m**3')
        self.add_input('data:structure:aerodynamics:C_D', val=np.nan, units=None)
        self.add_input('data:structure:geometry:top_surface', val=np.nan, units='m**2')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_input('specifications:k_maxthrust', val=np.nan, units=None)
        self.add_output('data:propeller:performances:hover_thrust_prop', units='N')
        self.add_output('data:propeller:performances:climb_thrust_prop', units='N')
        self.add_output('data:propeller:performances:max_thrust_prop', units='N')
        #self.add_output('optimization:objectives:mass_total_estimated', units='kg')
        self.add_output('data:propeller:prop_number', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_M = inputs['optimization:settings:k_M']
        M_load = inputs['specifications:load:mass']
        Npro_arm = inputs['data:propeller:prop_number_per_arm']
        Narm = inputs['data:structure:geometry:arms:arm_number']
        rho_air = inputs['specifications:rho_air']
        C_D = inputs['data:structure:aerodynamics:C_D']
        A_top = inputs['data:structure:geometry:top_surface']
        V_cl = inputs['specifications:climb_speed']
        k_maxthrust = inputs['specifications:k_maxthrust']

        Npro = Npro_arm * Narm  # Number of propellers
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        F_pro_hov = Mtotal_estimated * (9.81) / Npro  # [N] Thrust per propeller for hover
        F_pro_cl = (Mtotal_estimated * 9.81 + 0.5 * (rho_air) * (C_D) * (A_top) * (V_cl) ** 2) / Npro  # [N] Thrust per propeller for climbing
        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        outputs['data:propeller:prop_number'] = Npro
        #outputs['optimization:objectives:mass_total_estimated'] = Mtotal_estimated
        outputs['data:propeller:performances:hover_thrust_prop'] = F_pro_hov
        outputs['data:propeller:performances:climb_thrust_prop'] = F_pro_cl
        outputs['data:propeller:performances:max_thrust_prop'] = F_pro_to