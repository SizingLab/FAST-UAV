"""
Sizing scenarios definition
"""
import openmdao.api as om
import numpy as np
from fastoad.utils.physics.atmosphere import AtmosphereSI
from scipy.constants import g

class SizingScenarios(om.ExplicitComponent):
    """
    Sizing scenarios definition: Hover and Take-off
    """

    def setup(self):
        self.add_input('data:payload:settings:k_M', val=np.nan, units=None)
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number_per_arm', val=np.nan, units=None)
        self.add_input('data:structure:arms:arm_number', val=np.nan, units=None)
        self.add_input('specifications:altitude', val=np.nan, units='m')
        self.add_input('specifications:dISA', val=np.nan, units='K')
        self.add_input('data:structure:aerodynamics:C_D', val=np.nan, units=None)
        self.add_input('data:structure:body:surface:top', val=np.nan, units='m**2')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_input('specifications:k_maxthrust', val=np.nan, units=None)
        self.add_output('data:propeller:thrust:hover', units='N')
        self.add_output('data:propeller:thrust:climb', units='N')
        self.add_output('data:propeller:thrust:max', units='N')
        #self.add_output('optimization:objectives:mass_total_estimated', units='kg')
        self.add_output('data:propeller:prop_number', units=None)
        self.add_output('specifications:rho_air', units='kg/m**3')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_M = inputs['data:payload:settings:k_M']
        M_load = inputs['data:payload:mass']
        Npro_arm = inputs['data:propeller:prop_number_per_arm']
        Narm = inputs['data:structure:arms:arm_number']
        altitude = inputs['specifications:altitude']
        dISA = inputs['specifications:dISA']
        C_D = inputs['data:structure:aerodynamics:C_D']
        A_top = inputs['data:structure:body:surface:top']
        V_cl = inputs['specifications:climb_speed']
        k_maxthrust = inputs['specifications:k_maxthrust']

        rho_air = AtmosphereSI(altitude, dISA).density
        Npro = Npro_arm * Narm  # Number of propellers
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        F_pro_hov = Mtotal_estimated * g / Npro  # [N] Thrust per propeller for hover
        F_pro_cl = (Mtotal_estimated * g + 0.5 * (rho_air) * (C_D) * (A_top) * (V_cl) ** 2) / Npro  # [N] Thrust per propeller for climbing
        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        outputs['data:propeller:prop_number'] = Npro
        #outputs['optimization:objectives:mass_total_estimated'] = Mtotal_estimated
        outputs['data:propeller:thrust:hover'] = F_pro_hov
        outputs['data:propeller:thrust:climb'] = F_pro_cl
        outputs['data:propeller:thrust:max'] = F_pro_to
        outputs['specifications:rho_air'] = rho_air