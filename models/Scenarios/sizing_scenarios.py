"""
Sizing scenarios definition
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastoad.model_base.atmosphere import AtmosphereSI
from scipy.constants import g
from scipy.optimize import brentq


@oad.RegisterOpenMDAOSystem("multirotor.sizing_scenarios")
class SizingScenarios(om.ExplicitComponent):
    """
    Sizing scenarios definition: Hover and Take-off
    """

    def setup(self):
        self.add_input('data:system:settings:MTOW:k', val=np.nan, units=None)
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:prop_per_arm', val=np.nan, units=None)
        self.add_input('data:structure:arms:number', val=np.nan, units=None)
        self.add_input('specifications:altitude', val=np.nan, units='m')
        self.add_input('specifications:dISA', val=np.nan, units='K')
        self.add_input('data:structure:aerodynamics:Cd', val=np.nan, units=None)
        self.add_input('specifications:speed:climb', val=np.nan, units='m/s')
        self.add_input('data:mission:speed:forward', val=np.nan, units='m/s')
        self.add_input('specifications:maxthrust:k', val=np.nan, units=None)
        self.add_input('data:structure:reference:body:surface:top', val=np.nan, units='m**2')
        self.add_input('data:structure:reference:body:surface:front', val=np.nan, units='m**2')
        self.add_input('data:system:reference:MTOW', val=np.nan, units='kg')
        self.add_output('data:propeller:thrust:hover', units='N')
        self.add_output('data:propeller:thrust:climb', units='N')
        self.add_output('data:propeller:thrust:max', units='N')
        self.add_output('data:propeller:thrust:forward', units='N')
        self.add_output('data:mission:angle:forward', units='rad')
        self.add_output('data:mission:rho_air', units='kg/m**3')
        self.add_output('data:propeller:number', units=None)
        self.add_output('data:structure:body:surface:top', units='m**2')
        self.add_output('data:structure:body:surface:front', units='m**2')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_M = inputs['data:system:settings:MTOW:k']
        M_load = inputs['data:payload:mass']
        Npro_arm = inputs['data:structure:arms:prop_per_arm']
        Narm = inputs['data:structure:arms:number']
        altitude = inputs['specifications:altitude']
        dISA = inputs['specifications:dISA']
        C_D = inputs['data:structure:aerodynamics:Cd']
        V_cl = inputs['specifications:speed:climb']
        V_ff = inputs['data:mission:speed:forward']
        k_maxthrust = inputs['specifications:maxthrust:k']
        S_top_ref = inputs['data:structure:reference:body:surface:top']
        S_front_ref = inputs['data:structure:reference:body:surface:front']
        MTOW_ref = inputs['data:system:reference:MTOW']

        rho_air = AtmosphereSI(altitude, dISA).density
        Npro = Npro_arm * Narm  # Number of propellers
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)

        #S_top_estimated = (0.0243 * Mtotal_estimated) / 2 if (Mtotal_estimated < 25) else (0.0046 * Mtotal_estimated + 0.4485) / 2
        #S_front_estimated = (0.0179 * Mtotal_estimated) / 2 if (Mtotal_estimated < 25) else (0.0019 * Mtotal_estimated + 0.357) / 2
        S_top_estimated = S_top_ref * (Mtotal_estimated / MTOW_ref) ** (2/3)
        S_front_estimated = S_front_ref * (Mtotal_estimated / MTOW_ref) ** (2/3)

        # Thrust
        F_pro_hov = Mtotal_estimated * g / Npro  # [N] Thrust per propeller for hover
        F_pro_cl = (Mtotal_estimated * g + 0.5 * (rho_air) * (C_D) * (S_top_estimated) * (V_cl) ** 2) / Npro  # [N] Thrust per propeller for climbing
        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        # TODO: compute alpha with an implicit component or other approach ?
        func = lambda x: np.tan(x) - 0.5 * rho_air * C_D * (S_top_estimated * np.sin(x) + S_front_estimated * np.cos(x)) * V_ff ** 2 \
                         / (Mtotal_estimated * g)
        alpha = brentq(func, 0, np.pi / 2)
        S_ref = S_top_estimated * np.sin(alpha) + S_front_estimated * np.cos(alpha)
        Drag = 0.5 * rho_air * C_D * S_ref * V_ff ** 2
        F_pro_ff = ((Mtotal_estimated * g)**2 + (Drag)**2) ** (1/2) / Npro

        outputs['data:propeller:number'] = Npro
        outputs['data:propeller:thrust:hover'] = F_pro_hov
        outputs['data:propeller:thrust:climb'] = F_pro_cl
        outputs['data:propeller:thrust:max'] = F_pro_to
        outputs['data:propeller:thrust:forward'] = F_pro_ff
        outputs['data:mission:angle:forward'] = alpha
        outputs['data:mission:rho_air'] = rho_air
        outputs['data:structure:body:surface:top'] = S_top_estimated
        outputs['data:structure:body:surface:front'] = S_front_estimated