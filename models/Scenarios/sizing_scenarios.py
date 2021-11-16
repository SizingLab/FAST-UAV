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
class SizingScenarios(om.Group):
    """
    Sizing scenarios definition
    """
    def setup(self):
        self.add_subsystem("base", Base(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("forward", Forward(), promotes=["*"])


class Base(om.ExplicitComponent):
    """
    Computes base parameters
    """

    def setup(self):
        self.add_input('data:system:settings:MTOW:k', val=np.nan, units=None)
        self.add_input('specifications:payload:mass:max', val=np.nan, units='kg')
        self.add_input('data:structure:arms:prop_per_arm', val=np.nan, units=None)
        self.add_input('data:structure:arms:number', val=np.nan, units=None)
        self.add_input('mission:sizing_mission:takeoff:altitude', val=np.nan, units='m')
        self.add_input('mission:sizing_mission:dISA', val=np.nan, units='K')
        self.add_input('specifications:climb_height', val=np.nan, units='m')
        self.add_input('data:structure:reference:body:surface:top', val=np.nan, units='m**2')
        self.add_input('data:structure:reference:body:surface:front', val=np.nan, units='m**2')
        self.add_input('data:system:reference:MTOW', val=np.nan, units='kg')
        self.add_output('mission:sizing_mission:air_density', units='kg/m**3')
        self.add_output('mission:sizing_mission:forward:altitude', units='m')
        self.add_output('data:propeller:number', units=None)
        self.add_output('data:system:MTOW:guess', units='kg')
        self.add_output('data:structure:body:surface:top', units='m**2')
        self.add_output('data:structure:body:surface:front', units='m**2')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_M = inputs['data:system:settings:MTOW:k']
        M_load = inputs['specifications:payload:mass:max']
        Npro_arm = inputs['data:structure:arms:prop_per_arm']
        Narm = inputs['data:structure:arms:number']
        altitude_TO = inputs['mission:sizing_mission:takeoff:altitude']
        dISA = inputs['mission:sizing_mission:dISA']
        D_cl = inputs['specifications:climb_height']
        S_top_ref = inputs['data:structure:reference:body:surface:top']
        S_front_ref = inputs['data:structure:reference:body:surface:front']
        MTOW_ref = inputs['data:system:reference:MTOW']

        # Multirotor architecture
        Npro = Npro_arm * Narm  # [-] Number of propellers
        Mtotal_guess = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)

        # Flight conditions
        altitude_FF = altitude_TO + D_cl  # [m] Cruise altitude
        rho_air = AtmosphereSI(altitude_FF, dISA).density  # [kg/m3] Air density at cruise level
        # TODO: define rho_air for takeoff / climb / descent ? For now we are conservative

        # Surfaces scaling laws
        S_top_estimated = S_top_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] top surface estimation
        S_front_estimated = S_front_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] front surface estimation

        outputs['data:propeller:number'] = Npro
        outputs['data:system:MTOW:guess'] = Mtotal_guess
        outputs['mission:sizing_mission:air_density'] = rho_air
        outputs['mission:sizing_mission:forward:altitude'] = altitude_FF
        outputs['data:structure:body:surface:top'] = S_top_estimated
        outputs['data:structure:body:surface:front'] = S_front_estimated


class Hover(om.ExplicitComponent):
    """
    Hover scenario definition
    """

    def setup(self):
        self.add_input('data:system:MTOW:guess', val=np.nan, units='kg')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_output('data:propeller:thrust:hover', units='N')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs['data:system:MTOW:guess']
        Npro = inputs['data:propeller:number']

        F_pro_hov = Mtotal_guess * g / Npro  # [N] Thrust per propeller for hover

        outputs['data:propeller:thrust:hover'] = F_pro_hov


class TakeOff(om.ExplicitComponent):
    """
    Takeoff scenario definition
    """

    def setup(self):
        self.add_input('specifications:maxthrust:k', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:hover', val=np.nan, units='N')
        self.add_output('data:propeller:thrust:max', units='N')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        F_pro_hov = inputs['data:propeller:thrust:hover']
        k_maxthrust = inputs['specifications:maxthrust:k']

        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        outputs['data:propeller:thrust:max'] = F_pro_to


class Climb(om.ExplicitComponent):
    """
    Climb scenario definition
    """

    def setup(self):
        self.add_input('mission:sizing_mission:air_density', val=np.nan, units='kg/m**3')
        self.add_input('data:system:MTOW:guess', val=np.nan, units='kg')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:aerodynamics:Cd', val=np.nan, units=None)
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_input('data:structure:body:surface:top', val=np.nan, units='m**2')
        self.add_output('data:propeller:thrust:climb', units='N')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs['data:system:MTOW:guess']
        Npro = inputs['data:propeller:number']
        C_D = inputs['data:aerodynamics:Cd']
        V_cl = inputs['specifications:climb_speed']
        rho_air = inputs['mission:sizing_mission:air_density']
        S_top_estimated = inputs['data:structure:body:surface:top']

        F_pro_cl = (Mtotal_guess * g + 0.5 * rho_air * C_D * S_top_estimated * V_cl ** 2) / Npro

        # PROVISION FOR CLIMBING FORWARD FLIGHT (PATH ANGLE THETA)
        # theta = np.pi / 2  # [rad] flight path angle (vertical climb)
        # F_pro_cl, alpha_cl = FlightModel.get_thrust(Mtotal_guess, V_cl, theta, S_front_estimated, S_top_estimated, C_D, C_L0, rho_air)  # [N] required thrust (and angle of attack)
        # F_pro_cl = F_pro_cl / Npro  # [N] thrust per propeller

        outputs['data:propeller:thrust:climb'] = F_pro_cl


class Forward(om.ExplicitComponent):
    """
    Forward flight scenario definition
    """

    def setup(self):
        self.add_input('mission:sizing_mission:air_density', val=np.nan, units='kg/m**3')
        self.add_input('data:system:MTOW:guess', val=np.nan, units='kg')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:aerodynamics:Cd', val=np.nan, units=None)
        self.add_input('data:aerodynamics:Cl0', val=np.nan, units=None)
        self.add_input('mission:sizing_mission:forward:speed', val=np.nan, units='m/s')
        self.add_input('data:structure:body:surface:top', val=np.nan, units='m**2')
        self.add_input('data:structure:body:surface:front', val=np.nan, units='m**2')
        self.add_output('data:propeller:thrust:forward', units='N')
        self.add_output('mission:sizing_mission:forward:angle', units='rad')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs['data:system:MTOW:guess']
        Npro = inputs['data:propeller:number']
        C_D = inputs['data:aerodynamics:Cd']
        C_L0 = inputs['data:aerodynamics:Cl0']
        V_ff = inputs['mission:sizing_mission:forward:speed']
        rho_air = inputs['mission:sizing_mission:air_density']
        S_top_estimated = inputs['data:structure:body:surface:top']
        S_front_estimated = inputs['data:structure:body:surface:front']

        func = lambda x: np.tan(x) - 0.5 * rho_air * C_D * (S_top_estimated * np.sin(x) + S_front_estimated * np.cos(x)) * V_ff ** 2 \
                         / (Mtotal_guess * g + 0.5 * rho_air * C_L0 * (S_top_estimated * np.sin(x) + S_front_estimated * np.cos(x)) * V_ff ** 2)
        alpha = brentq(func, 0, np.pi / 2)  # [rad] angle of attack
        S_ref = S_top_estimated * np.sin(alpha) + S_front_estimated * np.cos(alpha)  # [m2] reference surface for drag and lift calculation
        drag = 0.5 * rho_air * C_D * S_ref * V_ff ** 2  # [N] drag
        lift = - 0.5 * rho_air * C_L0 * S_ref * V_ff ** 2  # [N] lift (downwards force)
        weight = Mtotal_guess * g  # [N] weight
        F_pro_ff = ((weight - lift)**2 + drag**2) ** (1/2) / Npro  # [N] thrust per propeller

        #lift = - 0.5 * rho_air * C_L0 * np.sin(2*alpha) * S_top_estimated * np.sin(alpha) * V_ff ** 2  # [N] downwards force (flat plate model)
        #F_pro_ff = ((weight - lift) ** 2 + drag ** 2) ** (1 / 2) / Npro  # [N] thrust per propeller

        # PROVISION FOR CLIMBING FORWARD FLIGHT (PATH ANGLE THETA)
        # theta = 0  # [rad] flight path angle (steady level flight)
        # F_pro_ff, alpha_ff = FlightModel.get_thrust(Mtotal_guess, V_ff, theta, S_front_estimated, S_top_estimated, C_D, C_L0, rho_air)  # [N] required thrust (and angle of attack)
        # F_pro_ff = F_pro_ff / Npro # [N] thrust per propeller

        outputs['data:propeller:thrust:forward'] = F_pro_ff
        outputs['mission:sizing_mission:forward:angle'] = alpha