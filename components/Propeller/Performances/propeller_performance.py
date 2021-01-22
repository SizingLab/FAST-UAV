"""
Propeller performances
"""
import openmdao.api as om
import numpy as np

class ComputePropellerPerfoMR(om.ExplicitComponent):
    """
    Performances calculation of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('data:mission:thrust:climb_thrust_prop', val=np.nan, units='N')
        self.add_input('data:mission:thrust:hover_thrust_prop', val=np.nan, units='N')
        self.add_input('data:mission:rho_air', val=np.nan, units='kg/m**3')
        self.add_input('data:propeller:aerodynamics:CT_static', val=np.nan)
        self.add_input('data:propeller:aerodynamics:CP_static', val=np.nan)
        self.add_input('data:propeller:aerodynamics:CT_dynamic', val=np.nan)
        self.add_input('data:propeller:aerodynamics:CP_dynamic', val=np.nan)
        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('settings:propeller:k_ND', val=np.nan)
        self.add_input('data:propeller:geometry:diameter', units='m')
        self.add_output('data:propeller:performances:rot_speed_takeoff', units='rad/s')
        self.add_output('data:propeller:performances:rot_speed_hover', units='rad/s')
        self.add_output('data:propeller:performances:rot_speed_climb', units='rad/s')
        self.add_output('data:propeller:performances:torque_takeoff', units='N*m')
        self.add_output('data:propeller:performances:torque_hover', units='N*m')
        self.add_output('data:propeller:performances:torque_climb', units='N*m')
        self.add_output('data:propeller:performances:power_takeoff', units='W')
        self.add_output('data:propeller:performances:power_hover', units='W')
        self.add_output('data:propeller:performances:power_climb', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        F_pro_cl = inputs['data:mission:thrust:climb_thrust_prop']
        F_pro_hov = inputs['data:mission:thrust:hover_thrust_prop']
        rho_air = inputs['data:mission:rho_air']
        NDmax = inputs['data:propeller:reference:nD_max']
        k_ND = inputs['settings:propeller:k_ND']
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_sta = inputs['data:propeller:aerodynamics:CT_static']
        C_t_dyn = inputs['data:propeller:aerodynamics:CT_dynamic']
        C_p_sta = inputs['data:propeller:aerodynamics:CP_static']
        C_p_dyn = inputs['data:propeller:aerodynamics:CP_dynamic']

        # Propeller torque& speed for take-off
        n_pro_to = NDmax * k_ND / Dpro  # [Hz] Propeller speed
        Wpro_to = n_pro_to * 2 * 3.14  # [rad/s] Propeller speed
        Ppro_to = C_p_sta * rho_air * n_pro_to ** 3 * Dpro ** 5  # [W] Power per propeller
        Qpro_to = Ppro_to / Wpro_to  # [N.m] Propeller torque

        # Propeller torque& speed for hover
        n_pro_hover = (F_pro_hov / (C_t_sta * rho_air * Dpro ** 4)) ** 0.5  # [Hz] hover speed
        Wpro_hover = n_pro_hover * 2 * 3.14  # [rad/s] Propeller speed
        Ppro_hover = C_p_sta * rho_air * n_pro_hover ** 3 * Dpro ** 5  # [W] Power per propeller
        Qpro_hover = Ppro_hover / Wpro_hover  # [N.m] Propeller torque

        # Propeller torque &speed for climbing
        n_pro_cl = (F_pro_cl / (C_t_dyn * rho_air * Dpro ** 4)) ** 0.5  # [Hz] climbing speed
        Wpro_cl = n_pro_cl * 2 * 3.14  # [rad/s] Propeller speed for climbing
        Ppro_cl = C_p_dyn * rho_air * n_pro_cl ** 3 * Dpro ** 5  # [W] Power per propeller for climbing
        Qpro_cl = Ppro_cl / Wpro_cl  # [N.m] Propeller torque for climbing

        outputs['data:propeller:performances:rot_speed_takeoff'] = Wpro_to
        outputs['data:propeller:performances:rot_speed_hover'] = Wpro_hover
        outputs['data:propeller:performances:rot_speed_climb'] = Wpro_cl
        outputs['data:propeller:performances:torque_takeoff'] = Qpro_to
        outputs['data:propeller:performances:torque_hover'] = Qpro_hover
        outputs['data:propeller:performances:torque_climb'] = Qpro_cl
        outputs['data:propeller:performances:power_takeoff'] = Ppro_to
        outputs['data:propeller:performances:power_hover'] = Ppro_hover
        outputs['data:propeller:performances:power_climb'] = Ppro_cl
