"""
Propeller performances
"""
import openmdao.api as om
import numpy as np

class ComputePropellerPerfoMR(om.ExplicitComponent):
    """
    Characteristics calculation of a Multi-Rotor Propeller
    """

    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        if self.options["use_catalogues"]:
            self.add_input('data:propeller:catalogue:diameter', val=np.nan, units='m')
            self.add_input('data:propeller:catalogue:aerodynamics:CT:static', val=np.nan, units=None)
            self.add_input('data:propeller:catalogue:aerodynamics:CP:static', val=np.nan, units=None)
            self.add_input('data:propeller:catalogue:aerodynamics:CT:dynamic', val=np.nan, units=None)
            self.add_input('data:propeller:catalogue:aerodynamics:CP:dynamic', val=np.nan, units=None)
        else:
            self.add_input('data:propeller:diameter', val=np.nan, units='m')
            self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
            self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
            self.add_input('data:propeller:aerodynamics:CT:dynamic', val=np.nan, units=None)
            self.add_input('data:propeller:aerodynamics:CP:dynamic', val=np.nan, units=None)

        self.add_input('data:propeller:thrust:climb', val=np.nan, units='N')
        self.add_input('data:propeller:thrust:hover', val=np.nan, units='N')
        self.add_input('data:air_density', val=np.nan, units='kg/m**3')
        self.add_input('data:propeller:reference:nD_max', val=np.nan, units='m/s')
        self.add_input('optimization:settings:k_ND', val=np.nan, units=None)
        self.add_output('data:propeller:speed:takeoff', units='rad/s')
        self.add_output('data:propeller:speed:hover', units='rad/s')
        self.add_output('data:propeller:speed:climb', units='rad/s')
        self.add_output('data:propeller:torque:takeoff', units='N*m')
        self.add_output('data:propeller:torque:hover', units='N*m')
        self.add_output('data:propeller:torque:climb', units='N*m')
        self.add_output('data:propeller:power:takeoff', units='W')
        self.add_output('data:propeller:power:hover', units='W')
        self.add_output('data:propeller:power:climb', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_catalogues"]:
            Dpro = inputs['data:propeller:catalogue:diameter']
            C_t_sta = inputs['data:propeller:catalogue:aerodynamics:CT:static']
            C_t_dyn = inputs['data:propeller:catalogue:aerodynamics:CT:dynamic']
            C_p_sta = inputs['data:propeller:catalogue:aerodynamics:CP:static']
            C_p_dyn = inputs['data:propeller:catalogue:aerodynamics:CP:dynamic']
        else:
            Dpro = inputs['data:propeller:diameter']
            C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
            C_t_dyn = inputs['data:propeller:aerodynamics:CT:dynamic']
            C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
            C_p_dyn = inputs['data:propeller:aerodynamics:CP:dynamic']

        F_pro_cl = inputs['data:propeller:thrust:climb']
        F_pro_hov = inputs['data:propeller:thrust:hover']
        rho_air = inputs['data:air_density']
        NDmax = inputs['data:propeller:reference:nD_max']
        k_ND = inputs['optimization:settings:k_ND']

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

        outputs['data:propeller:speed:takeoff'] = Wpro_to
        outputs['data:propeller:speed:hover'] = Wpro_hover
        outputs['data:propeller:speed:climb'] = Wpro_cl
        outputs['data:propeller:torque:takeoff'] = Qpro_to
        outputs['data:propeller:torque:hover'] = Qpro_hover
        outputs['data:propeller:torque:climb'] = Qpro_cl
        outputs['data:propeller:power:takeoff'] = Ppro_to
        outputs['data:propeller:power:hover'] = Ppro_hover
        outputs['data:propeller:power:climb'] = Ppro_cl
