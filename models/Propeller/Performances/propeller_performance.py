"""
Propeller performances
"""
import openmdao.api as om
import numpy as np

class ComputePropellerPerfoMR(om.ExplicitComponent):
    """
    Characteristics calculation of a Multi-Rotor Propeller
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:axial', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:axial', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:incidence', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:incidence', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:hover', val=np.nan, units='N')
        self.add_input('data:propeller:thrust:climb', val=np.nan, units='N')
        self.add_input('data:propeller:thrust:forward', val=np.nan, units='N')
        self.add_input('data:mission:rho_air', val=np.nan, units='kg/m**3')
        self.add_input('data:propeller:reference:ND:max', val=np.nan, units='m/s')
        self.add_input('data:propeller:settings:ND:k', val=np.nan, units=None)
        self.add_output('data:propeller:speed:takeoff', units='rad/s')
        self.add_output('data:propeller:speed:hover', units='rad/s')
        self.add_output('data:propeller:speed:climb', units='rad/s')
        self.add_output('data:propeller:speed:forward', units='rad/s')
        self.add_output('data:propeller:torque:takeoff', units='N*m')
        self.add_output('data:propeller:torque:hover', units='N*m')
        self.add_output('data:propeller:torque:climb', units='N*m')
        self.add_output('data:propeller:torque:forward', units='N*m')
        self.add_output('data:propeller:power:takeoff', units='W')
        self.add_output('data:propeller:power:hover', units='W')
        self.add_output('data:propeller:power:climb', units='W')
        self.add_output('data:propeller:power:forward', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
        C_t_axial = inputs['data:propeller:aerodynamics:CT:axial']
        C_t_inc = inputs['data:propeller:aerodynamics:CT:incidence']
        C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
        C_p_axial = inputs['data:propeller:aerodynamics:CP:axial']
        C_p_inc = inputs['data:propeller:aerodynamics:CP:incidence']
        F_pro_hov = inputs['data:propeller:thrust:hover']
        F_pro_cl = inputs['data:propeller:thrust:climb']
        F_pro_ff = inputs['data:propeller:thrust:forward']
        rho_air = inputs['data:mission:rho_air']
        NDmax = inputs['data:propeller:reference:ND:max']
        k_ND = inputs['data:propeller:settings:ND:k']

        # Propeller torque & speed for take-off
        n_pro_to = NDmax * k_ND / Dpro  # [Hz] Propeller speed
        Wpro_to = n_pro_to * 2 * np.pi  # [rad/s] Propeller speed
        Ppro_to = C_p_sta * rho_air * n_pro_to ** 3 * Dpro ** 5  # [W] Power per propeller
        Qpro_to = Ppro_to / Wpro_to  # [N.m] Propeller torque

        # Propeller torque & speed for hover
        n_pro_hover = (F_pro_hov / (C_t_sta * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for hover
        Wpro_hover = n_pro_hover * 2 * np.pi  # [rad/s] Propeller speed for hover
        Ppro_hover = C_p_sta * rho_air * n_pro_hover ** 3 * Dpro ** 5  # [W] Power per propeller
        Qpro_hover = Ppro_hover / Wpro_hover  # [N.m] Propeller torque

        # Propeller torque & speed for climbing
        n_pro_cl = (F_pro_cl / (C_t_axial * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for climbing
        Wpro_cl = n_pro_cl * 2 * np.pi  # [rad/s] Propeller speed for climbing
        Ppro_cl = C_p_axial * rho_air * n_pro_cl ** 3 * Dpro ** 5  # [W] Power per propeller for climbing
        Qpro_cl = Ppro_cl / Wpro_cl  # [N.m] Propeller torque for climbing

        # Propeller torque & speed for forward flight
        n_pro_ff = (F_pro_ff / (C_t_inc * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for forward flight
        Wpro_ff = n_pro_ff * 2 * np.pi  # [rad/s] Propeller speed for forward flight
        Ppro_ff = C_p_inc * rho_air * n_pro_ff ** 3 * Dpro ** 5  # [W] Power per propeller for forward flight
        Qpro_ff = Ppro_ff / Wpro_ff  # [N.m] Propeller torque for forward flight

        outputs['data:propeller:speed:takeoff'] = Wpro_to
        outputs['data:propeller:speed:hover'] = Wpro_hover
        outputs['data:propeller:speed:climb'] = Wpro_cl
        outputs['data:propeller:speed:forward'] = Wpro_ff
        outputs['data:propeller:torque:takeoff'] = Qpro_to
        outputs['data:propeller:torque:hover'] = Qpro_hover
        outputs['data:propeller:torque:climb'] = Qpro_cl
        outputs['data:propeller:torque:forward'] = Qpro_ff
        outputs['data:propeller:power:takeoff'] = Ppro_to
        outputs['data:propeller:power:hover'] = Ppro_hover
        outputs['data:propeller:power:climb'] = Ppro_cl
        outputs['data:propeller:power:forward'] = Ppro_ff
