"""
Propeller performances
"""
import openmdao.api as om
import numpy as np


class PropellerModel:
    """
    Propeller model for performances calculation
    """

    @staticmethod
    def speed(F_pro, D_pro, C_t, rho_air):
        n_pro = (F_pro / (C_t * rho_air * D_pro ** 4)) ** 0.5  # [Hz] Propeller speed
        W_pro = n_pro * 2 * np.pi  # [rad/s] Propeller speed
        return W_pro

    @staticmethod
    def power(W_pro, D_pro, C_p, rho_air):
        P_pro = C_p * rho_air * (W_pro / (2 * np.pi)) ** 3 * D_pro ** 5  # [W] Propeller power
        return P_pro

    @staticmethod
    def torque(P_pro, W_pro):
        Q_pro = P_pro / W_pro  # [N.m] Propeller torque
        return Q_pro

    @staticmethod
    def performances(F_pro, D_pro, C_t, C_p, rho_air):
        W_pro = PropellerModel.speed(F_pro, D_pro, C_t, rho_air)  # [rad/s] Propeller speed
        P_pro = PropellerModel.power(W_pro, D_pro, C_p, rho_air)  # [W] Propeller power
        Q_pro = PropellerModel.torque(P_pro, W_pro)  # [N.m] Propeller torque
        return W_pro, P_pro, Q_pro


class PropellerPerfos(om.Group):
    """
    Group containing the performance functions of the propeller
    """
    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("forward", Forward(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes performances of the propeller for takeoff
    """

    def setup(self):
        # self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        # self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
        # self.add_input('data:mission_design:air_density', val=np.nan, units='kg/m**3')
        # self.add_input('data:propeller:reference:ND:max', val=np.nan, units='m/s')
        # self.add_input('data:propeller:settings:ND:k', val=np.nan, units=None)

        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:max', val=np.nan, units='N')
        self.add_input('data:mission_design:air_density', val=np.nan, units='kg/m**3')
        self.add_output('data:propeller:speed:takeoff', units='rad/s')
        self.add_output('data:propeller:torque:takeoff', units='N*m')
        self.add_output('data:propeller:power:takeoff', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
        C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
        F_pro_to = inputs['data:propeller:thrust:max']
        rho_air = inputs['data:mission_design:air_density']

        # Dpro = inputs['data:propeller:geometry:diameter']
        # C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
        # rho_air = inputs['data:mission_design:air_density']
        # NDmax = inputs['data:propeller:reference:ND:max']
        # k_ND = inputs['data:propeller:settings:ND:k']

        # n_pro_to = (F_pro_to / (C_t_sta * rho_air * Dpro ** 4)) ** 0.5
        # n_pro_to = NDmax * k_ND / Dpro  # [Hz] Propeller speed
        # Wpro_to = n_pro_to * 2 * np.pi  # [rad/s] Propeller speed
        # Ppro_to = C_p_sta * rho_air * n_pro_to ** 3 * Dpro ** 5  # [W] Power per propeller
        # Qpro_to = Ppro_to / Wpro_to  # [N.m] Propeller torque

        Wpro_to = PropellerModel.speed(F_pro_to, Dpro, C_t_sta, rho_air)
        Ppro_to = PropellerModel.power(Wpro_to, Dpro, C_p_sta, rho_air)
        Qpro_to = PropellerModel.torque(Ppro_to, Wpro_to)

        outputs['data:propeller:speed:takeoff'] = Wpro_to
        outputs['data:propeller:torque:takeoff'] = Qpro_to
        outputs['data:propeller:power:takeoff'] = Ppro_to


class Hover(om.ExplicitComponent):
    """
    Computes performances of the propeller for hover
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:hover', val=np.nan, units='N')
        self.add_input('data:mission_design:air_density', val=np.nan, units='kg/m**3')
        self.add_output('data:propeller:speed:hover', units='rad/s')
        self.add_output('data:propeller:torque:hover', units='N*m')
        self.add_output('data:propeller:power:hover', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
        C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
        F_pro_hov = inputs['data:propeller:thrust:hover']
        rho_air = inputs['data:mission_design:air_density']

        # n_pro_hover = (F_pro_hov / (C_t_sta * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for hover
        # Wpro_hover = n_pro_hover * 2 * np.pi  # [rad/s] Propeller speed for hover
        # Ppro_hover = C_p_sta * rho_air * n_pro_hover ** 3 * Dpro ** 5  # [W] Power per propeller
        # Qpro_hover = Ppro_hover / Wpro_hover  # [N.m] Propeller torque

        Wpro_hover = PropellerModel.speed(F_pro_hov, Dpro, C_t_sta, rho_air)
        Ppro_hover = PropellerModel.power(Wpro_hover, Dpro, C_p_sta, rho_air)
        Qpro_hover = PropellerModel.torque(Ppro_hover, Wpro_hover)

        outputs['data:propeller:speed:hover'] = Wpro_hover
        outputs['data:propeller:torque:hover'] = Qpro_hover
        outputs['data:propeller:power:hover'] = Ppro_hover


class Climb(om.ExplicitComponent):
    """
    Computes performances of the propeller for climb
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:axial', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:axial', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:climb', val=np.nan, units='N')
        self.add_input('data:mission_design:air_density', val=np.nan, units='kg/m**3')
        self.add_output('data:propeller:speed:climb', units='rad/s')
        self.add_output('data:propeller:torque:climb', units='N*m')
        self.add_output('data:propeller:power:climb', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_axial = inputs['data:propeller:aerodynamics:CT:axial']
        C_p_axial = inputs['data:propeller:aerodynamics:CP:axial']
        F_pro_cl = inputs['data:propeller:thrust:climb']
        rho_air = inputs['data:mission_design:air_density']

        # n_pro_cl = (F_pro_cl / (C_t_axial * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for climbing
        # Wpro_cl = n_pro_cl * 2 * np.pi  # [rad/s] Propeller speed for climbing
        # Ppro_cl = C_p_axial * rho_air * n_pro_cl ** 3 * Dpro ** 5  # [W] Power per propeller for climbing
        # Qpro_cl = Ppro_cl / Wpro_cl  # [N.m] Propeller torque for climbing

        Wpro_cl = PropellerModel.speed(F_pro_cl, Dpro, C_t_axial, rho_air)
        Ppro_cl = PropellerModel.power(Wpro_cl, Dpro, C_p_axial, rho_air)
        Qpro_cl = PropellerModel.torque(Ppro_cl, Wpro_cl)

        outputs['data:propeller:speed:climb'] = Wpro_cl
        outputs['data:propeller:torque:climb'] = Qpro_cl
        outputs['data:propeller:power:climb'] = Ppro_cl


class Forward(om.ExplicitComponent):
    """
    Computes performances of the propeller for forward flight
    """

    def setup(self):
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:incidence', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:incidence', val=np.nan, units=None)
        self.add_input('data:propeller:thrust:forward', val=np.nan, units='N')
        self.add_input('data:mission_design:air_density', val=np.nan, units='kg/m**3')
        self.add_output('data:propeller:speed:forward', units='rad/s')
        self.add_output('data:propeller:torque:forward', units='N*m')
        self.add_output('data:propeller:power:forward', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Dpro = inputs['data:propeller:geometry:diameter']
        C_t_inc = inputs['data:propeller:aerodynamics:CT:incidence']
        C_p_inc = inputs['data:propeller:aerodynamics:CP:incidence']
        F_pro_ff = inputs['data:propeller:thrust:forward']
        rho_air = inputs['data:mission_design:air_density']

        # n_pro_ff = (F_pro_ff / (C_t_inc * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Propeller speed for forward flight
        # Wpro_ff = n_pro_ff * 2 * np.pi  # [rad/s] Propeller speed for forward flight
        # Ppro_ff = C_p_inc * rho_air * n_pro_ff ** 3 * Dpro ** 5  # [W] Power per propeller for forward flight
        # Qpro_ff = Ppro_ff / Wpro_ff  # [N.m] Propeller torque for forward flight

        Wpro_ff = PropellerModel.speed(F_pro_ff, Dpro, C_t_inc, rho_air)
        Ppro_ff = PropellerModel.power(Wpro_ff, Dpro, C_p_inc, rho_air)
        Qpro_ff = PropellerModel.torque(Ppro_ff, Wpro_ff)

        outputs['data:propeller:speed:forward'] = Wpro_ff
        outputs['data:propeller:torque:forward'] = Qpro_ff
        outputs['data:propeller:power:forward'] = Ppro_ff




