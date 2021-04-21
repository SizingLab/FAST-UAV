"""
Safety module
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from scipy.constants import g


@oad.RegisterOpenMDAOSystem("addons.safety")
class Safety(om.Group):
    """
    Group containing the safety requirements components
    """
    def setup(self):
        self.add_subsystem("motor_torque", MotorDegradedPerfos(), promotes=['*'])
        #self.add_subsystem("degraded_autonomy", degradedRange(), promotes=['*'])
        #self.add_subsystem("degraded_range", degradedRange(), promotes=['*'])


class MotorDegradedPerfos(om.ExplicitComponent):
    """
    Computes degraded motor perfos
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('safety:rotor_fail', val=np.nan, units=None)
        self.add_input('safety:k_uneven', val=np.nan, units=None)
        self.add_input('data:system:MTOW', val=np.nan, units=None)
        self.add_input('data:mission_nominal:air_density', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:static', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units=None)
        self.add_input('data:motor:torque:max', val=np.nan, units=None)
        self.add_output('safety:motor_constraint:hover', units=None)

    def setup_partials(self):
        # TODO : define partials for constraint
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # TODO : add forward flight scenario

        Npro = inputs['data:propeller:number']
        Npro_fail = inputs['safety:rotor_fail']
        k_asym = inputs['safety:k_asym']  # [-] even-to-uneven thrust oversizing ratio to maintain axis controllability
        Mtotal = inputs['data:system:MTOW']
        rho_air = inputs['data:mission_nominal:air_density']
        C_t_sta = inputs['data:propeller:aerodynamics:CT:static']
        C_p_sta = inputs['data:propeller:aerodynamics:CP:static']
        Dpro = inputs['data:propeller:geometry:diameter']
        Nred = 1.0 # TODO : take into account gearbox option
        Tmot_max = inputs['data:motor:torque:max']

        # Degraded scenario
        Npro_oper = Npro - Npro_fail  # [-] number of rotors operating
        F_pro_hov = k_asym * Mtotal * g / Npro_oper  # [N] Corrected thrust per propeller for hover

        # Degraded propeller performance
        n_pro_hover = (F_pro_hov / (C_t_sta * rho_air * Dpro ** 4)) ** 0.5  # [Hz] Corrected propeller speed for hover
        Wpro_hover = n_pro_hover * 2 * np.pi  # [rad/s] Corrected propeller speed for hover
        Ppro_hover = C_p_sta * rho_air * n_pro_hover ** 3 * Dpro ** 5  # [W] Corrected power per propeller
        Qpro_hover = Ppro_hover / Wpro_hover  # [N.m] Corrected propeller torque

        # Degraded motor performance
        Tmot_hover = Qpro_hover / Nred  # [N.m] motor nominal torque with reduction

        # Motor torque constraint
        motor_con_hover = (Tmot_max - Tmot_hover) / Tmot_max

        outputs['safety:motor_constraint:hover'] = motor_con_hover










