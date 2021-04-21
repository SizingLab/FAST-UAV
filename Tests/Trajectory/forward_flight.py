"""
Forward flight steady-state model
"""

import openmdao.api as om
import numpy as np
from scipy.constants import g
from scipy.optimize import fsolve, brentq


class ForwardFlight(om.ExplicitComponent):

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:geometry:beta', val=np.nan, units=None)
        self.add_input('data:system:MTOW', val=np.nan, units='kg')
        self.add_input('data:structure:body:surface:top', val=np.nan, units='m**2')
        self.add_input('data:structure:body:surface:front', val=np.nan, units='m**2')
        self.add_input('data:structure:aerodynamics:Cd', val=np.nan, units=None)
        self.add_input('data:mission_nominal:air_density', val=np.nan, units='kg/m**3')
        self.add_input('data:propeller:reference:ND:max', val=np.nan, units='m/s')
        self.add_input('data:motor:torque:max', val=np.nan, units='N*m')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:battery:discharge_limit', val=np.nan, units=None)
        self.add_input('specifications:range', val=np.nan, units='m')
        self.add_input('trajectory:forward_flight:velocity', val=np.nan, units='m/s')
        self.add_output('trajectory:forward_flight:power', units='W')
        self.add_output('trajectory:forward_flight:energy', units='J')
        self.add_output('trajectory:forward_flight:omega', units='rad/s')
        self.add_output('trajectory:forward_flight:constraints:propeller:speed', units=None)
        self.add_output('trajectory:forward_flight:constraints:motor:torque', units=None)
        self.add_output('trajectory:forward_flight:constraints:battery:energy', units=None)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        N_pro = inputs['data:propeller:number']
        D_pro = inputs['data:propeller:geometry:diameter']
        NDmax = inputs['data:propeller:reference:ND:max']
        beta = inputs['data:propeller:geometry:beta']
        Tmot_max = inputs['data:motor:torque:max']
        MTOW = inputs['data:system:MTOW']
        S_top = inputs['data:structure:body:surface:top']
        S_front = inputs['data:structure:body:surface:front']
        Cd_ref = inputs['data:structure:aerodynamics:Cd']
        rho_air = inputs['data:mission_nominal:air_density']
        C_bat = inputs['data:battery:capacity']
        U_bat = inputs['data:battery:voltage']
        distance = inputs['specifications:range']
        V_inf = inputs['trajectory:forward_flight:velocity']
        C_ratio = inputs['data:battery:discharge_limit']
        # TODO : add gearbox reduction ratio

        self._ForwardFlightModel = ForwardFlightModel(V_inf,
                                                      N_pro,
                                                      D_pro,
                                                      beta,
                                                      MTOW,
                                                      S_top,
                                                      S_front,
                                                      Cd_ref,
                                                      rho_air)

        P_ff, omega = self._ForwardFlightModel.compute_power()  # propeller power
        Tmot_ff = P_ff / omega  # torque per motor TODO : add gearbox reduction ratio
        E_ff = P_ff * N_pro * distance / V_inf  # TODO: replace by elec power (cf motor + eta_ESC)

        outputs['trajectory:forward_flight:energy'] = E_ff
        outputs['trajectory:forward_flight:power'] = P_ff # per propeller
        outputs['trajectory:forward_flight:omega'] = omega
        outputs['trajectory:forward_flight:constraints:propeller:speed'] = (NDmax - omega / 2 / np.pi * D_pro) / NDmax
        outputs['trajectory:forward_flight:constraints:motor:torque'] = (Tmot_max - Tmot_ff) / Tmot_max
        outputs['trajectory:forward_flight:constraints:battery:energy'] = (C_ratio * C_bat * U_bat - E_ff) / (C_ratio * C_bat * U_bat)


class ForwardFlightModel:
    """
    Straight and steady-level flight model

    usage:
        >> FF=ForwardFlightV_inf, (N_pro, D_pro, beta, MTOW, S_top, S_front, Cd_ref)
        >> P_ff = FF.compute_power()
    """

    def __init__(
        self,
        V_inf,
        N_pro,
        D_pro,
        beta,
        MTOW,
        S_top,
        S_front,
        Cd_ref,
        rho_air
    ):
        """
        :param V_inf: UAV velocity [m/s]
        :param N_pro: number of propellers [-]
        :param D_pro: propeller diameter [m]
        :param beta: propeller pitch angle [rad]
        :param MTOW: max. takeoff weight [kg]
        :param S_top: top surface of the UAV [m**2]
        :param S_front: side surface of the UAV [m**2]
        :param Cd_ref: body drag coefficient [-]
        """
        # Inputs
        self._V_inf = V_inf
        self._N_pro = N_pro
        self._R = D_pro / 2
        self._MTOW = MTOW
        self._S_top = S_top
        self._S_front = S_front
        self._Cd_ref = Cd_ref
        self._rho_air = rho_air

        # Propeller Model
        self._propeller_model = IncidencePropellerModel(beta)

        # Outputs
        self._v_i = None
        self._omega = None
        self._mu = None
        self._alpha = None
        self._thrust = None
        self._drag = None
        self._Ct = None
        self._Cp = None
        self._eta = None
        self._P_ind = None
        self._P_parasitic = None
        self._P_loss = None
        self._P_req = None


    def compute_ct(self, mu, alpha):
        """
        Thrust coefficient model (wrapper).
        Must return the coefficient with the helicopter notation: CT = T / (rho Omega^2 pi R^4)
        """
        Ct = self._propeller_model.compute_ct(mu * np.pi, alpha)
        return Ct

    def compute_cp(self, mu, alpha):
        """
        Power coefficient model (wrapper)
        Must return the coefficient with the helicopter notation: CP = P / (rho Omega^3 pi R^5)
        """
        Cp = self._propeller_model.compute_cp(mu * np.pi, alpha)
        return Cp

    def induced_velocity(self):
        """
        Computes the induced velocity from Glauert's model
        """
        func = lambda x: x - self._thrust / (2 * self._rho_air * self._N_pro * np.pi * self._R ** 2) / (
                    (self._V_inf * np.cos(self._alpha)) ** 2 + (self._V_inf * np.sin(self._alpha) + x) ** 2) ** (1 / 2)
        v_i = fsolve(func, x0=1)[0]
        return v_i

    def rotational_speed(self):
        """
        Solves the thrust equation for the rotational speed
        """
        # rotational speed solved with thrust equation

        func = lambda x: self._thrust - self._N_pro * self._rho_air * self._R ** 4 * np.pi * x ** 2 * self.compute_ct(self._V_inf / x / self._R,
                                                                                    self._alpha)
        omega = fsolve(func, x0=1000)[0]  # [rad/s]
        return omega

    def compute_drag(self, alpha):
        """
        Compute the drag of the UAV for a given speed and angle of attack
        """
        S_ref = self._S_top * np.sin(alpha) + self._S_front * np.cos(alpha)
        D = 0.5 * self._rho_air * self._Cd_ref * S_ref * self._V_inf ** 2  # drag [N]
        return D

    def angle_of_attack(self):
        """
        Finds the angle of attack required to satisfy equilibrium equations
        """
        W = self._MTOW * g  # weight [N]
        func = lambda x: np.tan(x) - self.compute_drag(x) / W
        alpha = brentq(func, 0, np.pi / 2)
        return alpha

    def efficiency(self):
        """
        Computes the propeller efficiency
        """
        eta = (self._V_inf * np.sin(self._alpha) + self._v_i) / (self._omega * self._R) * self._Ct / self._Cp  # propeller efficiency
        return eta

    def compute_power(self):
        """
        Computes the required propellers power for a forward flight at given velocity
        """
        #self._drag = 0.5 * self._rho_air * self._Cd_ref * self._S_ref * self._V_inf ** 2  # drag [N]
        #self._alpha = np.arctan(self._drag / (self._MTOW * g))  # incidence angle [rad]
        self._alpha = self.angle_of_attack()
        self._drag = self.compute_drag(self._alpha)
        self._thrust= ((self._MTOW * g) ** 2 + self._drag ** 2) ** (1 / 2)  # thrust [N]
        self._v_i = self.induced_velocity()  # induced velocity [m/s]
        self._omega = self.rotational_speed()  # rotational speed [rad/s]
        self._mu = self._V_inf / self._omega / self._R
        self._Ct = self.compute_ct(self._mu, self._alpha)
        self._Cp = self.compute_cp(self._mu, self._alpha)
        #mu_axial = self._mu * np.sin(self._alpha)  # axial advance ratio [-]
        #lmbda = (self._V_inf * np.sin(self._alpha) + self._v_i) / self._omega / self._R  # inflow ratio [-]
        self._eta = self.efficiency()  # propeller efficiency [-]

        # powers
        self._P_ind = self._thrust * self._v_i  # induced power [W]
        self._P_parasitic = self._drag * self._V_inf  # parasitic power [W]
        self._P_loss = (1 / self._eta - 1) * (self._P_ind + self._P_parasitic)  # propeller losses [W]
        self._P_req = self._P_ind + self._P_parasitic + self._P_loss  # total power [W]
        return self._P_req / self._N_pro, self._omega

    #def optimize_energy(self, distance):
    #    """
    #    DEPRECATED - Optimizes the velocity to minimize the energy required for the mission
    #    """
    #    func = lambda x: self.compute_power(x) * distance / x
    #    res = minimize(func, [10], method='SLSQP', bounds=Bounds(1, 30))
    #    self._V_inf = res.x
    #    self._E_req = res.fun
    #    self._P_req = self.compute_power(self._V_inf)
    #    return self._E_req, self._P_req / self._N_pro, self._V_inf, self._omega




class IncidencePropellerModel:
    """
    Incidence Propeller Model - Derived from Y. Leng

    usage:
        >> propeller_model = IncidencePropellerModel(N_pro, D_pro, beta)
        >> ct = propeller_model.compute_ct(J, alpha)
    """

    def __init__(
        self,
        beta,
    ):
        """
        :param beta: propeller pitch angle [rad]
        """
        # Inputs
        self._beta = beta

        # default inputs
        self._chord_to_radius = 0.15
        self._N_blades = 2
        self._r_norm = 0.75

        # Outputs
        self._Ct = None
        self._Cp = None

    def solidity_correction_factor(self, alpha):
        """
        Computes the High Incidence Thrust and Power Correction Factor
        """
        # sigma = N_blades * chord / (np.pi * r_norm * R)
        sigma = self._N_blades * self._chord_to_radius / np.pi
        delta = 3 / 2 * np.cos(self._beta) * (
                    1 + sigma / np.tan(self._beta) * (1 + np.sqrt(1 + 2 * np.tan(self._beta) / sigma)) * (1 - np.sin(alpha)))
        return delta

    def compute_ct_axial_conditions(self, J_axial):
        """
        Computes the thrust coefficient in axial flight conditions
        """
        # APC propellers
        Ct = 0.011 - 0.141 * J_axial + 0.282 * self._beta - 0.133 * J_axial ** 2 + 0.164 * J_axial * self._beta - 0.139 * self._beta ** 2
        # Ct= -0.043 - 0.201* J_axial + 0.555 * pitch - 0.145 * J_axial**2 + 0.294 * J_axial * pitch - 0.462 * pitch**2 # non-dominated props
        return Ct

    def compute_cp_axial_conditions(self, J_axial):
        """
        Computes the power coefficient in axial flight conditions
        """
        Cp = 0.046 + 0.036 * J_axial - 0.301 * self._beta - 0.017 * J_axial ** 2 - 0.241 * J_axial * self._beta + 1.054 * self._beta ** 2 - 0.062 * J_axial ** 3 - 0.088 * J_axial ** 2 * self._beta + 0.441 * J_axial * self._beta ** 2 - 0.890 * self._beta ** 3
        # Cp = 0.308 + 0.548 * J_axial - 2.322 * pitch + 0.331 * J_axial**2 - 2.881 * J_axial * pitch + 6.141 * pitch**2 - 0.060 * J_axial**3 - 0.714 * J_axial**2 * pitch + 3.519 * J_axial * pitch**2 - 5.021 * pitch**3 # non-dominated propellers
        return Cp

    def compute_J_0t_axial(self):
        """
        Computes the advance ratio where the thrust coefficient reaches zero, in axial flight conditions
        """
        J_0t_axial = 0.197 + 1.094 * self._beta
        # J_0t_axial = 0.080 + 1.310 * pitch # non-dominated propellers
        return J_0t_axial

    def compute_J_0p_axial(self):
        """
        Computes the advance ratio where the power coefficient reaches zero, in axial flight conditions
        """
        J_0p_axial = 0.286 + 0.993 * self._beta
        # J_0p_axial= -4141.155 + 56141.043 * pitch - 314474.039 * pitch**2 + 932418.159 * pitch**3 - 1544215.386 * pitch**4 + 1354985.876 * pitch**5 - 492301.163 * pitch**6 # non-dominated propellers (to be simplified ?)
        return J_0p_axial

    def compute_eta_t(self, alpha, J, J_0t_axial):
        """
        Computes the incidence thrust ratio
        """
        r_norm = 0.75  # representative radius position
        delta = self.solidity_correction_factor(alpha)
        eta = 1 + (J * np.cos(alpha) / np.pi / r_norm) ** 2 / 2 / (1 - J * np.sin(alpha) / J_0t_axial) * delta
        return eta

    def compute_eta_p(self, alpha, J, J_0p_axial):
        """
        Compute the incidence power ratio
        """
        r_norm = self._r_norm  # representative radius position
        delta = self.solidity_correction_factor(alpha)
        eta = 1 + (J * np.cos(alpha) / np.pi / r_norm) ** 2 / 2 / (1 - J / J_0p_axial * np.sin(alpha)) * delta
        return eta

    def compute_ct(self, J, alpha):
        """
        Incidence thrust coefficient (Y. Leng et al. model)
        """
        # Parameters at zero incidence propeller angle (vertical flight)
        J_axial = J * np.sin(alpha)
        Ct_axial_conditions = self.compute_ct_axial_conditions(J_axial)
        J_0t_axial = self.compute_J_0t_axial()

        # Thrust Ratio
        eta = self.compute_eta_t(alpha, J, J_0t_axial)

        # Thrust coefficient at non zero incidence angle
        Ct = Ct_axial_conditions * eta
        self._Ct = Ct * 4 / np.pi ** (3)  # helicopter notation

        return self._Ct

    def compute_cp(self, J, alpha):
        """
        Incidence power coefficient (Y. Leng et al. model)
        """
        # Parameters at zero incidence propeller angle (vertical flight)
        J_axial = J * np.sin(alpha)
        Cp_axial_conditions = self.compute_cp_axial_conditions(J_axial)
        J_0p_axial = self.compute_J_0p_axial()

        # Power Ratio
        eta = self.compute_eta_p(alpha, J, J_0p_axial)

        # Power coefficient at non zero incidence angle
        Cp = Cp_axial_conditions * eta
        self._Cp = Cp * 4 / np.pi ** (4)  # helicopter notation

        return self._Cp