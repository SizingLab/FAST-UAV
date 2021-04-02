"""
System parameters
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("multirotor.system")
class System(om.Group):
    """
    Group containing the system parameters
    """
    def setup(self):
        self.add_subsystem("MTOW", MTOW(), promotes=['*'])
        self.add_subsystem("hoverAutonomy", HoverAutonomy(), promotes=['*'])
        self.add_subsystem("range", Range(), promotes=['*'])
        self.add_subsystem("constraints", SystemConstraints(), promotes=['*'])


class MTOW(om.ExplicitComponent):
    """
    MTOW calculation
    """

    def setup(self):
        self.add_input('data:gearbox:mass', val=0.0, units='kg')
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:body:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:system:MTOW', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mgear = inputs['data:gearbox:mass']  # default value = .0 if use_gearbox = false
        Mmot = inputs['data:motor:mass']
        Mesc = inputs['data:ESC:mass']
        Mbat = inputs['data:battery:mass']
        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:number']
        M_load = inputs['data:payload:mass']
        Mfra = inputs['data:structure:body:mass']
        Marm = inputs['data:structure:arms:mass']

        # System mass
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        outputs['data:system:MTOW'] = Mtotal


class HoverAutonomy(om.ExplicitComponent):
    """
    Max. Hover autonomy calculation.
    Payload and avionics power consumption are taken into account.
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:battery:discharge_limit', val=0.8, units=None)
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('data:system:autonomy:hover', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        C_ratio = inputs['data:battery:discharge_limit']
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']

        I_bat = (P_el_hover * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery
        t_hov_max = C_ratio * C_bat / I_bat  # [s] Max. hover flight time

        outputs['data:system:autonomy:hover'] = t_hov_max / 60  # [min]


class Range(om.ExplicitComponent):
    """
    Range calculation at given V_ff speed.
    Range is calculated either including hover and climb requirements or not.
    Payload and avionics power consumption are also taken into account.
    """

    def setup(self):
        self.add_input('data:mission:speed:forward', val=np.nan, units='m/s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:forward', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:battery:discharge_limit', val=0.8, units=None)
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_input('data:mission:energy:hover', val=np.nan, units='J')
        self.add_input('data:mission:energy:climb', val=np.nan, units='J')
        self.add_input('specifications:time:hover', val=np.nan, units='s')
        self.add_input('data:mission:time:climb', val=np.nan, units='s')
        self.add_output('data:system:range:min', units='m')
        self.add_output('data:system:range:max', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_ff = inputs['data:motor:power:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        V_ff = inputs['data:mission:speed:forward']
        C_ratio = inputs['data:battery:discharge_limit']
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']
        E_hov = inputs['data:mission:energy:hover']
        E_cl = inputs['data:mission:energy:climb']
        t_hov = inputs['specifications:time:hover']
        t_cl = inputs['data:mission:time:climb']

        I_bat = (P_el_ff * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery during forward

        # Range without climb and hover requirements
        D_ff_max = V_ff * (C_ratio * C_bat) / I_bat  # [m] Range

        # Range with user defined climb and hover requirements
        C_hov = (E_hov + P_payload/eta_ESC * t_hov + P_avionics/eta_ESC * t_hov) / V_bat  # [Ah] Capacity consumption during hover
        C_cl  = (E_cl + P_payload/eta_ESC * t_cl + P_avionics/eta_ESC * t_cl) / V_bat  # [Ah] Capacity consumption during climb
        D_ff_min = V_ff * (C_ratio * C_bat - C_hov - C_cl) / I_bat  # [m] Range

        outputs['data:system:range:max'] = D_ff_max # [m]
        outputs['data:system:range:min'] = D_ff_min  # [m]


class SystemConstraints(om.ExplicitComponent):
    """
    System constraints
    """

    def setup(self):
        self.add_input('specifications:MTOW', val=np.nan, units='kg')
        self.add_input('data:system:MTOW', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:system:settings:MTOW:k', val=np.nan, units=None)
        self.add_output('data:system:constraints:mass:convergence', units=None)
        self.add_output('data:system:constraints:mass:MTOW', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        MTOW = inputs['specifications:MTOW']
        Mtotal = inputs['data:system:MTOW']
        M_load = inputs['data:payload:mass']
        k_M = inputs['data:system:settings:MTOW:k']
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)

        mass_con = (Mtotal_estimated - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        outputs['data:system:constraints:mass:convergence'] = mass_con
        outputs['data:system:constraints:mass:MTOW'] = MTOW_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        MTOW = inputs['specifications:MTOW']
        Mtotal = inputs['data:system:MTOW']
        M_load = inputs['data:payload:mass']
        k_M = inputs['data:system:settings:MTOW:k']
        Mtotal_estimated = k_M * M_load

        partials[
            'data:system:constraints:mass:convergence',
            'data:system:settings:MTOW:k',
        ] = M_load / Mtotal
        partials[
            'data:system:constraints:mass:convergence',
            'data:payload:mass',
        ] = k_M / Mtotal
        partials[
            'data:system:constraints:mass:convergence',
            'data:system:MTOW',
        ] = - Mtotal_estimated / Mtotal**2

        partials[
            'data:system:constraints:mass:MTOW',
            'specifications:MTOW',
        ] = 1.0 / Mtotal
        partials[
            'data:system:constraints:mass:MTOW',
            'data:system:MTOW',
        ] = - MTOW / Mtotal**2