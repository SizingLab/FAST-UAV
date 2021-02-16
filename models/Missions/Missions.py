"""
Missions definition
"""
import openmdao.api as om
import numpy as np


class Missions(om.Group):
    """
    Group containing the missions objectives and constraints
    """
    def setup(self):
        self.add_subsystem("objectives", MissionObjectives(), promotes=['*'])
        self.add_subsystem("constraints", MissionConstraints(), promotes=['*'])


class MissionObjectives(om.Group):
    def setup(self):
        self.add_subsystem("MTOW", MTOWObjective(), promotes=['*'])
        self.add_subsystem("hover_flight", HoverAutonomyObjective(), promotes=['*'])


class MissionConstraints(om.Group):
    def setup(self):
        self.add_subsystem("mass_convergence", MassConvergenceConstraint(), promotes=['*'])


class MTOWObjective(om.ExplicitComponent):
    """
    Weight objective with hover flight autonomy constraint
    """

    def setup(self):
        self.add_input('data:gearbox:mass', val=0.0, units='kg')
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:frame:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('mission:hover_time:specification', val=np.nan, units='min')
        self.add_input('data:battery:discharge_limit', val=0.8, units=None)
        self.add_output('data:system:MTOW', units='kg')
        self.add_output('constraints:system:flight_autonomy', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mgear = inputs['data:gearbox:mass']  # default value = .0 if use_gearbox = false
        Mmot = inputs['data:motor:mass']
        Mesc = inputs['data:ESC:mass']
        Mbat = inputs['data:battery:mass']
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']
        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['data:payload:mass']
        Mfra = inputs['data:structure:frame:mass']
        Marm = inputs['data:structure:arms:mass']
        t_h = inputs['mission:hover_time:specification']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        C_ratio = inputs['data:battery:discharge_limit']

        # Mission
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        I_bat = (P_el_hover * Npro) / eta_ESC / V_bat  # [I] Current of the battery
        t_hf = C_ratio * C_bat / I_bat / 60  # [min] Hover time

        # Constraints
        autonomy_con = (t_hf - t_h) / t_hf  # Min. hover flight autonomy, for weight minimization

        outputs['data:system:MTOW'] = Mtotal
        outputs['constraints:system:flight_autonomy'] = autonomy_con


class HoverAutonomyObjective(om.ExplicitComponent):
    """
    Hover flight autonomy objective with MTOW constraint
    """

    def setup(self):
        self.add_input('data:gearbox:mass', val=0.0, units='kg')
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:frame:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:system:MTOW:specification', val=np.nan, units='kg')
        self.add_input('data:battery:discharge_limit', val=0.8, units=None)
        self.add_output('mission:hover_time', units='min')
        self.add_output('constraints:system:MTOW', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mgear = inputs['data:gearbox:mass']  # default value = .0 if use_gearbox = false
        Mmot = inputs['data:motor:mass']
        Mesc = inputs['data:ESC:mass']
        Mbat = inputs['data:battery:mass']
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']
        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['data:payload:mass']
        Mfra = inputs['data:structure:frame:mass']
        Marm = inputs['data:structure:arms:mass']
        MTOW = inputs['data:system:MTOW:specification']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        C_ratio = inputs['data:battery:discharge_limit']

        # Mission
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        I_bat = (P_el_hover * Npro) / eta_ESC / V_bat  # [I] Current of the battery
        t_hf = C_ratio * C_bat / I_bat / 60  # [min] Hover time

        # Constraints
        MTOW_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        outputs['mission:hover_time'] = t_hf
        outputs['constraints:system:MTOW'] = MTOW_con


class MassConvergenceConstraint(om.ExplicitComponent):
    """
    Mass convergence constraint
    """

    def setup(self):
        self.add_input('data:gearbox:mass', val=0.0, units='kg')
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:frame:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:payload:settings:k_M', val=np.nan, units=None)
        self.add_output('constraints:system:mass_convergence', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mgear = inputs['data:gearbox:mass']  # default value = .0 if use_gearbox = false
        Mmot = inputs['data:motor:mass']
        Mesc = inputs['data:ESC:mass']
        Mbat = inputs['data:battery:mass']
        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['data:payload:mass']
        Mfra = inputs['data:structure:frame:mass']
        Marm = inputs['data:structure:arms:mass']
        k_M = inputs['data:payload:settings:k_M']

        # Missions
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        # Global constraint : mass convergence
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        mass_con = (Mtotal_estimated - Mtotal) / Mtotal  # mass convergence

        outputs['constraints:system:mass_convergence'] = mass_con
