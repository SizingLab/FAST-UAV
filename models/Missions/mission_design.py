"""
Design mission
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np


@oad.RegisterOpenMDAOSystem("multirotor.design_mission")
class Mission(om.Group):
    """
    Group containing the design mission parameters
    """
    def setup(self):
        self.add_subsystem("climb_segment", ClimbSegment(), promotes=['*'])
        self.add_subsystem("hover_segment", HoverSegment(), promotes=['*'])
        self.add_subsystem("forward_segment", ForwardSegment(), promotes=['*'])
        self.add_subsystem("mission", MissionComponent(), promotes=['*'])
        self.add_subsystem("constraints", MissionConstraints(), promotes=['*'])


class ClimbSegment(om.ExplicitComponent):
    """
    Climb segment
    """
    def setup(self):
        self.add_input('specifications:climb_height', val=np.nan, units='m')
        self.add_input('specifications:climb_speed', val=np.nan, units='m/s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:climb', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('specifications:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('mission:sizing_mission:climb:duration', units='min')
        self.add_output('mission:sizing_mission:climb:energy:propulsion', units='kJ')
        self.add_output('mission:sizing_mission:climb:energy:payload', units='kJ')
        self.add_output('mission:sizing_mission:climb:energy:avionics', units='kJ')
        self.add_output('mission:sizing_mission:climb:energy', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_cl = inputs['specifications:climb_height']
        Npro = inputs['data:propeller:number']
        P_el_cl = inputs['data:motor:power:climb']
        eta_ESC = inputs['data:ESC:efficiency']
        V_cl = inputs['specifications:climb_speed']
        P_payload = inputs['specifications:payload:power']
        P_avionics = inputs['data:avionics:power']

        t_cl = D_cl / V_cl  # [s]
        E_cl_pro = (P_el_cl * Npro) / eta_ESC * t_cl # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_cl  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_cl  # [J] consumed energy for avionics

        outputs['mission:sizing_mission:climb:duration'] = t_cl / 60  # [min]
        outputs['mission:sizing_mission:climb:energy:propulsion'] = E_cl_pro / 1000  # [kJ]
        outputs['mission:sizing_mission:climb:energy:payload'] = E_payload / 1000  # [kJ]
        outputs['mission:sizing_mission:climb:energy:avionics'] = E_avionics / 1000  # [kJ]
        outputs['mission:sizing_mission:climb:energy'] = (E_cl_pro + E_payload + E_avionics) / 1000  # [kJ]


class HoverSegment(om.ExplicitComponent):
    """
    Hover segment
    """

    def setup(self):
        self.add_input('specifications:hover_duration', val=np.nan, units='s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('specifications:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('mission:sizing_mission:hover:energy:propulsion', units='kJ')
        self.add_output('mission:sizing_mission:hover:energy:payload', units='kJ')
        self.add_output('mission:sizing_mission:hover:energy:avionics', units='kJ')
        self.add_output('mission:sizing_mission:hover:energy', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        t_hov = inputs['specifications:hover_duration']
        Npro = inputs['data:propeller:number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        P_payload = inputs['specifications:payload:power']
        P_avionics = inputs['data:avionics:power']

        E_hover_pro = (P_el_hover * Npro) / eta_ESC * t_hov  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_hov  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_hov  # [J] consumed energy for avionics

        outputs['mission:sizing_mission:hover:energy:propulsion'] = E_hover_pro / 1000  # [kJ]
        outputs['mission:sizing_mission:hover:energy:payload'] = E_payload / 1000  # [kJ]
        outputs['mission:sizing_mission:hover:energy:avionics'] = E_avionics / 1000  # [kJ]
        outputs['mission:sizing_mission:hover:energy'] = (E_hover_pro + E_payload + E_avionics) / 1000  # [kJ]


class ForwardSegment(om.ExplicitComponent):
    """
    Forward flight segment
    """

    def setup(self):
        self.add_input('specifications:range', val=np.nan, units='m')
        self.add_input('mission:sizing_mission:forward:speed', val=np.nan, units='m/s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:forward', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('specifications:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('mission:sizing_mission:forward:duration', units='min')
        self.add_output('mission:sizing_mission:forward:energy:propulsion', units='kJ')
        self.add_output('mission:sizing_mission:forward:energy:payload', units='kJ')
        self.add_output('mission:sizing_mission:forward:energy:avionics', units='kJ')
        self.add_output('mission:sizing_mission:forward:energy', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_ff = inputs['specifications:range']
        Npro = inputs['data:propeller:number']
        P_el_ff = inputs['data:motor:power:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        V_ff = inputs['mission:sizing_mission:forward:speed']
        P_payload = inputs['specifications:payload:power']
        P_avionics = inputs['data:avionics:power']

        t_ff = D_ff / V_ff  # [s]
        E_ff_pro = (P_el_ff * Npro) / eta_ESC * t_ff  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_ff  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_ff  # [J] consumed energy for avionics

        outputs['mission:sizing_mission:forward:duration'] = t_ff / 60  # [min]
        outputs['mission:sizing_mission:forward:energy:propulsion'] = E_ff_pro / 1000  # [kJ]
        outputs['mission:sizing_mission:forward:energy:payload'] = E_payload / 1000  # [kJ]
        outputs['mission:sizing_mission:forward:energy:avionics'] = E_avionics / 1000  # [kJ]
        outputs['mission:sizing_mission:forward:energy'] = (E_ff_pro + E_payload + E_avionics) / 1000  # [kJ]


class MissionComponent(om.ExplicitComponent):
    """
    Overall nominal mission - energy and duration
    """

    def setup(self):
        self.add_input('mission:sizing_mission:hover:energy', val=np.nan, units='kJ')
        self.add_input('mission:sizing_mission:climb:energy', val=np.nan, units='kJ')
        self.add_input('mission:sizing_mission:forward:energy', val=np.nan, units='kJ')
        self.add_input('specifications:hover_duration', val=np.nan, units='min')
        self.add_input('mission:sizing_mission:climb:duration', val=np.nan, units='min')
        self.add_input('mission:sizing_mission:forward:duration', val=np.nan, units='min')
        self.add_output('mission:sizing_mission:energy', units='kJ')
        self.add_output('mission:sizing_mission:duration', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        E_hov = inputs['mission:sizing_mission:hover:energy']
        E_cl = inputs['mission:sizing_mission:climb:energy']
        E_ff = inputs['mission:sizing_mission:forward:energy']
        t_hov = inputs['specifications:hover_duration']
        t_cl = inputs['mission:sizing_mission:climb:duration']
        t_ff = inputs['mission:sizing_mission:forward:duration']

        t_mission = t_hov + t_cl + t_ff
        E_mission = E_hov + E_cl + E_ff

        outputs['mission:sizing_mission:energy'] = E_mission
        outputs['mission:sizing_mission:duration'] = t_mission


class MissionConstraints(om.ExplicitComponent):
    """
    Nominal mission constraints
    """

    def setup(self):
        self.add_input('mission:sizing_mission:energy', val=np.nan, units='kJ')
        self.add_input('data:battery:energy', val=np.nan, units='kJ')
        self.add_input('data:battery:DoD:max', val=.8, units=None)
        self.add_output('mission:sizing_mission:constraints:energy', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        E_mission = inputs['mission:sizing_mission:energy']
        E_bat = inputs['data:battery:energy']
        C_ratio = inputs['data:battery:DoD:max']

        energy_con = (E_bat * C_ratio - E_mission) / (E_bat * C_ratio)

        outputs['mission:sizing_mission:constraints:energy'] = energy_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_mission = inputs['mission:sizing_mission:energy']
        E_bat = inputs['data:battery:energy']
        C_ratio = inputs['data:battery:DoD:max']

        partials[
            'mission:sizing_mission:constraints:energy',
            'mission:sizing_mission:energy',
        ] = -1.0 / (E_bat * C_ratio)
        partials[
            'mission:sizing_mission:constraints:energy',
            'data:battery:energy',
        ] = E_mission / (E_bat**2 * C_ratio)
        partials[
            'mission:sizing_mission:constraints:energy',
            'data:battery:DoD:max',
        ] = E_mission / (E_bat * C_ratio**2)


