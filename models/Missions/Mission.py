"""
Mission definition
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np

@oad.RegisterOpenMDAOSystem("multirotor.missions")
class Mission(om.Group):
    """
    Group containing the mission parameters
    """
    def setup(self):
        self.add_subsystem("hover", HoverFlight(), promotes=['*'])
        self.add_subsystem("climb", ClimbFlight(), promotes=['*'])
        self.add_subsystem("forward", ForwardFlight(), promotes=['*'])
        self.add_subsystem("payload", PayloadMission(), promotes=['*'])
        self.add_subsystem("avionics", AvionicsMission(), promotes=['*'])
        self.add_subsystem("total", TotalMission(), promotes=['*'])
        self.add_subsystem("constraints", MissionConstraints(), promotes=['*'])


class HoverFlight(om.ExplicitComponent):
    """
    Hover flight parameters
    """

    def setup(self):
        self.add_input('specifications:time:hover', val=np.nan, units='min')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:mission:energy:hover', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        t_hov = inputs['specifications:time:hover']
        Npro = inputs['data:propeller:number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']

        E_hover = (P_el_hover * Npro) / eta_ESC * (60 * t_hov)  # [J]

        outputs['data:mission:energy:hover'] = E_hover / 1000  # [kJ]


class ClimbFlight(om.ExplicitComponent):
    """
    Climb flight parameters
    """
    def setup(self):
        self.add_input('specifications:distance:climb', val=np.nan, units='m')
        self.add_input('specifications:speed:climb', val=np.nan, units='m/s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:climb', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:mission:energy:climb', units='kJ')
        self.add_output('data:mission:time:climb', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_cl = inputs['specifications:distance:climb']
        Npro = inputs['data:propeller:number']
        P_el_cl = inputs['data:motor:power:climb']
        eta_ESC = inputs['data:ESC:efficiency']
        V_cl = inputs['specifications:speed:climb']

        t_cl = D_cl / V_cl # [s]
        E_cl = (P_el_cl * Npro) / eta_ESC * t_cl # [J]

        outputs['data:mission:time:climb'] = t_cl / 60  # [min]
        outputs['data:mission:energy:climb'] = E_cl / 1000  # [kJ]


class ForwardFlight(om.ExplicitComponent):
    """
    Forward flight parameters
    """

    def setup(self):
        self.add_input('specifications:distance:forward', val=np.nan, units='m')
        self.add_input('data:mission:speed:forward', val=np.nan, units='m/s')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:forward', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:mission:energy:forward', units='kJ')
        self.add_output('data:mission:time:forward', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        D_ff = inputs['specifications:distance:forward']
        Npro = inputs['data:propeller:number']
        P_el_ff = inputs['data:motor:power:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        V_ff = inputs['data:mission:speed:forward']

        t_ff = D_ff / V_ff  # [s] travel time
        E_ff = (P_el_ff * Npro) / eta_ESC * t_ff  # [J] consumed energy

        outputs['data:mission:time:forward'] = t_ff / 60  # [min]
        outputs['data:mission:energy:forward'] = E_ff / 1000  # [kJ]


class PayloadMission(om.ExplicitComponent):
    """
    Payload energy consumption
    """

    def setup(self):
        self.add_input('specifications:time:hover', val=np.nan, units='min')
        self.add_input('data:mission:time:climb', val=np.nan, units='min')
        self.add_input('data:mission:time:forward', val=np.nan, units='min')
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:mission:energy:payload', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        t_hov = inputs['specifications:time:hover']
        t_cl = inputs['data:mission:time:climb']
        t_ff = inputs['data:mission:time:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        P_payload = inputs['data:payload:power']

        E_payload = P_payload / eta_ESC * (t_hov + t_cl + t_ff) * 60 / 1000

        outputs['data:mission:energy:payload'] = E_payload


class AvionicsMission(om.ExplicitComponent):
    """
    Avionics energy consumption
    """

    def setup(self):
        self.add_input('specifications:time:hover', val=np.nan, units='min')
        self.add_input('data:mission:time:climb', val=np.nan, units='min')
        self.add_input('data:mission:time:forward', val=np.nan, units='min')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:mission:energy:avionics', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        t_hov = inputs['specifications:time:hover']
        t_cl = inputs['data:mission:time:climb']
        t_ff = inputs['data:mission:time:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        P_avionics = inputs['data:avionics:power']

        E_avionics = P_avionics / eta_ESC * (t_hov + t_cl + t_ff) * 60 / 1000

        outputs['data:mission:energy:avionics'] = E_avionics


class TotalMission(om.ExplicitComponent):
    """
    Total Mission energy and duration
    """

    def setup(self):
        self.add_input('data:mission:energy:hover', val=np.nan, units='kJ')
        self.add_input('data:mission:energy:climb', val=np.nan, units='kJ')
        self.add_input('data:mission:energy:forward', val=np.nan, units='kJ')
        self.add_input('data:mission:energy:payload', val=.0, units='kJ')
        self.add_input('data:mission:energy:avionics', val=.0, units='kJ')
        self.add_input('specifications:time:hover', val=np.nan, units='min')
        self.add_input('data:mission:time:climb', val=np.nan, units='min')
        self.add_input('data:mission:time:forward', val=np.nan, units='min')
        self.add_output('data:mission:energy', units='kJ')
        self.add_output('data:mission:time', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        E_hov = inputs['data:mission:energy:hover']
        E_cl = inputs['data:mission:energy:climb']
        E_ff = inputs['data:mission:energy:forward']
        E_payload = inputs['data:mission:energy:payload']
        E_avionics = inputs['data:mission:energy:avionics']
        t_hov = inputs['specifications:time:hover']
        t_cl = inputs['data:mission:time:climb']
        t_ff = inputs['data:mission:time:forward']

        t_mission = t_hov + t_cl + t_ff
        E_mission = E_hov + E_cl + E_ff + E_payload + E_avionics

        outputs['data:mission:energy'] = E_mission
        outputs['data:mission:time'] = t_mission


class MissionConstraints(om.ExplicitComponent):
    """
    Mission constraints
    """

    def setup(self):
        self.add_input('data:mission:energy', val=np.nan, units='kJ')
        self.add_input('data:battery:energy', val=np.nan, units='kJ')
        self.add_input('data:battery:discharge_limit', val=.8, units=None)
        self.add_output('data:battery:constraints:energy', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        E_mission = inputs['data:mission:energy']
        E_bat = inputs['data:battery:energy']
        C_ratio = inputs['data:battery:discharge_limit']

        energy_con = (E_bat * C_ratio - E_mission) / (E_bat * C_ratio)

        outputs['data:battery:constraints:energy'] = energy_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_mission = inputs['data:mission:energy']
        E_bat = inputs['data:battery:energy']
        C_ratio = inputs['data:battery:discharge_limit']
        partials[
            'data:battery:constraints:energy',
            'data:battery:energy',
        ] = E_mission / C_ratio / E_bat ** 2
        partials[
            'data:battery:constraints:energy',
            'data:battery:discharge_limit',
        ] = E_mission / C_ratio ** 2 / E_bat
        partials[
            'data:battery:constraints:energy',
            'data:mission:energy',
        ] = - 1.0 / (E_bat * C_ratio)



