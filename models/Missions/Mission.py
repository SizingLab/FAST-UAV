"""
Mission definition
"""
import openmdao.api as om
import numpy as np


class Mission(om.Group):
    """
    Group containing the mission parameters
    """
    def setup(self):
        self.add_subsystem("hover_flight", HoverFlight(), promotes=['*'])
        self.add_subsystem("energy", MissionEnergy(), promotes=['*'])
        self.add_subsystem("mission_constraints", MissionConstraints(), promotes=['*'])


class HoverFlight(om.ExplicitComponent):
    """
    Hover flight autonomy
    """

    def setup(self):
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:discharge_limit', val=0.8, units=None)
        self.add_output('data:mission:hover:autonomy', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']
        Npro = inputs['data:propeller:prop_number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        C_ratio = inputs['data:battery:discharge_limit']

        I_bat = (P_el_hover * Npro) / eta_ESC / V_bat  # [I] Current of the battery
        t_hf = C_ratio * C_bat / I_bat / 60  # [min] Hover time

        outputs['data:mission:hover:autonomy'] = t_hf


class MissionConstraints(om.ExplicitComponent):
    """
    Mission constraints
    """

    def setup(self):
        self.add_input('specifications:hover:autonomy', val=np.nan, units='min')
        self.add_input('data:mission:hover:autonomy', val=np.nan, units='min')
        self.add_output('data:mission:constraints:hover:autonomy', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        t_h = inputs['specifications:hover:autonomy']
        t_hf = inputs['data:mission:hover:autonomy']

        # Constraints
        autonomy_con = (t_hf - t_h) / t_hf  # Min. hover flight autonomy

        outputs['data:mission:constraints:hover:autonomy'] = autonomy_con


class MissionEnergy(om.ExplicitComponent):
    """
        Energy required for the mission
        """

    def setup(self):
        self.add_input('data:mission:hover:autonomy', val=np.nan, units='min')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:payload:power', val=np.nan, units='W')
        self.add_input('data:avionics:power', val=np.nan, units='W')
        self.add_output('data:mission:energy:propulsion:hover', units='J')
        self.add_output('data:mission:energy:propulsion:climb', units='J')
        self.add_output('data:mission:energy:propulsion:forward', units='J')
        self.add_output('data:mission:energy:propulsion', units='J')
        self.add_output('data:mission:energy:payload', units='J')
        self.add_output('data:mission:energy:avionics', units='J')
        self.add_output('data:mission:energy', units='J')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:prop_number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        t_hf = inputs['data:mission:hover:autonomy'] * 60
        P_el_payload = inputs['data:payload:power']
        P_el_avionics = inputs['data:avionics:power']

        t_f = t_hf # [s] TODO: replace hover time by mission time (hover + climb + fwd...)

        # Propulsion energy
        E_hover = (P_el_hover * Npro) / eta_ESC * t_hf # [J]
        # TODO: add climb energy + forward energy + ...
        E_climb = 0 # [J]
        E_fwd   = 0 # [J]
        E_propulsion = E_hover + E_climb + E_fwd # [J]

        # Payload and avionics energy
        E_payload  = P_el_payload * t_f  # [J]
        E_avionics = P_el_avionics * t_f  # [J]

        # Total energy
        E_mission = E_propulsion + E_payload + E_avionics  # [J]

        outputs['data:mission:energy:propulsion:hover'] = E_hover
        outputs['data:mission:energy:propulsion:climb'] = E_climb
        outputs['data:mission:energy:propulsion:forward'] = E_fwd
        outputs['data:mission:energy:propulsion'] = E_propulsion
        outputs['data:mission:energy:payload'] = E_payload
        outputs['data:mission:energy:avionics'] = E_avionics
        outputs['data:mission:energy'] = E_mission