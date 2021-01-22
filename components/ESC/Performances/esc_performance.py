"""
ESC performances
"""
import openmdao.api as om
import numpy as np

class ComputeESCPerfo(om.ExplicitComponent):
    """
    Performances calculation of ESC (sized from max speed)
    """

    def setup(self):
        self.add_input('data:ESC:reference:power_ref', val=np.nan, units='W')
        self.add_input('data:ESC:reference:voltage_ref', val=np.nan, units='V')
        self.add_input('optimization:ESC:k_ESC', val=np.nan)
        self.add_input('data:motor:performances:elec_power_takeoff', val=np.nan, units='W')
        self.add_input('data:motor:performances:elec_power_climb', val=np.nan, units='W')
        self.add_input('data:motor:performances:voltage_takeoff', val=np.nan, units='V')
        self.add_input('data:motor:performances:voltage_climb', val=np.nan, units='V')
        self.add_input('data:battery:performances:voltage', val=np.nan, units='V')
        self.add_output('data:ESC:performances:power_max_thrust', units='W')
        self.add_output('data:ESC:performances:power_max_climb', units='W')
        self.add_output('data:ESC:performances:voltage_ESC', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # ESC sized from max speed
        Pesc_ref = inputs['data:ESC:reference:power_ref']
        Vesc_ref = inputs['data:ESC:reference:voltage_ref']
        k_ESC = inputs['optimization:ESC:k_ESC']
        P_el_to = inputs['data:motor:performances:elec_power_takeoff']
        P_el_cl = inputs['data:motor:performances:elec_power_climb']
        V_bat = inputs['data:battery:performances:voltage']
        Umot_to = inputs['data:motor:performances:voltage_takeoff']
        Umot_cl = inputs['data:motor:performances:voltage_climb']

        P_esc = k_ESC * (P_el_to * V_bat / Umot_to)  # [W] power electronic power max thrust
        P_esc_cl = P_el_cl * V_bat / Umot_cl  # [W] power electronic power max climb
        Vesc = Vesc_ref * (P_esc / Pesc_ref) ** (1 / 3)  # [V] ESC voltage

        outputs['data:ESC:performances:power_max_thrust'] = P_esc
        outputs['data:ESC:performances:power_max_climb'] = P_esc_cl
        outputs['data:ESC:performances:voltage_ESC'] = Vesc