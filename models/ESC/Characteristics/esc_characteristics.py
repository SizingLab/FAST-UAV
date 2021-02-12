"""
ESC characteristics
"""
import openmdao.api as om
import numpy as np

class ComputeESCCharacteristics(om.ExplicitComponent):
    """
    Characteristics calculation of ESC (sized from max speed)
    """

    def setup(self):
        self.add_input('data:ESC:reference:power', val=np.nan, units='W')
        self.add_input('data:ESC:reference:voltage', val=np.nan, units='V')
        self.add_input('data:ESC:settings:k_ESC', val=np.nan, units=None)
        self.add_input('data:motor:power:takeoff', val=np.nan, units='W')
        self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_output('data:ESC:power:max', units='W')
        self.add_output('data:ESC:voltage', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # ESC sized from max speed
        Pesc_ref = inputs['data:ESC:reference:power']
        Vesc_ref = inputs['data:ESC:reference:voltage']
        k_ESC = inputs['data:ESC:settings:k_ESC']
        P_el_to = inputs['data:motor:power:takeoff']
        V_bat = inputs['data:battery:voltage']
        Umot_to = inputs['data:motor:voltage:takeoff']


        P_esc = k_ESC * (P_el_to * V_bat / Umot_to)  # [W] power electronic power max thrust
        V_esc = Vesc_ref * (P_esc / Pesc_ref) ** (1 / 3)  # [V] ESC voltage

        outputs['data:ESC:power:max'] = P_esc
        outputs['data:ESC:voltage'] = V_esc