"""
ESC constraints
"""
import openmdao.api as om
import numpy as np

class ESCConstraints(om.ExplicitComponent):
    """
    Constraints definition of the ESC component
    """

    def setup(self):
        self.add_input('data:ESC:performances:power_max_thrust', val=np.nan, units='W')
        self.add_input('data:ESC:performances:power_max_climb', val=np.nan, units='W')
        self.add_input('data:ESC:performances:voltage_ESC', val=np.nan, units='V')
        self.add_input('data:battery:performances:voltage', val=np.nan, units='V')
        self.add_output('optimization:constraints:ESC:cons_climb_power', units=None)
        self.add_output('optimization:constraints:ESC:cons_bat_voltage', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        P_esc = inputs['data:ESC:performances:power_max_thrust']
        P_esc_cl = inputs['data:ESC:performances:power_max_climb']
        Vesc = inputs['data:ESC:performances:voltage_ESC']
        V_bat = inputs['data:battery:performances:voltage']

        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (V_bat - Vesc) / V_bat

        outputs['optimization:constraints:ESC:cons_climb_power'] = ESC_con1
        outputs['optimization:constraints:ESC:cons_bat_voltage'] = ESC_con2
