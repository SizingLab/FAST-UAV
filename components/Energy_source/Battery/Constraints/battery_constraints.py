"""
Battery constraints
"""
import openmdao.api as om
import numpy as np

class BatteryConstraints(om.ExplicitComponent):
    """
    Constraints definition of the Battery component
    """

    def setup(self):
        self.add_input('data:battery:performances:voltage', val=np.nan, units='V')
        self.add_input('data:motor:performances:voltage_takeoff', val=np.nan, units='V')
        self.add_input('data:motor:performances:voltage_climb', val=np.nan, units='V')
        self.add_input('data:battery:performances:max_current', val=np.nan, units='A')
        self.add_input('data:motor:performances:current_takeoff', val=np.nan, units='A')
        self.add_input('data:motor:performances:current_climb', val=np.nan, units='A')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:ESC:performances:efficiency', val=np.nan, units=None)
        self.add_output('optimization:constraints:battery:cons_takeoff_voltage', units=None)
        self.add_output('optimization:constraints:battery:cons_climb_voltage', units=None)
        self.add_output('optimization:constraints:battery:cons_takeoff_power', units=None)
        self.add_output('optimization:constraints:battery:cons_climb_power', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_bat = inputs['data:battery:performances:voltage']
        Umot_to = inputs['data:motor:performances:voltage_takeoff']
        Umot_cl = inputs['data:motor:performances:voltage_climb']
        Imax = inputs['data:battery:performances:max_current']
        Imot_to = inputs['data:motor:performances:current_takeoff']
        Imot_cl = inputs['data:motor:performances:current_climb']
        Npro = inputs['data:propeller:prop_number']
        eta_ESC = inputs['data:ESC:performances:efficiency']

        battery_con1 = (V_bat - Umot_to) / V_bat
        battery_con2 = (V_bat - Umot_cl) / V_bat
        battery_con3 = (V_bat * Imax - Umot_to * Imot_to * Npro / eta_ESC) / (V_bat * Imax)
        battery_con4 = (V_bat * Imax - Umot_cl * Imot_cl * Npro / eta_ESC) / (V_bat * Imax)

        outputs['optimization:constraints:battery:cons_takeoff_voltage'] = battery_con1
        outputs['optimization:constraints:battery:cons_climb_voltage'] = battery_con2
        outputs['optimization:constraints:battery:cons_takeoff_power'] = battery_con3
        outputs['optimization:constraints:battery:cons_climb_power'] = battery_con4
