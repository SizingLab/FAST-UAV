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
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:battery:current:max', val=np.nan, units='A')

        self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:motor:voltage:climb', val=np.nan, units='V')
        self.add_input('data:motor:current:takeoff', val=np.nan, units='A')
        self.add_input('data:motor:current:climb', val=np.nan, units='A')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('constraints:battery:voltage:takeoff', units=None)
        self.add_output('constraints:battery:voltage:climb', units=None)
        self.add_output('constraints:battery:power:takeoff', units=None)
        self.add_output('constraints:battery:power:climb', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_bat = inputs['data:battery:voltage']
        Imax = inputs['data:battery:current:max']

        Umot_to = inputs['data:motor:voltage:takeoff']
        Umot_cl = inputs['data:motor:voltage:climb']
        Imot_to = inputs['data:motor:current:takeoff']
        Imot_cl = inputs['data:motor:current:climb']
        Npro = inputs['data:propeller:prop_number']
        eta_ESC = inputs['data:ESC:efficiency']

        battery_con1 = (V_bat - Umot_to) / V_bat
        battery_con2 = (V_bat - Umot_cl) / V_bat
        battery_con3 = (V_bat * Imax - Umot_to * Imot_to * Npro / eta_ESC) / (V_bat * Imax)
        battery_con4 = (V_bat * Imax - Umot_cl * Imot_cl * Npro / eta_ESC) / (V_bat * Imax)

        outputs['constraints:battery:voltage:takeoff'] = battery_con1
        outputs['constraints:battery:voltage:climb'] = battery_con2
        outputs['constraints:battery:power:takeoff'] = battery_con3
        outputs['constraints:battery:power:climb'] = battery_con4

