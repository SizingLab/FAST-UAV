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
        self.add_input('data:motor:voltage:forward', val=np.nan, units='V')
        self.add_input('data:motor:current:takeoff', val=np.nan, units='A')
        self.add_input('data:motor:current:climb', val=np.nan, units='A')
        self.add_input('data:motor:current:forward', val=np.nan, units='A')
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:battery:constraints:voltage:takeoff', units=None)
        self.add_output('data:battery:constraints:voltage:climb', units=None)
        self.add_output('data:battery:constraints:voltage:forward', units=None)
        self.add_output('data:battery:constraints:power:takeoff', units=None)
        self.add_output('data:battery:constraints:power:climb', units=None)
        self.add_output('data:battery:constraints:power:forward', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_bat = inputs['data:battery:voltage']
        Imax = inputs['data:battery:current:max']
        Umot_to = inputs['data:motor:voltage:takeoff']
        Umot_cl = inputs['data:motor:voltage:climb']
        Umot_ff = inputs['data:motor:voltage:forward']
        Imot_to = inputs['data:motor:current:takeoff']
        Imot_cl = inputs['data:motor:current:climb']
        Imot_ff = inputs['data:motor:current:forward']
        Npro = inputs['data:propeller:number']
        eta_ESC = inputs['data:ESC:efficiency']

        battery_con1 = (V_bat - Umot_to) / V_bat
        battery_con2 = (V_bat - Umot_cl) / V_bat
        battery_con3 = (V_bat - Umot_ff) / V_bat
        battery_con4 = (V_bat * Imax - Umot_to * Imot_to * Npro / eta_ESC) / (V_bat * Imax)
        battery_con5 = (V_bat * Imax - Umot_cl * Imot_cl * Npro / eta_ESC) / (V_bat * Imax)
        battery_con6 = (V_bat * Imax - Umot_ff * Imot_ff * Npro / eta_ESC) / (V_bat * Imax)

        outputs['data:battery:constraints:voltage:takeoff'] = battery_con1
        outputs['data:battery:constraints:voltage:climb'] = battery_con2
        outputs['data:battery:constraints:voltage:forward'] = battery_con3
        outputs['data:battery:constraints:power:takeoff'] = battery_con4
        outputs['data:battery:constraints:power:climb'] = battery_con5
        outputs['data:battery:constraints:power:forward'] = battery_con6

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # TODO: add partials for power constraints
        V_bat = inputs['data:battery:voltage']
        Umot_to = inputs['data:motor:voltage:takeoff']
        Umot_cl = inputs['data:motor:voltage:climb']
        Umot_ff = inputs['data:motor:voltage:forward']

        partials[
            'data:battery:constraints:voltage:takeoff',
            'data:battery:voltage',
        ] = Umot_to / V_bat**2
        partials[
            'data:battery:constraints:voltage:takeoff',
            'data:motor:voltage:takeoff',
        ] = - 1.0 / V_bat

        partials[
            'data:battery:constraints:voltage:climb',
            'data:battery:voltage',
        ] = Umot_cl / V_bat**2
        partials[
            'data:battery:constraints:voltage:climb',
            'data:motor:voltage:climb',
        ] = - 1.0 / V_bat

        partials[
            'data:battery:constraints:voltage:forward',
            'data:battery:voltage',
        ] = Umot_ff / V_bat ** 2
        partials[
            'data:battery:constraints:voltage:forward',
            'data:motor:voltage:forward',
        ] = - 1.0 / V_bat