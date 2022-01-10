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
        self.add_input('data:ESC:efficiency:estimated', val=np.nan, units=None)
        self.add_output('data:battery:constraints:voltage:takeoff', units=None)
        self.add_output('data:battery:constraints:voltage:climb', units=None)
        self.add_output('data:battery:constraints:voltage:forward', units=None)
        self.add_output('data:battery:constraints:power:takeoff', units=None)
        self.add_output('data:battery:constraints:power:climb', units=None)
        self.add_output('data:battery:constraints:power:forward', units=None)

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

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
        eta_ESC = inputs['data:ESC:efficiency:estimated']  # TODO: replace by 'real' efficiency (ESC catalogue output, but be careful to algebraic loops...)

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

    def compute_partials(self, inputs, J, discrete_inputs=None):
        V_bat = inputs["data:battery:voltage"]
        Umot_to = inputs["data:motor:voltage:takeoff"]
        Umot_cl = inputs["data:motor:voltage:climb"]
        Umot_ff = inputs["data:motor:voltage:forward"]
        Imax = inputs["data:battery:current:max"]
        Imot_to = inputs["data:motor:current:takeoff"]
        Npro = inputs["data:propeller:number"]
        eta_ESC = inputs["data:ESC:efficiency:estimated"]
        Imot_cl = inputs["data:motor:current:climb"]
        Imot_ff = inputs["data:motor:current:forward"]

        J["data:battery:constraints:voltage:takeoff", "data:battery:voltage"] = (
                1 / V_bat - (-Umot_to + V_bat) / V_bat ** 2
        )
        J["data:battery:constraints:voltage:takeoff", "data:motor:voltage:takeoff"] = (
                -1 / V_bat
        )

        J["data:battery:constraints:voltage:climb", "data:battery:voltage"] = (
                1 / V_bat - (-Umot_cl + V_bat) / V_bat ** 2
        )
        J["data:battery:constraints:voltage:climb", "data:motor:voltage:climb"] = (
                -1 / V_bat
        )

        J["data:battery:constraints:voltage:forward", "data:battery:voltage"] = (
                1 / V_bat - (-Umot_ff + V_bat) / V_bat ** 2
        )
        J["data:battery:constraints:voltage:forward", "data:motor:voltage:forward"] = (
                -1 / V_bat
        )

        J[
            "data:battery:constraints:power:takeoff", "data:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_to * Npro * Umot_to / eta_ESC) / (
                Imax * V_bat ** 2
        )
        J[
            "data:battery:constraints:power:takeoff", "data:battery:current:max"
        ] = 1 / Imax - (Imax * V_bat - Imot_to * Npro * Umot_to / eta_ESC) / (
                Imax ** 2 * V_bat
        )
        J["data:battery:constraints:power:takeoff", "data:motor:voltage:takeoff"] = (
                -Imot_to * Npro / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:takeoff", "data:motor:current:takeoff"] = (
                -Npro * Umot_to / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:takeoff", "data:propeller:number"] = (
                -Imot_to * Umot_to / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:takeoff", "data:ESC:efficiency:estimated"] = (
                Imot_to * Npro * Umot_to / (Imax * V_bat * eta_ESC ** 2)
        )

        J[
            "data:battery:constraints:power:climb", "data:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_cl * Npro * Umot_cl / eta_ESC) / (
                Imax * V_bat ** 2
        )
        J[
            "data:battery:constraints:power:climb", "data:battery:current:max"
        ] = 1 / Imax - (Imax * V_bat - Imot_cl * Npro * Umot_cl / eta_ESC) / (
                Imax ** 2 * V_bat
        )
        J["data:battery:constraints:power:climb", "data:motor:voltage:climb"] = (
                -Imot_cl * Npro / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:climb", "data:motor:current:climb"] = (
                -Npro * Umot_cl / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:climb", "data:propeller:number"] = (
                -Imot_cl * Umot_cl / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:climb", "data:ESC:efficiency:estimated"] = (
                Imot_cl * Npro * Umot_cl / (Imax * V_bat * eta_ESC ** 2)
        )

        J[
            "data:battery:constraints:power:forward", "data:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_ff * Npro * Umot_ff / eta_ESC) / (
                Imax * V_bat ** 2
        )
        J[
            "data:battery:constraints:power:forward", "data:battery:current:max"
        ] = 1 / Imax - (Imax * V_bat - Imot_ff * Npro * Umot_ff / eta_ESC) / (
                Imax ** 2 * V_bat
        )
        J["data:battery:constraints:power:forward", "data:motor:voltage:forward"] = (
                -Imot_ff * Npro / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:forward", "data:motor:current:forward"] = (
                -Npro * Umot_ff / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:forward", "data:propeller:number"] = (
                -Imot_ff * Umot_ff / (Imax * V_bat * eta_ESC)
        )
        J["data:battery:constraints:power:forward", "data:ESC:efficiency:estimated"] = (
                Imot_ff * Npro * Umot_ff / (Imax * V_bat * eta_ESC ** 2)
        )