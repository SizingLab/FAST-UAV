"""
Battery performances
"""
import openmdao.api as om
import numpy as np


class BatteryPerfos(om.Group):
    """
    Group containing the performance functions of the battery
    """
    def setup(self):
        self.add_subsystem("takeoff", TakeOff(), promotes=["*"])
        self.add_subsystem("hover", Hover(), promotes=["*"])
        self.add_subsystem("climb", Climb(), promotes=["*"])
        self.add_subsystem("forward", Forward(), promotes=["*"])


class TakeOff(om.ExplicitComponent):
    """
    Computes performances of the battery for takeoff
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:takeoff', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('data:battery:current:takeoff', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_to = inputs['data:motor:power:takeoff']
        eta_ESC = inputs['data:ESC:efficiency']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']

        I_bat_to = (P_el_to * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery

        outputs['data:battery:current:takeoff'] = I_bat_to


class Hover(om.ExplicitComponent):
    """
    Computes performances of the battery for hover
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('data:battery:current:hover', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']

        I_bat_hov = (P_el_hover * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery

        outputs['data:battery:current:hover'] = I_bat_hov


class Climb(om.ExplicitComponent):
    """
    Computes performances of the battery for climb
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:climb', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('data:battery:current:climb', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_cl = inputs['data:motor:power:climb']
        eta_ESC = inputs['data:ESC:efficiency']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']

        I_bat_cl = (P_el_cl * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery

        outputs['data:battery:current:climb'] = I_bat_cl


class Forward(om.ExplicitComponent):
    """
    Computes performances of the battery for forward
    """

    def setup(self):
        self.add_input('data:propeller:number', val=np.nan, units=None)
        self.add_input('data:motor:power:forward', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:payload:power', val=.0, units='W')
        self.add_input('data:avionics:power', val=.0, units='W')
        self.add_output('data:battery:current:forward', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Npro = inputs['data:propeller:number']
        P_el_ff = inputs['data:motor:power:forward']
        eta_ESC = inputs['data:ESC:efficiency']
        V_bat = inputs['data:battery:voltage']
        P_payload = inputs['data:payload:power']
        P_avionics = inputs['data:avionics:power']

        I_bat_ff = (P_el_ff * Npro + P_payload + P_avionics) / eta_ESC / V_bat  # [I] Current of the battery

        outputs['data:battery:current:forward'] = I_bat_ff