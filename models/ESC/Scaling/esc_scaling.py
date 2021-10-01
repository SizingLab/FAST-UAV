"""
ESC scaling
"""
import openmdao.api as om
import numpy as np


class ESCScaling(om.Group):
    """
    Group containing the scaling functions of the ESC (sized from max speed)
    """
    def setup(self):
        self.add_subsystem("power", Power(), promotes=["*"])
        self.add_subsystem("voltage", Voltage(), promotes=["*"])
        self.add_subsystem("weight", Weight(), promotes=["*"])


class Power(om.ExplicitComponent):
    """
    Computes ESC power
    """

    def setup(self):
        self.add_input('data:ESC:settings:power:k', val=np.nan, units=None)
        self.add_input('data:motor:power:takeoff', val=np.nan, units='W')
        self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_output('data:ESC:power:max:estimated', units='W')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        k_ESC = inputs['data:ESC:settings:power:k']
        P_el_to = inputs['data:motor:power:takeoff']
        V_bat = inputs['data:battery:voltage']
        Umot_to = inputs['data:motor:voltage:takeoff']

        P_esc = k_ESC * (P_el_to * V_bat / Umot_to)  # [W] power electronic power max thrust

        outputs['data:ESC:power:max:estimated'] = P_esc


class Voltage(om.ExplicitComponent):
    """
    Computes ESC voltage
    """

    def setup(self):
        self.add_input('data:ESC:reference:power', val=np.nan, units='W')
        self.add_input('data:ESC:reference:voltage', val=np.nan, units='V')
        self.add_input('data:ESC:power:max:estimated', val=np.nan, units='W')
        self.add_output('data:ESC:voltage:estimated', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Pesc_ref = inputs['data:ESC:reference:power']
        Vesc_ref = inputs['data:ESC:reference:voltage']
        P_esc = inputs['data:ESC:power:max:estimated']

        V_esc = Vesc_ref * (P_esc / Pesc_ref) ** (1 / 3)  # [V] ESC voltage
        # V_esc = 1.84 * (P_esc) ** (0.36)  # [V] ESC voltage

        outputs['data:ESC:voltage:estimated'] = V_esc


class Weight(om.ExplicitComponent):
    """
    Computes ESC weight
    """

    def setup(self):
        self.add_input('data:ESC:reference:mass', val=np.nan, units='kg')
        self.add_input('data:ESC:reference:power', val=np.nan, units='W')
        self.add_input('data:ESC:power:max:estimated', val=np.nan, units='W')
        self.add_output('data:ESC:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mesc_ref = inputs['data:ESC:reference:mass']
        Pesc_ref = inputs['data:ESC:reference:power']
        P_esc = inputs['data:ESC:power:max:estimated']

        M_esc = Mesc_ref * (P_esc / Pesc_ref)  # [kg] Mass ESC

        outputs['data:ESC:mass:estimated'] = M_esc