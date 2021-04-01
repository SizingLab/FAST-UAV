"""
Battery Characteristics
"""
import openmdao.api as om
import numpy as np


class ComputeBatteryCharacteristics(om.ExplicitComponent):
    """
    Characteristics calculation of Battery (sized from hover)
    """

    def setup(self):
        self.add_input('data:battery:reference:mass', val=np.nan, units='kg')
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:voltage', val=np.nan, units='V')
        self.add_input('data:battery:reference:current:max', val=np.nan, units='A')
        self.add_input('data:battery:voltage:guess', val=np.nan, units='V')
        self.add_input('data:battery:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_output('data:battery:cell:number:estimated', units=None)
        self.add_output('data:battery:voltage:estimated', units='V')
        self.add_output('data:battery:capacity:estimated', units='A*s')
        self.add_output('data:battery:energy:estimated', units='kJ')
        self.add_output('data:battery:current:max:estimated', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Battery selection & scaling laws sized from hover
        Mbat_ref = inputs['data:battery:reference:mass']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Imax_ref = inputs['data:battery:reference:current:max']
        V_bat_guess = inputs['data:battery:voltage:guess']
        Mbat = inputs['data:battery:mass:estimated']
        Vcell = inputs['data:battery:cell:voltage:estimated']

        Ncel = np.ceil(V_bat_guess / Vcell)  # [-] Cell number, round (up value)
        V_bat = Vcell * Ncel  # [V] Battery voltage
        C_bat = Mbat / Mbat_ref * Cbat_ref / V_bat * Vbat_ref  # [A.s] Capacity  of the battery
        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery
        # E_bat = E_bat_ref * Mbat / Mbat_ref * (1 - C_ratio)
        E_bat = C_bat * V_bat / 1000  # [kJ] Stored energy

        outputs['data:battery:cell:number:estimated'] = Ncel
        outputs['data:battery:voltage:estimated'] = V_bat
        outputs['data:battery:capacity:estimated'] = C_bat
        outputs['data:battery:energy:estimated'] = E_bat
        outputs['data:battery:current:max:estimated'] = Imax
