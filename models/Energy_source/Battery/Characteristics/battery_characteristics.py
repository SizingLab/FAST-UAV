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
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_output('data:battery:cell_number', units=None)
        self.add_output('data:battery:voltage', units='V')
        self.add_output('data:battery:capacity', units='A*s')
        self.add_output('data:battery:current:max', units='A')

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
        Mbat = inputs['data:battery:mass']

        Ncel = np.ceil(V_bat_guess / 3.7)  # [-] Cell number, round (up value)
        V_bat = 3.7 * Ncel  # [V] Battery voltage

        # Hover --> autonomy
        C_bat = Mbat / Mbat_ref * Cbat_ref / V_bat * Vbat_ref  # [A.s] Capacity  of the battery
        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery

        outputs['data:battery:cell_number'] = Ncel
        outputs['data:battery:voltage'] = V_bat
        outputs['data:battery:capacity'] = C_bat
        outputs['data:battery:current:max'] = Imax