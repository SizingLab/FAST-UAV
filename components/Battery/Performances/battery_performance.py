"""
Battery performances
"""
import openmdao.api as om
import numpy as np

class ComputeBatteryPerfo(om.ExplicitComponent):
    """
    Performances calculation of Battery (sized from hover)
    """

    def setup(self):
        self.add_input('data:battery:reference:mass_ref', val=np.nan, units='kg')
        self.add_input('data:battery:reference:capacity_ref', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:voltage_ref', val=np.nan, units='V')
        self.add_input('data:battery:reference:max_current_ref', val=np.nan, units='A')
        self.add_input('data:battery:performances:voltage_estimation', val=np.nan, units='V')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:motor:performances:elec_power_hover', val=np.nan, units='W')
        self.add_input('data:propeller:prop_number', val=np.nan)
        self.add_input('data:ESC:performances:efficiency', val=np.nan)
        self.add_output('data:battery:cell_number')
        self.add_output('data:battery:performances:voltage', units='V')
        self.add_output('data:battery:performances:capacity', units='A*s')
        self.add_output('data:battery:performances:current', units='A')
        self.add_output('data:battery:performances:max_current', units='A')
        self.add_output('optimization:objectives:hover_time', units='min')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Battery selection & scaling laws sized from hover
        Mbat_ref = inputs['data:battery:reference:mass_ref']
        Cbat_ref = inputs['data:battery:reference:capacity_ref']
        Vbat_ref = inputs['data:battery:reference:voltage_ref']
        Imax_ref = inputs['data:battery:reference:max_current_ref']
        V_bat_est = inputs['data:battery:performances:voltage_estimation']
        Mbat = inputs['data:battery:mass']
        P_el_hover = inputs['data:motor:performances:elec_power_hover']
        Npro = inputs['data:propeller:prop_number']
        eta_ESC = inputs['data:ESC:performances:efficiency']

        Ncel = np.ceil(V_bat_est / 3.7)  # [-] Cell number, round (up value)
        V_bat = 3.7 * Ncel  # [V] Battery voltage

        # Hover --> autonomy
        C_bat = Mbat / Mbat_ref * Cbat_ref / V_bat * Vbat_ref  # [A.s] Capacity  of the battery
        I_bat = (P_el_hover * Npro) / eta_ESC / V_bat  # [I] Current of the battery
        t_hf = .8 * C_bat / I_bat / 60  # [min] Hover time
        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery

        outputs['data:battery:cell_number'] = Ncel
        outputs['data:battery:performances:voltage'] = V_bat
        outputs['data:battery:performances:capacity'] = C_bat
        outputs['data:battery:performances:current'] = I_bat
        outputs['data:battery:performances:max_current'] = Imax
        outputs['optimization:objectives:hover_time'] = t_hf