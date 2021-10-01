"""
Battery Scaling
"""
import openmdao.api as om
import numpy as np


class BatteryScaling(om.Group):
    """
    Group containing the scaling functions of the battery
    """
    def setup(self):
        self.add_subsystem("weight", Weight(), promotes=["*"])
        self.add_subsystem("cell_number", CellNumber(), promotes=["*"])
        self.add_subsystem("voltage", Voltage(), promotes=["*"])
        self.add_subsystem("capacity", Capacity(), promotes=["*"])
        self.add_subsystem("max_current", MaxCurrent(), promotes=["*"])
        self.add_subsystem("energy", Energy(), promotes=["*"])
        self.add_subsystem("geometry", Geometry(), promotes=["*"])


class Weight(om.ExplicitComponent):
    """
    Computes battery weight
    """

    def setup(self):
        self.add_input('specifications:payload:mass:max', val=np.nan, units='kg')
        self.add_input('data:battery:settings:mass:k', val=np.nan)
        self.add_output('data:battery:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        M_load = inputs['specifications:payload:mass:max']
        k_Mb = inputs['data:battery:settings:mass:k']

        Mbat = k_Mb * M_load  # Battery mass (estimated)

        outputs['data:battery:mass:estimated'] = Mbat


class Geometry(om.ExplicitComponent):
    """
    Computes battery geometry
    """

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:voltage', val=np.nan, units='V')
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:volume', val=np.nan, units='cm**3')
        self.add_output('data:battery:volume:estimated', units='cm**3')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_bat = inputs['data:battery:capacity:estimated']
        V_bat = inputs['data:battery:voltage:estimated']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Volbat_ref = inputs['data:battery:reference:volume']

        Vol_bat = Volbat_ref * (C_bat * V_bat / (Cbat_ref * Vbat_ref))  # [cm**3] Volume of the battery (estimated)

        outputs['data:battery:volume:estimated'] = Vol_bat


class CellNumber(om.ExplicitComponent):
    """
    Computes the number of cells of the battery
    """

    def setup(self):
        # self.add_input('data:battery:settings:voltage:k2', val=np.nan, units=None)
        # self.add_input('data:battery:voltage:guess', val=np.nan, units='V')
        self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_output('data:battery:cell:number:estimated', units=None)
        self.add_output('data:battery:cell:number:series:estimated', units=None)
        self.add_output('data:battery:cell:number:parallel:estimated', units=None)

    def setup_partials(self):
        # Approximate derivatives provided by user as analytic expressions to avoid issues with ceil function.
        self.declare_partials('*', '*', method='exact')
        # self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # V_bat_guess = inputs['data:battery:voltage:guess']
        # k_vb2 = inputs['data:battery:settings:voltage:k2']
        V_cell = inputs['data:battery:cell:voltage:estimated']
        Umot_to = inputs['data:motor:voltage:takeoff']

        # N_series = np.ceil(V_bat_guess / V_cell)  # [-] Number of series connections (for voltage upgrade)
        # N_series = np.ceil(k_vb2 * Umot_to / V_cell)  # [-] Number of series connections (for voltage upgrade)
        N_series = np.ceil(Umot_to / V_cell)  # [-] Number of series connections (for voltage upgrade)
        N_parallel = 1  # [-] Number of parallel connections (for capacity upgrade)
        N_cell = N_parallel * N_series

        outputs['data:battery:cell:number:series:estimated'] = N_series
        outputs['data:battery:cell:number:parallel:estimated'] = N_parallel
        outputs['data:battery:cell:number:estimated'] = N_cell

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        Umot_to = inputs['data:motor:voltage:takeoff']
        # k_vb2 = inputs['data:battery:settings:voltage:k2']

        partials[
            'data:battery:cell:number:estimated',
            'data:motor:voltage:takeoff',
        ] = 1 / V_cell # (1 + np.cos(2 * np.pi * Umot_to / V_cell)) / V_cell  # Smooth ceil function derivative

        partials[
            'data:battery:cell:number:series:estimated',
            'data:motor:voltage:takeoff',
        ] = 1 / V_cell # (1 + np.cos(2 * np.pi * Umot_to / V_cell)) / V_cell  # Smooth ceil function derivative

        partials[
            'data:battery:cell:number:estimated',
            'data:battery:cell:voltage:estimated',
        ] = - Umot_to / V_cell ** 2

        partials[
            'data:battery:cell:number:series:estimated',
            'data:battery:cell:voltage:estimated',
        ] = - Umot_to / V_cell ** 2


        # partials[
        #     'data:battery:cell:number:estimated',
        #     'data:battery:settings:voltage:k2',
        # ] = Umot_to / V_cell
        #
        # partials[
        #     'data:battery:cell:number:series:estimated',
        #     'data:battery:settings:voltage:k2',
        # ] = Umot_to / V_cell

class Voltage(om.ExplicitComponent):
    """
    Computes battery voltage
    """

    def setup(self):
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_input('data:battery:cell:number:series:estimated', val=np.nan, units=None)
        self.add_output('data:battery:voltage:estimated', units='V')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        N_series = inputs['data:battery:cell:number:series:estimated']

        V_bat = V_cell * N_series  # [V] Battery voltage

        outputs['data:battery:voltage:estimated'] = V_bat


class Capacity(om.ExplicitComponent):
    """
    Computes battery capacity
    """

    def setup(self):
        self.add_input('data:battery:reference:mass', val=np.nan, units='kg')
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:voltage', val=np.nan, units='V')
        self.add_input('data:battery:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_output('data:battery:capacity:estimated', units='A*s')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mbat_ref = inputs['data:battery:reference:mass']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Mbat = inputs['data:battery:mass:estimated']
        V_bat = inputs['data:battery:voltage:estimated']

        C_bat = Mbat / Mbat_ref * Cbat_ref / V_bat * Vbat_ref  # [A.s] Capacity  of the battery

        outputs['data:battery:capacity:estimated'] = C_bat


class MaxCurrent(om.ExplicitComponent):
    """
    Computes battery maximum current
    """

    def setup(self):
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:current:max', val=np.nan, units='A')
        self.add_input('data:battery:capacity:estimated', units='A*s')
        self.add_output('data:battery:current:max:estimated', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Cbat_ref = inputs['data:battery:reference:capacity']
        Imax_ref = inputs['data:battery:reference:current:max']
        C_bat = inputs['data:battery:capacity:estimated']

        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery

        outputs['data:battery:current:max:estimated'] = Imax


class Energy(om.ExplicitComponent):
    """
    Computes battery energy
    """

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_output('data:battery:energy:estimated', units='kJ')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_bat = inputs['data:battery:voltage:estimated']
        C_bat = inputs['data:battery:capacity:estimated']

        # E_bat = E_bat_ref * Mbat / Mbat_ref * (1 - C_ratio)
        E_bat = C_bat * V_bat / 1000  # [kJ] Stored energy

        outputs['data:battery:energy:estimated'] = E_bat