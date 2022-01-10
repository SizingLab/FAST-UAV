"""
Definition parameters for the battery.
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import add_subsystem_with_deviation


class BatteryDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the battery.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the battery are the voltage and the capacity (or, in an equivalent way, the voltage
    and the battery mass).
    """
    def setup(self):

        self.add_subsystem("cell_number", CellNumber(), promotes=["*"])

        add_subsystem_with_deviation(self, "voltage", Voltage(),
                                     uncertain_outputs={'data:battery:voltage:estimated': 'V'})

        self.add_subsystem("weight", Weight(), promotes=["*"])

        add_subsystem_with_deviation(self, "capacity", Capacity(),
                                     uncertain_outputs={'data:battery:capacity:estimated': 'A*s'})


class CellNumber(om.ExplicitComponent):
    """
    Computes the voltage of the battery. Also returns the number of cells.
    """

    def setup(self):
        self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_output('data:battery:cell:number:estimated', units=None)
        self.add_output('data:battery:cell:number:series:estimated', units=None)
        self.add_output('data:battery:cell:number:parallel:estimated', units=None)

    def setup_partials(self):
        # Derivatives provided by user as analytic expressions to avoid issues with ceil function.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        Umot_to = inputs['data:motor:voltage:takeoff']

        # N_series = np.ceil(V_bat_guess / V_cell)  # [-] Number of series connections (for voltage upgrade)
        N_series = np.ceil(Umot_to / V_cell)  # [-] Number of series connections (for voltage upgrade)
        N_parallel = 1  # [-] Number of parallel connections (for capacity upgrade)
        N_cell = N_parallel * N_series

        outputs['data:battery:cell:number:series:estimated'] = N_series
        outputs['data:battery:cell:number:parallel:estimated'] = N_parallel
        outputs['data:battery:cell:number:estimated'] = N_cell

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        Umot_to = inputs['data:motor:voltage:takeoff']

        partials[
            'data:battery:cell:number:estimated',
            'data:motor:voltage:takeoff',
        ] = 1 / V_cell  # (1 + np.cos(2 * np.pi * Umot_to / V_cell)) / V_cell  # Smooth ceil function derivative

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


class Voltage(om.ExplicitComponent):
    """
    Computes battery voltage
    """

    def setup(self):
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_input('data:battery:cell:number:series:estimated', val=np.nan, units=None)
        self.add_output('data:battery:voltage:estimated', units='V')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        N_series = inputs['data:battery:cell:number:series:estimated']

        V_bat = V_cell * N_series  # [V] Battery voltage

        outputs['data:battery:voltage:estimated'] = V_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cell = inputs['data:battery:cell:voltage:estimated']

        partials['data:battery:voltage:estimated',
                 'data:battery:cell:number:series:estimated'] = V_cell


class Weight(om.ExplicitComponent):
    """
    Computes battery weight
    """

    def setup(self):
        self.add_input('specifications:payload:mass:max', val=np.nan, units='kg')
        self.add_input('data:battery:settings:mass:k', val=np.nan)
        self.add_output('data:battery:mass:estimated', units='kg')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        M_load = inputs['specifications:payload:mass:max']
        k_Mb = inputs['data:battery:settings:mass:k']

        M_bat = k_Mb * M_load  # [kg] Battery mass (estimated)

        outputs['data:battery:mass:estimated'] = M_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        M_load = inputs['specifications:payload:mass:max']

        partials['data:battery:mass:estimated',
                 'data:battery:settings:mass:k'] = M_load


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
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        Mbat_ref = inputs['data:battery:reference:mass']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Mbat = inputs['data:battery:mass:estimated']
        V_bat = inputs['data:battery:voltage:estimated']

        C_bat = Mbat / Mbat_ref * Cbat_ref / V_bat * Vbat_ref  # [A.s] Capacity  of the battery

        outputs['data:battery:capacity:estimated'] = C_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mbat_ref = inputs['data:battery:reference:mass']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Mbat = inputs['data:battery:mass:estimated']
        V_bat = inputs['data:battery:voltage:estimated']

        partials['data:battery:capacity:estimated',
                 'data:battery:mass:estimated'] = 1 / Mbat_ref * Cbat_ref / V_bat * Vbat_ref

        partials['data:battery:capacity:estimated',
                 'data:battery:voltage:estimated'] = - Mbat / Mbat_ref * Cbat_ref / V_bat**2 * Vbat_ref
