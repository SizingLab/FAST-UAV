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
    and the energy).
    """
    def setup(self):

        self.add_subsystem("cell_number", CellNumber(), promotes=["*"])

        add_subsystem_with_deviation(self, "voltage", Voltage(),
                                     uncertain_outputs={'data:battery:voltage:estimated': 'V'})

        add_subsystem_with_deviation(self, "energy", Energy(),
                                     uncertain_outputs={'data:battery:energy:estimated': 'kJ'})

        self.add_subsystem("capacity", Capacity(), promotes=["*"])


class CellNumber(om.ExplicitComponent):
    """
    Computes the voltage of the battery. Also returns the number of cells.
    """

    def setup(self):
        # self.add_input('data:motor:voltage:takeoff', val=np.nan, units='V')
        self.add_input('data:battery:voltage:guess', val=np.nan, units='V')
        self.add_input('data:battery:cell:voltage:estimated', val=3.7, units='V')
        self.add_output('data:battery:cell:number:estimated', units=None)
        self.add_output('data:battery:cell:number:series:estimated', units=None)
        self.add_output('data:battery:cell:number:parallel:estimated', units=None)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        V_cell = inputs['data:battery:cell:voltage:estimated']
        # Umot_to = inputs['data:motor:voltage:takeoff']
        V_bat_guess = inputs['data:battery:voltage:guess']

        N_series = (V_bat_guess / V_cell)  # [-] Number of series connections (for voltage upgrade)
        # N_series = (Umot_to / V_cell)  # [-] Number of series connections (for voltage upgrade)
        N_parallel = 1  # [-] Number of parallel connections (for capacity upgrade)
        N_cell = N_parallel * N_series

        outputs['data:battery:cell:number:series:estimated'] = N_series
        outputs['data:battery:cell:number:parallel:estimated'] = N_parallel
        outputs['data:battery:cell:number:estimated'] = N_cell

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     """
    #     Defining approximates from partials helps avoiding local minima but may increase the number of iterations
    #     if the derivatives are not well defined (or cause convergence issues).
    #     """
    #
    #     V_cell = inputs['data:battery:cell:voltage:estimated']
    #     Umot_to = inputs['data:motor:voltage:takeoff']
    #     # V_bat_guess = inputs['data:battery:voltage:guess']
    #
    #     # partials[
    #     #     'data:battery:cell:number:estimated',
    #     #     'data:battery:voltage:guess',
    #     # ] = (1 + np.cos(2 * np.pi * V_bat_guess / V_cell)) / V_cell  # Smooth ceil function derivative
    #     #
    #     # partials[
    #     #     'data:battery:cell:number:series:estimated',
    #     #     'data:battery:voltage:guess',
    #     # ] = (1 + np.cos(2 * np.pi * V_bat_guess / V_cell)) / V_cell  # Smooth ceil function derivative
    #
    #     partials[
    #         'data:battery:cell:number:estimated',
    #         'data:motor:voltage:takeoff',
    #     ] = (1 + np.cos(2 * np.pi * Umot_to / V_cell)) / V_cell  # Smooth ceil function derivative
    #
    #     partials[
    #         'data:battery:cell:number:series:estimated',
    #         'data:motor:voltage:takeoff',
    #     ] = (1 + np.cos(2 * np.pi * Umot_to / V_cell)) / V_cell  # Smooth ceil function derivative


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

        partials[
            'data:battery:voltage:estimated',
            'data:battery:cell:number:series:estimated',
        ] = V_cell


class Energy(om.ExplicitComponent):
    """
    Computes battery energy
    """

    def setup(self):
        self.add_input('specifications:payload:mass:max', val=np.nan, units='kg')
        self.add_input('data:battery:settings:mass:k', val=np.nan, units=None)
        self.add_input('data:battery:reference:mass', val=np.nan, units='kg')
        self.add_input('data:battery:reference:energy', val=np.nan, units='kJ')
        self.add_output('data:battery:energy:estimated', units='kJ')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        k_Mb = inputs['data:battery:settings:mass:k']
        M_load = inputs['specifications:payload:mass:max']
        Mbat_ref = inputs['data:battery:reference:mass']
        Ebat_ref = inputs['data:battery:reference:energy']

        E_bat = k_Mb * M_load * Ebat_ref / Mbat_ref  # [kJ] Battery energy (estimated)

        outputs['data:battery:energy:estimated'] = E_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        M_load = inputs['specifications:payload:mass:max']
        Mbat_ref = inputs['data:battery:reference:mass']
        Ebat_ref = inputs['data:battery:reference:energy']

        partials['data:battery:energy:estimated',
                 'data:battery:settings:mass:k'] = M_load * Ebat_ref / Mbat_ref


class Capacity(om.ExplicitComponent):
    """
    Computes battery capacity
    """

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:energy:estimated', val=np.nan, units='J')
        self.add_output('data:battery:capacity:estimated', units='A*s')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        V_bat = inputs['data:battery:voltage:estimated']
        E_bat = inputs['data:battery:energy:estimated']

        C_bat = E_bat / V_bat  # [A.s] Capacity  of the battery

        outputs['data:battery:capacity:estimated'] = C_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_bat = inputs['data:battery:voltage:estimated']
        E_bat = inputs['data:battery:energy:estimated']

        partials['data:battery:capacity:estimated',
                 'data:battery:voltage:estimated'] = - E_bat / V_bat**2

        partials['data:battery:capacity:estimated',
                 'data:battery:energy:estimated'] = 1 / V_bat



