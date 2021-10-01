"""
Battery Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np
from scipy import stats

# Database import
path = './data/DecisionTrees/Batteries/'
DF = pd.read_csv(path + 'Non-Dominated-Augmented-Batteries.csv', sep=';')


class BatteryCatalogueSelection(om.Group):
    """
    Select either custom or off-the-shelf battery.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        name_str = 'off_the_shelf' if self.options["use_catalogue"] else 'custom'
        self.add_subsystem(name_str, BatteryDecisionTree(use_catalogue=self.options['use_catalogue']), promotes=["*"])


@ValidityDomainChecker(
    {
        'data:battery:capacity:estimated': (DF['Capacity_mAh'].min() * 3.6, DF['Capacity_mAh'].max() * 3.6),
    },
)
class BatteryDecisionTree(om.ExplicitComponent):
    """
    Get battery parameters from catalogue if asked by the user.
    Then, affect either catalogue values or estimated values to system parameters.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        C_bat_selection = 'next'
        V_bat_selection = 'average'
        self._DT = DecisionTrees(DF[['Capacity_mAh', 'Voltage_V']].values,
                                 DF[['Capacity_mAh', 'Voltage_V', 'Weight_kg', 'Volume_cm3', 'Imax [A]']].values,
                                 [C_bat_selection, V_bat_selection]).DT_handling(dist=1000000)

    def setup(self):
        # inputs: estimated values
        self.add_input('data:battery:cell:number:series:estimated', val=np.nan, units=None)
        self.add_input('data:battery:cell:number:parallel:estimated', val=np.nan, units=None)
        self.add_input('data:battery:cell:number:estimated', val=np.nan, units=None)
        self.add_input('data:battery:cell:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_input('data:battery:energy:estimated', val=np.nan, units='kJ')
        self.add_input('data:battery:current:max:estimated', val=np.nan, units='A')
        self.add_input('data:battery:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:battery:volume:estimated', val=np.nan, units='cm**3')
        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:battery:cell:number:series:catalogue', units=None)
            self.add_output('data:battery:cell:number:parallel:catalogue', units=None)
            self.add_output('data:battery:cell:number:catalogue', units=None)
            self.add_output('data:battery:cell:voltage:catalogue', units='V')
            self.add_output('data:battery:voltage:catalogue', units='V')
            self.add_output('data:battery:capacity:catalogue', units='A*s')
            self.add_output('data:battery:energy:catalogue', units='kJ')
            self.add_output('data:battery:current:max:catalogue', units='A')
            self.add_output('data:battery:mass:catalogue', units='kg')
            self.add_output('data:battery:volume:catalogue', units='cm**3')
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:battery:cell:number:series', units=None)
        self.add_output('data:battery:cell:number:parallel', units=None)
        self.add_output('data:battery:cell:number', units=None)
        self.add_output('data:battery:cell:voltage', units='V')
        self.add_output('data:battery:voltage', units='V')
        self.add_output('data:battery:capacity', units='A*s')
        self.add_output('data:battery:energy', units='kJ')
        self.add_output('data:battery:current:max', units='A')
        self.add_output('data:battery:mass', units='kg')
        self.add_output('data:battery:volume', units='cm**3')

    def setup_partials(self):
        self.declare_partials('data:battery:voltage', 'data:battery:voltage:estimated', val=1.)
        self.declare_partials('data:battery:capacity', 'data:battery:capacity:estimated', val=1.)
        self.declare_partials('data:battery:energy', 'data:battery:energy:estimated', val=1.)
        self.declare_partials('data:battery:current:max', 'data:battery:current:max:estimated', val=1.)
        self.declare_partials('data:battery:mass', 'data:battery:mass:estimated', val=1.)

    def compute(self, inputs, outputs):
        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Estimated parameters
            C_bat_est = inputs['data:battery:capacity:estimated']  # [A*s]
            V_bat_est = inputs['data:battery:voltage:estimated']  # [V]

            # Decision Tree
            y_pred = self._DT.predict([np.hstack((C_bat_est / 3.6, V_bat_est))])

            # Outputs
            C_cell = y_pred[0][0] * 3.6  # cell capacity [A*s]
            Imax_cell = y_pred[0][4]  # cell max current [A]
            N_parallel = 1 #max(np.ceil(C_bat_est / C_cell), np.ceil(Imax_est / Imax_cell))  # number of parallel connections to ensure sufficient current & capacity
            C_bat = N_parallel * C_cell  # battery pack capacity [As]
            Imax = N_parallel * Imax_cell  # max current [A]

            V_cell = y_pred[0][1]  # cell voltage [V]
            N_series = 1 #np.ceil(V_bat_est / V_cell)  # number of series connections to ensure sufficient voltage
            V_bat = N_series * V_cell  # battery pack voltage [V]

            N_cell = N_series * N_parallel  # number of cells
            M_bat = N_cell * y_pred[0][2]  # battery pack weight [kg]
            Vol_bat = N_cell * y_pred[0][3]  # battery pack volume [cm3]
            E_bat = C_bat * V_bat / 1000  # stored energy [kJ]

            outputs['data:battery:cell:number'] = outputs['data:battery:cell:number:catalogue'] = N_cell
            outputs['data:battery:cell:number:series'] = outputs['data:battery:cell:number:series:catalogue'] = N_series
            outputs['data:battery:cell:number:parallel'] = outputs['data:battery:cell:number:parallel:catalogue'] = N_parallel
            outputs['data:battery:voltage'] = outputs['data:battery:voltage:catalogue'] = V_bat
            outputs['data:battery:capacity'] = outputs['data:battery:capacity:catalogue'] = C_bat
            outputs['data:battery:energy'] = outputs['data:battery:energy:catalogue'] = E_bat
            outputs['data:battery:current:max'] = outputs['data:battery:current:max:catalogue'] = Imax
            outputs['data:battery:mass'] = outputs['data:battery:mass:catalogue'] = M_bat
            outputs['data:battery:cell:voltage'] = outputs['data:battery:cell:voltage:catalogue'] = V_cell
            outputs['data:battery:volume'] = outputs['data:battery:volume:catalogue'] = Vol_bat

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:battery:cell:number:series'] = inputs['data:battery:cell:number:series:estimated']
            outputs['data:battery:cell:number:parallel'] = inputs['data:battery:cell:number:parallel:estimated']
            outputs['data:battery:cell:number'] = inputs['data:battery:cell:number:estimated']
            outputs['data:battery:voltage'] = inputs['data:battery:voltage:estimated']
            outputs['data:battery:capacity'] = inputs['data:battery:capacity:estimated']
            outputs['data:battery:energy'] = inputs['data:battery:energy:estimated']
            outputs['data:battery:current:max'] = inputs['data:battery:current:max:estimated']
            outputs['data:battery:mass'] = inputs['data:battery:mass:estimated']
            outputs['data:battery:cell:voltage'] = inputs['data:battery:cell:voltage:estimated']
            outputs['data:battery:volume'] = inputs['data:battery:volume:estimated']
