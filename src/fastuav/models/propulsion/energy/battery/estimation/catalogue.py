"""
Off-the-shelf Battery selection.
"""
import openmdao.api as om
from utils.catalogues.estimators import NearestNeighbor
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np

# Database import
path = './data/catalogues/Batteries/'
DF = pd.read_csv(path + 'Non-Dominated-Augmented-Batteries.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:propulsion:battery:voltage:estimated': (DF['Voltage_V'].min(), DF['Voltage_V'].max()),
        'data:propulsion:battery:capacity:estimated': (DF['Capacity_As'].min(), DF['Capacity_As'].max()),
    },
)
class BatteryCatalogueSelection(om.ExplicitComponent):
    """
    Battery selection and component's parameters assignment:
            - If use_catalogue is True, a battery is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=False, types=bool)
        C_bat_selection = 'next'
        V_bat_selection = 'next'
        # E_bat_selection = 'next'
        self._clf = NearestNeighbor(df=DF, X_names=['Voltage_V', 'Capacity_As'],
                                     crits=[V_bat_selection, C_bat_selection])
        # self._clf = NearestNeighbor(df=DF, X_names=['Voltage_V', 'Energy_kJ'],
        #                             crits=[V_bat_selection, E_bat_selection])
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input('data:propulsion:battery:cell:number:series:estimated', val=np.nan, units=None)
        self.add_input('data:propulsion:battery:cell:number:parallel:estimated', val=np.nan, units=None)
        self.add_input('data:propulsion:battery:cell:number:estimated', val=np.nan, units=None)
        self.add_input('data:propulsion:battery:cell:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:propulsion:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:propulsion:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_input('data:propulsion:battery:energy:estimated', val=np.nan, units='kJ')
        self.add_input('data:propulsion:battery:current:max:estimated', val=np.nan, units='A')
        self.add_input('data:weights:battery:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:propulsion:battery:volume:estimated', val=np.nan, units='cm**3')
        self.add_input('data:propulsion:battery:DoD:max:estimated', val=np.nan, units=None)
        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:propulsion:battery:cell:number:series:catalogue', units=None)
            self.add_output('data:propulsion:battery:cell:number:parallel:catalogue', units=None)
            self.add_output('data:propulsion:battery:cell:number:catalogue', units=None)
            self.add_output('data:propulsion:battery:cell:voltage:catalogue', units='V')
            self.add_output('data:propulsion:battery:voltage:catalogue', units='V')
            self.add_output('data:propulsion:battery:capacity:catalogue', units='A*s')
            self.add_output('data:propulsion:battery:energy:catalogue', units='kJ')
            self.add_output('data:propulsion:battery:current:max:catalogue', units='A')
            self.add_output('data:weights:battery:mass:catalogue', units='kg')
            self.add_output('data:propulsion:battery:volume:catalogue', units='cm**3')
            # self.add_output('data:propulsion:battery:DoD:max:catalogue', units=None)
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:propulsion:battery:cell:number:series', units=None)
        self.add_output('data:propulsion:battery:cell:number:parallel', units=None)
        self.add_output('data:propulsion:battery:cell:number', units=None)
        self.add_output('data:propulsion:battery:cell:voltage', units='V')
        self.add_output('data:propulsion:battery:voltage', units='V')
        self.add_output('data:propulsion:battery:capacity', units='A*s')
        self.add_output('data:propulsion:battery:energy', units='kJ')
        self.add_output('data:propulsion:battery:current:max', units='A')
        self.add_output('data:weights:battery:mass', units='kg')
        self.add_output('data:propulsion:battery:volume', units='cm**3')
        self.add_output('data:propulsion:battery:DoD:max', units=None)

    def setup_partials(self):
        self.declare_partials('data:propulsion:battery:voltage', 'data:propulsion:battery:voltage:estimated', val=1.)
        self.declare_partials('data:propulsion:battery:capacity', 'data:propulsion:battery:capacity:estimated', val=1.)
        self.declare_partials('data:propulsion:battery:energy', 'data:propulsion:battery:energy:estimated', val=1.)
        self.declare_partials('data:propulsion:battery:current:max', 'data:propulsion:battery:current:max:estimated', val=1.)
        self.declare_partials('data:weights:battery:mass', 'data:weights:battery:mass:estimated', val=1.)
        self.declare_partials('data:propulsion:battery:DoD:max', 'data:propulsion:battery:DoD:max:estimated', val=1.)

    def compute(self, inputs, outputs):
        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for battery selection
            V_bat_opt = inputs['data:propulsion:battery:voltage:estimated']  # [V]
            C_bat_opt = inputs['data:propulsion:battery:capacity:estimated']  # [A*s]
            # E_bat_opt = inputs['data:propulsion:battery:energy:estimated']  # [kJ]

            # Get closest product
            df_y = self._clf.predict2([V_bat_opt, C_bat_opt])
            # df_y = self._clf.predict2([V_bat_opt, E_bat_opt])
            V_bat = df_y['Voltage_V'].iloc[0]  # battery pack voltage [V]
            C_bat = df_y['Capacity_As'].iloc[0]  # battery pack capacity [A*s]
            M_bat = df_y['Weight_kg'].iloc[0]  # battery pack weight [kg]
            Vol_bat = df_y['Volume_cm3'].iloc[0]  # battery pack volume [cm3]
            Imax = df_y['Imax_A'].iloc[0]  # max current [A]
            N_series = df_y['n_series'].iloc[0]  # number of series connections to ensure sufficient voltage
            N_parallel = df_y['n_parallel'].iloc[0]  # number of parallel connections
            N_cell = N_series * N_parallel  # number of cells
            E_bat = df_y['Energy_kJ'].iloc[0]  # C_bat * V_bat / 1000  # stored energy [kJ]

            # Outputs
            outputs['data:propulsion:battery:cell:number'] = outputs['data:propulsion:battery:cell:number:catalogue'] = N_cell
            outputs['data:propulsion:battery:cell:number:series'] = outputs['data:propulsion:battery:cell:number:series:catalogue'] = N_series
            outputs['data:propulsion:battery:cell:number:parallel'] = outputs['data:propulsion:battery:cell:number:parallel:catalogue'] = N_parallel
            outputs['data:propulsion:battery:voltage'] = outputs['data:propulsion:battery:voltage:catalogue'] = V_bat
            outputs['data:propulsion:battery:capacity'] = outputs['data:propulsion:battery:capacity:catalogue'] = C_bat
            outputs['data:propulsion:battery:energy'] = outputs['data:propulsion:battery:energy:catalogue'] = E_bat
            outputs['data:propulsion:battery:current:max'] = outputs['data:propulsion:battery:current:max:catalogue'] = Imax
            outputs['data:weights:battery:mass'] = outputs['data:weights:battery:mass:catalogue'] = M_bat
            outputs['data:propulsion:battery:cell:voltage'] = outputs['data:propulsion:battery:cell:voltage:catalogue'] = V_bat / N_series
            outputs['data:propulsion:battery:volume'] = outputs['data:propulsion:battery:volume:catalogue'] = Vol_bat
            outputs['data:propulsion:battery:DoD:max'] = inputs['data:propulsion:battery:DoD:max:estimated']

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:propulsion:battery:cell:number:series'] = inputs['data:propulsion:battery:cell:number:series:estimated']
            outputs['data:propulsion:battery:cell:number:parallel'] = inputs['data:propulsion:battery:cell:number:parallel:estimated']
            outputs['data:propulsion:battery:cell:number'] = inputs['data:propulsion:battery:cell:number:estimated']
            outputs['data:propulsion:battery:voltage'] = inputs['data:propulsion:battery:voltage:estimated']
            outputs['data:propulsion:battery:capacity'] = inputs['data:propulsion:battery:capacity:estimated']
            outputs['data:propulsion:battery:energy'] = inputs['data:propulsion:battery:energy:estimated']
            outputs['data:propulsion:battery:current:max'] = inputs['data:propulsion:battery:current:max:estimated']
            outputs['data:weights:battery:mass'] = inputs['data:weights:battery:mass:estimated']
            outputs['data:propulsion:battery:cell:voltage'] = inputs['data:propulsion:battery:cell:voltage:estimated']
            outputs['data:propulsion:battery:volume'] = inputs['data:propulsion:battery:volume:estimated']
            outputs['data:propulsion:battery:DoD:max'] = inputs['data:propulsion:battery:DoD:max:estimated']
