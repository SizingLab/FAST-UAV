"""
Battery Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np

path = './data/DecisionTrees/Batteries/'
DF = pd.read_csv(path + 'Non-Dominated-Batteries.csv', sep=';')


class BatteryCatalogueSelection(om.Group):
    """
    Get battery parameters from catalogue if asked by the user.
    Then, affect either catalogue values or estimated values to system parameters.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogue' = true)
        # And set parameters to 'estimated' or 'catalogue' for future use
        if self.options["use_catalogue"]:
            self.add_subsystem("getCatalogueValues", BatteryDecisionTree(), promotes=["*"])
            self.add_subsystem("keepCatalogueValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])
        else:
            self.add_subsystem("keepEstimatedValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])


@ValidityDomainChecker(
    {
        'data:battery:capacity:estimated': (DF['Capacity_mAh'].min() / 1000 * 3600, DF['Capacity_mAh'].max() / 1000 * 3600),
    },
)
class BatteryDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Battery Decision Tree
        """
        Cbat_selection = 'next'
        self._DT = DecisionTrees((DF[['Capacity_mAh']]),
                                 (DF[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Volume_mm3', 'Imax [A]']]),
                                 [Cbat_selection]).DT_handling(dist=100000)

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_output('data:battery:cell:number:catalogue', units=None)
        self.add_output('data:battery:voltage:catalogue', units='V')
        self.add_output('data:battery:capacity:catalogue', units='A*s')
        self.add_output('data:battery:energy:catalogue', units='kJ')
        self.add_output('data:battery:current:max:catalogue', units='A')
        self.add_output('data:battery:mass:catalogue', units='kg')
        self.add_output('data:battery:cell:voltage:catalogue', units='V')
        self.add_output('data:battery:volume:catalogue', units='cm**3')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        C_bat = inputs['data:battery:capacity:estimated']
        V_bat = inputs['data:battery:voltage:estimated']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((C_bat * 1000 / 3600))])
        C_bat = y_pred[0][0] / 1000 * 3600  # battery capacity [As]
        V_bat_data = y_pred[0][1]  # battery voltage [V]
        Ncel = np.ceil(V_bat / V_bat_data)  # number of cells
        Mbat = Ncel * y_pred[0][2] / 1000  # battery weight [kg]
        Vol_bat = y_pred[0][3] * 0.001 # battery volume [cm3]
        Imax = y_pred[0][4]  # max current [A]
        E_bat = C_bat * V_bat_data * Ncel / 1000  # stored energy [kJ]

        # Outputs
        outputs['data:battery:cell:number:catalogue'] = Ncel
        outputs['data:battery:voltage:catalogue'] = V_bat_data * Ncel
        outputs['data:battery:capacity:catalogue'] = C_bat
        outputs['data:battery:energy:catalogue'] = E_bat
        outputs['data:battery:current:max:catalogue'] = Imax
        outputs['data:battery:mass:catalogue'] = Mbat
        outputs['data:battery:cell:voltage:catalogue'] = V_bat_data
        outputs['data:battery:volume:catalogue'] = Vol_bat


class ValueSetter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self._str = ''

    def setup(self):
        if self.options["use_catalogue"]:  # discrete values from catalogues
            self._str = ':catalogue'
        else:  # estimated values
            self._str = ':estimated'
        self.add_input('data:battery:cell:number'+self._str, val=np.nan, units=None)
        self.add_input('data:battery:voltage'+self._str, val=np.nan, units='V')
        self.add_input('data:battery:capacity'+self._str, val=np.nan, units='A*s')
        self.add_input('data:battery:energy' + self._str, val=np.nan, units='kJ')
        self.add_input('data:battery:current:max'+self._str, val=np.nan, units='A')
        self.add_input('data:battery:mass'+self._str, val=np.nan, units='kg')
        self.add_input('data:battery:cell:voltage'+self._str, val=np.nan, units='V')
        self.add_input('data:battery:volume'+self._str, val=np.nan, units='cm**3')
        # 'real' values
        self.add_output('data:battery:cell:number', units=None)
        self.add_output('data:battery:voltage', units='V')
        self.add_output('data:battery:capacity', units='A*s')
        self.add_output('data:battery:energy', units='kJ')
        self.add_output('data:battery:current:max', units='A')
        self.add_output('data:battery:mass', units='kg')
        self.add_output('data:battery:cell:voltage', units='V')
        self.add_output('data:battery:volume', units='cm**3')

    def setup_partials(self):
        self.declare_partials('data:battery:cell:number', 'data:battery:cell:number'+self._str, val=1.)
        self.declare_partials('data:battery:voltage', 'data:battery:voltage'+self._str, val=1.)
        self.declare_partials('data:battery:capacity', 'data:battery:capacity'+self._str, val=1.)
        self.declare_partials('data:battery:energy', 'data:battery:energy' + self._str, val=1.)
        self.declare_partials('data:battery:current:max', 'data:battery:current:max'+self._str, val=1.)
        self.declare_partials('data:battery:mass', 'data:battery:mass'+self._str, val=1.)
        self.declare_partials('data:battery:cell:voltage', 'data:battery:cell:voltage'+self._str, val=1.)
        self.declare_partials('data:battery:volume', 'data:battery:volume' + self._str, val=1.)

    def compute(self, inputs, outputs):
        outputs['data:battery:cell:number'] = inputs['data:battery:cell:number'+self._str]
        outputs['data:battery:voltage'] = inputs['data:battery:voltage'+self._str]
        outputs['data:battery:capacity'] = inputs['data:battery:capacity'+self._str]
        outputs['data:battery:energy'] = inputs['data:battery:energy' + self._str]
        outputs['data:battery:current:max'] = inputs['data:battery:current:max'+self._str]
        outputs['data:battery:mass'] = inputs['data:battery:mass'+self._str]
        outputs['data:battery:cell:voltage'] = inputs['data:battery:cell:voltage'+self._str]
        outputs['data:battery:volume'] = inputs['data:battery:volume' + self._str]