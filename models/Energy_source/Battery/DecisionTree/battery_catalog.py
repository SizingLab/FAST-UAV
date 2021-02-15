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
DF = DF[DF['TYPE'] != 'KOKAM']  # remove KOKAM batteries (not the same techno)


class BatteryCatalogueSelection(om.Group):
    """
    Get battery parameters from catalogue if asked by the user.
    Otherwise, keep going with estimated parameters.
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
                                 (DF[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Imax [A]']]),
                                 [Cbat_selection]).DT_handling(dist=100000)

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_output('data:battery:cell:number:catalogue', units=None)
        self.add_output('data:battery:voltage:catalogue', units='V')
        self.add_output('data:battery:capacity:catalogue', units='A*s')
        self.add_output('data:battery:current:max:catalogue', units='A')
        self.add_output('data:battery:mass:catalogue', units='kg')
        self.add_output('data:battery:cell:voltage:catalogue', units='V')


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
        Ncel = np.ceil(V_bat / V_bat_data)
        Mbat = Ncel * y_pred[0][2] / 1000
        Imax = y_pred[0][3]

        # Outputs
        outputs['data:battery:cell:number:catalogue'] = Ncel
        outputs['data:battery:voltage:catalogue'] = V_bat_data * Ncel
        outputs['data:battery:capacity:catalogue'] = C_bat
        outputs['data:battery:current:max:catalogue'] = Imax
        outputs['data:battery:mass:catalogue'] = Mbat
        outputs['data:battery:cell:voltage:catalogue'] = V_bat_data


class ValueSetter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        if self.options["use_catalogue"]:  # discrete values from catalogues
            self.add_input('data:battery:cell:number:catalogue', val=np.nan, units=None)
            self.add_input('data:battery:voltage:catalogue', val=np.nan, units='V')
            self.add_input('data:battery:capacity:catalogue', val=np.nan, units='A*s')
            self.add_input('data:battery:current:max:catalogue', val=np.nan, units='A')
            self.add_input('data:battery:mass:catalogue', val=np.nan, units='kg')
            self.add_input('data:battery:cell:voltage:catalogue', val=np.nan, units='V')
        else:  # estimated values
            self.add_input('data:battery:cell:number:estimated', val=np.nan, units=None)
            self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
            self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
            self.add_input('data:battery:current:max:estimated', val=np.nan, units='A')
            self.add_input('data:battery:mass:estimated', val=np.nan, units='kg')
            self.add_input('data:battery:cell:voltage:estimated', val=np.nan, units='V')
        # real values
        self.add_output('data:battery:cell:number', units=None)
        self.add_output('data:battery:voltage', units='V')
        self.add_output('data:battery:capacity', units='A*s')
        self.add_output('data:battery:current:max', units='A')
        self.add_output('data:battery:mass', units='kg')
        self.add_output('data:battery:cell:voltage', units='V')

    def setup_partials(self):
        if self.options["use_catalogue"]:
            self.declare_partials('data:battery:cell:number', 'data:battery:cell:number:catalogue', val=1.)
            self.declare_partials('data:battery:voltage', 'data:battery:voltage:catalogue', val=1.)
            self.declare_partials('data:battery:capacity', 'data:battery:capacity:catalogue', val=1.)
            self.declare_partials('data:battery:current:max', 'data:battery:current:max:catalogue', val=1.)
            self.declare_partials('data:battery:mass', 'data:battery:mass:catalogue', val=1.)
            self.declare_partials('data:battery:cell:voltage', 'data:battery:cell:voltage:catalogue', val=1.)
        else:
            self.declare_partials('data:battery:cell:number', 'data:battery:cell:number:estimated', val=1.)
            self.declare_partials('data:battery:voltage', 'data:battery:voltage:estimated', val=1.)
            self.declare_partials('data:battery:capacity', 'data:battery:capacity:estimated', val=1.)
            self.declare_partials('data:battery:current:max', 'data:battery:current:max:estimated', val=1.)
            self.declare_partials('data:battery:mass', 'data:battery:mass:estimated', val=1.)
            self.declare_partials('data:battery:cell:voltage', 'data:battery:cell:voltage:estimated', val=1.)

    def compute(self, inputs, outputs):
        if self.options["use_catalogue"]:
            outputs['data:battery:cell:number'] = inputs['data:battery:cell:number:catalogue']
            outputs['data:battery:voltage'] = inputs['data:battery:voltage:catalogue']
            outputs['data:battery:capacity'] = inputs['data:battery:capacity:catalogue']
            outputs['data:battery:current:max'] = inputs['data:battery:current:max:catalogue']
            outputs['data:battery:mass'] = inputs['data:battery:mass:catalogue']
            outputs['data:battery:cell:voltage'] = inputs['data:battery:cell:voltage:catalogue']
        else:
            outputs['data:battery:cell:number'] = inputs['data:battery:cell:number:estimated']
            outputs['data:battery:voltage'] = inputs['data:battery:voltage:estimated']
            outputs['data:battery:capacity'] = inputs['data:battery:capacity:estimated']
            outputs['data:battery:current:max'] = inputs['data:battery:current:max:estimated']
            outputs['data:battery:mass'] = inputs['data:battery:mass:estimated']
            outputs['data:battery:cell:voltage'] = inputs['data:battery:cell:voltage:estimated']