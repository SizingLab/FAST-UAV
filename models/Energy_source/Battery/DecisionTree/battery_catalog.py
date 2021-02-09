"""
Battery Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np

path = './data/DecisionTrees/Batteries/'
df = pd.read_csv(path + 'Non-Dominated-Batteries.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:battery:capacity': (df['Capacity_mAh'].min() / 1000 * 3600, df['Capacity_mAh'].max() / 1000 * 3600),
    },
)
class BatteryDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Battery Decision Tree
        """
        Cbat_selection = 'next'
        self._DT = DecisionTrees((df[['Capacity_mAh']]),
                                 (df[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Imax [A]']]),
                                 [Cbat_selection]).DT_handling(dist=100000)

    def setup(self):
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_output('data:battery:catalogue:cell_number', units=None)
        self.add_output('data:battery:catalogue:voltage', units='V')
        self.add_output('data:battery:catalogue:capacity', units='A*s')
        self.add_output('data:battery:catalogue:current:max', units='A')
        self.add_output('data:battery:catalogue:mass', units='kg')


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        C_bat = inputs['data:battery:capacity']
        V_bat = inputs['data:battery:voltage']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((C_bat * 1000 / 3600))])
        C_bat = y_pred[0][0] / 1000 * 3600  # battery capacity [As]
        V_bat_data = y_pred[0][1]  # battery voltage[V]
        Ncel = np.ceil(V_bat / V_bat_data)
        Mbat = Ncel * y_pred[0][2] / 1000
        Imax = y_pred[0][3]


        # Outputs
        outputs['data:battery:catalogue:cell_number'] = Ncel
        outputs['data:battery:catalogue:voltage'] = V_bat_data * Ncel
        outputs['data:battery:catalogue:capacity'] = C_bat
        outputs['data:battery:catalogue:current:max'] = Imax
        outputs['data:battery:catalogue:mass'] = Mbat
