"""
ESC Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


path = './data/DecisionTrees/ESC/'
df = pd.read_csv(path + 'Non-Dominated-ESC.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:ESC:voltage': (df['Vmax_V'].min(), df['Vmax_V'].max()),
    },
)
class ESCDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the ESC Decision Tree
        """
        Vmax_selection = 'next'
        self._DT = DecisionTrees(df[['Vmax_V']],
                                 df[['Pmax_W', 'Vmax_V', 'Weight_g']], [Vmax_selection])\
            .DT_handling()

    def setup(self):
        self.add_input('data:ESC:voltage', val=np.nan, units='V')
        self.add_output('data:ESC:voltage:catalogue', units='V')
        self.add_output('data:ESC:power:max:catalogue', units='W')
        self.add_output('data:ESC:mass:catalogue', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        V_esc = inputs['data:ESC:voltage']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack(V_esc)])
        P_esc = y_pred[0][0]  # [W] ESC power
        V_esc = y_pred[0][1]  # [V] ESC voltage
        M_esc = y_pred[0][2] / 1000  # [kg] ESC mass

        # Outputs
        outputs['data:ESC:power:max:catalogue'] = P_esc
        outputs['data:ESC:voltage:catalogue'] = V_esc
        outputs['data:ESC:mass:catalogue'] = M_esc
