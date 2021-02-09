"""
Motor Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


path = './data/DecisionTrees/Motors/'
df = pd.read_csv(path + 'Non-Dominated-Motors.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:motor:torque:max': (df['Tmax_Nm'].min(), df['Tmax_Nm'].max()),
        'data:motor:torque_coefficient': (df['Kt_Nm_A'].min(), df['Kt_Nm_A'].max()),
    },
)
class MotorDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Propeller Decision Tree
        """
        Tmax_selection = 'next'
        Kt_selection = 'average'
        self._DT = DecisionTrees(df[['Tmax_Nm', 'Kt_Nm_A']].values,
                                 df[['Tnom_Nm', 'Kt_Nm_A', 'r_omn', 'Tmax_Nm', 'weight_g', 'Cf_Nm']].values,
                                 [Tmax_selection, Kt_selection]).DT_handling()

    def setup(self):
        self.add_input('data:motor:torque:max', val=np.nan, units='N*m')
        self.add_input('data:motor:torque_coefficient', val=np.nan, units='N*m/A')
        self.add_output('data:motor:catalogue:torque:max', units='N*m')
        self.add_output('data:motor:catalogue:torque_coefficient', units='N*m/A')
        self.add_output('data:motor:catalogue:torque:nominal', units='N*m')
        self.add_output('data:motor:catalogue:torque:friction', units='N*m')
        self.add_output('data:motor:catalogue:resistance', units='V/A')
        self.add_output('data:motor:catalogue:mass', units='kg')


    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        Tmot_max = inputs['data:motor:torque:max']
        Ktmot = inputs['data:motor:torque_coefficient']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((Tmot_max, Ktmot))])
        Tmot = y_pred[0][0]  # nominal torque [N.m]
        Ktmot = y_pred[0][1]  # Kt constant [N.m./A]
        Rmot = y_pred[0][2]  # motor resistance [ohm]
        Tmot_max = y_pred[0][3]  # max motor torque [Nm]
        Mmot = y_pred[0][4] / 1000  # motor mass [kg]
        Tfmot = y_pred[0][5]  # friction torque [Nm]

        # Outputs
        outputs['data:motor:catalogue:torque:max'] = Tmot_max
        outputs['data:motor:catalogue:torque_coefficient'] = Ktmot
        outputs['data:motor:catalogue:torque:nominal'] = Tmot
        outputs['data:motor:catalogue:torque:friction'] = Tfmot
        outputs['data:motor:catalogue:resistance'] = Rmot
        outputs['data:motor:catalogue:mass'] = Mmot
