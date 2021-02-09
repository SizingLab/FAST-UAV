"""
Propeller Decision Tree based on provided catalogue
"""
import openmdao.api as om
from openmdao.core.component import Component
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
import pandas as pd
import numpy as np


class PropellerDecisionTree:

    def __init__(self):
        """
        Creates and trains the Propeller Decision Tree
        """
        self._path = './data/DecisionTrees/Propeller/'
        self._df = pd.read_csv(self._path + 'Non-Dominated-Propeller.csv', sep=';')
        self._DT = DecisionTrees(self._df[['BETA', 'DIAMETER_IN']], self._df[['BETA', 'DIAMETER_IN']],
                                 ['next', 'next']).DT_handling()

    #def setup(self, component: Component):
    def setup(self):
        """
        Defines the needed OpenMDAO inputs for propulsion instantiation as done in :meth:`get_model`

        Use `add_inputs` and `declare_partials` methods of the provided `component`

        :param component:
        """



    def get_predict(self, beta, Dpro):
        """
        This method evaluates the decision tree

        :param beta
        :param Dpro
        :return: y_pred vector containing the discrete parameters beta and Dpro
        """
        y_pred = self._DT.predict([np.hstack((beta, Dpro/0.0253))])
        return y_pred