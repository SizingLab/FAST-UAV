"""
ESC Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


PATH = './data/DecisionTrees/ESC/'
DF = pd.read_csv(PATH + 'Non-Dominated-ESC.csv', sep=';')


class ESCCatalogueSelection(om.Group):
    """
    Get ESC parameters from catalogue if asked by the user.
    Then, affect either catalogue values or estimated values to system parameters.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogue' = true)
        # And set parameters to 'estimated' or 'catalogue' for future use
        if self.options["use_catalogue"]:
            self.add_subsystem("getCatalogueValues", ESCDecisionTree(), promotes=["*"])
            self.add_subsystem("keepCatalogueValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])
        else:
            self.add_subsystem("keepEstimatedValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])


@ValidityDomainChecker(
    {
        'data:ESC:voltage:estimated': (DF['Vmax_V'].min(), DF['Vmax_V'].max()),
    },
)
class ESCDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the ESC Decision Tree
        """
        Vmax_selection = 'next'
        self._DT = DecisionTrees(DF[['Vmax_V']],
                                 DF[['Pmax_W', 'Vmax_V', 'Weight_g']], [Vmax_selection])\
            .DT_handling()

    def setup(self):
        self.add_input('data:ESC:voltage:estimated', val=np.nan, units='V')
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
        V_esc = inputs['data:ESC:voltage:estimated']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack(V_esc)])
        P_esc = y_pred[0][0]  # [W] ESC power
        V_esc = y_pred[0][1]  # [V] ESC voltage
        M_esc = y_pred[0][2] / 1000  # [kg] ESC mass

        # Outputs
        outputs['data:ESC:power:max:catalogue'] = P_esc
        outputs['data:ESC:voltage:catalogue'] = V_esc
        outputs['data:ESC:mass:catalogue'] = M_esc


class ValueSetter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self._str = ''

    def setup(self):
        if self.options["use_catalogue"]:  # discrete values from catalogues
            self._str = ':catalogue'
        else:  # estimated values
            self._str = ':estimated'
        self.add_input('data:ESC:voltage'+self._str, val=np.nan, units='V')
        self.add_input('data:ESC:power:max'+self._str, val=np.nan, units='W')
        self.add_input('data:ESC:mass'+self._str, val=np.nan, units='kg')
        # 'real' values
        self.add_output('data:ESC:voltage', units='V')
        self.add_output('data:ESC:power:max', units='W')
        self.add_output('data:ESC:mass', units='kg')

    def setup_partials(self):
        self.declare_partials('data:ESC:power:max', 'data:ESC:power:max'+self._str, val=1.)
        self.declare_partials('data:ESC:voltage', 'data:ESC:voltage'+self._str, val=1.)
        self.declare_partials('data:ESC:mass', 'data:ESC:mass'+self._str, val=1.)

    def compute(self, inputs, outputs):
        outputs['data:ESC:power:max'] = inputs['data:ESC:power:max'+self._str]
        outputs['data:ESC:voltage'] = inputs['data:ESC:voltage'+self._str]
        outputs['data:ESC:mass'] = inputs['data:ESC:mass'+self._str]
