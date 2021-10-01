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
    Select either custom or off-the-shelf ESC.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        name_str = 'off_the_shelf' if self.options["use_catalogue"] else 'custom'
        self.add_subsystem(name_str, ESCDecisionTree(use_catalogue=self.options['use_catalogue']), promotes=["*"])


@ValidityDomainChecker(
    {
        'data:ESC:voltage:estimated': (DF['Vmax_V'].min(), DF['Vmax_V'].max()),
    },
)
class ESCDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Get ESC parameters from catalogue if asked by the user.
        Then, affect either catalogue values or estimated values to system parameters.
        """
        self.options.declare("use_catalogue", default=True, types=bool)
        Vmax_selection = 'next'
        self._DT = DecisionTrees(DF[['Vmax_V']],
                                 DF[['Pmax_W', 'Vmax_V', 'Weight_g']], [Vmax_selection]).DT_handling()

    def setup(self):
        # inputs: estimated values
        self.add_input('data:ESC:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:ESC:power:max:estimated', val=np.nan, units='W')
        self.add_input('data:ESC:mass:estimated', val=np.nan, units='kg')
        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:ESC:voltage:catalogue', units='V')
            self.add_output('data:ESC:power:max:catalogue', units='W')
            self.add_output('data:ESC:mass:catalogue', units='kg')
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:ESC:voltage', units='V')
        self.add_output('data:ESC:power:max', units='W')
        self.add_output('data:ESC:mass', units='kg')

    def setup_partials(self):
        self.declare_partials('data:ESC:voltage', 'data:ESC:voltage:estimated', val=1.)
        self.declare_partials('data:ESC:power:max', 'data:ESC:power:max:estimated', val=1.)
        self.declare_partials('data:ESC:mass', 'data:ESC:mass:estimated', val=1.)

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Continuous parameters
            V_esc = inputs['data:ESC:voltage:estimated']

            # Discrete parameters
            y_pred = self._DT.predict([np.hstack(V_esc)])  # TODO: add 2nd parameter to better discriminate.
            P_esc = y_pred[0][0]  # [W] ESC power
            V_esc = y_pred[0][1]  # [V] ESC voltage
            M_esc = y_pred[0][2] / 1000  # [kg] ESC mass

            # Outputs
            outputs['data:ESC:power:max'] = outputs['data:ESC:power:max:catalogue'] = P_esc
            outputs['data:ESC:voltage'] = outputs['data:ESC:voltage:catalogue'] = V_esc
            outputs['data:ESC:mass'] = outputs['data:ESC:mass:catalogue'] = M_esc

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:ESC:power:max'] = inputs['data:ESC:power:max:estimated']
            outputs['data:ESC:voltage'] = inputs['data:ESC:voltage:estimated']
            outputs['data:ESC:mass'] = inputs['data:ESC:mass:estimated']
