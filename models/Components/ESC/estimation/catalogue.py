"""
Off-the-shelf ESC selection.
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


PATH = './data/DecisionTrees/ESC/'
DF = pd.read_csv(PATH + 'Non-Dominated-ESC.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:ESC:power:max:estimated': (DF['Pmax_W'].min(), DF['Pmax_W'].max()),
        'data:ESC:voltage:estimated': (DF['Vmax_V'].min(), DF['Vmax_V'].max()),
    },
)
class ESCCatalogueSelection(om.ExplicitComponent):

    def initialize(self):
        """
        ESC selection and component's parameters assignment:
            - If use_catalogue is True, an ESC is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("use_catalogue", default=True, types=bool)
        Pmax_selection = 'average'
        Vmax_selection = 'next'
        self._DT = DecisionTrees(DF[['Pmax_W', 'Vmax_V']].values,
                                 DF[['Pmax_W', 'Vmax_V', 'Weight_g']].values,
                                 [Pmax_selection, Vmax_selection]).DT_handling()

    def setup(self):
        # inputs: estimated values
        self.add_input('data:ESC:power:max:estimated', val=np.nan, units='W')
        self.add_input('data:ESC:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:ESC:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:ESC:efficiency:estimated', val=np.nan, units=None)
        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:ESC:voltage:catalogue', units='V')
            self.add_output('data:ESC:power:max:catalogue', units='W')
            self.add_output('data:ESC:mass:catalogue', units='kg')
            # self.add_output('data:ESC:efficiency:catalogue', units=None)
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:ESC:voltage', units='V')
        self.add_output('data:ESC:power:max', units='W')
        self.add_output('data:ESC:mass', units='kg')
        self.add_output('data:ESC:efficiency', units=None)

    def setup_partials(self):
        self.declare_partials('data:ESC:voltage', 'data:ESC:voltage:estimated', val=1.)
        self.declare_partials('data:ESC:power:max', 'data:ESC:power:max:estimated', val=1.)
        self.declare_partials('data:ESC:mass', 'data:ESC:mass:estimated', val=1.)
        self.declare_partials('data:ESC:efficiency', 'data:ESC:efficiency:estimated', val=1.)

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for ESC selection
            P_esc_opt = inputs['data:ESC:power:max:estimated']
            V_esc_opt = inputs['data:ESC:voltage:estimated']

            # Decision tree
            y_pred = self._DT.predict([np.hstack((P_esc_opt, V_esc_opt))])
            P_esc = y_pred[0][0]  # [W] ESC power
            V_esc = y_pred[0][1]  # [V] ESC voltage
            M_esc = y_pred[0][2] / 1000  # [kg] ESC mass

            # Outputs
            outputs['data:ESC:power:max'] = outputs['data:ESC:power:max:catalogue'] = P_esc
            outputs['data:ESC:voltage'] = outputs['data:ESC:voltage:catalogue'] = V_esc
            outputs['data:ESC:mass'] = outputs['data:ESC:mass:catalogue'] = M_esc
            outputs['data:ESC:efficiency'] = inputs['data:ESC:efficiency:estimated']

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:ESC:power:max'] = inputs['data:ESC:power:max:estimated']
            outputs['data:ESC:voltage'] = inputs['data:ESC:voltage:estimated']
            outputs['data:ESC:mass'] = inputs['data:ESC:mass:estimated']
            outputs['data:ESC:efficiency'] = inputs['data:ESC:efficiency:estimated']
