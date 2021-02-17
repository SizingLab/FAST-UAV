"""
Motor Decision Tree based on provided catalogue
"""
import openmdao.api as om
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


PATH = './data/DecisionTrees/Motors/'
DF = pd.read_csv(PATH + 'Non-Dominated-Motors.csv', sep=';')


class MotorCatalogueSelection(om.Group):
    """
    Get motor parameters from catalogue if asked by the user.
    Then, affect either catalogue values or estimated values to system parameters.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogue' = true)
        # And set parameters to 'estimated' or 'catalogue' for future use
        if self.options["use_catalogue"]:
            self.add_subsystem("getCatalogueValues", MotorDecisionTree(), promotes=["*"])
            self.add_subsystem("keepCatalogueValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])
        else:
            self.add_subsystem("keepEstimatedValues", ValueSetter(use_catalogue=self.options['use_catalogue']),
                               promotes=["*"])


@ValidityDomainChecker(
    {
        'data:motor:torque:max:estimated': (DF['Tmax_Nm'].min(), DF['Tmax_Nm'].max()),
        'data:motor:torque_coefficient:estimated': (DF['Kt_Nm_A'].min(), DF['Kt_Nm_A'].max()),
    },
)
class MotorDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Propeller Decision Tree
        """
        Tmax_selection = 'next'
        Kt_selection = 'average'
        self._DT = DecisionTrees(DF[['Tmax_Nm', 'Kt_Nm_A']].values,
                                 DF[['Tnom_Nm', 'Kt_Nm_A', 'r_omn', 'Tmax_Nm', 'weight_g', 'Cf_Nm']].values,
                                 [Tmax_selection, Kt_selection]).DT_handling()

    def setup(self):
        self.add_input('data:motor:torque:max:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:torque_coefficient:estimated', val=np.nan, units='N*m/A')
        self.add_output('data:motor:torque:max:catalogue', units='N*m')
        self.add_output('data:motor:torque_coefficient:catalogue', units='N*m/A')
        self.add_output('data:motor:torque:nominal:catalogue', units='N*m')
        self.add_output('data:motor:torque:friction:catalogue', units='N*m')
        self.add_output('data:motor:resistance:catalogue', units='V/A')
        self.add_output('data:motor:mass:catalogue', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        Tmot_max = inputs['data:motor:torque:max:estimated']
        Ktmot = inputs['data:motor:torque_coefficient:estimated']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((Tmot_max, Ktmot))])
        Tmot = y_pred[0][0]  # nominal torque [N.m]
        Ktmot = y_pred[0][1]  # Kt constant [N.m./A]
        Rmot = y_pred[0][2]  # motor resistance [ohm]
        Tmot_max = y_pred[0][3]  # max motor torque [Nm]
        Mmot = y_pred[0][4] / 1000  # motor mass [kg]
        Tfmot = y_pred[0][5]  # friction torque [Nm]

        # Outputs
        outputs['data:motor:torque:max:catalogue'] = Tmot_max
        outputs['data:motor:torque_coefficient:catalogue'] = Ktmot
        outputs['data:motor:torque:nominal:catalogue'] = Tmot
        outputs['data:motor:torque:friction:catalogue'] = Tfmot
        outputs['data:motor:resistance:catalogue'] = Rmot
        outputs['data:motor:mass:catalogue'] = Mmot


class ValueSetter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self._str = ''

    def setup(self):
        if self.options["use_catalogue"]:  # discrete values from catalogues
            self._str = ':catalogue'
        else:  # estimated values
            self._str = ':estimated'
        self.add_input('data:motor:torque:max'+self._str, val=np.nan, units='N*m')
        self.add_input('data:motor:torque_coefficient'+self._str, val=np.nan, units='N*m/A')
        self.add_input('data:motor:torque:nominal'+self._str, val=np.nan, units='N*m')
        self.add_input('data:motor:torque:friction'+self._str, val=np.nan, units='N*m')
        self.add_input('data:motor:resistance'+self._str, val=np.nan, units='V/A')
        self.add_input('data:motor:mass'+self._str, val=np.nan, units='kg')
        # 'real' values
        self.add_output('data:motor:torque:max', units='N*m')
        self.add_output('data:motor:torque_coefficient', units='N*m/A')
        self.add_output('data:motor:torque:nominal', units='N*m')
        self.add_output('data:motor:torque:friction', units='N*m')
        self.add_output('data:motor:resistance', units='V/A')
        self.add_output('data:motor:mass', units='kg')

    def setup_partials(self):
        self.declare_partials('data:motor:torque:max', 'data:motor:torque:max'+self._str, val=1.)
        self.declare_partials('data:motor:torque_coefficient', 'data:motor:torque_coefficient'+self._str, val=1.)
        self.declare_partials('data:motor:torque:nominal', 'data:motor:torque:nominal'+self._str, val=1.)
        self.declare_partials('data:motor:torque:friction', 'data:motor:torque:friction'+self._str, val=1.)
        self.declare_partials('data:motor:resistance', 'data:motor:resistance'+self._str, val=1.)
        self.declare_partials('data:motor:mass', 'data:motor:mass'+self._str, val=1.)

    def compute(self, inputs, outputs):
        outputs['data:motor:torque:max'] = inputs['data:motor:torque:max'+self._str]
        outputs['data:motor:torque_coefficient'] = inputs['data:motor:torque_coefficient'+self._str]
        outputs['data:motor:torque:nominal'] = inputs['data:motor:torque:nominal'+self._str]
        outputs['data:motor:torque:friction'] = inputs['data:motor:torque:friction'+self._str]
        outputs['data:motor:resistance'] = inputs['data:motor:resistance'+self._str]
        outputs['data:motor:mass'] = inputs['data:motor:mass'+self._str]