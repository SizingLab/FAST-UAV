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
    Select either custom or off-the-shelf motor.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        name_str = 'off_the_shelf' if self.options["use_catalogue"] else 'custom'
        self.add_subsystem(name_str, MotorDecisionTree(use_catalogue=self.options['use_catalogue']), promotes=["*"])


@ValidityDomainChecker(
    {
        'data:motor:torque:max:estimated': (DF['Tmax_Nm'].min(), DF['Tmax_Nm'].max()),
        'data:motor:torque:coefficient:estimated': (DF['Kt_Nm_A'].min(), DF['Kt_Nm_A'].max()),
    },
)
class MotorDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Get motor parameters from catalogue if asked by the user.
        Then, affect either catalogue values or estimated values to system parameters.
        """
        self.options.declare("use_catalogue", default=True, types=bool)
        Tmax_selection = 'next'
        Kt_selection = 'average'
        self._DT = DecisionTrees(DF[['Tmax_Nm', 'Kt_Nm_A']].values,
                                 DF[['Tnom_Nm', 'Kt_Nm_A', 'R_ohm', 'Tmax_Nm', 'Mass_g', 'Cf_Nm']].values,
                                 [Tmax_selection, Kt_selection]).DT_handling(dist=1000)

    def setup(self):
        # inputs: estimated values
        self.add_input('data:motor:torque:max:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:coefficient:estimated', val=np.nan, units='N*m/A')
        self.add_input('data:motor:torque:nominal:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:torque:friction:estimated', val=np.nan, units='N*m')
        self.add_input('data:motor:resistance:estimated', val=np.nan, units='V/A')
        self.add_input('data:motor:mass:estimated', val=np.nan, units='kg')
        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:motor:torque:max:catalogue', units='N*m')
            self.add_output('data:motor:torque:coefficient:catalogue', units='N*m/A')
            self.add_output('data:motor:torque:nominal:catalogue', units='N*m')
            self.add_output('data:motor:torque:friction:catalogue', units='N*m')
            self.add_output('data:motor:resistance:catalogue', units='V/A')
            self.add_output('data:motor:mass:catalogue', units='kg')
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:motor:torque:max', units='N*m')
        self.add_output('data:motor:torque:coefficient', units='N*m/A')
        self.add_output('data:motor:torque:nominal', units='N*m')
        self.add_output('data:motor:torque:friction', units='N*m')
        self.add_output('data:motor:resistance', units='V/A')
        self.add_output('data:motor:mass', units='kg')

    def setup_partials(self):
        self.declare_partials('data:motor:torque:max', 'data:motor:torque:max:estimated', val=1.)
        self.declare_partials('data:motor:torque:coefficient', 'data:motor:torque:coefficient:estimated', val=1.)
        self.declare_partials('data:motor:torque:nominal', 'data:motor:torque:nominal:estimated', val=1.)
        self.declare_partials('data:motor:torque:friction', 'data:motor:torque:friction:estimated', val=1.)
        self.declare_partials('data:motor:resistance', 'data:motor:resistance:estimated', val=1.)
        self.declare_partials('data:motor:mass', 'data:motor:mass:estimated', val=1.)

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:

            # Continuous parameters
            Tmot_max_est = inputs['data:motor:torque:max:estimated']
            Ktmot_est = inputs['data:motor:torque:coefficient:estimated']

            # Discrete parameters
            y_pred = self._DT.predict([np.hstack((Tmot_max_est, Ktmot_est))])
            Tmot = y_pred[0][0]  # nominal torque [N.m]
            Ktmot = y_pred[0][1]  # Kt constant [N.m./A]
            Rmot = y_pred[0][2]  # motor resistance [ohm]
            Tmot_max = y_pred[0][3]  # max motor torque [Nm]
            Mmot = y_pred[0][4] / 1000  # motor mass [kg]
            Tfmot = y_pred[0][5]  # friction torque [Nm]

            # Outputs
            outputs['data:motor:torque:max'] = outputs['data:motor:torque:max:catalogue'] = Tmot_max
            outputs['data:motor:torque:coefficient'] = outputs['data:motor:torque:coefficient:catalogue'] = Ktmot
            outputs['data:motor:torque:nominal'] = outputs['data:motor:torque:nominal:catalogue'] = Tmot
            outputs['data:motor:torque:friction'] = outputs['data:motor:torque:friction:catalogue'] = Tfmot
            outputs['data:motor:resistance'] = outputs['data:motor:resistance:catalogue'] = Rmot
            outputs['data:motor:mass'] = outputs['data:motor:mass:catalogue'] = Mmot

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:motor:torque:max'] = inputs['data:motor:torque:max:estimated']
            outputs['data:motor:torque:coefficient'] = inputs['data:motor:torque:coefficient:estimated']
            outputs['data:motor:torque:nominal'] = inputs['data:motor:torque:nominal:estimated']
            outputs['data:motor:torque:friction'] = inputs['data:motor:torque:friction:estimated']
            outputs['data:motor:resistance'] = inputs['data:motor:resistance:estimated']
            outputs['data:motor:mass'] = inputs['data:motor:mass:estimated']
