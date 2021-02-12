"""
Propeller Decision Tree based on provided catalogue
"""
import openmdao.api as om
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from models.Propeller.Aerodynamics.propeller_aero import ComputePropellerAeroMR
import pandas as pd
import numpy as np


path = './data/DecisionTrees/Propeller/'
df = pd.read_csv(path + 'Non-Dominated-Propeller.csv', sep=';')


@ValidityDomainChecker(
    {
        'data:propeller:geometry:beta': (df['BETA'].min(), df['BETA'].max()),
        'data:propeller:geometry:diameter': (0.0254 * df['DIAMETER_IN'].min(), 0.0254 * df['DIAMETER_IN'].max()),
    },
)
class PropellerDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Propeller Decision Tree
        """
        beta_selection = 'average'
        Dpro_selection = 'next'
        self._DT = DecisionTrees(df[['BETA', 'DIAMETER_IN']], df[['BETA', 'DIAMETER_IN']],
                                 [beta_selection, Dpro_selection]).DT_handling()

    def setup(self):
        self.add_input('data:propeller:geometry:beta', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter', val=np.nan, units='m')
        self.add_input('data:propeller:settings:advance_ratio', val=np.nan, units=None)
        self.add_output('data:propeller:geometry:beta:catalogue', units=None)
        self.add_output('data:propeller:geometry:diameter:catalogue', units='m')
        self.add_output('data:propeller:aerodynamics:CT:static:catalogue', units=None)
        self.add_output('data:propeller:aerodynamics:CP:static:catalogue', units=None)
        self.add_output('data:propeller:aerodynamics:CT:dynamic:catalogue', units=None)
        self.add_output('data:propeller:aerodynamics:CP:dynamic:catalogue', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """
        # Continuous parameters
        beta = inputs['data:propeller:geometry:beta']
        Dpro = inputs['data:propeller:geometry:diameter']
        J = inputs['data:propeller:settings:advance_ratio']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((beta, Dpro/0.0254))])
        beta = y_pred[0][0]  # [-] beta
        Dpro = y_pred[0][1] * 0.0254  # [m] diameter expressed in meters

        # Update Ct and Cp with new parameters
        C_t_sta, C_p_sta, C_t_dyn, C_p_dyn = ComputePropellerAeroMR.aero_coefficients(beta, J)
        #C_t_sta = 4.27e-02 + 1.44e-01 * beta  # Thrust coef with T=C_T.rho.n^2.D^4
        #C_p_sta = -1.48e-03 + 9.72e-02 * beta  # Power coef with P=C_p.rho.n^3.D^5
        #C_t_dyn = 0.02791 - 0.06543 * J + 0.11867 * beta + 0.27334 * beta ** 2 - 0.28852 * beta ** 3 + 0.02104 * J ** 3 \
        #          - 0.23504 * J ** 2 + 0.18677 * beta * J ** 2  # thrust coef for APC props in dynamics
        #C_p_dyn = 0.01813 - 0.06218 * beta + 0.00343 * J + 0.35712 * beta ** 2 - 0.23774 * beta ** 3 + 0.07549 * beta \
        #          * J - 0.1235 * J ** 2  # power coef for APC props in dynamics


        # Outputs
        outputs['data:propeller:geometry:beta:catalogue'] = beta
        outputs['data:propeller:geometry:diameter:catalogue'] = Dpro
        outputs['data:propeller:aerodynamics:CT:static:catalogue'] = C_t_sta
        outputs['data:propeller:aerodynamics:CP:static:catalogue'] = C_p_sta
        outputs['data:propeller:aerodynamics:CT:dynamic:catalogue'] = C_t_dyn
        outputs['data:propeller:aerodynamics:CP:dynamic:catalogue'] = C_p_dyn


# class PropellerCatalogue(om.Group):
#
#     def setup(self):
#         self.add_subsystem("decision_tree", PropellerDecisionTree(), promotes=["*"])
#         self.add_subsystem("override_var", PropellerVariablesOverride(), promotes=["*"])
#

# class PropellerVariablesOverride(om.ExplicitComponent):
#
#     def setup(self):
#         self.add_input('catalogue:propeller:beta', units=None)
#         self.add_input('catalogue:propeller:diameter', units='m')
#         self.add_output('data:propeller:geometry:beta', val=np.nan, units=None)
#         self.add_output('data:propeller:geometry:diameter', val=np.nan, units='m')
#
#     def setup_partials(self):
#         # Finite difference all partials.
#         self.declare_partials('*', '*', method='fd')
#
#     def compute(self, inputs, outputs):
#         # Inputs
#         beta = inputs['catalogue:propeller:beta']
#         Dpro = inputs['catalogue:propeller:diameter']
#
#         # Outputs
#         outputs['data:propeller:geometry:beta'] = beta
#         outputs['data:propeller:geometry:diameter'] = Dpro