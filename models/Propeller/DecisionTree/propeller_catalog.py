"""
Propeller Decision Tree based on provided catalogue
"""
import openmdao.api as om
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from utils.DecisionTrees.predicted_values_DT import DecisionTrees
from models.Propeller.Aerodynamics.propeller_aero import ComputePropellerAeroMR
import pandas as pd
import numpy as np


PATH = './data/DecisionTrees/Propeller/'
DF = pd.read_csv(PATH + 'Non-Dominated-Propeller.csv', sep=';')


class PropellerCatalogueSelection(om.Group):
    """
    Get propeller parameters from catalogue if asked by the user.
    Then, affect either catalogue values or estimated values to system parameters.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogue' = true)
        # And set parameters to 'estimated' or 'catalogue' for future use
        if self.options["use_catalogue"]:
            self.add_subsystem("getCatalogueValues", PropellerDecisionTree(), promotes=["*"])
            self.add_subsystem("keepCatalogueValues", ValueSetter(use_catalogue=self.options['use_catalogue']), promotes=["*"])
        else:
            self.add_subsystem("keepEstimatedValues", ValueSetter(use_catalogue=self.options['use_catalogue']), promotes=["*"])


@ValidityDomainChecker(
    {
        'data:propeller:geometry:beta:estimated': (DF['BETA'].min(), DF['BETA'].max()),
        'data:propeller:geometry:diameter:estimated': (0.0254 * DF['DIAMETER_IN'].min(), 0.0254 * DF['DIAMETER_IN'].max()),
    },
)
class PropellerDecisionTree(om.ExplicitComponent):

    def initialize(self):
        """
        Creates and trains the Propeller Decision Tree
        """
        beta_selection = 'average'
        Dpro_selection = 'next'
        self._DT = DecisionTrees(DF[['BETA', 'DIAMETER_IN']], DF[['BETA', 'DIAMETER_IN']],
                                 [beta_selection, Dpro_selection]).DT_handling()

    def setup(self):
        self.add_input('data:propeller:geometry:beta:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter:estimated', val=np.nan, units='m')
        self.add_input('data:propeller:advance_ratio', val=np.nan, units=None)
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
        beta = inputs['data:propeller:geometry:beta:estimated']
        Dpro = inputs['data:propeller:geometry:diameter:estimated']
        J = inputs['data:propeller:advance_ratio']

        # Discrete parameters
        y_pred = self._DT.predict([np.hstack((beta, Dpro/0.0254))])
        beta = y_pred[0][0]  # [-] beta
        Dpro = y_pred[0][1] * 0.0254  # [m] diameter expressed in meters

        # Update Ct and Cp with new parameters
        C_t_sta, C_p_sta, C_t_dyn, C_p_dyn = ComputePropellerAeroMR.aero_coefficients(beta, J)

        # Outputs
        outputs['data:propeller:geometry:beta:catalogue'] = beta
        outputs['data:propeller:geometry:diameter:catalogue'] = Dpro
        outputs['data:propeller:aerodynamics:CT:static:catalogue'] = C_t_sta
        outputs['data:propeller:aerodynamics:CP:static:catalogue'] = C_p_sta
        outputs['data:propeller:aerodynamics:CT:dynamic:catalogue'] = C_t_dyn
        outputs['data:propeller:aerodynamics:CP:dynamic:catalogue'] = C_p_dyn


class ValueSetter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self._str = ''

    def setup(self):
        if self.options["use_catalogue"]:  # discrete values from catalogues
            self._str = ':catalogue'
        else: # estimated values
            self._str = ':estimated'
        self.add_input('data:propeller:geometry:beta'+self._str, val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter'+self._str, val=np.nan, units='m')
        self.add_input('data:propeller:aerodynamics:CT:static'+self._str, val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static'+self._str, val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:dynamic'+self._str, val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:dynamic'+self._str, val=np.nan, units=None)
        # 'real' values
        self.add_output('data:propeller:geometry:beta', units=None)
        self.add_output('data:propeller:geometry:diameter', units='m')
        self.add_output('data:propeller:aerodynamics:CT:static', units=None)
        self.add_output('data:propeller:aerodynamics:CP:static', units=None)
        self.add_output('data:propeller:aerodynamics:CT:dynamic', units=None)
        self.add_output('data:propeller:aerodynamics:CP:dynamic', units=None)

    def setup_partials(self):
        self.declare_partials('data:propeller:geometry:beta', 'data:propeller:geometry:beta'+self._str, val=1.)
        self.declare_partials('data:propeller:geometry:diameter', 'data:propeller:geometry:diameter'+self._str, val=1.)
        self.declare_partials('data:propeller:aerodynamics:CT:static', 'data:propeller:aerodynamics:CT:static'+self._str, val=1.)
        self.declare_partials('data:propeller:aerodynamics:CP:static', 'data:propeller:aerodynamics:CP:static'+self._str, val=1.)
        self.declare_partials('data:propeller:aerodynamics:CT:dynamic', 'data:propeller:aerodynamics:CT:dynamic'+self._str, val=1.)
        self.declare_partials('data:propeller:aerodynamics:CP:dynamic', 'data:propeller:aerodynamics:CP:dynamic'+self._str, val=1.)

    def compute(self, inputs, outputs):
        outputs['data:propeller:geometry:beta'] = inputs['data:propeller:geometry:beta'+self._str]
        outputs['data:propeller:geometry:diameter'] = inputs['data:propeller:geometry:diameter'+self._str]
        outputs['data:propeller:aerodynamics:CT:static'] = inputs['data:propeller:aerodynamics:CT:static'+self._str]
        outputs['data:propeller:aerodynamics:CP:static'] = inputs['data:propeller:aerodynamics:CP:static'+self._str]
        outputs['data:propeller:aerodynamics:CT:dynamic'] = inputs['data:propeller:aerodynamics:CT:dynamic'+self._str]
        outputs['data:propeller:aerodynamics:CP:dynamic'] = inputs['data:propeller:aerodynamics:CP:dynamic'+self._str]


# class ValueSetter(om.ExplicitComponent):
#     def initialize(self):
#         self.options.declare("use_catalogue", default=True, types=bool)
#
#     def setup(self):
#         if self.options["use_catalogue"]:  # discrete values from catalogues
#             self.add_input('data:propeller:geometry:beta:catalogue', val=np.nan, units=None)
#             self.add_input('data:propeller:geometry:diameter:catalogue', val=np.nan, units='m')
#             self.add_input('data:propeller:aerodynamics:CT:static:catalogue', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CP:static:catalogue', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CT:dynamic:catalogue', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CP:dynamic:catalogue', val=np.nan, units=None)
#         else:  # estimated values
#             self.add_input('data:propeller:geometry:beta:estimated', val=np.nan, units=None)
#             self.add_input('data:propeller:geometry:diameter:estimated', val=np.nan, units='m')
#             self.add_input('data:propeller:aerodynamics:CT:static:estimated', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CP:static:estimated', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CT:dynamic:estimated', val=np.nan, units=None)
#             self.add_input('data:propeller:aerodynamics:CP:dynamic:estimated', val=np.nan, units=None)
#         # real values
#         self.add_output('data:propeller:geometry:beta', units=None)
#         self.add_output('data:propeller:geometry:diameter', units='m')
#         self.add_output('data:propeller:aerodynamics:CT:static', units=None)
#         self.add_output('data:propeller:aerodynamics:CP:static', units=None)
#         self.add_output('data:propeller:aerodynamics:CT:dynamic', units=None)
#         self.add_output('data:propeller:aerodynamics:CP:dynamic', units=None)
#
#     def setup_partials(self):
#         if self.options["use_catalogue"]:
#             self.declare_partials('data:propeller:geometry:beta', 'data:propeller:geometry:beta:catalogue', val=1.)
#             self.declare_partials('data:propeller:geometry:diameter', 'data:propeller:geometry:diameter:catalogue', val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CT:static', 'data:propeller:aerodynamics:CT:static:catalogue', val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CP:static', 'data:propeller:aerodynamics:CP:static:catalogue', val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CT:dynamic', 'data:propeller:aerodynamics:CT:dynamic:catalogue', val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CP:dynamic', 'data:propeller:aerodynamics:CP:dynamic:catalogue', val=1.)
#         else:
#             self.declare_partials('data:propeller:geometry:beta', 'data:propeller:geometry:beta:estimated', val=1.)
#             self.declare_partials('data:propeller:geometry:diameter', 'data:propeller:geometry:diameter:estimated', val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CT:static', 'data:propeller:aerodynamics:CT:static:estimated',
#                                   val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CP:static', 'data:propeller:aerodynamics:CP:static:estimated',
#                                   val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CT:dynamic', 'data:propeller:aerodynamics:CT:dynamic:estimated',
#                                   val=1.)
#             self.declare_partials('data:propeller:aerodynamics:CP:dynamic', 'data:propeller:aerodynamics:CP:dynamic:estimated',
#                                   val=1.)
#
#     def compute(self, inputs, outputs):
#         if self.options["use_catalogue"]:
#             outputs['data:propeller:geometry:beta'] = inputs['data:propeller:geometry:beta:catalogue']
#             outputs['data:propeller:geometry:diameter'] = inputs['data:propeller:geometry:diameter:catalogue']
#             outputs['data:propeller:aerodynamics:CT:static'] = inputs[
#                 'data:propeller:aerodynamics:CT:static:catalogue']
#             outputs['data:propeller:aerodynamics:CP:static'] = inputs[
#                 'data:propeller:aerodynamics:CP:static:catalogue']
#             outputs['data:propeller:aerodynamics:CT:dynamic'] = inputs[
#                 'data:propeller:aerodynamics:CT:dynamic:catalogue']
#             outputs['data:propeller:aerodynamics:CP:dynamic'] = inputs[
#                 'data:propeller:aerodynamics:CP:dynamic:catalogue']
#         else:
#             outputs['data:propeller:geometry:beta'] = inputs['data:propeller:geometry:beta:estimated']
#             outputs['data:propeller:geometry:diameter'] = inputs['data:propeller:geometry:diameter:estimated']
#             outputs['data:propeller:aerodynamics:CT:static'] = inputs['data:propeller:aerodynamics:CT:static:estimated']
#             outputs['data:propeller:aerodynamics:CP:static'] = inputs['data:propeller:aerodynamics:CP:static:estimated']
#             outputs['data:propeller:aerodynamics:CP:dynamic'] = inputs['data:propeller:aerodynamics:CP:dynamic:estimated']
#             outputs['data:propeller:aerodynamics:CT:dynamic'] = inputs['data:propeller:aerodynamics:CT:dynamic:estimated']

