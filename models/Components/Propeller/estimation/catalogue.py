"""
Off-the-shelf propeller selection.
"""
import openmdao.api as om
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from utils.catalogues.estimators import NearestNeighbor
from models.Components.Propeller.estimation.models import PropellerAerodynamicsModel
from models.Uncertainty.uncertainty import add_subsystem_with_deviation
import pandas as pd
import numpy as np


PATH = './data/catalogues/Propeller/'
DF = pd.read_csv(PATH + 'APC_propellers_MR.csv', sep=';')


#@ValidityDomainChecker(
#    {
#        'data:propeller:geometry:beta:estimated': (DF['Pitch (-)'].min(), DF['Pitch (-)'].max()),
#        'data:propeller:geometry:diameter:estimated': (DF['Diameter (METERS)'].min(), DF['Diameter (METERS)'].max()),
#    },
#)
class PropellerCatalogueSelection(om.ExplicitComponent):

    def initialize(self):
        """
        Propeller selection and component's parameters assignment:
            - If use_catalogue is True, a propeller is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("use_catalogue", default=True, types=bool)
        beta_selection = 'average'
        Dpro_selection = 'next'
        self._clf = NearestNeighbor(df=DF, X_names=['Pitch (-)', 'Diameter (METERS)'],
                                    crits=[beta_selection, Dpro_selection])
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input('data:propeller:geometry:beta:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:geometry:diameter:estimated', val=np.nan, units='m')
        self.add_input('data:propeller:mass:estimated', val=np.nan, units='kg')
        self.add_input('data:propeller:aerodynamics:CT:static:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:static:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:axial:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:axial:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CT:incidence:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:aerodynamics:CP:incidence:estimated', val=np.nan, units=None)
        self.add_input('data:propeller:advance_ratio:climb', val=np.nan, units=None)
        self.add_input('data:propeller:advance_ratio:forward', val=np.nan, units=None)
        self.add_input('mission:sizing_mission:forward:angle', val=np.nan, units='rad')

        # outputs: catalogue values if use_catalogues is True
        if self.options['use_catalogue']:
            self.add_output('data:propeller:geometry:beta:catalogue', units=None)
            self.add_output('data:propeller:geometry:diameter:catalogue', units='m')
            self.add_output('data:propeller:mass:catalogue', units='kg')
            self.add_output('data:propeller:aerodynamics:CT:static:catalogue', units=None)
            self.add_output('data:propeller:aerodynamics:CP:static:catalogue', units=None)
            self.add_output('data:propeller:aerodynamics:CT:axial:catalogue', units=None)
            self.add_output('data:propeller:aerodynamics:CP:axial:catalogue', units=None)
            self.add_output('data:propeller:aerodynamics:CT:incidence:catalogue', units=None)
            self.add_output('data:propeller:aerodynamics:CP:incidence:catalogue', units=None)

        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output('data:propeller:geometry:beta', units=None)
        self.add_output('data:propeller:geometry:diameter', units='m')
        self.add_output('data:propeller:mass', units='kg')
        self.add_output('data:propeller:aerodynamics:CT:static', units=None)
        self.add_output('data:propeller:aerodynamics:CP:static', units=None)
        self.add_output('data:propeller:aerodynamics:CT:axial', units=None)
        self.add_output('data:propeller:aerodynamics:CP:axial', units=None)
        self.add_output('data:propeller:aerodynamics:CT:incidence', units=None)
        self.add_output('data:propeller:aerodynamics:CP:incidence', units=None)

    def setup_partials(self):
        self.declare_partials('data:propeller:geometry:beta', 'data:propeller:geometry:beta:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:geometry:diameter', 'data:propeller:geometry:diameter:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:mass', 'data:propeller:mass:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CT:static', 'data:propeller:aerodynamics:CT:static:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CP:static', 'data:propeller:aerodynamics:CP:static:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CT:axial', 'data:propeller:aerodynamics:CT:axial:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CP:axial', 'data:propeller:aerodynamics:CP:axial:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CT:incidence', 'data:propeller:aerodynamics:CT:incidence:estimated', val=1., method='fd')
        self.declare_partials('data:propeller:aerodynamics:CP:incidence', 'data:propeller:aerodynamics:CP:incidence:estimated', val=1., method='fd')

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree and updates aero parameters according to the new geometry
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for propeller selection
            beta_opt = inputs['data:propeller:geometry:beta:estimated']
            Dpro_opt = inputs['data:propeller:geometry:diameter:estimated']

            # Get closest product
            df_y = self._clf.predict([beta_opt, Dpro_opt])
            beta = df_y['Pitch (-)'].iloc[0]  # [-] beta
            Dpro = df_y['Diameter (METERS)'].iloc[0]  # [m] diameter
            Mpro = df_y['Weight (KG)'].iloc[0]  # [kg] mass

            # Update Ct and Cp with new parameters
            J_climb = inputs['data:propeller:advance_ratio:climb']
            J_forward = inputs['data:propeller:advance_ratio:forward']
            alpha = inputs['mission:sizing_mission:forward:angle']
            C_t_sta, C_p_sta = PropellerAerodynamicsModel.aero_coefficients_static(beta)
            C_t_axial, C_p_axial = PropellerAerodynamicsModel.aero_coefficients_axial(beta, J_climb)
            C_t_inc, C_p_inc = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J_forward, alpha)

            # Outputs
            outputs['data:propeller:geometry:beta'] = outputs['data:propeller:geometry:beta:catalogue'] = beta
            outputs['data:propeller:geometry:diameter'] = outputs['data:propeller:geometry:diameter:catalogue'] = Dpro
            outputs['data:propeller:mass'] = outputs['data:propeller:mass:catalogue'] = Mpro
            outputs['data:propeller:aerodynamics:CT:static'] = outputs['data:propeller:aerodynamics:CT:static:catalogue'] = C_t_sta
            outputs['data:propeller:aerodynamics:CP:static'] = outputs['data:propeller:aerodynamics:CP:static:catalogue'] = C_p_sta
            outputs['data:propeller:aerodynamics:CT:axial'] = outputs['data:propeller:aerodynamics:CT:axial:catalogue'] = C_t_axial
            outputs['data:propeller:aerodynamics:CP:axial'] = outputs['data:propeller:aerodynamics:CP:axial:catalogue'] = C_p_axial
            outputs['data:propeller:aerodynamics:CT:incidence'] = outputs['data:propeller:aerodynamics:CT:incidence:catalogue'] = C_t_inc
            outputs['data:propeller:aerodynamics:CP:incidence'] = outputs['data:propeller:aerodynamics:CP:incidence:catalogue'] = C_p_inc

        # CUSTOM COMPONENTS (no change)
        else:
            outputs['data:propeller:geometry:beta'] = inputs['data:propeller:geometry:beta:estimated']
            outputs['data:propeller:geometry:diameter'] = inputs['data:propeller:geometry:diameter:estimated']
            outputs['data:propeller:mass'] = inputs['data:propeller:mass:estimated']
            outputs['data:propeller:aerodynamics:CT:static'] = inputs['data:propeller:aerodynamics:CT:static:estimated']
            outputs['data:propeller:aerodynamics:CP:static'] = inputs['data:propeller:aerodynamics:CP:static:estimated']
            outputs['data:propeller:aerodynamics:CT:axial'] = inputs['data:propeller:aerodynamics:CT:axial:estimated']
            outputs['data:propeller:aerodynamics:CP:axial'] = inputs['data:propeller:aerodynamics:CP:axial:estimated']
            outputs['data:propeller:aerodynamics:CT:incidence'] = inputs['data:propeller:aerodynamics:CT:incidence:estimated']
            outputs['data:propeller:aerodynamics:CP:incidence'] = inputs['data:propeller:aerodynamics:CP:incidence:estimated']
