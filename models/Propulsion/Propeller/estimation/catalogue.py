"""
Off-the-shelf propeller selection.
"""
import openmdao.api as om
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from utils.catalogues.estimators import NearestNeighbor
import pandas as pd
import numpy as np


PATH = "./data/catalogues/Propeller/"
DF = pd.read_csv(PATH + "APC_propellers_MR.csv", sep=";")


# @ValidityDomainChecker(
#    {
#        'data:propeller:geometry:beta:estimated': (DF['Pitch (-)'].min(), DF['Pitch (-)'].max()),
#        'data:propeller:geometry:diameter:estimated': (DF['Diameter (METERS)'].min(), DF['Diameter (METERS)'].max()),
#    },
# )
class PropellerCatalogueSelection(om.ExplicitComponent):
    def initialize(self):
        """
        Propeller selection and component's parameters assignment:
            - If use_catalogue is True, a propeller is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("use_catalogue", default=True, types=bool)
        beta_selection = "average"
        Dpro_selection = "next"
        self._clf = NearestNeighbor(
            df=DF,
            X_names=["Pitch (-)", "Diameter (METERS)"],
            crits=[beta_selection, Dpro_selection],
        )
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input("data:propeller:geometry:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter:estimated", val=np.nan, units="m")
        self.add_input("data:propeller:mass:estimated", val=np.nan, units="kg")

        # outputs: catalogue values if use_catalogues is True
        if self.options["use_catalogue"]:
            self.add_output("data:propeller:geometry:beta:catalogue", units=None)
            self.add_output("data:propeller:geometry:diameter:catalogue", units="m")
            self.add_output("data:propeller:mass:catalogue", units="kg")

        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output("data:propeller:geometry:beta", units=None)
        self.add_output("data:propeller:geometry:diameter", units="m")
        self.add_output("data:propeller:mass", units="kg")

    def setup_partials(self):
        self.declare_partials(
            "data:propeller:geometry:beta",
            "data:propeller:geometry:beta:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propeller:geometry:diameter",
            "data:propeller:geometry:diameter:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propeller:mass", "data:propeller:mass:estimated", val=1.0, method="fd"
        )

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree and updates aero parameters according to the new geometry
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for propeller selection
            beta_opt = inputs["data:propeller:geometry:beta:estimated"]
            Dpro_opt = inputs["data:propeller:geometry:diameter:estimated"]

            # Get closest product
            df_y = self._clf.predict([beta_opt, Dpro_opt])
            beta = df_y["Pitch (-)"].iloc[0]  # [-] beta
            Dpro = df_y["Diameter (METERS)"].iloc[0]  # [m] diameter
            Mpro = df_y["Weight (KG)"].iloc[0]  # [kg] mass

            # Outputs
            outputs["data:propeller:geometry:beta"] = outputs[
                "data:propeller:geometry:beta:catalogue"
            ] = beta
            outputs["data:propeller:geometry:diameter"] = outputs[
                "data:propeller:geometry:diameter:catalogue"
            ] = Dpro
            outputs["data:propeller:mass"] = outputs[
                "data:propeller:mass:catalogue"
            ] = Mpro

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propeller:geometry:beta"] = inputs[
                "data:propeller:geometry:beta:estimated"
            ]
            outputs["data:propeller:geometry:diameter"] = inputs[
                "data:propeller:geometry:diameter:estimated"
            ]
            outputs["data:propeller:mass"] = inputs["data:propeller:mass:estimated"]
