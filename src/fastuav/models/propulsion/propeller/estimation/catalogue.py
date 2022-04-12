"""
Off-the-shelf propeller selection.
"""
import os.path as pth
import openmdao.api as om
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from fastuav.utils.catalogues.estimators import NearestNeighbor
import pandas as pd
import numpy as np


PATH = pth.join(
    pth.dirname(pth.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "data",
    "catalogues",
    "Propeller",
    "APC_propellers_MR.csv",
)
DF = pd.read_csv(PATH, sep=";")


# @ValidityDomainChecker(
#    {
#        'data:propulsion:propeller:beta:estimated': (DF['Pitch (-)'].min(), DF['Pitch (-)'].max()),
#        'data:propulsion:propeller:diameter:estimated': (DF['Diameter (METERS)'].min(), DF['Diameter (METERS)'].max()),
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
        self.options.declare("use_catalogue", default=False, types=bool)
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
        self.add_input("data:propulsion:propeller:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:diameter:estimated", val=np.nan, units="m")
        self.add_input("data:weights:propeller:mass:estimated", val=np.nan, units="kg")

        # outputs: catalogue values if use_catalogues is True
        if self.options["use_catalogue"]:
            self.add_output("data:propulsion:propeller:beta:catalogue", units=None)
            self.add_output("data:propulsion:propeller:diameter:catalogue", units="m")
            self.add_output("data:weights:propeller:mass:catalogue", units="kg")

        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output("data:propulsion:propeller:beta", units=None)
        self.add_output("data:propulsion:propeller:diameter", units="m")
        self.add_output("data:weights:propeller:mass", units="kg")

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:propeller:beta",
            "data:propulsion:propeller:beta:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:propeller:diameter",
            "data:propulsion:propeller:diameter:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:weights:propeller:mass",
            "data:weights:propeller:mass:estimated",
            val=1.0,
            method="fd",
        )

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree and updates aero parameters according to the new geometry
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for propeller selection
            beta_opt = inputs["data:propulsion:propeller:beta:estimated"]
            Dpro_opt = inputs["data:propulsion:propeller:diameter:estimated"]

            # Get closest product
            df_y = self._clf.predict([beta_opt, Dpro_opt])
            beta = df_y["Pitch (-)"].iloc[0]  # [-] beta
            Dpro = df_y["Diameter (METERS)"].iloc[0]  # [m] diameter
            Mpro = df_y["Weight (KG)"].iloc[0]  # [kg] mass

            # Outputs
            outputs["data:propulsion:propeller:beta"] = outputs[
                "data:propulsion:propeller:beta:catalogue"
            ] = beta
            outputs["data:propulsion:propeller:diameter"] = outputs[
                "data:propulsion:propeller:diameter:catalogue"
            ] = Dpro
            outputs["data:weights:propeller:mass"] = outputs[
                "data:weights:propeller:mass:catalogue"
            ] = Mpro

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:propeller:beta"] = inputs[
                "data:propulsion:propeller:beta:estimated"
            ]
            outputs["data:propulsion:propeller:diameter"] = inputs[
                "data:propulsion:propeller:diameter:estimated"
            ]
            outputs["data:weights:propeller:mass"] = inputs["data:weights:propeller:mass:estimated"]
