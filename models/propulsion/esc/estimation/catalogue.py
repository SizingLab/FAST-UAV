"""
Off-the-shelf ESC selection.
"""
import openmdao.api as om
from utils.catalogues.estimators import NearestNeighbor
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


PATH = "./data/catalogues/ESC/"
DF = pd.read_csv(PATH + "Non-Dominated-ESC.csv", sep=";")


@ValidityDomainChecker(
    {
        "data:propulsion:esc:power:max:estimated": (DF["Pmax_W"].min(), DF["Pmax_W"].max()),
        "data:propulsion:esc:voltage:estimated": (DF["Vmax_V"].min(), DF["Vmax_V"].max()),
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
        self.options.declare("use_catalogue", default=False, types=bool)
        Pmax_selection = "average"
        Vmax_selection = "average"
        self._clf = NearestNeighbor(
            df=DF, X_names=["Pmax_W", "Vmax_V"], crits=[Pmax_selection, Vmax_selection]
        )
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input("data:propulsion:esc:power:max:estimated", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:weights:esc:mass:estimated", val=np.nan, units="kg")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        # outputs: catalogue values if use_catalogues is True
        if self.options["use_catalogue"]:
            self.add_output("data:propulsion:esc:voltage:catalogue", units="V")
            self.add_output("data:propulsion:esc:power:max:catalogue", units="W")
            self.add_output("data:weights:esc:mass:catalogue", units="kg")
            # self.add_output('data:propulsion:esc:efficiency:catalogue', units=None)
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output("data:propulsion:esc:voltage", units="V")
        self.add_output("data:propulsion:esc:power:max", units="W")
        self.add_output("data:weights:esc:mass", units="kg")
        self.add_output("data:propulsion:esc:efficiency", units=None)

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:esc:voltage", "data:propulsion:esc:voltage:estimated", val=1.0,
        )
        self.declare_partials(
            "data:propulsion:esc:power:max", "data:propulsion:esc:power:max:estimated", val=1.0,
        )
        self.declare_partials(
            "data:weights:esc:mass", "data:weights:esc:mass:estimated", val=1.0,
        )
        self.declare_partials(
            "data:propulsion:esc:efficiency", "data:propulsion:esc:efficiency:estimated", val=1.0,
        )

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for ESC selection
            P_esc_opt = inputs["data:propulsion:esc:power:max:estimated"]
            V_esc_opt = inputs["data:propulsion:esc:voltage:estimated"]

            # Get closest product
            df_y = self._clf.predict([P_esc_opt, V_esc_opt])
            P_esc = df_y["Pmax_W"].iloc[0]  # [W] ESC power
            V_esc = df_y["Vmax_V"].iloc[0]  # [V] ESC voltage
            M_esc = df_y["Weight_g"].iloc[0] / 1000  # [kg] ESC mass

            # Outputs
            outputs["data:propulsion:esc:power:max"] = outputs[
                "data:propulsion:esc:power:max:catalogue"
            ] = P_esc
            outputs["data:propulsion:esc:voltage"] = outputs["data:propulsion:esc:voltage:catalogue"] = V_esc
            outputs["data:weights:esc:mass"] = outputs["data:weights:esc:mass:catalogue"] = M_esc
            outputs["data:propulsion:esc:efficiency"] = inputs["data:propulsion:esc:efficiency:estimated"]

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:esc:power:max"] = inputs["data:propulsion:esc:power:max:estimated"]
            outputs["data:propulsion:esc:voltage"] = inputs["data:propulsion:esc:voltage:estimated"]
            outputs["data:weights:esc:mass"] = inputs["data:weights:esc:mass:estimated"]
            outputs["data:propulsion:esc:efficiency"] = inputs["data:propulsion:esc:efficiency:estimated"]
