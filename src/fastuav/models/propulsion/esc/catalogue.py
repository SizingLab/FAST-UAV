"""
Off-the-shelf ESC selection.
"""
import os.path as pth
import openmdao.api as om
from fastuav.utils.catalogues.estimators import NearestNeighbor
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np


PATH = pth.join(
    pth.dirname(pth.abspath(__file__)),
    "",
    "..",
    "..",
    "..",
    "data",
    "catalogues",
    "ESC",
    "Non-Dominated-ESC.csv",
)
DF = pd.read_csv(PATH, sep=";")


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
            - If off_the_shelf is True, an ESC is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("off_the_shelf", default=False, types=bool)
        Pmax_selection = "next"
        Vmax_selection = "next"
        self._clf = NearestNeighbor(
            df=DF, X_names=["Pmax_W", "Vmax_V"], crits=[Pmax_selection, Vmax_selection]
        )
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input("data:propulsion:esc:power:max:estimated", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:weight:propulsion:esc:mass:estimated", val=np.nan, units="kg")
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        # outputs: catalogue values if off_the_shelf is True
        if self.options["off_the_shelf"]:
            self.add_output("data:propulsion:esc:power:max:catalogue", units="W")
            self.add_output("data:propulsion:esc:voltage:catalogue", units="V")
            self.add_output("data:weight:propulsion:esc:mass:catalogue", units="kg")
            # self.add_output('data:propulsion:esc:efficiency:catalogue', units=None)
        # outputs: 'real' values (= estimated values if off_the_shelf is False, catalogue values else)
        self.add_output("data:propulsion:esc:power:max", units="W")
        self.add_output("data:propulsion:esc:voltage", units="V")
        self.add_output("data:weight:propulsion:esc:mass", units="kg")
        self.add_output("data:propulsion:esc:efficiency", units=None)

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:esc:power:max",
            "data:propulsion:esc:power:max:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:esc:voltage",
            "data:propulsion:esc:voltage:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:weight:propulsion:esc:mass",
            "data:weight:propulsion:esc:mass:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:esc:efficiency",
            "data:propulsion:esc:efficiency:estimated",
            val=1.0,
        )

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["off_the_shelf"]:
            # Definition parameters for ESC selection
            P_esc_opt = inputs["data:propulsion:esc:power:max:estimated"]
            U_esc_opt = inputs["data:propulsion:esc:voltage:estimated"]

            # Get closest product
            df_y = self._clf.predict([P_esc_opt, U_esc_opt])
            P_esc = df_y["Pmax_W"].iloc[0]  # [W] ESC power
            U_esc = df_y["Vmax_V"].iloc[0]  # [V] ESC voltage
            m_esc = df_y["Weight_g"].iloc[0] / 1000  # [kg] ESC mass

            # Outputs
            outputs["data:propulsion:esc:power:max"] = outputs[
                "data:propulsion:esc:power:max:catalogue"
            ] = P_esc
            outputs["data:propulsion:esc:voltage"] = outputs[
                "data:propulsion:esc:voltage:catalogue"
            ] = U_esc
            outputs["data:weight:propulsion:esc:mass"] = outputs["data:weight:propulsion:esc:mass:catalogue"] = m_esc
            outputs["data:propulsion:esc:efficiency"] = inputs[
                "data:propulsion:esc:efficiency:estimated"
            ]

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:esc:power:max"] = inputs[
                "data:propulsion:esc:power:max:estimated"
            ]
            outputs["data:propulsion:esc:voltage"] = inputs["data:propulsion:esc:voltage:estimated"]
            outputs["data:weight:propulsion:esc:mass"] = inputs["data:weight:propulsion:esc:mass:estimated"]
            outputs["data:propulsion:esc:efficiency"] = inputs[
                "data:propulsion:esc:efficiency:estimated"
            ]
