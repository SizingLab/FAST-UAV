"""
Off-the-shelf motor selection.
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
    "Motors",
    "Non-Dominated-Motors.csv",
)
DF = pd.read_csv(PATH, sep=";")


@ValidityDomainChecker(
    {
        "data:propulsion:motor:torque:nominal:estimated": (
            DF["Tnom_Nm"].min(),
            DF["Tnom_Nm"].max(),
        ),
        "data:propulsion:motor:torque:coefficient:estimated": (
            DF["Kt_Nm_A"].min(),
            DF["Kt_Nm_A"].max(),
        ),
    },
)
class MotorCatalogueSelection(om.ExplicitComponent):
    def initialize(self):
        """
        Motor selection and component's parameters assignment:
            - If off_the_shelf is True, a motor is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("off_the_shelf", default=False, types=bool)
        T_selection = "next"
        Kt_selection = "average"
        self._clf = NearestNeighbor(
            df=DF, X_names=["Tmax_Nm", "Kt_Nm_A"], crits=[T_selection, Kt_selection]
        )
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input(
            "data:propulsion:motor:torque:coefficient:estimated", val=np.nan, units="N*m/A"
        )
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:friction:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance:estimated", val=np.nan, units="V/A")
        self.add_input("data:weights:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        # outputs: catalogue values if off_the_shelfs is True
        if self.options["off_the_shelf"]:
            self.add_output("data:propulsion:motor:torque:max:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:torque:coefficient:catalogue", units="N*m/A")
            self.add_output("data:propulsion:motor:torque:nominal:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:torque:friction:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:resistance:catalogue", units="V/A")
            self.add_output("data:weights:propulsion:motor:mass:catalogue", units="kg")
        # outputs: 'real' values (= estimated values if off_the_shelf is False, catalogue values else)
        self.add_output("data:propulsion:motor:torque:max", units="N*m")
        self.add_output("data:propulsion:motor:torque:coefficient", units="N*m/A")
        self.add_output("data:propulsion:motor:torque:nominal", units="N*m")
        self.add_output("data:propulsion:motor:torque:friction", units="N*m")
        self.add_output("data:propulsion:motor:resistance", units="V/A")
        self.add_output("data:weights:propulsion:motor:mass", units="kg")

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:motor:torque:max",
            "data:propulsion:motor:torque:max:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:motor:torque:coefficient",
            "data:propulsion:motor:torque:coefficient:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:motor:torque:nominal",
            "data:propulsion:motor:torque:nominal:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:motor:torque:friction",
            "data:propulsion:motor:torque:friction:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:motor:resistance",
            "data:propulsion:motor:resistance:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:weights:propulsion:motor:mass",
            "data:weights:propulsion:motor:mass:estimated",
            val=1.0,
        )

    def compute(self, inputs, outputs):
        """
        This method evaluates the decision tree
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["off_the_shelf"]:

            # Definition parameters for motor selection
            Tmax_opt = inputs["data:propulsion:motor:torque:max:estimated"]
            Tnom_opt = inputs["data:propulsion:motor:torque:nominal:estimated"]
            Ktmot_opt = inputs["data:propulsion:motor:torque:coefficient:estimated"]

            # Get closest product
            df_y = self._clf.predict([Tmax_opt, Ktmot_opt])
            Tnom = df_y["Tnom_Nm"].iloc[0]  # nominal torque [N.m]
            Ktmot = df_y["Kt_Nm_A"].iloc[0]  # Kt constant [N.m./A]
            Rmot = df_y["R_ohm"].iloc[0]  # motor resistance [ohm]
            Tmax = df_y["Tmax_Nm"].iloc[0]  # max motor torque [Nm]
            Mmot = df_y["Mass_g"].iloc[0] / 1000  # motor mass [kg]
            Tfmot = df_y["Cf_Nm"].iloc[0]  # friction torque [Nm]

            # Outputs
            outputs["data:propulsion:motor:torque:max"] = outputs[
                "data:propulsion:motor:torque:max:catalogue"
            ] = Tmax
            outputs["data:propulsion:motor:torque:coefficient"] = outputs[
                "data:propulsion:motor:torque:coefficient:catalogue"
            ] = Ktmot
            outputs["data:propulsion:motor:torque:nominal"] = outputs[
                "data:propulsion:motor:torque:nominal:catalogue"
            ] = Tnom
            outputs["data:propulsion:motor:torque:friction"] = outputs[
                "data:propulsion:motor:torque:friction:catalogue"
            ] = Tfmot
            outputs["data:propulsion:motor:resistance"] = outputs[
                "data:propulsion:motor:resistance:catalogue"
            ] = Rmot
            outputs["data:weights:propulsion:motor:mass"] = outputs["data:weights:propulsion:motor:mass:catalogue"] = Mmot

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:motor:torque:max"] = inputs[
                "data:propulsion:motor:torque:max:estimated"
            ]
            outputs["data:propulsion:motor:torque:coefficient"] = inputs[
                "data:propulsion:motor:torque:coefficient:estimated"
            ]
            outputs["data:propulsion:motor:torque:nominal"] = inputs[
                "data:propulsion:motor:torque:nominal:estimated"
            ]
            outputs["data:propulsion:motor:torque:friction"] = inputs[
                "data:propulsion:motor:torque:friction:estimated"
            ]
            outputs["data:propulsion:motor:resistance"] = inputs[
                "data:propulsion:motor:resistance:estimated"
            ]
            outputs["data:weights:propulsion:motor:mass"] = inputs["data:weights:propulsion:motor:mass:estimated"]
