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
        "data:propulsion:motor:torque:max:estimated": (
            DF["Tmax_Nm"].min(),
            DF["Tmax_Nm"].max(),
        ),
        "data:propulsion:motor:speed:constant:estimated": (
            DF["Kv_SI"].min(),
            DF["Kv_SI"].max(),
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
        Kv_selection = "average"
        self._clf = NearestNeighbor(
            df=DF, X_names=["Tmax_Nm", "Kv_SI"], crits=[T_selection, Kv_selection]
        )
        self._clf.train()

    def setup(self):
        # inputs: estimated values
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input(
            "data:propulsion:motor:speed:constant:estimated", val=np.nan, units="rad/V/s"
        )
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:friction:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance:estimated", val=np.nan, units="V/A")
        self.add_input("data:weight:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        # outputs: catalogue values if off_the_shelf is True
        if self.options["off_the_shelf"]:
            self.add_output("data:propulsion:motor:torque:max:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:speed:constant:catalogue", units="rad/V/s")
            self.add_output("data:propulsion:motor:torque:nominal:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:torque:friction:catalogue", units="N*m")
            self.add_output("data:propulsion:motor:resistance:catalogue", units="V/A")
            self.add_output("data:weight:propulsion:motor:mass:catalogue", units="kg")
        # outputs: 'real' values (= estimated values if off_the_shelf is False, catalogue values else)
        self.add_output("data:propulsion:motor:torque:max", units="N*m")
        self.add_output("data:propulsion:motor:speed:constant", units="rad/V/s")
        self.add_output("data:propulsion:motor:torque:nominal", units="N*m")
        self.add_output("data:propulsion:motor:torque:friction", units="N*m")
        self.add_output("data:propulsion:motor:resistance", units="V/A")
        self.add_output("data:weight:propulsion:motor:mass", units="kg")

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:motor:torque:max",
            "data:propulsion:motor:torque:max:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:motor:speed:constant",
            "data:propulsion:motor:speed:constant:estimated",
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
            "data:weight:propulsion:motor:mass",
            "data:weight:propulsion:motor:mass:estimated",
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
            # Tnom_opt = inputs["data:propulsion:motor:torque:nominal:estimated"]
            Kv_opt = inputs["data:propulsion:motor:speed:constant:estimated"]

            # Get closest product
            df_y = self._clf.predict([Tmax_opt, Kv_opt])
            Tnom = df_y["Tnom_Nm"].iloc[0]  # nominal torque [N.m]
            Kv = df_y["Kv_SI"].iloc[0]  # speed constant [rad/V/s]
            R = df_y["R_ohm"].iloc[0]  # motor resistance [ohm]
            Tmax = df_y["Tmax_Nm"].iloc[0]  # max motor torque [Nm]
            m_mot = df_y["Mass_g"].iloc[0] / 1000  # motor mass [kg]
            Tf = df_y["Cf_Nm"].iloc[0]  # friction torque [Nm]

            # Outputs
            outputs["data:propulsion:motor:torque:max"] = outputs[
                "data:propulsion:motor:torque:max:catalogue"
            ] = Tmax
            outputs["data:propulsion:motor:speed:constant"] = outputs[
                "data:propulsion:motor:speed:constant:catalogue"
            ] = Kv
            outputs["data:propulsion:motor:torque:nominal"] = outputs[
                "data:propulsion:motor:torque:nominal:catalogue"
            ] = Tnom
            outputs["data:propulsion:motor:torque:friction"] = outputs[
                "data:propulsion:motor:torque:friction:catalogue"
            ] = Tf
            outputs["data:propulsion:motor:resistance"] = outputs[
                "data:propulsion:motor:resistance:catalogue"
            ] = R
            outputs["data:weight:propulsion:motor:mass"] = outputs["data:weight:propulsion:motor:mass:catalogue"] = m_mot

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:motor:torque:max"] = inputs[
                "data:propulsion:motor:torque:max:estimated"
            ]
            outputs["data:propulsion:motor:speed:constant"] = inputs[
                "data:propulsion:motor:speed:constant:estimated"
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
            outputs["data:weight:propulsion:motor:mass"] = inputs["data:weight:propulsion:motor:mass:estimated"]
