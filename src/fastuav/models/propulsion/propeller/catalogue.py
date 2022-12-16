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
    "",
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
            - If off_the_shelf is True, a propeller is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
        """
        self.options.declare("off_the_shelf", default=False, types=bool)
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
        self.add_input("data:propulsion:propeller:Ct:model:static:estimated",
                       shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:model:static:estimated",
                       shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:model:dynamic:estimated",
                       shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:model:dynamic:estimated",
                       shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:weight:propulsion:propeller:mass:estimated", val=np.nan, units="kg")

        # outputs: catalogue values if off_the_shelf is True
        if self.options["off_the_shelf"]:
            self.add_output("data:propulsion:propeller:beta:catalogue", units=None)
            self.add_output("data:propulsion:propeller:diameter:catalogue", units="m")
            self.add_output("data:propulsion:propeller:Ct:model:static:catalogue",
                            copy_shape="data:propulsion:propeller:Ct:model:static:estimated",
                            units=None)
            self.add_output("data:propulsion:propeller:Cp:model:static:catalogue",
                            copy_shape="data:propulsion:propeller:Cp:model:static:estimated",
                            units=None)
            self.add_output("data:propulsion:propeller:Ct:model:dynamic:catalogue",
                            copy_shape="data:propulsion:propeller:Ct:model:dynamic:estimated",
                            units=None)
            self.add_output("data:propulsion:propeller:Cp:model:dynamic:catalogue",
                            copy_shape="data:propulsion:propeller:Cp:model:dynamic:estimated",
                            units=None)
            self.add_output("data:weight:propulsion:propeller:mass:catalogue", units="kg")
            self.add_discrete_output("data:propulsion:propeller:product_name", val='')

        # outputs: estimated values if off_the_shelf is False, catalogue values else
        self.add_output("data:propulsion:propeller:beta", units=None)
        self.add_output("data:propulsion:propeller:diameter", units="m")
        self.add_output("data:propulsion:propeller:Ct:model:static",
                        copy_shape="data:propulsion:propeller:Ct:model:static:estimated",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:model:static",
                        copy_shape="data:propulsion:propeller:Cp:model:static:estimated",
                        units=None)
        self.add_output("data:propulsion:propeller:Ct:model:dynamic",
                        copy_shape="data:propulsion:propeller:Ct:model:dynamic:estimated",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:model:dynamic",
                        copy_shape="data:propulsion:propeller:Cp:model:dynamic:estimated",
                        units=None)
        self.add_output("data:weight:propulsion:propeller:mass", units="kg")

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
            "data:propulsion:propeller:Ct:model:static",
            "data:propulsion:propeller:Ct:model:static:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:model:static",
            "data:propulsion:propeller:Cp:model:static:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:propeller:Ct:model:dynamic",
            "data:propulsion:propeller:Ct:model:dynamic:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:model:dynamic",
            "data:propulsion:propeller:Cp:model:dynamic:estimated",
            val=1.0,
        )
        self.declare_partials(
            "data:weight:propulsion:propeller:mass",
            "data:weight:propulsion:propeller:mass:estimated",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        This method evaluates the decision tree and updates aero parameters according to the new geometry
        """

        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["off_the_shelf"]:
            # Definition parameters for propeller selection
            beta_opt = inputs["data:propulsion:propeller:beta:estimated"]
            Dpro_opt = inputs["data:propulsion:propeller:diameter:estimated"]

            # Get closest product
            df_y = self._clf.predict([beta_opt, Dpro_opt])
            beta = df_y["Pitch (-)"].iloc[0]  # [-] pitch-to-diameter ratio
            Dpro = df_y["Diameter (METERS)"].iloc[0]  # [m] diameter
            m_pro = df_y["Weight (KG)"].iloc[0]  # [kg] mass
            ct_static = df_y["Ct_static"].iloc[0]  # [-] static thrust coefficient
            cp_static = df_y["Cp_static"].iloc[0]  # [-] static power coefficient

            # FIXME: FAST-OAD variable viewer does not allow for str, will be converted to int
            product_name = str(df_y["Product Name"].iloc[0])  # product name

            # Outputs
            outputs["data:propulsion:propeller:beta"] = outputs[
                "data:propulsion:propeller:beta:catalogue"
            ] = beta
            outputs["data:propulsion:propeller:diameter"] = outputs[
                "data:propulsion:propeller:diameter:catalogue"
            ] = Dpro
            outputs["data:weight:propulsion:propeller:mass"] = outputs[
                "data:weight:propulsion:propeller:mass:catalogue"
            ] = m_pro
            outputs["data:propulsion:propeller:Ct:model:static"] = outputs[
                "data:propulsion:propeller:Ct:model:static:catalogue"
            ] = np.array([ct_static, 0])
            outputs["data:propulsion:propeller:Cp:model:static"] = outputs[
                "data:propulsion:propeller:Cp:model:static:catalogue"
            ] = np.array([cp_static, 0])
            outputs["data:propulsion:propeller:Ct:model:dynamic"] = outputs[
                "data:propulsion:propeller:Ct:model:dynamic:catalogue"
            ] = inputs["data:propulsion:propeller:Ct:model:dynamic:estimated"]  # No usage of catalogue values yet
            outputs["data:propulsion:propeller:Cp:model:dynamic"] = outputs[
                "data:propulsion:propeller:Cp:model:dynamic:catalogue"
            ] = inputs["data:propulsion:propeller:Cp:model:dynamic:estimated"]  # No usage of catalogue values yet
            discrete_outputs["data:propulsion:propeller:product_name"] = product_name

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:propulsion:propeller:beta"] = inputs[
                "data:propulsion:propeller:beta:estimated"
            ]
            outputs["data:propulsion:propeller:diameter"] = inputs[
                "data:propulsion:propeller:diameter:estimated"
            ]
            outputs["data:propulsion:propeller:Ct:model:static"] = inputs[
                "data:propulsion:propeller:Ct:model:static:estimated"
            ]
            outputs["data:propulsion:propeller:Cp:model:static"] = inputs[
                "data:propulsion:propeller:Cp:model:static:estimated"
            ]
            outputs["data:propulsion:propeller:Ct:model:dynamic"] = inputs[
                "data:propulsion:propeller:Ct:model:dynamic:estimated"
            ]
            outputs["data:propulsion:propeller:Cp:model:dynamic"] = inputs[
                "data:propulsion:propeller:Cp:model:dynamic:estimated"
            ]
            outputs["data:weight:propulsion:propeller:mass"] = inputs[
                "data:weight:propulsion:propeller:mass:estimated"
            ]
