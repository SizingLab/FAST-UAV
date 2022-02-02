"""
Off-the-shelf Battery selection.
"""
import openmdao.api as om
from utils.catalogues.estimators import DecisionTreeRgr
from fastoad.openmdao.validity_checker import ValidityDomainChecker
import pandas as pd
import numpy as np

# Database import
path = "./data/product_selection/Batteries/"
DF = pd.read_csv(path + "Non-Dominated-Augmented-Batteries.csv", sep=";")


@ValidityDomainChecker(
    {
        "data:battery:voltage:estimated": (
            DF["Voltage_V"].min(),
            DF["Voltage_V"].max(),
        ),
        "data:battery:capacity:estimated": (
            DF["Capacity_As"].min(),
            DF["Capacity_As"].max(),
        ),
    },
)
class BatteryCatalogueSelection(om.ExplicitComponent):
    """
    Battery selection and component's parameters assignment:
            - If use_catalogue is True, a battery is selected from the provided catalogue, according to the definition
               parameters. The component is then fully described by the manufacturer's data.
            - Otherwise, the previously estimated parameters are kept to describe the component.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        C_bat_selection = "average"
        V_bat_selection = "average"
        self._DT = DecisionTreeRgr(
            DF[["Voltage_V", "Capacity_As"]].values,
            DF[
                [
                    "Voltage_V",
                    "Capacity_As",
                    "Weight_kg",
                    "Volume_cm3",
                    "Imax [A]",
                    "n_series",
                    "n_parallel",
                ]
            ].values,
            [V_bat_selection, C_bat_selection],
        ).DT_handling(dist=1000000)

    def setup(self):
        # inputs: estimated values
        self.add_input(
            "data:battery:cell:number:series:estimated", val=np.nan, units=None
        )
        self.add_input(
            "data:battery:cell:number:parallel:estimated", val=np.nan, units=None
        )
        self.add_input("data:battery:cell:number:estimated", val=np.nan, units=None)
        self.add_input("data:battery:cell:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:battery:capacity:estimated", val=np.nan, units="A*s")
        self.add_input("data:battery:energy:estimated", val=np.nan, units="kJ")
        self.add_input("data:battery:current:max:estimated", val=np.nan, units="A")
        self.add_input("data:battery:mass:estimated", val=np.nan, units="kg")
        self.add_input("data:battery:volume:estimated", val=np.nan, units="cm**3")
        self.add_input("data:battery:DoD:max:estimated", val=np.nan, units=None)
        # outputs: catalogue values if use_catalogues is True
        if self.options["use_catalogue"]:
            self.add_output("data:battery:cell:number:series:catalogue", units=None)
            self.add_output("data:battery:cell:number:parallel:catalogue", units=None)
            self.add_output("data:battery:cell:number:catalogue", units=None)
            self.add_output("data:battery:cell:voltage:catalogue", units="V")
            self.add_output("data:battery:voltage:catalogue", units="V")
            self.add_output("data:battery:capacity:catalogue", units="A*s")
            self.add_output("data:battery:energy:catalogue", units="kJ")
            self.add_output("data:battery:current:max:catalogue", units="A")
            self.add_output("data:battery:mass:catalogue", units="kg")
            self.add_output("data:battery:volume:catalogue", units="cm**3")
            # self.add_output('data:battery:DoD:max:catalogue', units=None)
        # outputs: 'real' values (= estimated values if use_catalogue is False, catalogue values else)
        self.add_output("data:battery:cell:number:series", units=None)
        self.add_output("data:battery:cell:number:parallel", units=None)
        self.add_output("data:battery:cell:number", units=None)
        self.add_output("data:battery:cell:voltage", units="V")
        self.add_output("data:battery:voltage", units="V")
        self.add_output("data:battery:capacity", units="A*s")
        self.add_output("data:battery:energy", units="kJ")
        self.add_output("data:battery:current:max", units="A")
        self.add_output("data:battery:mass", units="kg")
        self.add_output("data:battery:volume", units="cm**3")
        self.add_output("data:battery:DoD:max", units=None)

    def setup_partials(self):
        self.declare_partials(
            "data:battery:voltage", "data:battery:voltage:estimated", val=1.0
        )
        self.declare_partials(
            "data:battery:capacity", "data:battery:capacity:estimated", val=1.0
        )
        self.declare_partials(
            "data:battery:energy", "data:battery:energy:estimated", val=1.0
        )
        self.declare_partials(
            "data:battery:current:max", "data:battery:current:max:estimated", val=1.0
        )
        self.declare_partials(
            "data:battery:mass", "data:battery:mass:estimated", val=1.0
        )
        self.declare_partials(
            "data:battery:DoD:max", "data:battery:DoD:max:estimated", val=1.0
        )

    def compute(self, inputs, outputs):
        # OFF-THE-SHELF COMPONENTS SELECTION
        if self.options["use_catalogue"]:
            # Definition parameters for battery selection
            V_bat_opt = inputs["data:battery:voltage:estimated"]  # [V]
            C_bat_opt = inputs["data:battery:capacity:estimated"]  # [A*s]

            # Decision Tree
            y_pred = self._DT.predict([np.hstack((V_bat_opt, C_bat_opt))])

            # Outputs
            V_bat = y_pred[0][0]  # battery pack voltage [V]
            C_bat = y_pred[0][1]  # battery pack capacity [A*s]
            M_bat = y_pred[0][2]  # battery pack weight [kg]
            Vol_bat = y_pred[0][3]  # battery pack volume [cm3]
            Imax = y_pred[0][4]  # max current [A]
            N_series = y_pred[0][
                5
            ]  # number of series connections to ensure sufficient voltage
            N_parallel = y_pred[0][6]  # number of parallel connections
            N_cell = N_series * N_parallel  # number of cells
            E_bat = C_bat * V_bat / 1000  # stored energy [kJ]

            outputs["data:battery:cell:number"] = outputs[
                "data:battery:cell:number:catalogue"
            ] = N_cell
            outputs["data:battery:cell:number:series"] = outputs[
                "data:battery:cell:number:series:catalogue"
            ] = N_series
            outputs["data:battery:cell:number:parallel"] = outputs[
                "data:battery:cell:number:parallel:catalogue"
            ] = N_parallel
            outputs["data:battery:voltage"] = outputs[
                "data:battery:voltage:catalogue"
            ] = V_bat
            outputs["data:battery:capacity"] = outputs[
                "data:battery:capacity:catalogue"
            ] = C_bat
            outputs["data:battery:energy"] = outputs[
                "data:battery:energy:catalogue"
            ] = E_bat
            outputs["data:battery:current:max"] = outputs[
                "data:battery:current:max:catalogue"
            ] = Imax
            outputs["data:battery:mass"] = outputs[
                "data:battery:mass:catalogue"
            ] = M_bat
            outputs["data:battery:cell:voltage"] = outputs[
                "data:battery:cell:voltage:catalogue"
            ] = (V_bat / N_series)
            outputs["data:battery:volume"] = outputs[
                "data:battery:volume:catalogue"
            ] = Vol_bat
            outputs["data:battery:DoD:max"] = inputs["data:battery:DoD:max:estimated"]

        # CUSTOM COMPONENTS (no change)
        else:
            outputs["data:battery:cell:number:series"] = inputs[
                "data:battery:cell:number:series:estimated"
            ]
            outputs["data:battery:cell:number:parallel"] = inputs[
                "data:battery:cell:number:parallel:estimated"
            ]
            outputs["data:battery:cell:number"] = inputs[
                "data:battery:cell:number:estimated"
            ]
            outputs["data:battery:voltage"] = inputs["data:battery:voltage:estimated"]
            outputs["data:battery:capacity"] = inputs["data:battery:capacity:estimated"]
            outputs["data:battery:energy"] = inputs["data:battery:energy:estimated"]
            outputs["data:battery:current:max"] = inputs[
                "data:battery:current:max:estimated"
            ]
            outputs["data:battery:mass"] = inputs["data:battery:mass:estimated"]
            outputs["data:battery:cell:voltage"] = inputs[
                "data:battery:cell:voltage:estimated"
            ]
            outputs["data:battery:volume"] = inputs["data:battery:volume:estimated"]
            outputs["data:battery:DoD:max"] = inputs["data:battery:DoD:max:estimated"]
