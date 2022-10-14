"""
Definition parameters for the Electronic Speed Controller (ESC).
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class ESCDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the ESC.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the ESC are the voltage and the apparent power.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "power",
            Power(),
            uncertain_outputs={"data:propulsion:esc:power:estimated": "W"},
        )

        add_subsystem_with_deviation(
            self,
            "voltage",
            Voltage(),
            uncertain_outputs={"data:propulsion:esc:voltage:estimated": "V"},
        )


class Power(om.ExplicitComponent):
    """
    Computes ESC maximum power.
    """

    def setup(self):
        self.add_input("data:propulsion:esc:power:k", val=1.0, units=None)
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:estimated", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_pesc = inputs["data:propulsion:esc:power:k"]
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        U_bat = inputs["data:propulsion:battery:voltage"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]

        P_esc = k_pesc * (P_mot_to / U_mot_to) * U_bat  # [W] power electronic power max thrust

        outputs["data:propulsion:esc:power:estimated"] = P_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_pesc = inputs["data:propulsion:esc:power:k"]
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        U_bat = inputs["data:propulsion:battery:voltage"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]

        partials["data:propulsion:esc:power:estimated", "data:propulsion:esc:power:k"] = (
            P_mot_to * U_bat / U_mot_to
        )

        partials[
            "data:propulsion:esc:power:estimated", "data:propulsion:motor:power:takeoff"
        ] = (k_pesc * U_bat / U_mot_to)

        partials["data:propulsion:esc:power:estimated", "data:propulsion:battery:voltage"] = (
            k_pesc * P_mot_to / U_mot_to
        )

        partials[
            "data:propulsion:esc:power:estimated", "data:propulsion:motor:voltage:takeoff"
        ] = (-k_pesc * P_mot_to * U_bat / U_mot_to**2)


class Voltage(om.ExplicitComponent):
    """
    Computes ESC voltage
    """

    def setup(self):
        self.add_input("data:propulsion:esc:voltage:k", val=1.0, units=None)
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:voltage:estimated", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_vesc = inputs["data:propulsion:esc:voltage:k"]
        U_bat = inputs["data:propulsion:battery:voltage"]

        U_esc = k_vesc * U_bat  # [V] ESC voltage rating

        outputs["data:propulsion:esc:voltage:estimated"] = U_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_vesc = inputs["data:propulsion:esc:voltage:k"]
        U_bat = inputs["data:propulsion:battery:voltage"]

        partials[
            "data:propulsion:esc:voltage:estimated", "data:propulsion:esc:voltage:k"
        ] = U_bat

        partials[
            "data:propulsion:esc:voltage:estimated", "data:propulsion:battery:voltage"
        ] = k_vesc


class Voltage_2(om.ExplicitComponent):
    """
    Computes ESC voltage
    """

    def setup(self):
        self.add_input("data:propulsion:esc:power:reference", val=3180.0, units="W")
        self.add_input("data:propulsion:esc:voltage:reference", val=44.4, units="V")
        self.add_input("data:propulsion:esc:power:estimated", val=np.nan, units="W")
        self.add_output("data:propulsion:esc:voltage:estimated", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        P_esc_ref = inputs["data:propulsion:esc:power:reference"]
        U_esc_ref = inputs["data:propulsion:esc:voltage:reference"]
        P_esc = inputs["data:propulsion:esc:power:estimated"]

        U_esc = U_esc_ref * (P_esc / P_esc_ref) ** (1 / 3)  # [V] ESC voltage
        # U_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage

        outputs["data:propulsion:esc:voltage:estimated"] = U_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        P_esc_ref = inputs["data:propulsion:esc:power:reference"]
        U_esc_ref = inputs["data:propulsion:esc:voltage:reference"]
        P_esc = inputs["data:propulsion:esc:power:estimated"]

        partials[
            "data:propulsion:esc:voltage:estimated", "data:propulsion:esc:power:estimated"
        ] = (1 / 3) * U_esc_ref / P_esc_ref ** (1 / 3) / P_esc ** (2 / 3)
        partials[
            "data:propulsion:esc:voltage:estimated", "data:propulsion:esc:power:reference"
        ] = - (1 / 3) * U_esc_ref * P_esc ** (1 / 3) / P_esc_ref ** (4 / 3)
        partials[
            "data:propulsion:esc:voltage:estimated", "data:propulsion:esc:voltage:reference"
        ] = (P_esc / P_esc_ref) ** (1 / 3)
