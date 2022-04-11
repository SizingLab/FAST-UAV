"""
Definition parameters for the Electronic Speed Controller (ESC).
"""
import openmdao.api as om
import numpy as np
from fastuav.models.uncertainty.uncertainty import add_subsystem_with_deviation


class ESCDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the ESC.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the ESC are the voltage and the maximum Power.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "power",
            Power(),
            uncertain_outputs={"data:propulsion:esc:power:max:estimated": "W"},
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
        self.add_input("data:propulsion:esc:power:k", val=np.nan, units=None)
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_output("data:propulsion:esc:power:max:estimated", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_ESC = inputs["data:propulsion:esc:power:k"]
        P_el_to = inputs["data:propulsion:motor:power:takeoff"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        Umot_to = inputs["data:propulsion:motor:voltage:takeoff"]

        P_esc = (
            k_ESC * (P_el_to / Umot_to) * V_bat
        )  # [W] power electronic power max thrust

        outputs["data:propulsion:esc:power:max:estimated"] = P_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_ESC = inputs["data:propulsion:esc:power:k"]
        P_el_to = inputs["data:propulsion:motor:power:takeoff"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        Umot_to = inputs["data:propulsion:motor:voltage:takeoff"]

        partials["data:propulsion:esc:power:max:estimated", "data:propulsion:esc:power:k"] = (
            P_el_to * V_bat / Umot_to
        )

        partials["data:propulsion:esc:power:max:estimated", "data:propulsion:motor:power:takeoff"] = (
            k_ESC * V_bat / Umot_to
        )

        partials["data:propulsion:esc:power:max:estimated", "data:propulsion:battery:voltage"] = (
            k_ESC * P_el_to / Umot_to
        )

        partials["data:propulsion:esc:power:max:estimated", "data:propulsion:motor:voltage:takeoff"] = (
            -k_ESC * P_el_to * V_bat / Umot_to**2
        )


class Voltage(om.ExplicitComponent):
    """
    Computes ESC voltage
    """

    def setup(self):
        self.add_input("data:propulsion:esc:power:reference", val=3180.0, units="W")
        self.add_input("data:propulsion:esc:voltage:reference", val=44.4, units="V")
        self.add_input("data:propulsion:esc:power:max:estimated", val=np.nan, units="W")
        self.add_output("data:propulsion:esc:voltage:estimated", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Pesc_ref = inputs["data:propulsion:esc:power:reference"]
        Vesc_ref = inputs["data:propulsion:esc:voltage:reference"]
        P_esc = inputs["data:propulsion:esc:power:max:estimated"]

        V_esc = Vesc_ref * (P_esc / Pesc_ref) ** (1 / 3)  # [V] ESC voltage
        # V_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage

        outputs["data:propulsion:esc:voltage:estimated"] = V_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Pesc_ref = inputs["data:propulsion:esc:power:reference"]
        Vesc_ref = inputs["data:propulsion:esc:voltage:reference"]
        P_esc = inputs["data:propulsion:esc:power:max:estimated"]

        partials["data:propulsion:esc:voltage:estimated", "data:propulsion:esc:power:max:estimated"] = (
            (1 / 3) * Vesc_ref / Pesc_ref ** (1 / 3) / P_esc ** (2 / 3)
        )


# class Voltage(om.ExplicitComponent):
#     """
#     Computes ESC voltage
#     """
#
#     def setup(self):
#         self.add_input('data:propulsion:battery:voltage', val=np.nan, units='V')
#         self.add_output('data:propulsion:esc:voltage:estimated', units='V')
#
#     def setup_partials(self):
#         self.declare_partials('*', '*', method='exact')
#
#     def compute(self, inputs, outputs):
#         Vbat = inputs['data:propulsion:battery:voltage']
#
#         V_esc = Vbat  # [V] ESC voltage
#         #V_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage
#
#         outputs['data:propulsion:esc:voltage:estimated'] = V_esc
#
#     def compute_partials(self, inputs, partials, discrete_inputs=None):
#         partials['data:propulsion:esc:voltage:estimated',
#                  'data:propulsion:battery:voltage'] = 1
