"""
Definition parameters for the Electronic Speed Controller (ESC).
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import add_subsystem_with_deviation


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
            uncertain_outputs={"data:ESC:power:max:estimated": "W"},
        )

        add_subsystem_with_deviation(
            self,
            "voltage",
            Voltage(),
            uncertain_outputs={"data:ESC:voltage:estimated": "V"},
        )


class Power(om.ExplicitComponent):
    """
    Computes ESC maximum power.
    """

    def setup(self):
        self.add_input("data:ESC:power:k", val=np.nan, units=None)
        self.add_input("data:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_output("data:ESC:power:max:estimated", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_ESC = inputs["data:ESC:power:k"]
        P_el_to = inputs["data:motor:power:takeoff"]
        V_bat = inputs["data:battery:voltage"]
        Umot_to = inputs["data:motor:voltage:takeoff"]

        P_esc = (
            k_ESC * (P_el_to / Umot_to) * V_bat
        )  # [W] power electronic power max thrust

        outputs["data:ESC:power:max:estimated"] = P_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_ESC = inputs["data:ESC:power:k"]
        P_el_to = inputs["data:motor:power:takeoff"]
        V_bat = inputs["data:battery:voltage"]
        Umot_to = inputs["data:motor:voltage:takeoff"]

        partials["data:ESC:power:max:estimated", "data:ESC:power:k"] = (
            P_el_to * V_bat / Umot_to
        )

        partials["data:ESC:power:max:estimated", "data:motor:power:takeoff"] = (
            k_ESC * V_bat / Umot_to
        )

        partials["data:ESC:power:max:estimated", "data:battery:voltage"] = (
            k_ESC * P_el_to / Umot_to
        )

        partials["data:ESC:power:max:estimated", "data:motor:voltage:takeoff"] = (
            -k_ESC * P_el_to * V_bat / Umot_to**2
        )


class Voltage(om.ExplicitComponent):
    """
    Computes ESC voltage
    """

    def setup(self):
        self.add_input("data:ESC:reference:power", val=3180.0, units="W")
        self.add_input("data:ESC:reference:voltage", val=44.4, units="V")
        self.add_input("data:ESC:power:max:estimated", val=np.nan, units="W")
        self.add_output("data:ESC:voltage:estimated", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Pesc_ref = inputs["data:ESC:reference:power"]
        Vesc_ref = inputs["data:ESC:reference:voltage"]
        P_esc = inputs["data:ESC:power:max:estimated"]

        V_esc = Vesc_ref * (P_esc / Pesc_ref) ** (1 / 3)  # [V] ESC voltage
        # V_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage

        outputs["data:ESC:voltage:estimated"] = V_esc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Pesc_ref = inputs["data:ESC:reference:power"]
        Vesc_ref = inputs["data:ESC:reference:voltage"]
        P_esc = inputs["data:ESC:power:max:estimated"]

        partials["data:ESC:voltage:estimated", "data:ESC:power:max:estimated"] = (
            (1 / 3) * Vesc_ref / Pesc_ref ** (1 / 3) / P_esc ** (2 / 3)
        )


# class Voltage(om.ExplicitComponent):
#     """
#     Computes ESC voltage
#     """
#
#     def setup(self):
#         self.add_input('data:battery:voltage', val=np.nan, units='V')
#         self.add_output('data:ESC:voltage:estimated', units='V')
#
#     def setup_partials(self):
#         self.declare_partials('*', '*', method='exact')
#
#     def compute(self, inputs, outputs):
#         Vbat = inputs['data:battery:voltage']
#
#         V_esc = Vbat  # [V] ESC voltage
#         #V_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage
#
#         outputs['data:ESC:voltage:estimated'] = V_esc
#
#     def compute_partials(self, inputs, partials, discrete_inputs=None):
#         partials['data:ESC:voltage:estimated',
#                  'data:battery:voltage'] = 1
