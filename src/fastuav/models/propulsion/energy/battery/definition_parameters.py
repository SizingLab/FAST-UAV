"""
Definition parameters for the battery.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class BatteryDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the battery.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the battery are the voltage and the capacity (or, alternatively, the voltage
    and the energy).
    """

    def setup(self):

        # add_subsystem_with_deviation(
        #     self,
        #     "power",
        #     Power(),
        #     uncertain_outputs={"data:propulsion:battery:power:max:estimated": "W"},
        # )

        add_subsystem_with_deviation(
            self,
            "energy",
            Energy(),
            uncertain_outputs={"data:propulsion:battery:energy:estimated": "kJ"},
        )

        self.add_subsystem("cell_number", CellNumber(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "voltage",
            Voltage(),
            uncertain_outputs={"data:propulsion:battery:voltage:estimated": "V"},
        )


class Power(om.ExplicitComponent):
    """
    Computes battery power
    """

    def setup(self):
        self.add_input("optimization:variables:propulsion:battery:power:k", val=1.0, units=None)
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("models:propulsion:esc:efficiency:reference", val=0.95, units=None)
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("mission:sizing:payload:power", val=np.nan, units="W")
        self.add_output("data:propulsion:battery:power:max:estimated", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        k_pb = inputs["optimization:variables:propulsion:battery:power:k"]
        N_pro = inputs["data:propulsion:propeller:number"]
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        eta_ESC = inputs["models:propulsion:esc:efficiency:reference"]
        P_payload = inputs["mission:sizing:payload:power"]

        P_bat_max = k_pb * N_pro * P_mot_to / eta_ESC + P_payload  # [W]

        outputs["data:propulsion:battery:power:max:estimated"] = P_bat_max


class CellNumber(om.ExplicitComponent):
    """
    Computes the voltage of the battery. Also returns the number of cells.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("optimization:variables:propulsion:battery:voltage:k", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:cell:voltage:estimated", val=3.7, units="V")
        self.add_output("data:propulsion:battery:cell:number:estimated", units=None)
        self.add_output("data:propulsion:battery:cell:number:series:estimated", units=None)
        self.add_output("data:propulsion:battery:cell:number:parallel:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        U_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        k_vb = inputs["optimization:variables:propulsion:battery:voltage:k"]

        N_series = k_vb * (
            U_mot_to / U_cell
        )  # [-] Number of series connections (for voltage upgrade)
        N_parallel = 1  # [-] Number of parallel connections (for capacity upgrade)
        N_cell = N_parallel * N_series

        outputs["data:propulsion:battery:cell:number:series:estimated"] = N_series
        outputs["data:propulsion:battery:cell:number:parallel:estimated"] = N_parallel
        outputs["data:propulsion:battery:cell:number:estimated"] = N_cell

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     """
    #     Defining approximates from partials helps avoiding local minima but may increase the number of iterations
    #     if the derivatives are not well defined (or cause convergence issues).
    #     """
    #
    #     U_cell = inputs['data:propulsion:battery:cell:voltage:estimated']
    #     k_vb = inputs['optimization:variables:propulsion:battery:voltage:k']
    #     U_mot_to = inputs['data:propulsion:motor:voltage:takeoff']
    #
    #     partials[
    #         'data:propulsion:battery:cell:number:series:estimated',
    #         'data:propulsion:motor:voltage:takeoff',
    #     ] = (1 + np.cos(2 * np.pi * k_vb * U_mot_to / U_cell)) * k_vb / U_cell  # Smooth ceil function derivative
    #
    #     partials[
    #         'data:propulsion:battery:cell:number:series:estimated',
    #         'optimization:variables:propulsion:battery:voltage:k',
    #     ] = (1 + np.cos(2 * np.pi * k_vb * U_mot_to / U_cell)) * U_mot_to / U_cell  # Smooth ceil function derivative


class Voltage(om.ExplicitComponent):
    """
    Computes battery voltage
    """

    def setup(self):
        self.add_input("data:propulsion:battery:cell:voltage:estimated", val=3.7, units="V")
        self.add_input(
            "data:propulsion:battery:cell:number:series:estimated", val=np.nan, units=None
        )
        self.add_output("data:propulsion:battery:voltage:estimated", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        U_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        N_series = inputs["data:propulsion:battery:cell:number:series:estimated"]

        U_bat = U_cell * N_series  # [V] Battery voltage

        outputs["data:propulsion:battery:voltage:estimated"] = U_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        U_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        N_series = inputs["data:propulsion:battery:cell:number:series:estimated"]

        partials[
            "data:propulsion:battery:voltage:estimated",
            "data:propulsion:battery:cell:number:series:estimated",
        ] = U_cell

        partials[
            "data:propulsion:battery:voltage:estimated",
            "data:propulsion:battery:cell:voltage:estimated",
        ] = N_series


class Capacity(om.ExplicitComponent):
    """
    Computes battery capacity
    """

    def setup(self):
        self.add_input("mission:sizing:payload:mass", val=np.nan, units="kg")
        self.add_input("optimization:variables:propulsion:battery:capacity:k", val=np.nan, units=None)
        self.add_input("models:weight:propulsion:battery:mass:reference", val=np.nan, units="kg")
        self.add_input("models:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_output("data:propulsion:battery:capacity:estimated", units="A*s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_mb = inputs["optimization:variables:propulsion:battery:capacity:k"]
        m_load = inputs["mission:sizing:payload:mass"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        C_bat_ref = inputs["models:propulsion:battery:capacity:reference"]

        C_bat = k_mb * m_load * C_bat_ref / m_bat_ref  # [A.s] Capacity  of the battery

        outputs["data:propulsion:battery:capacity:estimated"] = C_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_mb = inputs["optimization:variables:propulsion:battery:capacity:k"]
        m_load = inputs["mission:sizing:payload:mass"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        C_bat_ref = inputs["models:propulsion:battery:capacity:reference"]

        partials["data:propulsion:battery:capacity:estimated",
                 "optimization:variables:propulsion:battery:capacity:k"] = m_load * C_bat_ref / m_bat_ref
        partials["data:propulsion:battery:capacity:estimated",
                 "mission:sizing:payload:mass"] = k_mb * C_bat_ref / m_bat_ref
        partials["data:propulsion:battery:capacity:estimated",
                 "models:weight:propulsion:battery:mass:reference"] = - k_mb * m_load * C_bat_ref / m_bat_ref ** 2
        partials["data:propulsion:battery:capacity:estimated",
                 "models:propulsion:battery:capacity:reference"] = k_mb * m_load / m_bat_ref


class Energy(om.ExplicitComponent):
    """
    Computes battery energy
    """

    def setup(self):
        self.add_input("mission:sizing:payload:mass", val=np.nan, units="kg")
        self.add_input("optimization:variables:propulsion:battery:energy:k", val=np.nan, units=None)
        self.add_input("models:weight:propulsion:battery:mass:reference", val=np.nan, units="kg")
        self.add_input("models:propulsion:battery:energy:reference", val=np.nan, units="kJ")
        self.add_output("data:propulsion:battery:energy:estimated", units="kJ")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_mb = inputs["optimization:variables:propulsion:battery:energy:k"]
        m_load = inputs["mission:sizing:payload:mass"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]

        E_bat = k_mb * m_load * E_bat_ref / m_bat_ref  # [kJ] Energy  of the battery

        outputs["data:propulsion:battery:energy:estimated"] = E_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_mb = inputs["optimization:variables:propulsion:battery:energy:k"]
        m_load = inputs["mission:sizing:payload:mass"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]

        partials["data:propulsion:battery:energy:estimated",
                 "optimization:variables:propulsion:battery:energy:k"] = m_load * E_bat_ref / m_bat_ref
        partials["data:propulsion:battery:energy:estimated",
                 "mission:sizing:payload:mass"] = k_mb * E_bat_ref / m_bat_ref
        partials["data:propulsion:battery:energy:estimated",
                 "models:weight:propulsion:battery:mass:reference"] = - k_mb * m_load * E_bat_ref / m_bat_ref ** 2
        partials["data:propulsion:battery:energy:estimated",
                 "models:propulsion:battery:energy:reference"] = k_mb * m_load / m_bat_ref
