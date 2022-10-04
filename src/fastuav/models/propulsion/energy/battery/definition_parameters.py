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
    The definition parameters for the battery are the voltage and the capacity (or, in an equivalent way, the voltage
    and the energy).
    """

    def setup(self):

        self.add_subsystem("cell_number", CellNumber(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "voltage",
            Voltage(),
            uncertain_outputs={"data:propulsion:battery:voltage:estimated": "V"},
        )

        add_subsystem_with_deviation(
            self,
            "capacity",
            Capacity(),
            uncertain_outputs={"data:propulsion:battery:capacity:estimated": "A*s"},
        )

        self.add_subsystem("energy", Energy(), promotes=["*"])


class CellNumber(om.ExplicitComponent):
    """
    Computes the voltage of the battery. Also returns the number of cells.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage:k", val=np.nan, units=None)
        self.add_input("data:propulsion:battery:cell:voltage:estimated", val=3.7, units="V")
        self.add_output("data:propulsion:battery:cell:number:estimated", units=None)
        self.add_output("data:propulsion:battery:cell:number:series:estimated", units=None)
        self.add_output("data:propulsion:battery:cell:number:parallel:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        V_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        k_vb = inputs["data:propulsion:battery:voltage:k"]

        N_series = k_vb * (
            U_mot_to / V_cell
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
    #     V_cell = inputs['data:propulsion:battery:cell:voltage:estimated']
    #     k_vb = inputs['data:propulsion:battery:voltage:k']
    #     U_mot_to = inputs['data:propulsion:motor:voltage:takeoff']
    #
    #     partials[
    #         'data:propulsion:battery:cell:number:series:estimated',
    #         'data:propulsion:motor:voltage:takeoff',
    #     ] = (1 + np.cos(2 * np.pi * k_vb * U_mot_to / V_cell)) * k_vb / V_cell  # Smooth ceil function derivative
    #
    #     partials[
    #         'data:propulsion:battery:cell:number:series:estimated',
    #         'data:propulsion:battery:voltage:k',
    #     ] = (1 + np.cos(2 * np.pi * k_vb * U_mot_to / V_cell)) * U_mot_to / V_cell  # Smooth ceil function derivative


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
        V_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        N_series = inputs["data:propulsion:battery:cell:number:series:estimated"]

        V_bat = V_cell * N_series  # [V] Battery voltage

        outputs["data:propulsion:battery:voltage:estimated"] = V_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_cell = inputs["data:propulsion:battery:cell:voltage:estimated"]
        N_series = inputs["data:propulsion:battery:cell:number:series:estimated"]

        partials[
            "data:propulsion:battery:voltage:estimated",
            "data:propulsion:battery:cell:number:series:estimated",
        ] = V_cell

        partials[
            "data:propulsion:battery:voltage:estimated",
            "data:propulsion:battery:cell:voltage:estimated",
        ] = N_series


class Capacity(om.ExplicitComponent):
    """
    Computes battery capacity
    """

    def setup(self):
        self.add_input("data:scenarios:payload:mass", val=np.nan, units="kg")
        self.add_input("data:propulsion:battery:capacity:k", val=np.nan, units=None)
        self.add_input("data:weights:propulsion:battery:mass:reference", val=np.nan, units="kg")
        self.add_input("data:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_output("data:propulsion:battery:capacity:estimated", units="A*s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_Mb = inputs["data:propulsion:battery:capacity:k"]
        M_load = inputs["data:scenarios:payload:mass"]
        Mbat_ref = inputs["data:weights:propulsion:battery:mass:reference"]
        Cbat_ref = inputs["data:propulsion:battery:capacity:reference"]

        C_bat = k_Mb * M_load * Cbat_ref / Mbat_ref  # [A.s] Capacity  of the battery

        outputs["data:propulsion:battery:capacity:estimated"] = C_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_Mb = inputs["data:propulsion:battery:capacity:k"]
        M_load = inputs["data:scenarios:payload:mass"]
        Mbat_ref = inputs["data:weights:propulsion:battery:mass:reference"]
        Cbat_ref = inputs["data:propulsion:battery:capacity:reference"]

        partials["data:propulsion:battery:capacity:estimated",
                 "data:propulsion:battery:capacity:k"] = M_load * Cbat_ref / Mbat_ref
        partials["data:propulsion:battery:capacity:estimated",
                 "data:scenarios:payload:mass"] = k_Mb * Cbat_ref / Mbat_ref
        partials["data:propulsion:battery:capacity:estimated",
                 "data:weights:propulsion:battery:mass:reference"] = - k_Mb * M_load * Cbat_ref / Mbat_ref ** 2
        partials["data:propulsion:battery:capacity:estimated",
                 "data:propulsion:battery:capacity:reference"] = k_Mb * M_load / Mbat_ref


class Energy(om.ExplicitComponent):
    """
    Computes battery energy
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:capacity:estimated", val=np.nan, units="kA*s")
        self.add_output("data:propulsion:battery:energy:estimated", units="kJ")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        V_bat = inputs["data:propulsion:battery:voltage:estimated"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        E_bat = C_bat * V_bat  # [kJ] total energy stored

        outputs["data:propulsion:battery:energy:estimated"] = E_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        V_bat = inputs["data:propulsion:battery:voltage:estimated"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        partials[
            "data:propulsion:battery:energy:estimated", "data:propulsion:battery:voltage:estimated"
        ] = C_bat

        partials[
            "data:propulsion:battery:energy:estimated", "data:propulsion:battery:capacity:estimated"
        ] = V_bat
