"""
Estimation models for the battery.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class BatteryEstimationModels(om.Group):
    """
    Group containing the estimation models for the battery.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        # add_subsystem_with_deviation(
        #     self,
        #     "energy",
        #     Energy(),
        #     uncertain_outputs={"data:propulsion:battery:energy:estimated": "kJ"},
        # )

        add_subsystem_with_deviation(
            self,
            "power",
            Power(),
            uncertain_outputs={"data:propulsion:battery:power:max:estimated": "W"},
        )

        self.add_subsystem("capacity", Capacity(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "max_current",
            MaxCurrent(),
            uncertain_outputs={"data:propulsion:battery:current:max:estimated": "A"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weight:propulsion:battery:mass:estimated": "kg"},
        )

        self.add_subsystem("geometry", Geometry(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "max_DoD",
            MaxDepthOfDischarge(),
            uncertain_outputs={"data:propulsion:battery:DoD:max:estimated": None},
        )

        add_subsystem_with_deviation(
            self,
            "esc_efficiency",
            ESCEfficiency(),
            uncertain_outputs={"data:propulsion:esc:efficiency:estimated": None},
        )


class Energy(om.ExplicitComponent):
    """
    Computes battery energy
    """

    def setup(self):
        self.add_input("data:propulsion:battery:power:max:estimated", val=np.nan, units="W")
        self.add_input("models:propulsion:battery:current:max:reference", val=np.nan, units="A")
        self.add_input("models:propulsion:battery:voltage:reference", val=np.nan, units="V")
        self.add_input("models:propulsion:battery:energy:reference", val=np.nan, units="kJ")
        self.add_output("data:propulsion:battery:energy:estimated", units="kJ")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        P_bat_max = inputs["data:propulsion:battery:power:max:estimated"]
        U_bat_ref = inputs["models:propulsion:battery:voltage:reference"]
        I_bat_max_ref = inputs["models:propulsion:battery:current:max:reference"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]

        E_bat = E_bat_ref * P_bat_max / (U_bat_ref * I_bat_max_ref)  # [kJ]

        outputs["data:propulsion:battery:energy:estimated"] = E_bat


class Power(om.ExplicitComponent):
    """
    Computes battery power
    """

    def setup(self):
        self.add_input("data:propulsion:battery:energy:estimated", val=np.nan, units="kJ")
        self.add_input("models:propulsion:battery:current:max:reference", val=np.nan, units="A")
        self.add_input("models:propulsion:battery:voltage:reference", val=np.nan, units="V")
        self.add_input("models:propulsion:battery:energy:reference", val=np.nan, units="kJ")
        self.add_output("data:propulsion:battery:power:max:estimated", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        E_bat = inputs["data:propulsion:battery:energy:estimated"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]
        U_bat_ref = inputs["models:propulsion:battery:voltage:reference"]
        I_bat_max_ref = inputs["models:propulsion:battery:current:max:reference"]

        P_bat_max = (U_bat_ref * I_bat_max_ref) * E_bat / E_bat_ref  # [W]

        outputs["data:propulsion:battery:power:max:estimated"] = P_bat_max

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_bat = inputs["data:propulsion:battery:energy:estimated"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]
        U_bat_ref = inputs["models:propulsion:battery:voltage:reference"]
        I_bat_max_ref = inputs["models:propulsion:battery:current:max:reference"]

        partials["data:propulsion:battery:power:max:estimated",
                 "data:propulsion:battery:energy:estimated"] = (U_bat_ref * I_bat_max_ref) / E_bat_ref

        partials["data:propulsion:battery:power:max:estimated",
                 "models:propulsion:battery:energy:reference"] = - (U_bat_ref * I_bat_max_ref) * E_bat / E_bat_ref ** 2

        partials["data:propulsion:battery:power:max:estimated",
                 "models:propulsion:battery:voltage:reference"] = I_bat_max_ref * E_bat / E_bat_ref

        partials["data:propulsion:battery:power:max:estimated",
                 "models:propulsion:battery:current:max:reference"] = U_bat_ref * E_bat / E_bat_ref


class Capacity(om.ExplicitComponent):
    """
    Computes battery capacity
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:energy:estimated", val=np.nan, units="kJ")
        self.add_output("data:propulsion:battery:capacity:estimated", units="kA*s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        U_bat = inputs["data:propulsion:battery:voltage:estimated"]
        E_bat = inputs["data:propulsion:battery:energy:estimated"]

        C_bat = E_bat / U_bat  # [kA*s]

        outputs["data:propulsion:battery:capacity:estimated"] = C_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        U_bat = inputs["data:propulsion:battery:voltage:estimated"]
        E_bat = inputs["data:propulsion:battery:energy:estimated"]

        partials["data:propulsion:battery:capacity:estimated",
                 "data:propulsion:battery:voltage:estimated"] = - E_bat / U_bat ** 2

        partials["data:propulsion:battery:capacity:estimated",
                 "data:propulsion:battery:energy:estimated"] = 1 / U_bat


class MaxCurrent(om.ExplicitComponent):
    """
    Computes battery maximum current
    """

    def setup(self):
        self.add_input("models:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_input("models:propulsion:battery:current:max:reference", val=np.nan, units="A")
        self.add_input("data:propulsion:battery:capacity:estimated", units="A*s")
        self.add_output("data:propulsion:battery:current:max:estimated", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_bat_ref = inputs["models:propulsion:battery:capacity:reference"]
        I_bat_max_ref = inputs["models:propulsion:battery:current:max:reference"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        I_bat_max = I_bat_max_ref * C_bat / C_bat_ref  # [A] max current battery

        outputs["data:propulsion:battery:current:max:estimated"] = I_bat_max


class Weight(om.ExplicitComponent):
    """
    Computes battery weight
    """

    def setup(self):
        self.add_input("data:propulsion:battery:energy:estimated", val=np.nan, units="kJ")
        self.add_input("models:weight:propulsion:battery:mass:reference", val=np.nan, units="kg")
        self.add_input("models:propulsion:battery:energy:reference", val=np.nan, units="kJ")
        self.add_output("data:weight:propulsion:battery:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        E_bat = inputs["data:propulsion:battery:energy:estimated"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]

        m_bat = m_bat_ref * E_bat / E_bat_ref  # [kg] estimated battery mass

        outputs["data:weight:propulsion:battery:mass:estimated"] = m_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_bat = inputs["data:propulsion:battery:energy:estimated"]
        m_bat_ref = inputs["models:weight:propulsion:battery:mass:reference"]
        E_bat_ref = inputs["models:propulsion:battery:energy:reference"]

        partials[
            "data:weight:propulsion:battery:mass:estimated",
            "data:propulsion:battery:energy:estimated"
        ] = m_bat_ref / E_bat_ref
        partials[
            "data:weight:propulsion:battery:mass:estimated",
            "models:weight:propulsion:battery:mass:reference"
        ] = E_bat / E_bat_ref
        partials[
            "data:weight:propulsion:battery:mass:estimated",
            "models:propulsion:battery:energy:reference"
        ] = - m_bat_ref * E_bat / E_bat_ref ** 2


class Geometry(om.ExplicitComponent):
    """
    Computes battery geometry
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:capacity:estimated", val=np.nan, units="A*s")
        self.add_input("models:propulsion:battery:voltage:reference", val=np.nan, units="V")
        self.add_input("models:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_input("models:propulsion:battery:volume:reference", val=np.nan, units="cm**3")
        self.add_output("data:propulsion:battery:volume:estimated", units="cm**3")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]
        U_bat = inputs["data:propulsion:battery:voltage:estimated"]
        C_bat_ref = inputs["models:propulsion:battery:capacity:reference"]
        U_bat_ref = inputs["models:propulsion:battery:voltage:reference"]
        Volbat_ref = inputs["models:propulsion:battery:volume:reference"]

        Vol_bat = Volbat_ref * (
            C_bat * U_bat / (C_bat_ref * U_bat_ref)
        )  # [cm**3] Volume of the battery (estimated)

        outputs["data:propulsion:battery:volume:estimated"] = Vol_bat


class MaxDepthOfDischarge(om.ExplicitComponent):
    """
    Computes max. depth of discharge of the battery  TODO: find a model
    """

    def setup(self):
        self.add_input("models:propulsion:battery:DoD:max:reference", val=0.8, units=None)
        self.add_output("data:propulsion:battery:DoD:max:estimated", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_ratio_ref = inputs["models:propulsion:battery:DoD:max:reference"]

        # Model to be defined
        C_ratio = C_ratio_ref

        outputs["data:propulsion:battery:DoD:max:estimated"] = C_ratio


class ESCEfficiency(om.ExplicitComponent):
    """
    Computes efficiency of the ESC  TODO: find a model
    """

    def setup(self):
        self.add_input("models:propulsion:esc:efficiency:reference", val=0.95, units=None)
        self.add_output("data:propulsion:esc:efficiency:estimated", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        eta_ref = inputs["models:propulsion:esc:efficiency:reference"]

        # Model to be defined
        eta = eta_ref

        outputs["data:propulsion:esc:efficiency:estimated"] = eta


class Energy2(om.ExplicitComponent):
    """
    Computes battery energy from capacity and voltage
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:capacity:estimated", val=np.nan, units="kA*s")
        self.add_output("data:propulsion:battery:energy:estimated", units="kJ")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        U_bat = inputs["data:propulsion:battery:voltage:estimated"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        E_bat = C_bat * U_bat  # [kJ] total energy stored

        outputs["data:propulsion:battery:energy:estimated"] = E_bat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        U_bat = inputs["data:propulsion:battery:voltage:estimated"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        partials[
            "data:propulsion:battery:energy:estimated", "data:propulsion:battery:voltage:estimated"
        ] = C_bat

        partials[
            "data:propulsion:battery:energy:estimated", "data:propulsion:battery:capacity:estimated"
        ] = U_bat