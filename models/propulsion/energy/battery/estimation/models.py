"""
Estimation models for the battery.
"""
import openmdao.api as om
import numpy as np
from models.uncertainty.uncertainty import add_subsystem_with_deviation


class BatteryEstimationModels(om.Group):
    """
    Group containing the estimation models for the battery.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
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
            uncertain_outputs={"data:weights:battery:mass:estimated": "kg"},
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


class MaxCurrent(om.ExplicitComponent):
    """
    Computes battery maximum current
    """

    def setup(self):
        self.add_input("data:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_input("data:propulsion:battery:current:max:reference", val=np.nan, units="A")
        self.add_input("data:propulsion:battery:capacity:estimated", units="A*s")
        self.add_output("data:propulsion:battery:current:max:estimated", units="A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Cbat_ref = inputs["data:propulsion:battery:capacity:reference"]
        Imax_ref = inputs["data:propulsion:battery:current:max:reference"]
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]

        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery

        outputs["data:propulsion:battery:current:max:estimated"] = Imax


class Weight(om.ExplicitComponent):
    """
    Computes battery weight
    """

    def setup(self):
        self.add_input("data:propulsion:battery:energy:estimated", val=np.nan, units="kJ")
        self.add_input("data:weights:battery:mass:reference", val=np.nan, units="kg")
        self.add_input("data:propulsion:battery:energy:reference", val=np.nan, units="kJ")
        self.add_output("data:weights:battery:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Ebat = inputs["data:propulsion:battery:energy:estimated"]
        Mbat_ref = inputs["data:weights:battery:mass:reference"]
        Ebat_ref = inputs["data:propulsion:battery:energy:reference"]

        Mbat = Mbat_ref * Ebat / Ebat_ref  # [kg] estimated battery mass

        outputs["data:weights:battery:mass:estimated"] = Mbat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mbat_ref = inputs["data:weights:battery:mass:reference"]
        Ebat_ref = inputs["data:propulsion:battery:energy:reference"]

        partials["data:weights:battery:mass:estimated", "data:propulsion:battery:energy:estimated"] = (
            Mbat_ref / Ebat_ref
        )


class Geometry(om.ExplicitComponent):
    """
    Computes battery geometry
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage:estimated", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:capacity:estimated", val=np.nan, units="A*s")
        self.add_input("data:propulsion:battery:voltage:reference", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:capacity:reference", val=np.nan, units="A*s")
        self.add_input("data:propulsion:battery:volume:reference", val=np.nan, units="cm**3")
        self.add_output("data:propulsion:battery:volume:estimated", units="cm**3")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_bat = inputs["data:propulsion:battery:capacity:estimated"]
        V_bat = inputs["data:propulsion:battery:voltage:estimated"]
        Cbat_ref = inputs["data:propulsion:battery:capacity:reference"]
        Vbat_ref = inputs["data:propulsion:battery:voltage:reference"]
        Volbat_ref = inputs["data:propulsion:battery:volume:reference"]

        Vol_bat = Volbat_ref * (
            C_bat * V_bat / (Cbat_ref * Vbat_ref)
        )  # [cm**3] Volume of the battery (estimated)

        outputs["data:propulsion:battery:volume:estimated"] = Vol_bat


class MaxDepthOfDischarge(om.ExplicitComponent):
    """
    Computes max. depth of discharge of the battery  TODO: find a model
    """

    def setup(self):
        self.add_input("data:propulsion:battery:DoD:max:reference", val=0.8, units=None)
        self.add_output("data:propulsion:battery:DoD:max:estimated", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        C_ratio_ref = inputs["data:propulsion:battery:DoD:max:reference"]

        # Model to be defined
        C_ratio = C_ratio_ref

        outputs["data:propulsion:battery:DoD:max:estimated"] = C_ratio


class ESCEfficiency(om.ExplicitComponent):
    """
    Computes efficiency of the ESC  TODO: find a model
    """

    def setup(self):
        self.add_input("data:propulsion:esc:efficiency:reference", val=0.95, units=None)
        self.add_output("data:propulsion:esc:efficiency:estimated", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        eta_ref = inputs["data:propulsion:esc:efficiency:reference"]

        # Model to be defined
        eta = eta_ref

        outputs["data:propulsion:esc:efficiency:estimated"] = eta
