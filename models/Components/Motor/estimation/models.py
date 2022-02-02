"""
Estimation models for the motor
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import add_subsystem_with_deviation


class MotorEstimationModels(om.Group):
    """
    Group containing the estimation models for the motor.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "max_torque",
            MaxTorque(),
            uncertain_outputs={"data:motor:torque:max:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "friction_torque",
            FrictionTorque(),
            uncertain_outputs={"data:motor:torque:friction:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "resistance",
            Resistance(),
            uncertain_outputs={"data:motor:resistance:estimated": "V/A"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:motor:mass:estimated": "kg"},
        )

        self.add_subsystem("geometry", Geometry(), promotes=["*"])


class MaxTorque(om.ExplicitComponent):
    """
    Compute maximum torque
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:max", val=np.nan, units="N*m")
        self.add_output("data:motor:torque:max:estimated", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:motor:reference:torque:nominal"]
        Tmot_max_ref = inputs["data:motor:reference:torque:max"]
        Tmot = inputs["data:motor:torque:nominal:estimated"]

        Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref)  # [N.m] max torque

        outputs["data:motor:torque:max:estimated"] = Tmot_max


class FrictionTorque(om.ExplicitComponent):
    """
    Computes friction torque.
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:friction", val=np.nan, units="N*m")
        self.add_output("data:motor:torque:friction:estimated", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:motor:reference:torque:nominal"]
        Tfmot_ref = inputs["data:motor:reference:torque:friction"]
        Tmot = inputs["data:motor:torque:nominal:estimated"]

        Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs["data:motor:torque:friction:estimated"] = Tfmot


class Resistance(om.ExplicitComponent):
    """
    Computes motor resistance.
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input(
            "data:motor:torque:coefficient:estimated", val=np.nan, units="N*m/A"
        )
        self.add_input("data:motor:reference:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:resistance", val=np.nan, units="V/A")
        self.add_input(
            "data:motor:reference:torque:coefficient", val=np.nan, units="N*m/A"
        )
        self.add_output("data:motor:resistance:estimated", units="V/A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:motor:reference:torque:nominal"]
        Rmot_ref = inputs["data:motor:reference:resistance"]
        Ktmot_ref = inputs["data:motor:reference:torque:coefficient"]
        Tmot = inputs["data:motor:torque:nominal:estimated"]
        Ktmot = inputs["data:motor:torque:coefficient:estimated"]

        Rmot = (
            Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2
        )  # [Ohm] motor resistance

        outputs["data:motor:resistance:estimated"] = Rmot


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:mass", val=np.nan, units="kg")
        self.add_output("data:motor:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot = inputs["data:motor:torque:nominal:estimated"]
        Tmot_ref = inputs["data:motor:reference:torque:nominal"]
        Mmot_ref = inputs["data:motor:reference:mass"]

        Mmot = Mmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs["data:motor:mass:estimated"] = Mmot


class Geometry(om.ExplicitComponent):
    """
    Computes motor geometry
    """

    def setup(self):
        self.add_input("data:motor:reference:length", val=np.nan, units="m")
        self.add_input("data:motor:reference:mass", val=np.nan, units="kg")
        self.add_input("data:motor:mass:estimated", val=np.nan, units="kg")
        self.add_output("data:motor:length:estimated", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Lmot_ref = inputs["data:motor:reference:length"]
        Mmot_ref = inputs["data:motor:reference:mass"]
        Mmot = inputs["data:motor:mass:estimated"]

        Lmot = Lmot_ref * (Mmot / Mmot_ref) ** (1 / 3)  # [m] Motor length (estimated)

        outputs["data:motor:length:estimated"] = Lmot
