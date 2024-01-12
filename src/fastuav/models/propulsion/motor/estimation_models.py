"""
Estimation models for the motor
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class MotorEstimationModels(om.Group):
    """
    Group containing the estimation models for the motor.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "nominal_torque",
            NominalTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:nominal:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "friction_torque",
            FrictionTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:friction:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "resistance",
            Resistance(),
            uncertain_outputs={"data:propulsion:motor:resistance:estimated": "V/A"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weight:propulsion:motor:mass:estimated": "kg"},
        )

        self.add_subsystem("geometry", Geometry(), promotes=["*"])


class NominalTorque(om.ExplicitComponent):
    """
    Compute nominal torque
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:nominal:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_nom_ref = inputs["models:propulsion:motor:torque:nominal:reference"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        T_mot_nom = T_mot_nom_ref * T_mot_max / T_mot_max_ref  # [N.m] nominal torque

        outputs["data:propulsion:motor:torque:nominal:estimated"] = T_mot_nom

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_nom_ref = inputs["models:propulsion:motor:torque:nominal:reference"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "models:propulsion:motor:torque:nominal:reference"] = T_mot_max / T_mot_max_ref

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "models:propulsion:motor:torque:max:reference"] = - T_mot_nom_ref * T_mot_max / T_mot_max_ref ** 2

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "data:propulsion:motor:torque:max:estimated"] = T_mot_nom_ref / T_mot_max_ref


class FrictionTorque(om.ExplicitComponent):
    """
    Computes friction torque.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:friction:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:friction:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        Tf_ref = inputs["models:propulsion:motor:torque:friction:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        Tf = Tf_ref * (T_mot_max / T_mot_max_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs["data:propulsion:motor:torque:friction:estimated"] = Tf

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        Tf_ref = inputs["models:propulsion:motor:torque:friction:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:friction:estimated",
                 "models:propulsion:motor:torque:friction:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (3 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * Tf_ref * T_mot_max ** (3 / 3.5) / T_mot_max_ref ** (6.5 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * Tf_ref / T_mot_max_ref ** (3 / 3.5) * T_mot_max ** (- 0.5 / 3.5)


class Resistance(om.ExplicitComponent):
    """
    Computes motor resistance.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:speed:constant:estimated", val=np.nan, units="rad/V/s")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:resistance:reference", val=np.nan, units="V/A")
        self.add_input("models:propulsion:motor:speed:constant:reference", val=np.nan, units="rad/V/s")
        self.add_output("data:propulsion:motor:resistance:estimated", units="V/A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        R_ref = inputs["models:propulsion:motor:resistance:reference"]
        Kv_ref = inputs["models:propulsion:motor:speed:constant:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Kv = inputs["data:propulsion:motor:speed:constant:estimated"]

        R = (
            R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * (Kv / Kv_ref) ** (-2)
        )  # [Ohm] motor resistance

        outputs["data:propulsion:motor:resistance:estimated"] = R

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        R_ref = inputs["models:propulsion:motor:resistance:reference"]
        Kv_ref = inputs["models:propulsion:motor:speed:constant:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Kv = inputs["data:propulsion:motor:speed:constant:estimated"]

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = (5 / 3.5) * R_ref * T_mot_max ** (-5 / 3.5) * T_mot_max_ref ** (1.5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:resistance:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:speed:constant:reference"
        ] = 2 * R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * Kv_ref / Kv ** 2

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (-5 / 3.5) * R_ref / T_mot_max_ref ** (-5 / 3.5) * T_mot_max ** (-8.5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:speed:constant:estimated"
        ] = -2 * R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * Kv_ref ** 2 / Kv ** 3


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:weight:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weight:propulsion:motor:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]

        m_mot = m_mot_ref * (T_mot_max / T_mot_max_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs["data:weight:propulsion:motor:mass:estimated"] = m_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]

        partials["data:weight:propulsion:motor:mass:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * m_mot_ref / T_mot_max_ref ** (3 / 3.5) * T_mot_max ** (-0.5 / 3.5)

        partials["data:weight:propulsion:motor:mass:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * m_mot_ref * T_mot_max ** (3 / 3.5) / T_mot_max_ref ** (6.5 / 3.5)

        partials["data:weight:propulsion:motor:mass:estimated",
                 "models:weight:propulsion:motor:mass:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (3 / 3.5)


class Geometry(om.ExplicitComponent):
    """
    Computes motor geometry
    """

    def setup(self):
        self.add_input("models:propulsion:motor:length:reference", val=np.nan, units="m")
        self.add_input("models:weight:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        self.add_output("data:propulsion:motor:length:estimated", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        L_mot_ref = inputs["models:propulsion:motor:length:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        m_mot = inputs["data:weight:propulsion:motor:mass:estimated"]

        L_mot = L_mot_ref * (m_mot / m_mot_ref) ** (1 / 3)  # [m] Motor length (estimated)

        outputs["data:propulsion:motor:length:estimated"] = L_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        L_mot_ref = inputs["models:propulsion:motor:length:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        m_mot = inputs["data:weight:propulsion:motor:mass:estimated"]

        partials["data:propulsion:motor:length:estimated",
                 "models:propulsion:motor:length:reference"
        ] = (m_mot / m_mot_ref) ** (1 / 3)

        partials["data:propulsion:motor:length:estimated",
                 "data:weight:propulsion:motor:mass:estimated"
        ] = (1 / 3) * L_mot_ref / m_mot_ref ** (1 / 3) * m_mot ** (- 2 / 3)

        partials["data:propulsion:motor:length:estimated",
                 "models:weight:propulsion:motor:mass:reference"
        ] = - (1 / 3) * L_mot_ref * m_mot ** (1 / 3) / m_mot_ref ** (4 / 3)

