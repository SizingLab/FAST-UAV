"""
Safety module. For Multirotors Only.
DEPRECATED.
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
from models.Propulsion.Propeller.performances import PropellerPerfoModel
from models.Propulsion.Motor.performances import MotorPerfoModel
from models.Propulsion.Propeller.estimation.models import PropellerAerodynamicsModel


@oad.RegisterOpenMDAOSystem("addons.safety")
class Safety(om.Group):
    """
    Group containing the performances and requirements in case of rotor failure.
    """

    def setup(self):
        self.add_subsystem("hover_torque", EmergencyMotorTorque_hover(), promotes=["*"])
        self.add_subsystem(
            "cruise_torque", EmergencyMotorTorque_cruise(), promotes=["*"]
        )
        self.add_subsystem("constraints", EmergencyMotorConstraints(), promotes=["*"])
        # self.add_subsystem("degraded_autonomy", degradedRange(), promotes=['*'])
        # self.add_subsystem("degraded_range", degradedRange(), promotes=['*'])


class EmergencyMotorTorque_hover(om.ExplicitComponent):
    """
    Computes motor torque constraint in emergency mode, in hover.
    Assumptions:
        - At least one rotor has failed, which leads to a new rotor arrangement to keep controllability
        - k_thrust represents the max. thrust increase with respect to the nominal case (no failure)
        - Drone is fully loaded (i.e. design payload)
    Under these assumptions, the motor must be able to maintain the increase in thrust in steady state.
    """

    def initialize(self):
        self.options.declare(
            "use_gearbox", default=False, types=bool
        )  # TODO: define gearbox option in conf file?

    def setup(self):
        # System parameters
        self.add_input("data:propeller:thrust:hover", val=np.nan, units="N")
        self.add_input(
            "mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3"
        )
        self.add_input("addons:safety:k_thrust", val=np.nan, units=None)
        # Propeller parameters
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        # Motor parameters
        if self.options["use_gearbox"]:
            self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")

        self.add_output("addons:safety:motor:torque:hover", units="N*m")

        # self.add_input('data:motor:torque:nominal', val=np.nan, units='N*m')
        # self.add_output('addons:safety:constraints:torque:hover', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_nom = inputs["data:propeller:thrust:hover"]
        k_thrust = inputs[
            "addons:safety:k_thrust"
        ]  # [-] thrust ratio of failure case to normal operation
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]

        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]

        N_red = inputs["data:gearbox:N_red"] if self.options["use_gearbox"] else 1.0
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]
        # Tmot_nom = inputs['data:motor:torque:nominal']

        # Hover
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta)
        F_pro = k_thrust * F_pro_nom  # [N] emergency thrust per propeller
        W_pro, P_pro, Q_pro = PropellerPerfoModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorPerfoModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        outputs["addons:safety:motor:torque:hover"] = T_mot

        # motor_con_hover = (Tmot_nom - T_mot) / Tmot_nom  # [-]
        # outputs['addons:safety:constraints:torque:hover'] = motor_con_hover


class EmergencyMotorTorque_cruise(om.ExplicitComponent):
    """
    Computes maximum motor torque in emergency mode, in cruise.
    Assumptions:
        - At least one rotor has failed, which leads to a new rotor arrangement to keep controllability
        - k_thrust represents the max. thrust increase with respect to the nominal case (no failure)
        - Drone is fully loaded (i.e. design payload)
        - No change in flight velocity and angle of attack
        - Propellers thrust and power coefficients are considered unchanged
    Under these assumptions, the motor must be able to maintain the increase in thrust in steady state.
    """

    def initialize(self):
        self.options.declare(
            "use_gearbox", default=False, types=bool
        )  # TODO: define gearbox option in conf file?

    def setup(self):
        # System parameters
        self.add_input("data:propeller:thrust:cruise", val=np.nan, units="N")
        self.add_input(
            "mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3"
        )
        self.add_input("addons:safety:k_thrust", val=np.nan, units=None)
        # Propeller parameters
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:cruise", val=np.nan, units=None)
        self.add_input("mission:design_mission:cruise:AoA", val=np.nan, units="rad")
        # Motor parameters
        if self.options["use_gearbox"]:
            self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")

        self.add_output("addons:safety:motor:torque:cruise", units="N*m")

        # self.add_input('data:motor:torque:nominal', val=np.nan, units='N*m')
        # self.add_output('addons:safety:constraints:torque:cruise', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_nom = inputs["data:propeller:thrust:cruise"]
        k_thrust = inputs[
            "addons:safety:k_thrust"
        ]  # [-] thrust ratio of failure case to normal operation
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]

        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_cr = inputs["data:propeller:advance_ratio:cruise"]
        alpha = inputs["mission:design_mission:cruise:AoA"]

        N_red = inputs["data:gearbox:N_red"] if self.options["use_gearbox"] else 1.0
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]
        # Tmot_nom = inputs['data:motor:torque:nominal']

        # Cruise
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J_cr, alpha)
        F_pro = k_thrust * F_pro_nom  # [N] emergency thrust per propeller
        W_pro, P_pro, Q_pro = PropellerPerfoModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorPerfoModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        outputs["addons:safety:motor:torque:cruise"] = T_mot

        # motor_con_cruise = (Tmot_nom - T_mot) / Tmot_nom  # [-]
        # outputs['addons:safety:constraints:torque:cruise'] = motor_con_cruise


class EmergencyMotorConstraints(om.ExplicitComponent):
    """
    The motor must be able to maintain the increase in thrust from an emergency situation in steady state.
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal", val=np.nan, units="N*m")
        self.add_input("addons:safety:motor:torque:hover", val=np.nan, units="N*m")
        self.add_input("addons:safety:motor:torque:cruise", val=np.nan, units="N*m")
        self.add_output("addons:safety:constraints:torque:hover", units=None)
        self.add_output("addons:safety:constraints:torque:cruise", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["addons:safety:motor:torque:hover"]
        Tmot_cr = inputs["addons:safety:motor:torque:cruise"]

        # Motor torque constraint
        motor_con_hover = (Tmot_nom - Tmot_hov) / Tmot_nom  # [-]
        motor_con_cruise = (Tmot_nom - Tmot_cr) / Tmot_nom  # [-]

        outputs["addons:safety:constraints:torque:hover"] = motor_con_hover
        outputs["addons:safety:constraints:torque:cruise"] = motor_con_cruise

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["addons:safety:motor:torque:hover"]
        Tmot_cr = inputs["addons:safety:motor:torque:cruise"]

        partials[
            "addons:safety:constraints:torque:hover",
            "data:motor:torque:nominal",
        ] = (
            Tmot_hov / Tmot_nom**2
        )
        partials[
            "addons:safety:constraints:torque:hover",
            "addons:safety:motor:torque:hover",
        ] = (
            -1.0 / Tmot_nom
        )
        partials[
            "addons:safety:constraints:torque:hover",
            "addons:safety:motor:torque:cruise",
        ] = 0.0

        partials[
            "addons:safety:constraints:torque:cruise",
            "data:motor:torque:nominal",
        ] = (
            Tmot_cr / Tmot_nom**2
        )
        partials[
            "addons:safety:constraints:torque:cruise",
            "addons:safety:motor:torque:cruise",
        ] = (
            -1.0 / Tmot_nom
        )
        partials[
            "addons:safety:constraints:torque:cruise",
            "addons:safety:motor:torque:hover",
        ] = 0.0
