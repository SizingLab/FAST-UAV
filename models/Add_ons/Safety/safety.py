"""
Safety module
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
from models.Components.Propeller.performances import PropellerModel
from models.Components.Motor.performances import MotorModel


@oad.RegisterOpenMDAOSystem("addons.safety")
class Safety(om.Group):
    """
    Group containing the performances and requirements in case of rotor failure.
    """

    def setup(self):
        self.add_subsystem("hover_torque", EmergencyMotorTorque_hover(), promotes=["*"])
        self.add_subsystem(
            "forward_torque", EmergencyMotorTorque_forward(), promotes=["*"]
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
            "mission:sizing_mission:air_density", val=np.nan, units="kg/m**3"
        )
        self.add_input("addons:safety:k_thrust", val=np.nan, units=None)
        # Propeller parameters
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:aerodynamics:CT:static", val=np.nan, units=None)
        self.add_input("data:propeller:aerodynamics:CP:static", val=np.nan, units=None)
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
        rho_air = inputs["mission:sizing_mission:air_density"]

        D_pro = inputs["data:propeller:geometry:diameter"]
        C_t = inputs["data:propeller:aerodynamics:CT:static"]
        C_p = inputs["data:propeller:aerodynamics:CP:static"]

        N_red = inputs["data:gearbox:N_red"] if self.options["use_gearbox"] else 1.0
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]
        # Tmot_nom = inputs['data:motor:torque:nominal']

        # Hover
        F_pro = k_thrust * F_pro_nom  # [N] emergency thrust per propeller
        W_pro, P_pro, Q_pro = PropellerModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        outputs["addons:safety:motor:torque:hover"] = T_mot

        # motor_con_hover = (Tmot_nom - T_mot) / Tmot_nom  # [-]
        # outputs['addons:safety:constraints:torque:hover'] = motor_con_hover


class EmergencyMotorTorque_forward(om.ExplicitComponent):
    """
    Computes maximum motor torque in emergency mode, in forward flight.
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
        self.add_input("data:propeller:thrust:forward", val=np.nan, units="N")
        self.add_input(
            "mission:sizing_mission:air_density", val=np.nan, units="kg/m**3"
        )
        self.add_input("addons:safety:k_thrust", val=np.nan, units=None)
        # Propeller parameters
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input(
            "data:propeller:aerodynamics:CT:incidence", val=np.nan, units=None
        )
        self.add_input(
            "data:propeller:aerodynamics:CP:incidence", val=np.nan, units=None
        )
        # Motor parameters
        if self.options["use_gearbox"]:
            self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")

        self.add_output("addons:safety:motor:torque:forward", units="N*m")

        # self.add_input('data:motor:torque:nominal', val=np.nan, units='N*m')
        # self.add_output('addons:safety:constraints:torque:forward', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_nom = inputs["data:propeller:thrust:forward"]
        k_thrust = inputs[
            "addons:safety:k_thrust"
        ]  # [-] thrust ratio of failure case to normal operation
        rho_air = inputs["mission:sizing_mission:air_density"]

        D_pro = inputs["data:propeller:geometry:diameter"]
        C_t = inputs["data:propeller:aerodynamics:CT:incidence"]
        C_p = inputs["data:propeller:aerodynamics:CP:incidence"]

        N_red = inputs["data:gearbox:N_red"] if self.options["use_gearbox"] else 1.0
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]
        # Tmot_nom = inputs['data:motor:torque:nominal']

        # Forward
        F_pro = k_thrust * F_pro_nom  # [N] emergency thrust per propeller
        W_pro, P_pro, Q_pro = PropellerModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        outputs["addons:safety:motor:torque:forward"] = T_mot

        # motor_con_forward = (Tmot_nom - T_mot) / Tmot_nom  # [-]
        # outputs['addons:safety:constraints:torque:forward'] = motor_con_forward


class EmergencyMotorConstraints(om.ExplicitComponent):
    """
    The motor must be able to maintain the increase in thrust from an emergency situation in steady state.
    """

    def setup(self):
        self.add_input("data:motor:torque:nominal", val=np.nan, units="N*m")
        self.add_input("addons:safety:motor:torque:hover", val=np.nan, units="N*m")
        self.add_input("addons:safety:motor:torque:forward", val=np.nan, units="N*m")
        self.add_output("addons:safety:constraints:torque:hover", units=None)
        self.add_output("addons:safety:constraints:torque:forward", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["addons:safety:motor:torque:hover"]
        Tmot_ff = inputs["addons:safety:motor:torque:forward"]

        # Motor torque constraint
        motor_con_hover = (Tmot_nom - Tmot_hov) / Tmot_nom  # [-]
        motor_con_forward = (Tmot_nom - Tmot_ff) / Tmot_nom  # [-]

        outputs["addons:safety:constraints:torque:hover"] = motor_con_hover
        outputs["addons:safety:constraints:torque:forward"] = motor_con_forward

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["addons:safety:motor:torque:hover"]
        Tmot_ff = inputs["addons:safety:motor:torque:forward"]

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
            "addons:safety:motor:torque:forward",
        ] = 0.0

        partials[
            "addons:safety:constraints:torque:forward",
            "data:motor:torque:nominal",
        ] = (
            Tmot_ff / Tmot_nom**2
        )
        partials[
            "addons:safety:constraints:torque:forward",
            "addons:safety:motor:torque:forward",
        ] = (
            -1.0 / Tmot_nom
        )
        partials[
            "addons:safety:constraints:torque:forward",
            "addons:safety:motor:torque:hover",
        ] = 0.0
