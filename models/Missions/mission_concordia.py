"""
Organ delivery mission. For Multirotors only.
A future version may include Fixed Wing drones capability.
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
from scipy.constants import g
from models.Scenarios.multirotor.flight_model import MultirotorFlightModel
from models.Propulsion.Propeller.performances import PropellerPerfoModel
from models.Propulsion.Motor.performances import MotorPerfoModel
from models.Propulsion.Propeller.estimation.models import PropellerAerodynamicsModel


@oad.RegisterOpenMDAOSystem("mission.concordia")
class Mission(om.Group):
    """
    Organ delivery mission definition
    """

    def setup(self):
        route_1 = self.add_subsystem("route_1", om.Group(), promotes=["*"])
        route_1.add_subsystem("tow", ComputeTOW(route="_1"), promotes=["*"])
        route_1.add_subsystem("climb", ClimbSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem("cruise", CruiseSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem("hover", HoverSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem(
            "route",
            RouteComponent(route="_1", segments_list=["climb", "cruise", "hover"]),
            promotes=["*"],
        )

        route_2 = self.add_subsystem("route_2", om.Group(), promotes=["*"])
        route_2.add_subsystem("tow", ComputeTOW(route="_2"), promotes=["*"])
        route_2.add_subsystem("climb", ClimbSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem("cruise", CruiseSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem("hover", HoverSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem(
            "route",
            RouteComponent(route="_2", segments_list=["climb", "cruise", "hover"]),
            promotes=["*"],
        )

        diversion = self.add_subsystem("diversion", om.Group(), promotes=["*"])
        diversion.add_subsystem("tow", ComputeTOW(route="_diversion"), promotes=["*"])
        diversion.add_subsystem(
            "cruise", CruiseSegment(route="_diversion"), promotes=["*"]
        )
        diversion.add_subsystem(
            "hover", HoverSegment(route="_diversion"), promotes=["*"]
        )
        diversion.add_subsystem(
            "route",
            RouteComponent(route="_diversion", segments_list=["cruise", "hover"]),
            promotes=["*"],
        )

        mission = self.add_subsystem(
            "mission",
            MissionComponent(routes_list=["_1", "_2", "_diversion"]),
            promotes=["*"],
        )
        constraints = self.add_subsystem(
            "constraints", MissionConstraints(), promotes=["*"]
        )


class ComputeTOW(om.ExplicitComponent):
    """
    Computes TOW from MTOW, design paylaod and mission payload
    """

    def initialize(self):
        self.options.declare("route", default="", types=str)

    def setup(self):
        self.add_input("data:system:MTOW", val=np.nan, units="kg")
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_input(
            "mission:concordia_study:route%s:payload:mass" % self.options["route"],
            val=np.nan,
            units="kg",
        )
        self.add_output(
            "mission:concordia_study:route%s:tow" % self.options["route"], units="kg"
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mtow = inputs["data:system:MTOW"]
        m_pay_design = inputs["specifications:payload:mass"]
        m_pay_mission = inputs[
            "mission:concordia_study:route%s:payload:mass" % self.options["route"]
        ]

        tow = mtow - m_pay_design + m_pay_mission

        outputs["mission:concordia_study:route%s:tow" % self.options["route"]] = tow


class ClimbSegment(om.ExplicitComponent):
    """
    Climb segment
    """

    def initialize(self):
        self.options.declare("route", default="", types=str)

    def setup(self):
        # System parameters
        self.add_input(
            "mission:concordia_study:route%s:tow" % self.options["route"],
            val=np.nan,
            units="kg",
        )
        self.add_input(
            "mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3"
        )
        self.add_input("data:airframe:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:airframe:body:surface:top", val=np.nan, units="m**2")
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:climb", val=np.nan, units=None)
        # Motor parameters
        self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")
        # Mission parameters
        self.add_input(
            "mission:concordia_study:route%s:climb:height" % self.options["route"],
            val=np.nan,
            units="m",
        )
        self.add_input("mission:concordia_study:climb:speed", val=np.nan, units="m/s")
        self.add_input("data:ESC:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        # Outputs: energy
        self.add_output(
            "mission:concordia_study:route%s:climb:duration" % self.options["route"],
            units="min",
        )
        self.add_output(
            "mission:concordia_study:route%s:climb:energy:propulsion"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:climb:energy:payload"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:climb:energy:avionics"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:climb:energy" % self.options["route"],
            units="kJ",
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal = inputs["mission:concordia_study:route%s:tow" % self.options["route"]]
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]
        S_top = inputs["data:airframe:body:surface:top"]
        C_D = inputs["data:airframe:aerodynamics:CD0"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_cl = inputs["data:propeller:advance_ratio:climb"] # TODO: compute new J_cr based on cruise speed and propeller diameter (requires solver to get C_t, C_p as well)

        N_red = inputs["data:gearbox:N_red"]
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]

        D_cl = inputs[
            "mission:concordia_study:route%s:climb:height" % self.options["route"]
        ]
        V_cl = inputs["mission:concordia_study:climb:speed"]
        eta_ESC = inputs["data:ESC:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # Compute new flight parameters
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_axial(beta, J_cl)
        F_pro = (Mtotal * g + 0.5 * rho_air * C_D * S_top * V_cl**2) / Npro
        W_pro, P_pro, Q_pro = PropellerPerfoModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorPerfoModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        # Energy consumption
        t_cl = D_cl / V_cl  # [s]
        E_cl_pro = (P_el * Npro) / eta_ESC * t_cl  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_cl  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_cl  # [J] consumed energy for avionics

        outputs[
            "mission:concordia_study:route%s:climb:duration" % self.options["route"]
        ] = (
            t_cl / 60
        )  # [min]
        outputs[
            "mission:concordia_study:route%s:climb:energy:propulsion"
            % self.options["route"]
        ] = (
            E_cl_pro / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:climb:energy:payload"
            % self.options["route"]
        ] = (
            E_payload / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:climb:energy:avionics"
            % self.options["route"]
        ] = (
            E_avionics / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:climb:energy" % self.options["route"]
        ] = (
            E_cl_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class HoverSegment(om.ExplicitComponent):
    """
    Hover segment
    """

    def initialize(self):
        self.options.declare("route", default="", types=str)

    def setup(self):
        # System parameters
        self.add_input(
            "mission:concordia_study:route%s:tow" % self.options["route"],
            val=np.nan,
            units="kg",
        )
        self.add_input(
            "mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3"
        )
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        # Motor parameters
        self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")
        # Mission parameters
        self.add_input(
            "mission:concordia_study:route%s:hover:duration" % self.options["route"],
            val=np.nan,
            units="min",
        )
        self.add_input("data:ESC:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        # Outputs: energy
        self.add_output(
            "mission:concordia_study:route%s:hover:energy:propulsion"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:hover:energy:payload"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:hover:energy:avionics"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:hover:energy" % self.options["route"],
            units="kJ",
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal = inputs["mission:concordia_study:route%s:tow" % self.options["route"]]
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]

        N_red = inputs["data:gearbox:N_red"]
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]

        t_hov = (
            inputs[
                "mission:concordia_study:route%s:hover:duration" % self.options["route"]
            ]
            * 60
        )  # [s]
        eta_ESC = inputs["data:ESC:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # Compute new flight parameters
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta)
        F_pro = Mtotal * g / Npro
        W_pro, P_pro, Q_pro = PropellerPerfoModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorPerfoModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )

        # Energy consumption
        E_hover_pro = (
            (P_el * Npro) / eta_ESC * t_hov
        )  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_hov  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_hov  # [J] consumed energy for avionics

        outputs[
            "mission:concordia_study:route%s:hover:energy:propulsion"
            % self.options["route"]
        ] = (
            E_hover_pro / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:hover:energy:payload"
            % self.options["route"]
        ] = (
            E_payload / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:hover:energy:avionics"
            % self.options["route"]
        ] = (
            E_avionics / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:hover:energy" % self.options["route"]
        ] = (
            E_hover_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class CruiseSegment(om.ExplicitComponent):
    """
    Cruise segment
    Assumption: cruise velocity is unchanged from design scenario.
    """

    def initialize(self):
        self.options.declare("route", default="", types=str)

    def setup(self):
        # System parameters
        self.add_input(
            "mission:concordia_study:route%s:tow" % self.options["route"],
            val=np.nan,
            units="kg",
        )
        self.add_input(
            "mission:design_mission:cruise:atmosphere:density", val=np.nan, units="kg/m**3"
        )
        self.add_input("data:airframe:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:airframe:aerodynamics:CLmax", val=np.nan, units=None)
        self.add_input("data:airframe:body:surface:top", val=np.nan, units="m**2")
        self.add_input("data:airframe:body:surface:front", val=np.nan, units="m**2")
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:cruise", val=np.nan, units=None)
        # Motor parameters
        self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")
        # Mission parameters
        self.add_input(
            "mission:concordia_study:route%s:cruise:distance" % self.options["route"],
            val=np.nan,
            units="m",
        )
        self.add_input("mission:concordia_study:cruise:speed", val=np.nan, units="m/s")
        self.add_input("data:ESC:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        # Outputs: energy
        self.add_output(
            "mission:concordia_study:route%s:cruise:duration" % self.options["route"],
            units="min",
        )
        self.add_output(
            "mission:concordia_study:route%s:cruise:energy:propulsion"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:cruise:energy:payload"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:cruise:energy:avionics"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:cruise:energy" % self.options["route"],
            units="kJ",
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal = inputs["mission:concordia_study:route%s:tow" % self.options["route"]]
        rho_air = inputs["mission:design_mission:cruise:atmosphere:density"]
        C_D = inputs["data:airframe:aerodynamics:CD0"]
        C_L0 = inputs["data:airframe:aerodynamics:CLmax"]
        S_top = inputs["data:airframe:body:surface:top"]
        S_front = inputs["data:airframe:body:surface:front"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_cr = inputs[
            "data:propeller:advance_ratio:cruise"
        ]  # TODO: compute new J_cr based on cruise speed and propeller diameter (requires solver to get C_t, C_p as well)

        N_red = inputs["data:gearbox:N_red"]
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]

        D_cr = inputs[
            "mission:concordia_study:route%s:cruise:distance" % self.options["route"]
        ]
        V_cr = inputs[
            "mission:concordia_study:cruise:speed"
        ]  # TODO: compute optimal speed ? Can be achieved through optimizer at system level
        t_cr = D_cr / V_cr  # [s]
        eta_ESC = inputs["data:ESC:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # Compute new flight parameters
        thrust, alpha = MultirotorFlightModel.get_thrust(
            Mtotal, V_cr, 0, S_front, S_top, C_D, C_L0, rho_air
        )  # flight parameters
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(beta, J_cr, alpha)
        F_pro = thrust / Npro  # [N] thrust per propeller
        W_pro, P_pro, Q_pro = PropellerPerfoModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )  # propeller performances
        T_mot, W_mot, I_mot, U_mot, P_el = MotorPerfoModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )  # motor performances

        # Energy consumption
        E_cr_pro = (P_el * Npro) / eta_ESC * t_cr  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_cr  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_cr  # [J] consumed energy for avionics

        outputs[
            "mission:concordia_study:route%s:cruise:duration" % self.options["route"]
        ] = (
            t_cr / 60
        )  # [min]
        outputs[
            "mission:concordia_study:route%s:cruise:energy:propulsion"
            % self.options["route"]
        ] = (
            E_cr_pro / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:cruise:energy:payload"
            % self.options["route"]
        ] = (
            E_payload / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:cruise:energy:avionics"
            % self.options["route"]
        ] = (
            E_avionics / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:cruise:energy" % self.options["route"]
        ] = (
            E_cr_pro + E_payload + E_avionics
        ) / 1000  # [kJ]


class RouteComponent(om.ExplicitComponent):
    """
    Computes route energy and duration
    """

    def initialize(self):
        self.options.declare("route", default="", types=str)
        self.options.declare("segments_list", default=[], types=list)

    def setup(self):
        for segment in self.options["segments_list"]:
            self.add_input(
                "mission:concordia_study:route%s:%s:energy"
                % (self.options["route"], segment),
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:%s:energy:propulsion"
                % (self.options["route"], segment),
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:%s:energy:payload"
                % (self.options["route"], segment),
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:%s:energy:avionics"
                % (self.options["route"], segment),
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:%s:duration"
                % (self.options["route"], segment),
                val=np.nan,
                units="min",
            )
        self.add_output(
            "mission:concordia_study:route%s:energy" % self.options["route"], units="kJ"
        )
        self.add_output(
            "mission:concordia_study:route%s:energy:propulsion" % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:energy:payload" % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:energy:avionics" % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:duration" % self.options["route"],
            units="min",
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        E_mission = sum(
            inputs[
                "mission:concordia_study:route%s:%s:energy"
                % (self.options["route"], segment)
            ]
            for segment in self.options["segments_list"]
        )
        E_propulsion = sum(
            inputs[
                "mission:concordia_study:route%s:%s:energy:propulsion"
                % (self.options["route"], segment)
            ]
            for segment in self.options["segments_list"]
        )
        E_payload = sum(
            inputs[
                "mission:concordia_study:route%s:%s:energy:payload"
                % (self.options["route"], segment)
            ]
            for segment in self.options["segments_list"]
        )
        E_avionics = sum(
            inputs[
                "mission:concordia_study:route%s:%s:energy:avionics"
                % (self.options["route"], segment)
            ]
            for segment in self.options["segments_list"]
        )
        t_mission = sum(
            inputs[
                "mission:concordia_study:route%s:%s:duration"
                % (self.options["route"], segment)
            ]
            for segment in self.options["segments_list"]
        )

        outputs[
            "mission:concordia_study:route%s:energy" % self.options["route"]
        ] = E_mission
        outputs[
            "mission:concordia_study:route%s:energy:propulsion" % self.options["route"]
        ] = E_propulsion
        outputs[
            "mission:concordia_study:route%s:energy:payload" % self.options["route"]
        ] = E_payload
        outputs[
            "mission:concordia_study:route%s:energy:avionics" % self.options["route"]
        ] = E_avionics
        outputs[
            "mission:concordia_study:route%s:duration" % self.options["route"]
        ] = t_mission


class MissionComponent(om.ExplicitComponent):
    """
    Computes mission energy and duration
    """

    def initialize(self):
        self.options.declare("routes_list", default=[], types=list)

    def setup(self):
        for route in self.options["routes_list"]:
            self.add_input(
                "mission:concordia_study:route%s:energy" % route, val=np.nan, units="kJ"
            )
            self.add_input(
                "mission:concordia_study:route%s:energy:propulsion" % route,
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:energy:payload" % route,
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:energy:avionics" % route,
                val=np.nan,
                units="kJ",
            )
            self.add_input(
                "mission:concordia_study:route%s:duration" % route,
                val=np.nan,
                units="min",
            )
        self.add_output("mission:concordia_study:energy", units="kJ")
        self.add_output("mission:concordia_study:energy:propulsion", units="kJ")
        self.add_output("mission:concordia_study:energy:payload", units="kJ")
        self.add_output("mission:concordia_study:energy:avionics", units="kJ")
        self.add_output("mission:concordia_study:duration", units="min")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        E_mission = sum(
            inputs["mission:concordia_study:route%s:energy" % route]
            for route in self.options["routes_list"]
        )
        E_propulsion = sum(
            inputs["mission:concordia_study:route%s:energy:propulsion" % route]
            for route in self.options["routes_list"]
        )
        E_payload = sum(
            inputs["mission:concordia_study:route%s:energy:payload" % route]
            for route in self.options["routes_list"]
        )
        E_avionics = sum(
            inputs["mission:concordia_study:route%s:energy:avionics" % route]
            for route in self.options["routes_list"]
        )
        t_mission = sum(
            inputs["mission:concordia_study:route%s:duration" % route]
            for route in self.options["routes_list"]
        )

        outputs["mission:concordia_study:energy"] = E_mission
        outputs["mission:concordia_study:energy:propulsion"] = E_propulsion
        outputs["mission:concordia_study:energy:payload"] = E_payload
        outputs["mission:concordia_study:energy:avionics"] = E_avionics
        outputs["mission:concordia_study:duration"] = t_mission


class MissionConstraints(om.ExplicitComponent):
    """
    Computes mission constraints
    """

    def setup(self):
        self.add_input("mission:concordia_study:energy", val=np.nan, units="kJ")
        self.add_input("data:battery:energy", val=np.nan, units="kJ")
        self.add_input("data:battery:DoD:max", val=0.8, units=None)
        self.add_output("mission:concordia_study:constraints:energy", units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        E_mission = inputs["mission:concordia_study:energy"]
        E_bat = inputs["data:battery:energy"]
        C_ratio = inputs["data:battery:DoD:max"]

        energy_con = (E_bat * C_ratio - E_mission) / (E_bat * C_ratio)

        outputs["mission:concordia_study:constraints:energy"] = energy_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E_mission = inputs["mission:concordia_study:energy"]
        E_bat = inputs["data:battery:energy"]
        C_ratio = inputs["data:battery:DoD:max"]

        partials[
            "mission:concordia_study:constraints:energy",
            "mission:concordia_study:energy",
        ] = -1.0 / (E_bat * C_ratio)
        partials[
            "mission:concordia_study:constraints:energy",
            "data:battery:energy",
        ] = E_mission / (E_bat**2 * C_ratio)
        partials[
            "mission:concordia_study:constraints:energy",
            "data:battery:DoD:max",
        ] = E_mission / (E_bat * C_ratio**2)
