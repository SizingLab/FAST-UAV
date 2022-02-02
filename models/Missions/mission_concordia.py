"""
Organ delivery mission
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
from scipy.constants import g
from models.Scenarios.flight_model import FlightModel
from models.Components.Propeller.performances import PropellerModel
from models.Components.Motor.performances import MotorModel
from models.Components.Propeller.estimation.models import PropellerAerodynamicsModel


@oad.RegisterOpenMDAOSystem("multirotor.mission_concordia")
class Mission(om.Group):
    """
    Organ delivery mission definition
    """

    def setup(self):
        route_1 = self.add_subsystem("route_1", om.Group(), promotes=["*"])
        route_1.add_subsystem("tow", ComputeTOW(route="_1"), promotes=["*"])
        route_1.add_subsystem("climb", ClimbSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem("forward", ForwardSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem("hover", HoverSegment(route="_1"), promotes=["*"])
        route_1.add_subsystem(
            "route",
            RouteComponent(route="_1", segments_list=["climb", "forward", "hover"]),
            promotes=["*"],
        )

        route_2 = self.add_subsystem("route_2", om.Group(), promotes=["*"])
        route_2.add_subsystem("tow", ComputeTOW(route="_2"), promotes=["*"])
        route_2.add_subsystem("climb", ClimbSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem("forward", ForwardSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem("hover", HoverSegment(route="_2"), promotes=["*"])
        route_2.add_subsystem(
            "route",
            RouteComponent(route="_2", segments_list=["climb", "forward", "hover"]),
            promotes=["*"],
        )

        diversion = self.add_subsystem("diversion", om.Group(), promotes=["*"])
        diversion.add_subsystem("tow", ComputeTOW(route="_diversion"), promotes=["*"])
        diversion.add_subsystem(
            "forward", ForwardSegment(route="_diversion"), promotes=["*"]
        )
        diversion.add_subsystem(
            "hover", HoverSegment(route="_diversion"), promotes=["*"]
        )
        diversion.add_subsystem(
            "route",
            RouteComponent(route="_diversion", segments_list=["forward", "hover"]),
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
        self.add_input("specifications:payload:mass:max", val=np.nan, units="kg")
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
        m_pay_design = inputs["specifications:payload:mass:max"]
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
            "mission:sizing_mission:air_density", val=np.nan, units="kg/m**3"
        )
        self.add_input("data:aerodynamics:Cd", val=np.nan, units=None)
        self.add_input("data:structure:body:surface:top", val=np.nan, units="m**2")
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:aerodynamics:CT:axial", val=np.nan, units=None)
        self.add_input("data:propeller:aerodynamics:CP:axial", val=np.nan, units=None)
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
        self.add_input("specifications:climb_speed", val=np.nan, units="m/s")
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
        rho_air = inputs["mission:sizing_mission:air_density"]
        S_top = inputs["data:structure:body:surface:top"]
        C_D = inputs["data:aerodynamics:Cd"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        C_t = inputs[
            "data:propeller:aerodynamics:CT:axial"
        ]  # TODO: compute new C_t based on climb speed and propeller diameter
        C_p = inputs[
            "data:propeller:aerodynamics:CP:axial"
        ]  # TODO: compute new C_p based on climb speed and propeller diameter

        N_red = inputs["data:gearbox:N_red"]
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]

        D_cl = inputs[
            "mission:concordia_study:route%s:climb:height" % self.options["route"]
        ]
        V_cl = inputs["specifications:climb_speed"]
        eta_ESC = inputs["data:ESC:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # Compute new flight parameters
        F_pro = (Mtotal * g + 0.5 * rho_air * C_D * S_top * V_cl**2) / Npro
        W_pro, P_pro, Q_pro = PropellerModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorModel.performances(
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
            "mission:sizing_mission:air_density", val=np.nan, units="kg/m**3"
        )
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:aerodynamics:CT:static", val=np.nan, units=None)
        self.add_input("data:propeller:aerodynamics:CP:static", val=np.nan, units=None)
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
        rho_air = inputs["mission:sizing_mission:air_density"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        C_t = inputs["data:propeller:aerodynamics:CT:static"]
        C_p = inputs["data:propeller:aerodynamics:CP:static"]

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
        F_pro = Mtotal * g / Npro
        W_pro, P_pro, Q_pro = PropellerModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )
        T_mot, W_mot, I_mot, U_mot, P_el = MotorModel.performances(
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


class ForwardSegment(om.ExplicitComponent):
    """
    Forward flight segment
    Assumption: forward velocity is unchanged from design scenario.
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
            "mission:sizing_mission:air_density", val=np.nan, units="kg/m**3"
        )
        self.add_input("data:aerodynamics:Cd", val=np.nan, units=None)
        self.add_input("data:aerodynamics:Cl0", val=np.nan, units=None)
        self.add_input("data:structure:body:surface:top", val=np.nan, units="m**2")
        self.add_input("data:structure:body:surface:front", val=np.nan, units="m**2")
        # Propeller parameters
        self.add_input("data:propeller:number", val=np.nan, units=None)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:geometry:beta", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:forward", val=np.nan, units=None)
        # self.add_input('data:propeller:aerodynamics:CT:incidence', val=np.nan, units=None)
        # self.add_input('data:propeller:aerodynamics:CP:incidence', val=np.nan, units=None)
        # Motor parameters
        self.add_input("data:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:motor:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:resistance", val=np.nan, units="V/A")
        self.add_input("data:motor:torque:coefficient", val=np.nan, units="N*m/A")
        # Mission parameters
        self.add_input(
            "mission:concordia_study:route%s:forward:range" % self.options["route"],
            val=np.nan,
            units="m",
        )
        self.add_input("mission:sizing_mission:forward:speed", val=np.nan, units="m/s")
        self.add_input("data:ESC:efficiency", val=np.nan, units=None)
        self.add_input("specifications:payload:power", val=0.0, units="W")
        self.add_input("data:avionics:power", val=0.0, units="W")
        # Outputs: energy
        self.add_output(
            "mission:concordia_study:route%s:forward:duration" % self.options["route"],
            units="min",
        )
        self.add_output(
            "mission:concordia_study:route%s:forward:energy:propulsion"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:forward:energy:payload"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:forward:energy:avionics"
            % self.options["route"],
            units="kJ",
        )
        self.add_output(
            "mission:concordia_study:route%s:forward:energy" % self.options["route"],
            units="kJ",
        )

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal = inputs["mission:concordia_study:route%s:tow" % self.options["route"]]
        rho_air = inputs["mission:sizing_mission:air_density"]
        C_D = inputs["data:aerodynamics:Cd"]
        C_L0 = inputs["data:aerodynamics:Cl0"]
        S_top = inputs["data:structure:body:surface:top"]
        S_front = inputs["data:structure:body:surface:front"]

        Npro = inputs["data:propeller:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        beta = inputs["data:propeller:geometry:beta"]
        J_ff = inputs[
            "data:propeller:advance_ratio:forward"
        ]  # TODO: compute new J_ff based on cruise speed and propeller diameter (requires solver to get C_t, C_p as well)

        N_red = inputs["data:gearbox:N_red"]
        Tf_mot = inputs["data:motor:torque:friction"]
        Kt_mot = inputs["data:motor:torque:coefficient"]
        R_mot = inputs["data:motor:resistance"]

        D_ff = inputs[
            "mission:concordia_study:route%s:forward:range" % self.options["route"]
        ]
        V_ff = inputs[
            "mission:sizing_mission:forward:speed"
        ]  # TODO: compute optimal speed
        t_ff = D_ff / V_ff  # [s]
        eta_ESC = inputs["data:ESC:efficiency"]
        P_payload = inputs["specifications:payload:power"]
        P_avionics = inputs["data:avionics:power"]

        # Compute new flight parameters
        thrust, alpha = FlightModel.get_thrust(
            Mtotal, V_ff, 0, S_front, S_top, C_D, C_L0, rho_air
        )  # flight parameters
        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_incidence(
            beta, J_ff, alpha
        )
        F_pro = thrust / Npro  # [N] thrust per propeller
        W_pro, P_pro, Q_pro = PropellerModel.performances(
            F_pro, D_pro, C_t, C_p, rho_air
        )  # propeller performances
        T_mot, W_mot, I_mot, U_mot, P_el = MotorModel.performances(
            Q_pro, W_pro, N_red, Tf_mot, Kt_mot, R_mot
        )  # motor performances

        # Energy consumption
        E_ff_pro = (P_el * Npro) / eta_ESC * t_ff  # [J] consumed energy for propulsion
        E_payload = P_payload / eta_ESC * t_ff  # [J] consumed energy for payload
        E_avionics = P_avionics / eta_ESC * t_ff  # [J] consumed energy for avionics

        outputs[
            "mission:concordia_study:route%s:forward:duration" % self.options["route"]
        ] = (
            t_ff / 60
        )  # [min]
        outputs[
            "mission:concordia_study:route%s:forward:energy:propulsion"
            % self.options["route"]
        ] = (
            E_ff_pro / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:forward:energy:payload"
            % self.options["route"]
        ] = (
            E_payload / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:forward:energy:avionics"
            % self.options["route"]
        ] = (
            E_avionics / 1000
        )  # [kJ]
        outputs[
            "mission:concordia_study:route%s:forward:energy" % self.options["route"]
        ] = (
            E_ff_pro + E_payload + E_avionics
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
