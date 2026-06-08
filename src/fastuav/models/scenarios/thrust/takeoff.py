"""
Takeoff scenarios
"""

import numpy as np
from scipy.constants import g
import openmdao.api as om
from stdatm import AtmosphereSI
from fastuav.constants import FW_PROPULSION, MR_PROPULSION

T0 = 288.15  # Sea level temperature (K)
L = 0.0065   # Lapse rate (K/m)


class VerticalTakeoffThrust(om.ExplicitComponent):
    """
    Thrust for the desired vertical takeoff acceleration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:thrust_weight_ratio:%s" % propulsion_id, val=np.nan, units=None)
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_output("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, units="N")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        k_maxthrust = inputs["mission:sizing:thrust_weight_ratio:%s" % propulsion_id]
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        F_pro_to = m_uav_guess * g / Npro * k_maxthrust  # [N] Thrust per propeller

        outputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id] = F_pro_to
    def compute_partials(self, inputs, partials):
        propulsion_id = self.options["propulsion_id"]
        k_maxthrust = inputs["mission:sizing:thrust_weight_ratio:%s" % propulsion_id]
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        partials["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, "mission:sizing:thrust_weight_ratio:%s" % propulsion_id] = m_uav_guess * g / Npro
        partials["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, "optimization:variables:weight:mtow:guess"] = g / Npro * k_maxthrust
        partials["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, "data:propulsion:%s:propeller:number" % propulsion_id] = -m_uav_guess * g / Npro**2 * k_maxthrust


class LauncherTakeoff(om.ExplicitComponent):
    """
    Thrust required for takeoff assuming the use of a rail launcher or bungee, in fixed wing configuration.
    The launching system brings the UAV at the required speed for takeoff (10% margin on stall speed).
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("mission:sizing:main_route:takeoff:altitude", val=0.0, units="m")
        self.add_input("mission:sizing:main_route:stall:speed:%s" % propulsion_id, val=np.nan, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("optimization:variables:aerodynamics:CDi:K:guess", val=0.035, units=None)
        self.add_output("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, units="N")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]

        # Flight parameters
        V_stall_ID = "mission:sizing:main_route:stall:speed:%s" % propulsion_id
        V_stall = inputs[V_stall_ID]
        altitude_takeoff = inputs["mission:sizing:main_route:takeoff:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_takeoff, dISA)
        atm.true_airspeed = 1.1 * V_stall  # 10% margin on the stall speed [kg/ms2]
        q_takeoff = atm.dynamic_pressure

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Weight = m_uav_guess * g  # [N]

        # Induced drag parameters
        K = inputs["optimization:variables:aerodynamics:CDi:K:guess"]

        # Parasitic drag parameters
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Thrust calculation (equilibrium)
        TW_takeoff = (
            q_takeoff * CD_0_guess / WS + K / q_takeoff * WS
        )  # thrust-to-weight ratio at takeoff  [-]
        F_pro_takeoff = TW_takeoff * Weight / Npro  # [N] Thrust per propeller for takeoff

        outputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id] = F_pro_takeoff

    def compute_partials(self, inputs, partials):
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]
        out_id = "data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id

        # Flight parameters
        V_stall = inputs["mission:sizing:main_route:stall:speed:%s" % propulsion_id]
        altitude_takeoff = inputs["mission:sizing:main_route:takeoff:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_takeoff, dISA)
        atm.true_airspeed = 1.1 * V_stall  # 10% margin on the stall speed [kg/ms2]
        q_takeoff = atm.dynamic_pressure
        rho = atm.density

        # ===== DENSITY DERIVATIVES (Analytic, Troposphere) =====
        # Temperature profile: T = T0 - L * h + dISA
        # Density varies as: rho ∝ (T/T0)^(g/RL - 1) --> (g/(R*L) - 1) ≈ -5.256 for ISA
        from stdatm import AtmosphereWithPartials
        datm = AtmosphereWithPartials(altitude_takeoff, dISA, altitude_in_feet=False)
        # drho = datm.density
        # d(rho)/d(h):   chain rule through T
        # d(rho)/d(h) = d(rho)/d(T) * d(T)/d(h) = (g/(R*L) - 1) * rho / T * (-L)
        drho_dh = datm.partial_density_altitude # d(rho)/d(altitude)
        # d(rho)/d(dISA): stdatm treats dISA as a pure temperature offset at fixed pressure
        drho_ddISA = -rho / (T0 - L * altitude_takeoff + dISA)

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Weight = m_uav_guess * g  # [N]

        # Induced drag parameters
        K = inputs["optimization:variables:aerodynamics:CDi:K:guess"]

        # Parasitic drag parameters
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Thrust calculation (equilibrium)
        TW_takeoff = q_takeoff * CD_0_guess / WS + K / q_takeoff * WS  # thrust-to-weight ratio at takeoff  [-]

        partials[out_id, "optimization:variables:aerodynamics:CD0:guess"] = q_takeoff / WS * Weight / Npro

        partials[out_id, "optimization:variables:aerodynamics:CDi:K:guess"] = WS / q_takeoff * Weight / Npro
        

        # ∂(q_stall)/∂(V_stall) = ∂(0.5 * ρ * V^2)/∂(V_stall) = ρ * V_stall
        partials[out_id, "mission:sizing:main_route:stall:speed:%s" % propulsion_id] = rho * 1.1**2 * V_stall*(CD_0_guess / WS - K / (q_takeoff**2) * WS)* Weight / Npro
        # ∂(q_stall)/∂(h) = 0.5 * V^2 * d(rho)/d(dISA)
        partials[out_id, "mission:sizing:main_route:takeoff:altitude"] = 0.5 * (1.1 * V_stall)**2 * drho_dh * (CD_0_guess / WS - K / (q_takeoff**2) * WS)* Weight / Npro
        # ∂(q_stall)/∂(dISA) = 0.5 * V^2 * d(rho)/d(dISA)
        partials[out_id, "mission:sizing:dISA"] = 0.5 * (1.1 * V_stall)**2 * drho_ddISA * (CD_0_guess / WS - K / (q_takeoff**2) * WS)* Weight / Npro


        partials[out_id, "data:geometry:wing:loading"] = (-q_takeoff*CD_0_guess / WS**2 + K / q_takeoff)* Weight / Npro

        

        partials[out_id, "optimization:variables:weight:mtow:guess"] = TW_takeoff * g / Npro
        partials[out_id, "data:propulsion:%s:propeller:number" % propulsion_id] = -TW_takeoff * Weight / Npro**2


