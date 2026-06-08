"""
Computes Mach number and unitary Reynolds.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from stdatm import Atmosphere


class ComputeUnitReynolds(ExplicitComponent):
    """
    Computes the mach number and reynolds number based on inputs and the ISA model.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if self.options["low_speed_aero"]:
            self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
            self.add_output("data:aerodynamics:low_speed:mach")
            self.add_output("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
        else:
            self.add_input("mission:sizing:main_route:cruise:speed:fixedwing", val=np.nan, units="m/s")
            self.add_input("mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
            self.add_output("data:aerodynamics:cruise:mach")
            self.add_output("data:aerodynamics:cruise:unit_reynolds", units="m**-1")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = inputs["data:TLAR:v_approach"] / Atmosphere(altitude).speed_of_sound
        else:
            altitude = float(inputs["mission:sizing:main_route:cruise:altitude"])
            mach = (
                inputs["mission:sizing:main_route:cruise:speed:fixedwing"]
                / Atmosphere(altitude, altitude_in_feet=False).speed_of_sound
            )

        atm = Atmosphere(altitude, altitude_in_feet=False)
        atm.mach = mach
        unit_reynolds = atm.unitary_reynolds

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:low_speed:mach"] = mach
            outputs["data:aerodynamics:low_speed:unit_reynolds"] = unit_reynolds
        else:
            outputs["data:aerodynamics:cruise:mach"] = mach
            outputs["data:aerodynamics:cruise:unit_reynolds"] = unit_reynolds

    
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        from stdatm import Atmosphere as AtmIsa, AtmosphereWithPartials
        
        if self.options["low_speed_aero"]:
            # Low-speed case: altitude = 0 (sea level)
            V = inputs["data:TLAR:v_approach"]
            altitude = 0.0
            
            atm = AtmIsa(altitude)
            a = atm.speed_of_sound
            rho = atm.density
            mu = atm.dynamic_viscosity
            
            mach = V / a
            
            # dMach/dV_approach = 1/a
            partials["data:aerodynamics:low_speed:mach", "data:TLAR:v_approach"] = 1.0 / a
            
            # dunit_reynolds/dV_approach = rho/mu
            partials["data:aerodynamics:low_speed:unit_reynolds", "data:TLAR:v_approach"] = rho / mu
            
        else:
            # Cruise case: altitude and speed are inputs
            V = inputs["mission:sizing:main_route:cruise:speed:fixedwing"]
            altitude = float(inputs["mission:sizing:main_route:cruise:altitude"])
            
            atm = AtmIsa(altitude, altitude_in_feet=False)
            a = atm.speed_of_sound
            rho = atm.density
            mu = atm.dynamic_viscosity
            T = atm.temperature
            
            mach = V / a
            unit_reynolds = rho * V / mu
            
            # dMach/dV = 1/a
            partials["data:aerodynamics:cruise:mach", 
                    "mission:sizing:main_route:cruise:speed:fixedwing"] = 1.0 / a
            
            # dMach/daltitude = -V * (da/daltitude) / a^2 = -mach * (da/daltitude) / a
            datm = AtmosphereWithPartials(altitude, 0.0, altitude_in_feet=False)
            da_dh = datm.partial_speed_of_sound_altitude
            partials["data:aerodynamics:cruise:mach",
                    "mission:sizing:main_route:cruise:altitude"] = -mach * da_dh / a
            
            # dunit_reynolds/dV = rho/mu
            partials["data:aerodynamics:cruise:unit_reynolds",
                    "mission:sizing:main_route:cruise:speed:fixedwing"] = rho / mu
            
            # dunit_reynolds/daltitude = d(rho/mu)/daltitude * V
            # d(rho/mu)/daltitude = (drho/dh * mu - rho * dmu/dh) / mu^2
            drho_dh = datm.partial_density_altitude
            
            # dmu/dh = mu * (1.5/T - 1/(T+S)) * dT/dh
            # where dT/dh = -L (temperature lapse rate = -0.0065 K/m)
            S_sutherland = 110.4
            L = 0.0065  # K/m
            dmu_dT = mu * (1.5 / T - 1.0 / (T + S_sutherland))
            dmu_dh = dmu_dT * (-L)
            
            d_rho_over_mu_dh = (drho_dh * mu - rho * dmu_dh) / (mu ** 2)
            partials["data:aerodynamics:cruise:unit_reynolds",
                    "mission:sizing:main_route:cruise:altitude"] = d_rho_over_mu_dh * V
    
