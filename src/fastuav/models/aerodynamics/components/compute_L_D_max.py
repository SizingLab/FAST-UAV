"""Estimation of the optimal aerodynamics configuration for the aircraft in cruise condition."""
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

import fastoad.api as oad
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

from ..constants import SUBMODEL_MAX_L_D


@oad.RegisterSubmodel(SUBMODEL_MAX_L_D, "fastga.submodel.aerodynamics.aircraft.l_d_max.legacy")
class ComputeLDMax(ExplicitComponent):
    """
    Computes optimal CL/CD aerodynamic performance of the aircraft in cruise conditions.

    """

    def setup(self):
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CDi:k", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:cruise:L_D_max")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CL")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CD")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")

        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:optimal_CL",
            [
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:wing:cruise:CDi:k",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:optimal_CD",
            [
                "data:aerodynamics:aircraft:cruise:CD0",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:L_D_max",
            [
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:wing:cruise:CDi:k",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:optimal_alpha",
            [
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:wing:cruise:CDi:k",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:wing:cruise:CL0_clean",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # TODO: to be written with momentum equilibrium formula to consider htp drag
        cl0_clean = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coeff_k = inputs["data:aerodynamics:wing:cruise:CDi:k"]

        cl_opt = np.sqrt(cd0 / coeff_k)
        alpha_opt = (cl_opt - cl0_clean) / cl_alpha * 180 / np.pi
        cd_opt = cd0 + coeff_k * cl_opt**2

        outputs["data:aerodynamics:aircraft:cruise:L_D_max"] = cl_opt / cd_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_CL"] = cl_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_CD"] = cd_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_alpha"] = alpha_opt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cl0_clean = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coeff_k = inputs["data:aerodynamics:wing:cruise:CDi:k"]

        cl_opt = np.sqrt(cd0 / coeff_k)
        d_cl_opt_d_cd0 = 0.5 * np.sqrt(1.0 / (coeff_k * cd0))
        d_cl_opt_d_coeff_k = -0.5 * np.sqrt(cd0) * coeff_k**-1.5

        partials[
            "data:aerodynamics:aircraft:cruise:optimal_CL", "data:aerodynamics:aircraft:cruise:CD0"
        ] = d_cl_opt_d_cd0
        partials[
            "data:aerodynamics:aircraft:cruise:optimal_CL",
            "data:aerodynamics:wing:cruise:CDi:k",
        ] = d_cl_opt_d_coeff_k

        partials[
            "data:aerodynamics:aircraft:cruise:optimal_CD", "data:aerodynamics:aircraft:cruise:CD0"
        ] = 2.0

        partials[
            "data:aerodynamics:aircraft:cruise:L_D_max", "data:aerodynamics:aircraft:cruise:CD0"
        ] = -0.25 * coeff_k**-0.5 * cd0**-1.5
        partials[
            "data:aerodynamics:aircraft:cruise:L_D_max",
            "data:aerodynamics:wing:cruise:CDi:k",
        ] = -0.25 * coeff_k**-1.5 * cd0**-0.5

        partials[
            "data:aerodynamics:aircraft:cruise:optimal_alpha",
            "data:aerodynamics:aircraft:cruise:CD0",
        ] = d_cl_opt_d_cd0 / cl_alpha * 180 / np.pi
        partials[
            "data:aerodynamics:aircraft:cruise:optimal_alpha",
            "data:aerodynamics:wing:cruise:CDi:k",
        ] = d_cl_opt_d_coeff_k / cl_alpha * 180 / np.pi
        partials[
            "data:aerodynamics:aircraft:cruise:optimal_alpha",
            "data:aerodynamics:wing:cruise:CL0_clean",
        ] = -1.0 / cl_alpha * 180 / np.pi
        partials[
            "data:aerodynamics:aircraft:cruise:optimal_alpha",
            "data:aerodynamics:wing:cruise:CL_alpha",
        ] = -(cl_opt - cl0_clean) / cl_alpha**2.0 * 180.0 / np.pi
