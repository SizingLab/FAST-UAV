"""
Estimation of aero coefficients using OPENVSP.
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

import logging

import numpy as np
from openmdao.core.group import Group

from .openvsp import DEFAULT_HTP_AIRFOIL, DEFAULT_WING_AIRFOIL, OpenVSPSimpleGeometry
from ...components.compute_reynolds import ComputeUnitReynolds
from ...constants import DEFAULT_INPUT_AOA, MACH_NB_PTS, SPAN_MESH_POINT

_LOGGER = logging.getLogger(__name__)


class ComputeAeroOpenVSP(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("compute_mach_interpolation", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare("input_angle_of_attack", default=DEFAULT_INPUT_AOA, types=float)

    def setup(self):
        self.add_subsystem(
            "comp_unit_reynolds",
            ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "aero_openvsp",
            _ComputeAeroOpenVSP(
                low_speed_aero=self.options["low_speed_aero"],
                compute_mach_interpolation=self.options["compute_mach_interpolation"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                htp_airfoil_file=self.options["htp_airfoil_file"],
                input_angle_of_attack=self.options["input_angle_of_attack"],
            ),
            promotes=["*"],
        )


class _ComputeAeroOpenVSP(OpenVSPSimpleGeometry):
    def initialize(self):
        super().initialize()
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("compute_mach_interpolation", default=False, types=bool)
        self.options.declare("input_angle_of_attack", default=DEFAULT_INPUT_AOA, types=float)

    def setup(self):
        super().setup()
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:CL_ref")
            self.add_output("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:CM0_clean")
            self.add_output(
                "data:aerodynamics:wing:low_speed:Y_vector", shape=SPAN_MESH_POINT, units="m"
            )
            self.add_output("data:aerodynamics:wing:low_speed:CL_vector", shape=SPAN_MESH_POINT)
            self.add_output(
                "data:aerodynamics:wing:low_speed:chord_vector", shape=SPAN_MESH_POINT, units="m"
            )
            self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
            )
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:CL_vector", shape=SPAN_MESH_POINT
            )
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient")
        else:
            self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CL_ref")
            self.add_output("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CDi:k")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
            self.add_output("data:aerodynamics:tail:horizontal:cruise:CL_alpha", units="rad**-1")
            self.add_output(
                "data:aerodynamics:tail:horizontal:cruise:CL_alpha_isolated", units="rad**-1"
            )
            self.add_output("data:aerodynamics:tail:horizontal:cruise:CDi:k")
            if self.options["compute_mach_interpolation"]:
                self.add_output(
                    "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                    shape=MACH_NB_PTS + 1,
                )
                self.add_output(
                    "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                    shape=MACH_NB_PTS + 1,
                    units="rad**-1",
                )

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):
        _LOGGER.debug("Entering aerodynamic computation")
        input_aoa = self.options["input_angle_of_attack"]

        # Check AOA input is float
        if not isinstance(input_aoa, float):
            raise TypeError("Option input_angle_of_attack should be a float!")

        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["mission:sizing:main_route:cruise:altitude"]
            mach = inputs["data:aerodynamics:cruise:mach"]

        (
            cl_0_wing,
            cl_ref_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_X_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
            _,
        ) = self.compute_aero_coeff(inputs, outputs, altitude, mach, input_aoa)

        if not self.options["low_speed_aero"] and self.options["compute_mach_interpolation"]:
            mach_interp, cl_alpha_interp = self.compute_cl_alpha_mach(
                inputs, outputs, input_aoa, altitude, mach
            )
            outputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"] = mach_interp
            outputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"] = (
                cl_alpha_interp
            )

        # Defining outputs
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:CL0_clean"] = cl_0_wing
            outputs["data:aerodynamics:wing:low_speed:CL_ref"] = cl_ref_wing
            outputs["data:aerodynamics:wing:low_speed:CL_alpha"] = cl_alpha_wing
            outputs["data:aerodynamics:wing:low_speed:CM0_clean"] = cm_0_wing
            outputs["data:aerodynamics:wing:low_speed:Y_vector"] = y_vector_wing
            outputs["data:aerodynamics:wing:low_speed:CL_vector"] = cl_vector_wing
            outputs["data:aerodynamics:wing:low_speed:chord_vector"] = chord_vector_wing
            outputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] = coef_k_wing
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL0"] = cl_0_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_ref"] = cl_X_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"] = cl_alpha_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"] = (
                cl_alpha_htp_isolated
            )
            outputs["data:aerodynamics:horizontal_tail:low_speed:Y_vector"] = y_vector_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_vector"] = cl_vector_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"] = (
                coef_k_htp
            )
        else:
            outputs["data:aerodynamics:wing:cruise:CL0_clean"] = cl_0_wing
            outputs["data:aerodynamics:wing:cruise:CL_ref"] = cl_ref_wing
            outputs["data:aerodynamics:wing:cruise:CL_alpha"] = cl_alpha_wing
            outputs["data:aerodynamics:wing:cruise:CM0_clean"] = cm_0_wing
            outputs["data:aerodynamics:wing:cruise:CDi:k"] = coef_k_wing
            outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = cl_0_htp
            outputs["data:aerodynamics:tail:horizontal:cruise:CL_alpha"] = cl_alpha_htp
            outputs["data:aerodynamics:tail:horizontal:cruise:CL_alpha_isolated"] = (
                cl_alpha_htp_isolated
            )
            outputs["data:aerodynamics:tail:horizontal:cruise:CDi:k"] = coef_k_htp
