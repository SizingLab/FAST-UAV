"""
Estimation of cl/cm/oswald aero coefficients using VLM.
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
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

from .vlm import VLMSimpleGeometry
from ..xfoil.xfoil_polar import XfoilPolar
from ..neuralfoil.neuralfoil_polar import NeuralfoilPolar
from ...constants import SPAN_MESH_POINT, MACH_NB_PTS, DEFAULT_INPUT_AOA

_LOGGER = logging.getLogger(__name__)

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"


class ComputeAeroVLM(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        #self.options.declare("compute_mach_interpolation", default=False, types=bool)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare("input_angle_of_attack", default=DEFAULT_INPUT_AOA, types=float)
        self.options.declare("use_neuralfoil", default=False, types=bool)

    def setup(self):
        from ...components.compute_reynolds import ComputeUnitReynolds
        self.add_subsystem(
            "comp_unit_reynolds",
            ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_local_reynolds",
            ComputeLocalReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        # Selects the tool for airfoil analysis: uses NeuralFoil if 'use_neuralfoil' is True;
        # otherwise, uses Xfoil
        airfoil_polar = NeuralfoilPolar if self.options["use_neuralfoil"] else XfoilPolar
        if self.options["low_speed_aero"]:
            self.add_subsystem(
                "wing_polar_ls",
                airfoil_polar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["wing_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.add_subsystem(
                "htp_polar_ls",
                airfoil_polar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["htp_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
        else:
            self.add_subsystem(
                "wing_polar_hs",
                airfoil_polar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["wing_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.add_subsystem(
                "htp_polar_hs",
                airfoil_polar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["htp_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
        self.add_subsystem(
            "aero_vlm",
            _ComputeAeroVLM(
                low_speed_aero=self.options["low_speed_aero"],
                result_folder_path=self.options["result_folder_path"],
                #compute_mach_interpolation=self.options["compute_mach_interpolation"],
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                htp_airfoil_file=self.options["htp_airfoil_file"],
                input_angle_of_attack=self.options["input_angle_of_attack"],
            ),
            promotes=["*"],
        )

        if self.options["low_speed_aero"]:
            self.connect("data:aerodynamics:low_speed:mach", "wing_polar_ls.mach")
            self.connect(
                "data:aerodynamics:wing:low_speed:reynolds",
                "wing_polar_ls.reynolds",
            )
            self.connect("wing_polar_ls.CL", "data:aerodynamics:wing:low_speed:CL")
            self.connect("wing_polar_ls.CDp", "data:aerodynamics:wing:low_speed:CDp")
            self.connect("data:aerodynamics:low_speed:mach", "htp_polar_ls.mach")
            self.connect(
                "data:aerodynamics:horizontal_tail:low_speed:reynolds",
                "htp_polar_ls.reynolds",
            )
            self.connect(
                "htp_polar_ls.CL",
                "data:aerodynamics:horizontal_tail:low_speed:CL",
            )
            self.connect(
                "htp_polar_ls.CDp",
                "data:aerodynamics:horizontal_tail:low_speed:CDp",
            )
        else:
            self.connect("data:aerodynamics:cruise:mach", "wing_polar_hs.mach")
            self.connect(
                "data:aerodynamics:wing:cruise:reynolds",
                "wing_polar_hs.reynolds",
            )
            self.connect("wing_polar_hs.CL", "data:aerodynamics:wing:cruise:CL")
            self.connect("wing_polar_hs.CDp", "data:aerodynamics:wing:cruise:CDp")
            self.connect("data:aerodynamics:cruise:mach", "htp_polar_hs.mach")
            self.connect(
                "data:aerodynamics:horizontal_tail:cruise:reynolds",
                "htp_polar_hs.reynolds",
            )
            self.connect(
                "htp_polar_hs.CL",
                "data:aerodynamics:horizontal_tail:cruise:CL",
            )
            self.connect(
                "htp_polar_hs.CDp",
                "data:aerodynamics:horizontal_tail:cruise:CDp",
            )


class ComputeLocalReynolds(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        else:
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:MAC:length", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:reynolds")
        else:
            self.add_output("data:aerodynamics:wing:cruise:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:reynolds")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:reynolds"] = (
                inputs["data:aerodynamics:low_speed:unit_reynolds"]
                * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:low_speed:reynolds"] = (
                inputs["data:aerodynamics:low_speed:unit_reynolds"]
                * inputs["data:geometry:tail:horizontal:MAC:length"]#* inputs["data:geometry:horizontal_tail:MAC:length"]
            )
        else:
            outputs["data:aerodynamics:wing:cruise:reynolds"] = (
                inputs["data:aerodynamics:cruise:unit_reynolds"]
                * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:cruise:reynolds"] = (
                inputs["data:aerodynamics:cruise:unit_reynolds"]
                * inputs["data:geometry:tail:horizontal:MAC:length"]#* inputs["data:geometry:horizontal_tail:MAC:length"]
            )
    
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if self.options["low_speed_aero"]:
            Re_unit = inputs["data:aerodynamics:low_speed:unit_reynolds"]
            c_MAC_w = inputs["data:geometry:wing:MAC:length"]
            c_MAC_ht = inputs["data:geometry:tail:horizontal:MAC:length"]

            # Wing Reynolds: Re_w = Re_unit * c_MAC_w
            out_w = "data:aerodynamics:wing:low_speed:reynolds"
            partials[out_w, "data:aerodynamics:low_speed:unit_reynolds"] = c_MAC_w
            partials[out_w, "data:geometry:wing:MAC:length"] = Re_unit
            partials[out_w, "data:geometry:tail:horizontal:MAC:length"] = 0.0

            # Horizontal tail Reynolds: Re_ht = Re_unit * c_MAC_ht
            out_ht = "data:aerodynamics:horizontal_tail:low_speed:reynolds"
            partials[out_ht, "data:aerodynamics:low_speed:unit_reynolds"] = c_MAC_ht
            partials[out_ht, "data:geometry:wing:MAC:length"] = 0.0
            partials[out_ht, "data:geometry:tail:horizontal:MAC:length"] = Re_unit

        else:
            Re_unit = inputs["data:aerodynamics:cruise:unit_reynolds"]
            c_MAC_w = inputs["data:geometry:wing:MAC:length"]
            c_MAC_ht = inputs["data:geometry:tail:horizontal:MAC:length"]

            # Wing Reynolds: Re_w = Re_unit * c_MAC_w
            out_w = "data:aerodynamics:wing:cruise:reynolds"
            partials[out_w, "data:aerodynamics:cruise:unit_reynolds"] = c_MAC_w
            partials[out_w, "data:geometry:wing:MAC:length"] = Re_unit
            partials[out_w, "data:geometry:tail:horizontal:MAC:length"] = 0.0

            # Horizontal tail Reynolds: Re_ht = Re_unit * c_MAC_ht
            out_ht = "data:aerodynamics:horizontal_tail:cruise:reynolds"
            partials[out_ht, "data:aerodynamics:cruise:unit_reynolds"] = c_MAC_ht
            partials[out_ht, "data:geometry:wing:MAC:length"] = 0.0
            partials[out_ht, "data:geometry:tail:horizontal:MAC:length"] = Re_unit


class _ComputeAeroVLM(VLMSimpleGeometry):
    def initialize(self):
        super().initialize()
        self.options.declare("result_folder_path", default="", types=str)
        #self.options.declare("compute_mach_interpolation", default=False, types=bool)
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
            self.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient")
            # if self.options["compute_mach_interpolation"]:
            #     self.add_output(
            #         "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
            #         shape=MACH_NB_PTS + 1,
            #     )
            #     self.add_output(
            #         "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
            #         shape=MACH_NB_PTS + 1,
            #         units="rad**-1",
            #     )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
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
        ) = self.compute_aero_coeff(inputs, altitude, mach, input_aoa)

        # if self.options["low_speed_aero"]:
        #     pass
        # else:
        #     if self.options["compute_mach_interpolation"]:
        #         mach_interp, cl_alpha_interp = self.compute_cl_alpha_mach(
        #             inputs, input_aoa, altitude, mach
        #         )

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
            outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = (
                coef_k_htp
            )
            # if self.options["compute_mach_interpolation"]:
            #     outputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"] = mach_interp
            #     outputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"] = (
            #         cl_alpha_interp
            #     )
