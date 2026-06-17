"""
Estimation of the slope of the airfoil of the lifting surface using the results of Xfoil
/ Neuralfoil runs.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from ..constants import POLAR_POINT_COUNT, SUBMODEL_AIRFOIL_LIFT_SLOPE
from ..external.neuralfoil.neuralfoil_polar import NeuralfoilPolar
from ..external.xfoil.xfoil_polar import XfoilPolar

ALPHA_START_LINEAR = np.deg2rad(-5.0)
ALPHA_END_LINEAR = np.deg2rad(10.0)

_LOGGER = logging.getLogger(__name__)

oad.RegisterSubmodel.active_models[SUBMODEL_AIRFOIL_LIFT_SLOPE] = (
    "fastga.submodel.aerodynamics.airfoil.all.lift_curve_slope"
)


@oad.RegisterSubmodel(
    SUBMODEL_AIRFOIL_LIFT_SLOPE, "fastga.submodel.aerodynamics.airfoil.all.lift_curve_slope"
)
class ComputeAirfoilLiftCurveSlope(om.Group):
    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default="naca23012.af", types=str, allow_none=True
        )
        self.options.declare("htp_airfoil_file", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil_file", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("use_neuralfoil", default=False, types=bool)

    def setup(self):
        self.add_subsystem(
            "comp_local_reynolds_airfoil",
            ComputeLocalReynolds(),
            promotes=[
                "data:aerodynamics:low_speed:mach",
                "data:aerodynamics:low_speed:unit_reynolds",
                "data:geometry:wing:MAC:length",
                "data:geometry:tail:horizontal:MAC:length",  # "data:geometry:horizontal_tail:MAC:length"
                "data:aerodynamics:horizontal_tail:efficiency",
                "data:geometry:tail:vertical:MAC:length",  # "data:geometry:vertical_tail:MAC:length"
                "data:aerodynamics:wing:MAC:low_speed:reynolds",
                "data:aerodynamics:horizontal_tail:MAC:low_speed:reynolds",
                "data:aerodynamics:vertical_tail:MAC:low_speed:reynolds",
            ],
        )

        # Selects the tool for airfoil analysis: uses NeuralFoil if 'use_neuralfoil' is True;
        # otherwise, uses Xfoil
        airfoil_polar = NeuralfoilPolar if self.options["use_neuralfoil"] else XfoilPolar

        self.add_subsystem(
            "wing_airfoil_slope",
            airfoil_polar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                airfoil_file=self.options["wing_airfoil_file"],
                activate_negative_angle=True,
                alpha_end=20.0,
            ),
            promotes=[],
        )
        self.add_subsystem(
            "htp_airfoil_slope",
            airfoil_polar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                airfoil_file=self.options["htp_airfoil_file"],
                activate_negative_angle=True,
                alpha_end=20.0,
            ),
            promotes=[],
        )
        self.add_subsystem(
            "vtp_airfoil_slope",
            airfoil_polar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                airfoil_file=self.options["vtp_airfoil_file"],
                activate_negative_angle=True,
                alpha_end=20.0,
            ),
            promotes=[],
        )

        self.add_subsystem(
            "airfoil_lift_slope",
            _ComputeAirfoilLiftCurveSlope(),
            promotes=["*"],
        )

        self.connect("comp_local_reynolds_airfoil.mach", "wing_airfoil_slope.mach")
        self.connect(
            "data:aerodynamics:wing:MAC:low_speed:reynolds",
            "wing_airfoil_slope.reynolds",
        )
        self.connect("comp_local_reynolds_airfoil.mach", "htp_airfoil_slope.mach")
        self.connect(
            "data:aerodynamics:horizontal_tail:MAC:low_speed:reynolds",
            "htp_airfoil_slope.reynolds",
        )
        self.connect("comp_local_reynolds_airfoil.mach", "vtp_airfoil_slope.mach")
        self.connect(
            "data:aerodynamics:vertical_tail:MAC:low_speed:reynolds",
            "vtp_airfoil_slope.reynolds",
        )

        self.connect("wing_airfoil_slope.alpha", "wing:alpha")
        self.connect("wing_airfoil_slope.CL", "wing:CL")
        self.connect("htp_airfoil_slope.alpha", "horizontal_tail:alpha")
        self.connect("htp_airfoil_slope.CL", "horizontal_tail:CL")
        self.connect("vtp_airfoil_slope.alpha", "vertical_tail:alpha")
        self.connect("vtp_airfoil_slope.CL", "vertical_tail:CL")


class ComputeLocalReynolds(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:MAC:length", val=np.nan, units="m")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=0.9)
        # self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:vertical:MAC:length", val=np.nan, units="m")

        self.add_output("data:aerodynamics:wing:MAC:low_speed:reynolds")
        self.add_output("data:aerodynamics:horizontal_tail:MAC:low_speed:reynolds")
        self.add_output("data:aerodynamics:vertical_tail:MAC:low_speed:reynolds")

        self.add_output(name="mach")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:wing:MAC:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs["data:geometry:wing:MAC:length"]
        )
        outputs["data:aerodynamics:horizontal_tail:MAC:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs[
                "data:geometry:tail:horizontal:MAC:length"
            ]  # inputs["data:geometry:horizontal_tail:MAC:length"]
            * np.sqrt(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        )
        outputs["data:aerodynamics:vertical_tail:MAC:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs[
                "data:geometry:tail:vertical:MAC:length"
            ]  # * inputs["data:geometry:vertical_tail:MAC:length"]
            * np.sqrt(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        )
        outputs["mach"] = inputs["data:aerodynamics:low_speed:mach"]


class _ComputeAirfoilLiftCurveSlope(om.ExplicitComponent):
    """Lift curve slope coefficient from Xfoil or Neuralfoil polars."""

    def setup(self):
        nans_array = np.full(POLAR_POINT_COUNT, np.nan)

        self.add_input(name="wing:alpha", val=nans_array, shape=POLAR_POINT_COUNT, units="rad")
        self.add_input(name="wing:CL", val=nans_array, shape=POLAR_POINT_COUNT)
        self.add_input(
            name="horizontal_tail:alpha",
            val=nans_array,
            shape=POLAR_POINT_COUNT,
            units="rad",
        )
        self.add_input(name="horizontal_tail:CL", val=nans_array, shape=POLAR_POINT_COUNT)
        self.add_input(
            name="vertical_tail:alpha",
            val=nans_array,
            shape=POLAR_POINT_COUNT,
            units="rad",
        )
        self.add_input(name="vertical_tail:CL", val=nans_array, shape=POLAR_POINT_COUNT)

        self.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1")
        self.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1")
        self.add_output("data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_cl_orig = inputs["wing:CL"]
        wing_alpha_orig = inputs["wing:alpha"]
        wing_alpha, wing_cl = self.delete_additional_zeros(
            np.array(wing_alpha_orig), np.array(wing_cl_orig)
        )
        wing_alpha, wing_cl = self.rearrange_data(wing_alpha, wing_cl)
        index_start_wing = int(np.min(np.where(wing_alpha >= ALPHA_START_LINEAR)))
        index_end_wing = int(np.max(np.where(wing_alpha <= ALPHA_END_LINEAR)))
        wing_airfoil_cl_alpha_array = (
            wing_cl[index_start_wing + 1 : index_end_wing] - wing_cl[index_start_wing]
        ) / (wing_alpha[index_start_wing + 1 : index_end_wing] - wing_alpha[index_start_wing])
        wing_airfoil_cl_alpha = np.mean(wing_airfoil_cl_alpha_array)

        htp_cl_orig = inputs["horizontal_tail:CL"]
        htp_alpha_orig = inputs["horizontal_tail:alpha"]
        htp_alpha, htp_cl = self.delete_additional_zeros(
            np.array(htp_alpha_orig), np.array(htp_cl_orig)
        )
        htp_alpha, htp_cl = self.rearrange_data(htp_alpha, htp_cl)
        index_start_htp = int(np.min(np.where(htp_alpha >= ALPHA_START_LINEAR)))
        index_end_htp = int(np.max(np.where(htp_alpha <= ALPHA_END_LINEAR)))
        htp_airfoil_cl_alpha_array = (
            htp_cl[index_start_htp + 1 : index_end_htp] - htp_cl[index_start_htp]
        ) / (htp_alpha[index_start_htp + 1 : index_end_htp] - htp_alpha[index_start_htp])
        htp_airfoil_cl_alpha = np.mean(htp_airfoil_cl_alpha_array)

        vtp_cl_orig = inputs["horizontal_tail:CL"]
        vtp_alpha_orig = inputs["horizontal_tail:alpha"]
        vtp_alpha, vtp_cl = self.delete_additional_zeros(
            np.array(vtp_alpha_orig), np.array(vtp_cl_orig)
        )
        vtp_alpha, vtp_cl = self.rearrange_data(vtp_alpha, vtp_cl)
        index_start_vtp = int(np.min(np.where(vtp_alpha >= ALPHA_START_LINEAR)))
        index_end_vtp = int(np.max(np.where(vtp_alpha <= ALPHA_END_LINEAR)))
        vtp_airfoil_cl_alpha_array = (
            vtp_cl[index_start_vtp + 1 : index_end_vtp] - vtp_cl[index_start_vtp]
        ) / (vtp_alpha[index_start_vtp + 1 : index_end_vtp] - vtp_alpha[index_start_vtp])
        vtp_airfoil_cl_alpha = np.mean(vtp_airfoil_cl_alpha_array)

        outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = htp_airfoil_cl_alpha
        outputs["data:aerodynamics:vertical_tail:airfoil:CL_alpha"] = vtp_airfoil_cl_alpha
        outputs["data:aerodynamics:wing:airfoil:CL_alpha"] = wing_airfoil_cl_alpha

    @staticmethod
    def delete_additional_zeros(array_alpha, array_cl):
        """
        Function that delete the additional zeros we had to add to fit the format imposed by
        OpenMDAO in both the alpha and CL array simultaneously

        @param array_alpha: an array with the alpha values and the additional zeros we want to
        delete
        @param array_cl: the corresponding Cl array @return: final_array_alpha an array containing
        the same alphas of the initial array but with the additional zeros deleted
        @return: final_array_CL an array containing the corresponding CL.
        """

        non_zero_array = np.where(array_alpha != 0)
        if array_alpha[0] == 0.0:
            valid_data_index = np.insert(non_zero_array, 0, 0)
        else:
            valid_data_index = non_zero_array

        final_array_alpha = array_alpha[valid_data_index]
        final_array_cl = array_cl[valid_data_index]

        return final_array_alpha, final_array_cl

    @staticmethod
    def rearrange_data(array_alpha, array_cl):
        """
        Function that rearrange the data so that the alpha array is sorted and the cl array is
        rearranged accordingly

        @param array_alpha: an array with the alpha values in potentially the wrong order
        @param array_cl: the corresponding Cl array
        @return: final_array_alpha an array containing the same alphas of the initial array but
        sorted
        @return: final_array_CL an array containing the corresponding CL in the corresponding
        position.
        """

        sorter = np.zeros((len(array_alpha), 2))
        sorter[:, 0] = array_alpha
        sorter[:, 1] = array_cl
        sorted_alpha = np.sort(sorter, axis=0)
        for alpha in sorted_alpha[:, 0]:
            index_alpha_new = np.where(sorted_alpha[:, 0] == alpha)
            index_alpha_orig = np.where(array_alpha == alpha)
            sorted_alpha[index_alpha_new, 1] = array_cl[index_alpha_orig]

        return sorted_alpha[:, 0], sorted_alpha[:, 1]
