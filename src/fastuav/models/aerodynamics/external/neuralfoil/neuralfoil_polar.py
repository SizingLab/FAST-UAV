"""
Computation of the airfoil aerodynamic properties using Neuralfoil from :cite:`neuralfoil:2023`.
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
import numpy as np
import neuralfoil as nf
from typing import Tuple
import tempfile
import os

import openmdao.api as om
from pathlib import Path
from fastuav.models.aerodynamics import airfoil_folder
from ...constants import (
    POLAR_POINT_COUNT,
    OPTION_ALPHA_START,
    OPTION_ALPHA_END,
    OPTION_COMP_NEG_AIR_SYM,
    _DEFAULT_AIRFOIL_FILE,
    ALPHA_STEP,
    DEFAULT_2D_CL_MAX,
    DEFAULT_2D_CL_MIN,
)


_LOGGER = logging.getLogger(__name__)


class NeuralfoilPolar(om.ExplicitComponent):
    """Runs a polar computation with Neuralfoil and returns 2D aerodynamic coefficients."""

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=90.0, types=float)
        self.options.declare(OPTION_COMP_NEG_AIR_SYM, default=False, types=bool)
        self.options.declare(
            "single_AoA",
            default=False,
            types=bool,
            desc="If only one angle of attack is required, Cl_max_2D and Cl_min_2D won't be "
            "returned and results won't be written in the resources nor will they be read from "
            "them. In addition to that, options "
            + OPTION_ALPHA_START
            + " and "
            + OPTION_ALPHA_END
            + " must match",
        )

    def setup(self):
        """
        Set up inputs and outputs required for this operation.The pressure drag coefficient
        (CDp) is set to zero in all computations, as NeuralFoil does not provide it, ensuring
        backward compatibility.
        """

        multiple_aoa = not self.options["single_AoA"]

        self.add_input("reynolds", val=np.nan)
        self.add_input("mach", val=np.nan)

        if multiple_aoa:
            self.add_output("alpha", shape=POLAR_POINT_COUNT, units="deg")
            self.add_output("CL", shape=POLAR_POINT_COUNT)
            self.add_output("CD", shape=POLAR_POINT_COUNT)
            self.add_output("CDp", val=np.zeros(POLAR_POINT_COUNT))
            self.add_output("CM", shape=POLAR_POINT_COUNT)
            self.add_output("CL_max_2D")
            self.add_output("CL_min_2D")
            self.add_output("CD_min_2D")

        else:
            self.add_output("alpha", units="deg")
            self.add_output("CL")
            self.add_output("CD")
            self.add_output("CDp", val=0.0)
            self.add_output("CM")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Function that computes airfoil aerodynamics with NeuralFoil and returns the different 2D
        aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        """

        # Get inputs and initialise outputs
        mach = inputs["mach"]
        reynolds = inputs["reynolds"]

        # Compressibility correction
        beta = np.sqrt(1.0 - mach**2.0)

        # Modify file type respect to negative AoA/inviscid/single AoA options
        multiple_aoa = not self.options["single_AoA"]
        alpha_start = self.options[OPTION_ALPHA_START]
        alpha_end = self.options[OPTION_ALPHA_END]

        alpha = self.options[OPTION_ALPHA_START]
        if self.options["airfoil_folder_path"] is None:
            self.options["airfoil_folder_path"] = airfoil_folder.__path__[0]
        airfoil_path = Path(self.options["airfoil_folder_path"]) / self.options["airfoil_file"]
        airfoil_path = self._create_temp_airfoil_file(airfoil_path)

        if multiple_aoa:
            alpha = np.arange(
                -alpha_end if self.options[OPTION_COMP_NEG_AIR_SYM] else alpha_start,
                alpha_end + ALPHA_STEP,
                ALPHA_STEP,
            )
        _LOGGER.info("Entering Neuralfoil computation")
        results = nf.get_aero_from_dat_file(airfoil_path, alpha=alpha, Re=reynolds)
        # Only apply for Cl and Cm based on the documentation of Neuralfoil
        cl = results["CL"] / (
            beta + mach**2.0 / beta * 0.5 * results["CL"] * (1.0 + 0.2 * mach**2.0)
        )
        cd = results["CD"]
        cm = results["CM"] / (
            beta + mach**2.0 / beta * 0.5 * results["CM"] * (1.0 + 0.2 * mach**2.0)
        )

        if multiple_aoa:
            results["AoA"] = alpha
            cl_max_2d, _ = self._get_max_cl(alpha, cl)
            cl_min_2d, _ = self._get_min_cl(alpha, cl)
            cd_min_2d = np.min(cd)
            alpha, cl, cd, cm = self._fix_calculation_result_length(results)

        # Defining outputs -------------------------------------------------------------------------
        outputs["alpha"] = alpha
        outputs["CL"] = cl
        outputs["CD"] = cd
        outputs["CM"] = cm
        if multiple_aoa:
            outputs["CL_max_2D"] = cl_max_2d
            outputs["CL_min_2D"] = cl_min_2d
            outputs["CD_min_2D"] = cd_min_2d

        if os.path.exists(airfoil_path):
            os.remove(airfoil_path)

    @staticmethod
    def _fix_calculation_result_length(computed_result):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise filled with zeros.

        :param interpolated_result: result dataset with labels and values

        :return: length-modified aerodynamic characteristic array
        """

        # Extract results

        alpha = computed_result["AoA"]
        cl = computed_result["CL"]
        cd = computed_result["CD"]
        cm = computed_result["CM"]

        # Modify vector length if necessary
        if POLAR_POINT_COUNT < len(alpha):
            alpha = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha, alpha, cl)
            cd = np.interp(alpha, alpha, cd)
            cm = np.interp(alpha, alpha, cm)
        else:
            filler = np.zeros(POLAR_POINT_COUNT - len(alpha))
            alpha = np.append(alpha, filler)
            cl = np.append(cl, filler)
            cd = np.append(cd, filler)
            cm = np.append(cm, filler)

        return alpha, cl, cd, cm

    def _get_max_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """
        :param alpha:
        :param lift_coeff: CL
        :return: max CL within +/- 0.3 around linear zone if enough alpha computed, or default value
        otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]

        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            alpha = self._take_second_half(alpha)
            lift_coeff = self._take_second_half(lift_coeff)

        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if np.abs(covered_range / alpha_range) >= 0.4:

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

                delta = np.abs(lift_coeff - lift_fct(alpha))
                filtered = lift_coeff[delta <= 0.3]
                if len(filtered) > 0:
                    return max(filtered), False

        _LOGGER.warning(
            "2D CL max not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MAX,
        )
        return DEFAULT_2D_CL_MAX, True

    def _get_min_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """
        :param alpha:
        :param lift_coeff: CL

        :return: min CL +/- 0.3 around linear zone if enough alpha computed, or default value
        otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]

        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            alpha = self._take_first_half(alpha)
            lift_coeff = self._take_first_half(lift_coeff)

        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if covered_range / alpha_range >= 0.4:

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

                delta = np.abs(lift_coeff - lift_fct(alpha))
                filtered = lift_coeff[delta <= 0.3]
                if len(filtered) > 0:
                    return min(filtered), False

        _LOGGER.warning(
            "2D CL min not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MIN,
        )
        return DEFAULT_2D_CL_MIN, True

    def _create_temp_airfoil_file(self, original_file_path):
        """
        Convert an airfoil coordinate file to Selig format and create a temporary file.

        This function reads airfoil coordinates from a file, separates them into upper and lower
        surfaces, and writes them in Selig format to a temporary file. The Selig format consists
        of upper surface coordinates (trailing edge to leading edge) followed by a blank line,
        then lower surface coordinates (leading edge to trailing edge).

        Args:
            original_file_path (str): Path to the original airfoil coordinate file.
                                     File should contain coordinate pairs (x, y) on separate lines.

        Returns:
            str: Path to the temporary file containing the converted airfoil in Selig format.

        Raises:
            ValueError: If the file contains fewer than 3 coordinate points.
            FileNotFoundError: If the original file path does not exist.

        """

        def is_coordinate_line(line):
            """Check if line contains valid airfoil coordinate pair (x, y)."""
            parts = line.strip().split()
            if len(parts) != 2:
                return False
            try:
                x, y = float(parts[0]), float(parts[1])
                # Additional check: coordinates should be reasonable airfoil values
                # x typically 0-1, y typically -0.5 to 0.5
                if not (0.0 <= x <= 1.0 and -0.5 <= y <= 0.5):
                    return False
                return True
            except ValueError:
                return False

        # Read and filter coordinate lines from file
        with open(original_file_path, "r") as f:
            coord_lines = [line.strip() for line in f if is_coordinate_line(line)]

        # Convert string coordinates to float tuples
        coords = [tuple(map(float, line.split())) for line in coord_lines]
        if len(coords) < 3:
            raise ValueError("Insufficient coordinate points.")

        # Separate upper and lower surfaces based on y values
        # Use small tolerance to handle floating-point precision issues like -0.0000000
        upper_coords = []
        lower_coords = []
        leading_edge_points = []  # Collect points at (0,0)

        for pt in coords:
            if pt[1] > 1e-8:  # Clearly positive y - upper surface
                upper_coords.append(pt)
            elif pt[1] < -1e-8:  # Clearly negative y - lower surface
                lower_coords.append(pt)
            else:  # Near zero y values - need special handling
                if abs(pt[0]) < 1e-6:  # Leading edge point (0,0)
                    leading_edge_points.append(pt)
                else:  # All other points with near-zero y
                    upper_coords.append(pt)

        # Handle leading edge points: distribute between upper and lower surfaces
        # Put one in lower surface, rest in upper to maintain proper connectivity
        if leading_edge_points:
            lower_coords.append(leading_edge_points[-1])  # Last one to lower
            if len(leading_edge_points) > 1:
                upper_coords.extend(leading_edge_points[:-1])  # Rest to upper

        # Sort upper surface from trailing edge to leading edge (x decreasing)
        upper = sorted(upper_coords, key=lambda pt: pt[0], reverse=True)

        # Sort lower surface from leading edge to trailing edge (x increasing)
        lower = sorted(lower_coords, key=lambda pt: pt[0])

        # Handle duplicate trailing edge points [1,0] in upper surface
        trailing_edge_points = []
        upper_filtered = []

        for pt in upper:
            # Check if point is approximately x=1.0 (trailing edge)
            if abs(pt[0] - 1.0) < 1e-6:
                trailing_edge_points.append(pt)
            else:
                upper_filtered.append(pt)

        # Consolidate trailing edge points: keep one at beginning of upper surface
        # Move extras to end of lower surface to maintain proper airfoil closure
        if len(trailing_edge_points) > 1:
            upper = [trailing_edge_points[0]] + upper_filtered  # First [1,0] at beginning
            lower.extend(trailing_edge_points[1:2])  # Move rest to end of lower
        elif len(trailing_edge_points) == 1:
            upper = [trailing_edge_points[0]] + upper_filtered  # Put [1,0] at beginning
        else:
            upper = [(1.0, 0.0)] + upper_filtered  # No trailing edge points found, add one at [1,0]

        # Generate Selig format: name line, upper coordinates, blank line, lower coordinates
        selig_lines = ["ConvertedAirfoil"]
        selig_lines += [f"{x:.7f} {y:.7f}" for x, y in upper]
        selig_lines.append("")  # Blank line separator between upper and lower surfaces
        selig_lines += [f"{x:.7f} {y:.7f}" for x, y in lower]

        # Create temporary file and write Selig format data
        tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".af", delete=False)
        tmp_file.write("\n".join(selig_lines))
        tmp_file_path = tmp_file.name
        tmp_file.close()

        return tmp_file_path

    def _take_second_half(self, arr):
        """
        Splits the input NumPy array into two halves and returns the second half.
        If the array has an odd length, the extra element goes to the second half.
        """
        mid = len(arr) // 2
        return arr[mid:]

    def _take_first_half(self, arr):
        """
        Splits the input NumPy array into two halves and returns the first half.
        If the array has an odd length, the extra element goes to the second half.
        """
        mid = len(arr) // 2
        return arr[:mid]
