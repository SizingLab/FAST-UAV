"""Computation of the airfoil aerodynamic properties using Xfoil."""
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
import os
import os.path as pth
import shutil
import sys
import warnings
from importlib.resources import path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from fastoad._utils.resource_management.copy import copy_resource
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator

from fastuav.command.api import string_to_array
from fastuav.models.aerodynamics.external.xfoil import xfoil699
from fastuav.models.geometry.profiles.get_profile import get_profile

from . import resources as local_resources
from ...constants import (
    ALPHA_STEP,
    DEFAULT_2D_CL_MAX,
    DEFAULT_2D_CL_MIN,
    OPTION_ALPHA_END,
    OPTION_ALPHA_START,
    OPTION_COMP_NEG_AIR_SYM,
    POLAR_POINT_COUNT,
    _DEFAULT_AIRFOIL_FILE,
)

OPTION_RESULT_POLAR_FILENAME = "result_polar_filename"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"
OPTION_XFOIL_EXE_PATH = "xfoil_exe_path"
OPTION_ITER_LIMIT = "iter_limit"

_INPUT_FILE_NAME = "polar_session.txt"
_STDOUT_FILE_NAME = "polar_calc.log"
_STDERR_FILE_NAME = "polar_calc.err"
_TMP_PROFILE_FILE_NAME = "in"  # as short as possible to avoid problems of path length
_TMP_RESULT_FILE_NAME = "out"  # as short as possible to avoid problems of path length
XFOIL_EXE_NAME = "xfoil.exe"  # name of embedded XFoil executable

_LOGGER = logging.getLogger(__name__)

_XFOIL_PATH_LIMIT = 64


class XfoilPolar(ExternalCodeComp):
    """Runs a polar computation with XFOIL and returns the 2D max lift coefficient."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Column names in XFOIL polar result
        self._xfoil_output_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]

    def initialize(self):
        self.options.declare(OPTION_XFOIL_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_RESULT_POLAR_FILENAME, default="polar_result.txt", types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=90.0, types=float)
        self.options.declare(OPTION_ITER_LIMIT, default=100, types=int)
        self.options.declare(OPTION_COMP_NEG_AIR_SYM, default=False, types=bool)
        self.options.declare("inviscid_calculation", default=False, types=bool)
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
        Set up inputs and outputs required for this operation
        """

        multiple_aoa = not self.options["single_AoA"]

        self.add_input("mach", val=np.nan)
        self.add_input("reynolds", val=np.nan)

        if multiple_aoa:
            self.add_output("alpha", shape=POLAR_POINT_COUNT, units="deg")
            self.add_output("CL", shape=POLAR_POINT_COUNT)
            self.add_output("CD", shape=POLAR_POINT_COUNT)
            self.add_output("CDp", shape=POLAR_POINT_COUNT)
            self.add_output("CM", shape=POLAR_POINT_COUNT)
            self.add_output("CL_max_2D")
            self.add_output("CL_min_2D")
            self.add_output("CD_min_2D")

        else:
            self.add_output("alpha", units="deg")
            self.add_output("CL")
            self.add_output("CD")
            self.add_output("CDp")
            self.add_output("CM")

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):
        """
        Function that computes airfoil aerodynamics with XFoil and returns the different 2D
        aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        """

        # Define timeout for the function
        self.options["timeout"] = 15.0

        # Get inputs and initialise outputs
        mach = np.round(inputs["mach"], 4).item()
        reynolds = np.round(inputs["reynolds"]).item()

        # Search if data already stored for this profile and mach with reynolds values bounding
        # current value. If so, use linear interpolation with the nearest upper/lower reynolds
        no_file = True
        data_saved = None
        interpolated_result = None

        # Modify file type respect to negative AoA/inviscid/single AoA options
        result_file = self._define_result_file_path()
        multiple_aoa = not self.options["single_AoA"]

        if pth.exists(result_file):
            no_file = False
            interpolated_result, data_saved = self._interpolation_for_exist_data(
                result_file, mach, reynolds
            )

        if interpolated_result is None:
            (
                result_array_p,
                result_array_n,
                result_folder_path,
                tmp_directory,
                tmp_result_file_path,
            ) = self._run_xfoil(inputs, outputs, reynolds, mach)

            (
                alpha,
                cl,
                cd,
                cdp,
                cm,
                cl_max_2d,
                cl_min_2d,
                cd_min_2d,
                error,
            ) = self._post_processing_fill_value(result_array_p, result_array_n)

            if multiple_aoa:
                # We chose not to save the results for now when there is a single AoA
                # Fix output length if needed
                alpha, cl, cd, cdp, cm = self.fix_multiple_aoa_output_length(alpha, cl, cd, cdp, cm)

                # Save results to defined path
                if not error:
                    results, labels = self._give_data_labels(
                        alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, mach, reynolds
                    )
                    if no_file or (data_saved is None):
                        data = pd.DataFrame(results, index=labels)
                    else:
                        data = pd.DataFrame(np.c_[data_saved, results], index=labels)
                    # noinspection PyBroadException
                    try:
                        data.to_csv(result_file)
                    except PermissionError:
                        warnings.warn(
                            "Unable to save XFoil results to *.csv file: writing permission denied "
                            "for %s folder!" % local_resources.__path__[0]
                        )

            # Getting output files if needed
            if self.options[OPTION_RESULT_FOLDER_PATH] != "":
                self._get_output_files(result_folder_path, tmp_result_file_path)
            # Try to delete the temp directory, if process not finished correctly try to
            # close files before removing directory for second attempt
            # noinspection PyBroadException
            try:
                tmp_directory.cleanup()
            except PermissionError:
                for file_path in os.listdir(tmp_directory.name):
                    if os.path.isfile(file_path):
                        # noinspection PyBroadException
                        try:
                            file = os.open(file_path, os.O_WRONLY)
                            os.close(file)
                        except:  # noqa: E722
                            _LOGGER.info("Error while trying to close %s file!", file_path)
                # noinspection PyBroadException
                try:
                    tmp_directory.cleanup()
                except PermissionError:
                    _LOGGER.info(
                        "Error while trying to erase %s temporary directory!", tmp_directory.name
                    )

        else:
            # adjust the interpolated results size for other model use

            (
                alpha,
                cl,
                cd,
                cdp,
                cm,
                cl_max_2d,
                cl_min_2d,
                cd_min_2d,
            ) = self._extract_fix_interpolated_result_length(interpolated_result)

        # Defining outputs -------------------------------------------------------------------------
        outputs["alpha"] = alpha
        outputs["CL"] = cl
        outputs["CD"] = cd
        outputs["CDp"] = cdp
        outputs["CM"] = cm
        if multiple_aoa:
            outputs["CL_max_2D"] = cl_max_2d
            outputs["CL_min_2D"] = cl_min_2d
            outputs["CD_min_2D"] = cd_min_2d

    def _write_script_file(
        self,
        reynolds,
        mach,
        tmp_profile_file_path,
        tmp_result_file_path,
        alpha_start,
        alpha_end,
        step,
    ):
        """
        Create command script to run XFoil for obtaining results

        :param reynolds: Reynolds number
        :param mach: Mach number
        :param tmp_profile_file_path: temporary profile path
        :param tmp_result_file_path: temporary result path
        :param alpha_start: starting angle of attack in XFoil calculation
        :param alpha_end: ending angle of attack in XFoil calculation
        :param step: step between each angle of attack in XFoil calculation
        """

        parser = InputFileGenerator()
        inviscid = self.options["inviscid_calculation"]
        single_aoa = self.options["single_AoA"]

        # Check the computation options and select different script templates
        if not single_aoa:
            if inviscid:
                input_file_name = "polar_session_inv.txt"
            else:
                input_file_name = "polar_session.txt"
        else:
            if inviscid:
                input_file_name = "polar_session_single_AoA_inv.txt"
            else:
                input_file_name = "polar_session_single_AoA.txt"

        # input command to run XFoil
        with path(local_resources, input_file_name) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(self.stdin)
            if not inviscid:
                parser.mark_anchor("RE")
                parser.transfer_var(float(reynolds), 1, 1)
            parser.mark_anchor("M")
            parser.transfer_var(float(mach), 1, 1)
            parser.mark_anchor("ITER")
            parser.transfer_var(self.options[OPTION_ITER_LIMIT], 1, 1)

            if single_aoa:
                parser.mark_anchor("ALFA")
                parser.transfer_var(alpha_start, 1, 1)
            else:
                parser.mark_anchor("ASEQ")
                parser.transfer_var(alpha_start, 1, 1)
                parser.transfer_var(alpha_end, 2, 1)

            parser.transfer_var(step, 3, 1)
            parser.reset_anchor()
            parser.mark_anchor("/profile")
            parser.transfer_var(tmp_profile_file_path, 0, 1)
            parser.mark_anchor("/polar_result")
            parser.transfer_var(tmp_result_file_path, 0, 1)
            parser.generate()

    def _read_polar(self, xfoil_result_file_path: str) -> np.ndarray:
        """
        :param xfoil_result_file_path:
        :return: numpy array with XFoil polar results
        """
        if os.path.isfile(xfoil_result_file_path):
            dtypes = [(name, "f8") for name in self._xfoil_output_names]
            result_array = np.genfromtxt(xfoil_result_file_path, skip_header=12, dtype=dtypes)
            return result_array

        _LOGGER.error("XFOIL results file not found")
        return np.array([])

    def _get_max_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """

        :param alpha:
        :param lift_coeff: CL
        :return: max CL within +/- 0.3 around linear zone if enough alpha computed, or default value
        otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if np.abs(covered_range / alpha_range) >= 0.4:

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

                delta = np.abs(lift_coeff - lift_fct(alpha))
                return max(lift_coeff[delta <= 0.3]), False

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
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if covered_range / alpha_range >= 0.4:

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

                delta = np.abs(lift_coeff - lift_fct(alpha))
                return min(lift_coeff[delta <= 0.3]), False

        _LOGGER.warning(
            "2D CL min not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MIN,
        )
        return DEFAULT_2D_CL_MIN, True

    @staticmethod
    def _reshape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Delete ending 0.0 values
        for idx in range(len(x)):
            if np.sum(x[idx : len(x)] == 0.0) == (len(x) - idx):
                y = y[0:idx]
                break
        return y

    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        """
        Dev Note: XFOIL fails if length of provided file path exceeds 64 characters.
        Changing working directory to the tmp dir would allow to just provide file name,
        but it is not really safe (at least, it does mess with the coverage report).
        Then the point is to get a tmp directory with a short path.
        On Windows, the default (user-dependent) tmp dir can exceed the limit.
        Therefore, as a second choice, tmp dir is created as close of user home
        directory as possible.
        """
        tmp_candidates = []
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = TemporaryDirectory(prefix="x", dir=tmp_base_path)
            tmp_candidates.append(tmp_directory.name)
            tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
            tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)

            if max(len(tmp_profile_file_path), len(tmp_result_file_path)) <= _XFOIL_PATH_LIMIT:
                # tmp_directory is OK. Stop there
                break
            # tmp_directory has a too long path. Erase and continue...
            tmp_directory.cleanup()

        if max(len(tmp_profile_file_path), len(tmp_result_file_path)) > _XFOIL_PATH_LIMIT:
            raise IOError(
                "Could not create a tmp directory where file path will respects XFOIL "
                "limitation (%i): tried %s" % (_XFOIL_PATH_LIMIT, tmp_candidates)
            )

        return tmp_directory

    def _run_xfoil(self, inputs, outputs, reynolds, mach):
        """
        Run Xfoil to obtain the 2D aerodynamics of selected airfoil.

        :param inputs: inputs in the OpenMDAO format
        :param outputs: outputs in the OpenMDAO format
        :param reynolds: Reynolds number
        :param mach: Mach number

        :return: list of results for post-processing
        """

        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
        if result_folder_path != "":
            os.makedirs(result_folder_path, exist_ok=True)

        # Pre-processing (populating temp directory)
        # XFoil exe
        tmp_directory = self._create_tmp_directory()
        if self.options[OPTION_XFOIL_EXE_PATH]:
            # if a path for Xfoil has been provided, simply use it
            self.options["command"] = [self.options[OPTION_XFOIL_EXE_PATH]]
        else:
            # otherwise, copy the embedded resource in tmp dir
            # noinspection PyTypeChecker
            copy_resource(xfoil699, XFOIL_EXE_NAME, tmp_directory.name)
            self.options["command"] = [pth.join(tmp_directory.name, XFOIL_EXE_NAME)]

        # I/O files
        self.stdin = pth.join(tmp_directory.name, _INPUT_FILE_NAME)
        self.stdout = pth.join(tmp_directory.name, _STDOUT_FILE_NAME)
        self.stderr = pth.join(tmp_directory.name, _STDERR_FILE_NAME)

        # profile file
        tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
        profile = get_profile(
            airfoil_folder_path=self.options["airfoil_folder_path"],
            file_name=self.options["airfoil_file"],
        ).get_sides()
        # noinspection PyTypeChecker
        np.savetxt(
            tmp_profile_file_path,
            profile.to_numpy(),
            fmt="%.15f",
            delimiter=" ",
            header="Wing",
            comments="",
        )

        # standard input file
        tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)
        self._write_script_file(
            reynolds,
            mach,
            tmp_profile_file_path,
            tmp_result_file_path,
            self.options[OPTION_ALPHA_START],
            self.options[OPTION_ALPHA_END],
            ALPHA_STEP,
        )

        # Run XFOIL
        self.options["external_input_files"] = [self.stdin, tmp_profile_file_path]
        self.options["external_output_files"] = [tmp_result_file_path]
        # noinspection PyBroadException
        try:
            super().compute(inputs, outputs)
            result_array_p = self._read_polar(tmp_result_file_path)
        except:  # noqa: E722
            # catch the error and try to read result file for non-convergence on higher angles
            error = sys.exc_info()[1]
            try:
                result_array_p = self._read_polar(tmp_result_file_path)
            except:  # noqa: E722
                raise TimeoutError("<p>Error: %s</p>" % error)
        result_array_n = np.array([])

        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            os.remove(self.stdin)
            os.remove(self.stdout)
            os.remove(self.stderr)
            if os.path.exists(tmp_result_file_path):
                os.remove(tmp_result_file_path)
            if len(result_array_p) > 0:
                alpha_start = min(-1 * self.options[OPTION_ALPHA_START], -ALPHA_STEP)
                self._write_script_file(
                    reynolds,
                    mach,
                    tmp_profile_file_path,
                    tmp_result_file_path,
                    alpha_start,
                    -1 * self.options[OPTION_ALPHA_END],
                    -ALPHA_STEP,
                )
                # noinspection PyBroadException
                try:
                    super().compute(inputs, outputs)
                    result_array_n = self._read_polar(tmp_result_file_path)
                except:  # noqa: E722
                    # catch the error and try to read result file for non-convergence on higher
                    # angles
                    e = sys.exc_info()[1]
                    try:
                        result_array_n = self._read_polar(tmp_result_file_path)
                    except:  # noqa: E722
                        raise TimeoutError("<p>Error: %s</p>" % e)

        return (
            result_array_p,
            result_array_n,
            result_folder_path,
            tmp_directory,
            tmp_result_file_path,
        )

    @staticmethod
    def _interpolation_for_exist_data(result_file, mach, reynolds):
        """
        If a result file exist, we then check if the proper Mach number exists. If it does we
        check if a Reynolds number below and above it exist, in which case we interpolate between
        the two. Even if interpolated results can't be obtained (Reynold too low or too high),
        if we enter this function, it means the corresponding airfoil exists ad the Xfoil run we
        are about to do should be added to the existing results.

        :return interpolated_result: the interpolated results if they exists, None otherwise.
        :return data_saved: existing results
        """

        interpolated_result = None

        data_saved = pd.read_csv(result_file)

        # Pre-processing of the dataframe
        values = data_saved.to_numpy()[:, 1 : len(data_saved.to_numpy()[0])]
        labels = data_saved.to_numpy()[:, 0].tolist()
        data_saved = pd.DataFrame(values, index=labels)

        # Look for existing mach or one close enough
        saved_mach_list = data_saved.loc["mach", :].to_numpy().astype(float)
        index_near_mach = np.where(abs(saved_mach_list - mach) < 0.03)[0]
        near_mach = []
        distance_to_mach = []
        # Check if there is a velocity (Mach) value that is close to this one
        for index in index_near_mach:
            if saved_mach_list[index] not in near_mach:
                near_mach.append(saved_mach_list[index])
                distance_to_mach.append(abs(saved_mach_list[index] - mach))
        if not near_mach:
            index_mach = np.where(data_saved.loc["mach", :].to_numpy() == str(mach))[
                0
            ]  # TODO: I would change it to = []
        else:
            selected_mach_index = distance_to_mach.index(min(distance_to_mach))
            index_mach = np.where(saved_mach_list == near_mach[selected_mach_index])[0]
        data_reduced = data_saved.loc[labels, index_mach]

        # Search if this exact reynolds has been computed and save results
        reynolds_vect = data_reduced.loc["reynolds", :].to_numpy().astype(float)

        index_reynolds = index_mach[np.where(reynolds_vect == reynolds)]
        if len(index_reynolds) == 1:
            interpolated_result = data_reduced.loc[labels, index_reynolds]

        # Else search for lower/upper Reynolds
        else:
            lower_reynolds = reynolds_vect[np.where(reynolds_vect < reynolds)[0]]
            upper_reynolds = reynolds_vect[np.where(reynolds_vect > reynolds)[0]]

            if not (len(lower_reynolds) == 0 or len(upper_reynolds) == 0):
                index_lower_reynolds = index_mach[np.where(reynolds_vect == max(lower_reynolds))[0]]
                index_upper_reynolds = index_mach[np.where(reynolds_vect == min(upper_reynolds))[0]]
                lower_values = data_reduced.loc[labels, index_lower_reynolds]
                upper_values = data_reduced.loc[labels, index_upper_reynolds]

                # Initialise values with lower reynolds
                interpolated_result = lower_values
                # Calculate reynolds ratio split for linear interpolation
                x_ratio = (min(upper_reynolds) - reynolds) / (
                    min(upper_reynolds) - max(lower_reynolds)
                )

                # Search for common alpha range for linear interpolation
                alpha_lower = list(
                    string_to_array(lower_values.loc["alpha", index_lower_reynolds].to_numpy()[0])
                )
                alpha_upper = list(
                    string_to_array(upper_values.loc["alpha", index_upper_reynolds].to_numpy()[0])
                )
                alpha_shared = np.array(list(set(alpha_upper).intersection(alpha_lower)))
                interpolated_result.loc["alpha", index_lower_reynolds] = str(alpha_shared.tolist())

                labels.remove("alpha")

                # Calculate average values (cd, cl...) with linear interpolation
                for label in labels:
                    lower_value = string_to_array(
                        lower_values.loc[label, index_lower_reynolds].to_numpy()[0]
                    ).astype(float)
                    upper_value = string_to_array(
                        upper_values.loc[label, index_upper_reynolds].to_numpy()[0]
                    ).astype(float)

                    # If values relative to alpha vector, performs interpolation with shared
                    # vector
                    # Justification: There are some values which are not alpha dependant (cl max, cl min, cd min) and thus are not interpolated with alpha but directly with reynolds. For these values, we don't want to interpolate with the shared alpha vector but directly with the reynolds ratio.
                    # TODO: Is this really necessary? Having a different alpha implies having a different Reynolds for example look at naca4412_30S.csv vs naca23018_20S.csv
                    if np.size(lower_value) == len(alpha_lower):
                        lower_value = np.interp(
                            alpha_shared,
                            np.array(alpha_lower),
                            lower_value,
                        )
                        upper_value = np.interp(
                            alpha_shared,
                            np.array(alpha_upper),
                            upper_value,
                        )

                    value = (lower_value * x_ratio + upper_value * (1 - x_ratio)).tolist()

                    interpolated_result.loc[label, index_lower_reynolds] = str(value)

        return interpolated_result, data_saved

    def _define_result_file_path(self):
        """
        Each computation option (airfoil name, max AOA, single AOA, ...) will lead to unique file
        name for stored results. We can thus check if results already exists if the file name
        already exists. This function generate file name corresponding to the computation options.

        :return result_file: path to result file if it exists
        """

        # Generate tags for the different options
        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            negative_angle_tag = "_" + str(int(np.ceil(self.options[OPTION_ALPHA_END]))) + "S"
        else:
            negative_angle_tag = ""

        if self.options["single_AoA"]:
            single_aoa_tag = "_1_AOA"
        else:
            single_aoa_tag = ""

        if self.options["inviscid_calculation"]:
            inviscid_tag = "_inv"
        else:
            inviscid_tag = ""

        naming = negative_angle_tag + single_aoa_tag + inviscid_tag

        result_file = pth.join(
            pth.split(os.path.realpath(__file__))[0],
            "resources",
            self.options["airfoil_file"].replace(".af", naming) + ".csv",
        )

        return result_file

    def _post_processing_fill_value(self, result_array_p, result_array_n):
        """
        Prepare value for post-processing. If positive and negative angle were computed, concatenate
        results. If only one AOA is required, max and min CL/CD will be empty.

        :param result_array_p: results with positive angle of attacks
        :param result_array_n: results with negative angle of attacks

        :return: aerodynamic characteristics of the airfoil
        """
        if len(result_array_p) == 0:
            _LOGGER.warning("XFoil produced no polar results: using default CL max/min values.")
            return [], [], [], [], [], DEFAULT_2D_CL_MAX, DEFAULT_2D_CL_MIN, 0.0, True

        if self.options["single_AoA"]:
            alpha = result_array_p["alpha"].tolist()
            cl = result_array_p["CL"].tolist()
            cd = result_array_p["CD"].tolist()
            cdp = result_array_p["CDp"].tolist()
            cm = result_array_p["CM"].tolist()
            cl_max_2d = []
            cl_min_2d = []
            cd_min_2d = []
            error = []

        elif self.options[OPTION_COMP_NEG_AIR_SYM]:
            cl_max_2d, error = self._get_max_cl(result_array_p["alpha"], result_array_p["CL"])
            # noinspection PyUnboundLocalVariable
            cl_min_2d, _ = self._get_min_cl(result_array_n["alpha"], result_array_n["CL"])
            cd_min_2d = np.min(result_array_p["CD"])
            alpha = result_array_n["alpha"].tolist()
            alpha.reverse()
            alpha.extend(result_array_p["alpha"].tolist())
            cl = result_array_n["CL"].tolist()
            cl.reverse()
            cl.extend(result_array_p["CL"].tolist())
            cd = result_array_n["CD"].tolist()
            cd.reverse()
            cd.extend(result_array_p["CD"].tolist())
            cdp = result_array_n["CDp"].tolist()
            cdp.reverse()
            cdp.extend(result_array_p["CDp"].tolist())
            cm = result_array_n["CM"].tolist()
            cm.reverse()
            cm.extend(result_array_p["CM"].tolist())

        else:
            cl_max_2d, error = self._get_max_cl(result_array_p["alpha"], result_array_p["CL"])
            cl_min_2d, _ = self._get_min_cl(result_array_p["alpha"], result_array_p["CL"])
            cd_min_2d = np.min(result_array_p["CD"])
            alpha = result_array_p["alpha"].tolist()
            cl = result_array_p["CL"].tolist()
            cd = result_array_p["CD"].tolist()
            cdp = result_array_p["CDp"].tolist()
            cm = result_array_p["CM"].tolist()

        return alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, error

    @staticmethod
    def fix_multiple_aoa_output_length(alpha, cl, cd, cdp, cm):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise we complete with zeros.
        Used when alpha, cl, cd, cdp and cm are arrays. If inside a dataframe use
        _extract_fix_interpolated_result_length

        :param alpha: angle of attach array
        :param cl: lift coefficient array
        :param cd: drag coefficient array
        :param cdp: pressure drag coefficient array
        :param cm: moment coefficient array

        :return: the input array with the proper size for OpenMDAO
        """

        # use interpolation to fill missing values and add zero for values that are out of range
        if POLAR_POINT_COUNT < len(alpha):
            alpha_interp = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha_interp, alpha, cl)
            cd = np.interp(alpha_interp, alpha, cd)
            cdp = np.interp(alpha_interp, alpha, cdp)
            cm = np.interp(alpha_interp, alpha, cm)
            alpha = alpha_interp
            warnings.warn("Defined polar point in fast aerodynamics\\constants.py exceeded!")
        else:
            filler = np.zeros(POLAR_POINT_COUNT - len(alpha))
            alpha = np.append(alpha, filler)
            cl = np.append(cl, filler)
            cd = np.append(cd, filler)
            cdp = np.append(cdp, filler)
            cm = np.append(cm, filler)

        return alpha, cl, cd, cdp, cm

    @staticmethod
    def _extract_fix_interpolated_result_length(interpolated_result):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise we complete with zeros.

        :param interpolated_result: result dataset with labels and values

        :return: length-modified aerodynamic characteristic array
        """

        # Extract results
        cl_max_2d = string_to_array(interpolated_result.loc["cl_max_2d", :].values[0])
        cl_min_2d = string_to_array(interpolated_result.loc["cl_min_2d", :].values[0])
        alpha = string_to_array(interpolated_result.loc["alpha", :].values[0])
        cl = string_to_array(interpolated_result.loc["cl", :].values[0])
        cd = string_to_array(interpolated_result.loc["cd", :].values[0])
        cdp = string_to_array(interpolated_result.loc["cdp", :].values[0])
        cm = string_to_array(interpolated_result.loc["cm", :].values[0])
        cd_min_2d = np.min(cd)

        # Modify vector length if necessary
        if POLAR_POINT_COUNT < len(alpha):
            alpha = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha, alpha, cl)
            cd = np.interp(alpha, alpha, cd)
            cdp = np.interp(alpha, alpha, cdp)
            cm = np.interp(alpha, alpha, cm)
        else:
            filler = np.zeros(POLAR_POINT_COUNT - len(alpha))
            alpha = np.append(alpha, filler)
            cl = np.append(cl, filler)
            cd = np.append(cd, filler)
            cdp = np.append(cdp, filler)
            cm = np.append(cm, filler)

        return alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d

    def _give_data_labels(
        self, alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, mach, reynolds
    ):
        """
        Five data labels and prepare list for later writing to the file
        """

        results = [
            np.array(mach),
            np.array(reynolds),
            np.array(cl_max_2d),
            np.array(cl_min_2d),
            np.array(cd_min_2d),
            str(self._reshape(alpha, alpha).tolist()),
            str(self._reshape(alpha, cl).tolist()),
            str(self._reshape(alpha, cd).tolist()),
            str(self._reshape(alpha, cdp).tolist()),
            str(self._reshape(alpha, cm).tolist()),
        ]
        labels = [
            "mach",
            "reynolds",
            "cl_max_2d",
            "cl_min_2d",
            "cd_min_2d",
            "alpha",
            "cl",
            "cd",
            "cdp",
            "cm",
        ]
        return results, labels

    def _get_output_files(self, result_folder_path, tmp_result_file_path):
        if pth.exists(tmp_result_file_path):
            polar_file_path = pth.join(
                result_folder_path, self.options[OPTION_RESULT_POLAR_FILENAME]
            )
            shutil.move(tmp_result_file_path, polar_file_path)

        if pth.exists(self.stdout):
            stdout_file_path = pth.join(result_folder_path, _STDOUT_FILE_NAME)
            shutil.move(self.stdout, stdout_file_path)

        if pth.exists(self.stderr):
            stderr_file_path = pth.join(result_folder_path, _STDERR_FILE_NAME)
            shutil.move(self.stderr, stderr_file_path)
