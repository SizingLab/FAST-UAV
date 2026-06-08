"""API."""
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
import warnings
import os.path as pth
import os
import shutil
import inspect
import importlib
import tempfile
from tempfile import TemporaryDirectory
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Union, List
from platform import system

import numpy as np
from deprecated import deprecated
import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.system import System

import fastoad.api as oad
from fastoad.cmd.exceptions import FastPathExistsError
from fastoad.io import IVariableIOFormatter, VariableIO
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

# noinspection PyProtectedMember
from fastoad.cmd.api import _get_simple_system_list

from fastuav.utils.warnings import VariableDescriptionWarning

from . import resources

_LOGGER = logging.getLogger(__name__)

SAMPLE_FILENAME = "fastga.yml"
SAMPLE_XML_NAME = "fastga.xml"
BOOLEAN_OPTIONS = [
    "use_openvsp",
    "compute_mach_interpolation",
    "compute_slipstream",
    "low_speed_aero",
]


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory."""
    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def file_temporary_transfer(file_path: str):
    """
    Put a copy of original python file into temporary directory and remove plugin registration
    from current file.
    """
    tmp_folder = _create_tmp_directory()
    file_name = pth.split(file_path)[-1]
    shutil.copy(file_path, pth.join(tmp_folder.name, file_name))
    file = open(file_path, "r")
    lines = file.read()
    lines = lines.split("\n")
    idx_to_remove = []
    for idx, _ in enumerate(lines):
        if "@oad.RegisterOpenMDAOSystem" in lines[idx]:
            idx_to_remove.append(idx)
    for idx in sorted(idx_to_remove, reverse=True):
        del lines[idx]
    file.close()
    file = open(file_path, "w")
    for line in lines:
        file.write(line + "\n")
    file.close()

    return tmp_folder


def retrieve_original_file(tmp_folder, file_path: str):
    """Retrieve the original file."""
    file_name = pth.split(file_path)[-1]
    shutil.copy(pth.join(tmp_folder.name, file_name), file_path)

    tmp_folder.cleanup()


def generate_variables_description(subpackage_path: str, overwrite: bool = False):
    """
    Generates/append the variable descriptions file for a given subpackage.

    To use it simply type:
    from fastga.command.api import generate_variables_description
    import my_package

    generate_variables_description(my_package.__path__[0], overwrite=True).

    :param subpackage_path: the path of the subpackage to explore
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastPathExistsError: if overwrite==False and subpackage_path already exists.
    """

    if not overwrite and pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
        # noinspection PyStringFormat
        raise FastPathExistsError(
            "Variable descriptions file is not written because it already exists. "
            "Use overwrite=True to bypass."
            % pth.join(subpackage_path, "variable_descriptions.txt"),
            pth.join(subpackage_path, "variable_descriptions.txt"),
        )

    if not pth.exists(subpackage_path):
        _LOGGER.info("Sub-package path %s not found!", subpackage_path)
    else:
        # Read file and construct dictionary of variables name index
        saved_dict = {}
        if pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
            file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "r")
            for line in file:
                if line[0] != "#" and len(line.split("||")) == 2:
                    variable_name, variable_description = line.split("||")
                    variable_name_length = len(variable_name)
                    variable_name = variable_name.replace(" ", "")
                    while variable_name_length != len(variable_name):
                        variable_name = variable_name.replace(" ", "")
                        variable_name_length = len(variable_name)
                    saved_dict[variable_name] = (variable_description, subpackage_path)
            file.close()

        # If path point to ./models directory list output variables described in the different
        # models
        if pth.split(subpackage_path)[-1] == "models":
            for root, _, files in os.walk(subpackage_path, topdown=False):
                vd_file_empty_description = False
                empty_description_variables = []
                for name in files:
                    if name == "variable_descriptions.txt":
                        file = open(pth.join(root, name), "r")
                        for line in file:
                            if line[0] != "#" and len(line.split("||")) == 2:
                                variable_name, variable_description = line.split("||")
                                if variable_description.replace(" ", "") == "\n":
                                    vd_file_empty_description = True
                                    empty_description_variables.append(
                                        variable_name.replace(" ", "")
                                    )
                                variable_name_length = len(variable_name)
                                variable_name = variable_name.replace(" ", "")
                                while variable_name_length != len(variable_name):
                                    variable_name = variable_name.replace(" ", "")
                                    variable_name_length = len(variable_name)
                                if variable_name not in saved_dict.keys():
                                    saved_dict[variable_name] = (variable_description, root)
                                else:
                                    if not (
                                        pth.split(root)[-1]
                                        == pth.split(saved_dict[variable_name][1])[-1]
                                    ):
                                        warnings.warn(
                                            "file variable_descriptions.txt from subpackage "
                                            + pth.split(root)[-1]
                                            + " contains parameter "
                                            + variable_name
                                            + " already saved in "
                                            + pth.split(saved_dict[variable_name][1])[-1]
                                            + " subpackage!",
                                            category=VariableDescriptionWarning,
                                        )
                        file.close()
                if vd_file_empty_description:
                    warnings.warn(
                        "file variable_descriptions.txt from %s subpackage contains empty"
                        " descriptions! \n"
                        % pth.split(root)[-1]
                        + "\tFollowing variables have empty descriptions : "
                        + ", ".join(empty_description_variables),
                        category=VariableDescriptionWarning,
                    )

        # Explore subpackage models and find the output variables and store them in a dictionary
        dict_to_be_saved = {}
        for root, _, files in os.walk(subpackage_path, topdown=False):
            for name in files:
                if name[-3:] == ".py":
                    spec = importlib.util.spec_from_file_location(
                        name.replace(".py", ""), pth.join(root, name)
                    )
                    module = importlib.util.module_from_spec(spec)
                    tmp_folder = None
                    # noinspection PyBroadException
                    try:
                        # if register decorator in module, temporary replace file removing
                        # decorators
                        # noinspection PyBroadException
                        try:
                            spec.loader.exec_module(module)
                        except Exception:  # noqa: E722
                            _LOGGER.info(
                                "Trying to load %s, but it is not a module!", pth.join(root, name)
                            )
                        if "oad.RegisterOpenMDAOSystem" in dir(module):
                            tmp_folder = file_temporary_transfer(pth.join(root, name))
                        spec.loader.exec_module(module)
                        total_class_list = [
                            x for x in dir(module) if inspect.isclass(getattr(module, x))
                        ]
                        class_list = []
                        for class_name in total_class_list:
                            address = getattr(module, class_name).__module__
                            if len(address.split(".")) <= 2:
                                class_list.append(class_name)
                            else:
                                if pth.split(subpackage_path)[-1] == address.split(".")[2]:
                                    class_list.append(class_name)
                        # noinspection PyUnboundLocalVariable
                        retrieve_original_file(tmp_folder, pth.join(root, name))
                        if system() != "Windows":
                            root_lib = ".".join(root.split("/")[root.split("/").index("fastga") :])
                        else:
                            root_lib = ".".join(
                                root.split("\\")[root.split("\\").index("fastga") :]
                            )
                        root_lib += "." + name.replace(".py", "")
                        for class_name in class_list:
                            # noinspection PyBroadException
                            try:
                                my_class = getattr(importlib.import_module(root_lib), class_name)
                                options_dictionary = {}
                                if "propulsion_id" in my_class().options:
                                    available_id_list = _get_simple_system_list()
                                    idx_to_remove = []
                                    for idx, _ in enumerate(available_id_list):
                                        available_id_list[idx] = available_id_list[idx][0]
                                        if "PROPULSION" in available_id_list[idx]:
                                            idx_to_remove.extend(list(range(idx + 1)))
                                        if "fastga" not in available_id_list[idx]:
                                            idx_to_remove.append(idx)
                                    idx_to_remove = list(dict.fromkeys(idx_to_remove))
                                    for idx in sorted(idx_to_remove, reverse=True):
                                        del available_id_list[idx]
                                    options_dictionary["propulsion_id"] = available_id_list[0]
                                variables = list_variables(my_class(**options_dictionary))
                                local_options = []
                                for option_name in BOOLEAN_OPTIONS:
                                    # noinspection PyProtectedMember
                                    if option_name in my_class().options._dict.keys():
                                        local_options.append(option_name)
                                # If no boolean options alternatives to be tested, search for
                                # input variables in models and output variables for subpackages
                                # (including ivc)
                                if not local_options:
                                    if pth.split(subpackage_path)[-1] == "models":
                                        var_names = [var.name for var in variables if var.is_input]
                                    else:
                                        var_names = [
                                            var.name for var in variables if not var.is_input
                                        ]
                                        if list_ivc_outputs_name(my_class(**options_dictionary)):
                                            var_names.append(
                                                list_ivc_outputs_name(
                                                    my_class(**options_dictionary)
                                                )
                                            )
                                    # Remove duplicates
                                    var_names = list(dict.fromkeys(var_names))
                                    # Add to dictionary only variable name including data:,
                                    # settings: or tuning:
                                    for key in var_names:
                                        if (
                                            ("data:" in key)
                                            or ("settings:" in key)
                                            or ("tuning:" in key)
                                        ):
                                            if key not in dict_to_be_saved.keys():
                                                dict_to_be_saved[key] = ""
                                # If boolean options alternatives encountered, all alternatives
                                # have to be tested to ensure complete coverage of variables.
                                # Working principle is similar to previous one.
                                else:
                                    for options_tuple in list(
                                        product([True, False], repeat=len(local_options))
                                    ):
                                        # Define local option dictionary
                                        for idx, _ in enumerate(local_options):
                                            options_dictionary[local_options[idx]] = options_tuple[
                                                idx
                                            ]
                                        variables = list_variables(my_class(**options_dictionary))
                                        if pth.split(subpackage_path)[-1] == "models":
                                            var_names = [
                                                var.name for var in variables if var.is_input
                                            ]
                                        else:
                                            var_names = [
                                                var.name for var in variables if not var.is_input
                                            ]
                                            if (
                                                len(
                                                    list_ivc_outputs_name(
                                                        my_class(**options_dictionary)
                                                    )
                                                )
                                                != 0
                                            ):
                                                var_names.append(
                                                    list_ivc_outputs_name(
                                                        my_class(**options_dictionary)
                                                    )
                                                )
                                        # Remove duplicates
                                        var_names = list(dict.fromkeys(var_names))
                                        # Add to dictionary only variable name including data:,
                                        # settings: or tuning:
                                        for key in var_names:
                                            if (
                                                ("data:" in key)
                                                or ("settings:" in key)
                                                or ("tuning:" in key)
                                            ):
                                                if key not in dict_to_be_saved.keys():
                                                    dict_to_be_saved[key] = ""
                            except Exception:  # noqa: E722
                                _LOGGER.info(
                                    "Failed to read %s.%s class parameters!", root_lib, class_name
                                )
                    except Exception:  # noqa: E722
                        if tmp_folder is not None:
                            # noinspection PyUnboundLocalVariable
                            retrieve_original_file(tmp_folder, pth.join(root, name))

        # Complete the variable descriptions file with missing outputs
        if pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
            file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "a")
            if len(
                set(list(dict_to_be_saved.keys())).intersection(set(list(saved_dict.keys())))
            ) != len(dict_to_be_saved.keys()):
                file.write("\n")
            file.close()
        else:
            if dict_to_be_saved.keys():
                file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "w")
                file.write("# Documentation of variables used in FAST-GA models\n")
                file.write("# Each line should be like:\n")
                file.write(
                    "# my:variable||The description of my:variable, as long as needed, but on one "
                    "line.\n "
                )
                file.write(
                    '# The separator "||" can be surrounded with spaces (that will be ignored)\n\n'
                )
                file.close()
        if len(dict_to_be_saved.keys()) != 0:
            file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "a")
            sorted_keys = sorted(dict_to_be_saved.keys(), key=lambda x: x.lower())
            added_key = False
            added_key_names = []
            for key in sorted_keys:
                if key not in saved_dict:
                    # noinspection PyUnboundLocalVariable
                    file.write(key + " || \n")
                    added_key = True
                    added_key_names.append(key)
            file.close()
            if added_key:
                warnings.warn(
                    "file variable_descriptions.txt from {} subpackage contains empty "
                    "descriptions! \n".format(pth.split(subpackage_path)[-1])
                    + "\tFollowing variables have empty descriptions : "
                    + ", ".join(added_key_names),
                    category=VariableDescriptionWarning,
                )


@deprecated(
    version="0.2.0",
    reason="Will be removed in version 1.0. Please use the generate_configuration_file from "
    'fast-oad-core api with the distribution_name="fast-oad-cs23" instead',
)
def generate_configuration_file(configuration_file_path: str, overwrite: bool = False):
    """
    Generates a sample configuration file.

    :param configuration_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastPathExistsError: if overwrite==False and configuration_file_path already exists
    """
    if not overwrite and pth.exists(configuration_file_path):
        raise FastPathExistsError(
            "Configuration file is not written because it already exists. "
            "Use overwrite=True to bypass." % configuration_file_path,
            configuration_file_path,
        )

    if not pth.exists(pth.split(configuration_file_path)[0]):
        os.mkdir(pth.split(configuration_file_path)[0])
    shutil.copy(pth.join(resources.__path__[0], SAMPLE_FILENAME), configuration_file_path)

    _LOGGER.info("Sample configuration written in %s", configuration_file_path)


def generate_xml_file(xml_file_path: str, overwrite: bool = False):
    """
    Generates a sample XML file.

    :param xml_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastPathExistsError: if overwrite==False and configuration_file_path already exists
    """
    if not overwrite and pth.exists(xml_file_path):
        raise FastPathExistsError(
            "Configuration file is not written because it already exists. "
            "Use overwrite=True to bypass." % xml_file_path,
            xml_file_path,
        )

    if not pth.exists(pth.split(xml_file_path)[0]):
        os.mkdir(pth.split(xml_file_path)[0])
    shutil.copy(pth.join(resources.__path__[0], SAMPLE_XML_NAME), xml_file_path)

    _LOGGER.info("Sample configuration written in %s", xml_file_path)


def write_needed_inputs(
    problem: oad.FASTOADProblem, xml_file_path: str, source_formatter: IVariableIOFormatter = None
):
    """
    Writes the input file of the problem with unconnected inputs of the configured problem.
    Written value of each variable will be taken:
        1. from input_data if it contains the variable
        2. from defined default values in component definitions
    :param problem: problem we want to write the inputs of
    :param xml_file_path: if provided, variable values will be read from it
    :param source_formatter: the class that defines format of input file. if
                             not provided, expected format will be the default one.
    """
    variables = oad.DataFile(xml_file_path)

    unconnected_inputs = oad.VariableList.from_problem(
        problem,
        use_initial_values=True,
        get_promoted_names=True,
        promoted_only=True,
        io_status="inputs",
    )

    variables.update(
        unconnected_inputs,
        add_variables=True,
    )
    if xml_file_path:
        ref_vars = oad.DataFile(xml_file_path, source_formatter)
        variables.update(ref_vars)
        for var in variables:
            var.is_input = True
    variables.save()


def list_ivc_outputs_name(local_system: Union[ExplicitComponent, ImplicitComponent, Group]):
    """
    List all "root" components in the systems, meaning the components that don't have any
    subcomponents.
    """
    group = AutoUnitsDefaultGroup()
    group.add_subsystem("system", local_system, promotes=["*"])
    problem = oad.FASTOADProblem()
    problem.model = group
    try:
        problem.setup()
    except RuntimeError:
        _LOGGER.info(
            "Some problem occurred while setting-up the problem without input file probably "
            "because shape_by_conn variables exist!"
        )
    model = problem.model
    dict_sub_system = {}
    dict_sub_system = list_all_subsystem(model, "model", dict_sub_system)
    ivc_outputs_names = []

    # Find the outputs of all of those systems that are IndepVarComp
    for sub_system_keys in dict_sub_system.keys():
        if (
            dict_sub_system[sub_system_keys] == "IndepVarComp"
            and sub_system_keys.split(".")[-1] != "fastoad_shaper"
        ):
            actual_attribute_name = sub_system_keys.replace("model.system.", "")
            address_levels = actual_attribute_name.split(".")
            component = model.system
            for next_level in address_levels:
                component = getattr(component, next_level)
            component_output = component.list_outputs()
            for outputs in component_output:
                ivc_outputs_names.append(outputs[0])

    return ivc_outputs_names


def generate_block_analysis(
    local_system: Union[ExplicitComponent, ImplicitComponent, Group, str],
    var_inputs: List,
    xml_file_path: str,
    options: dict = None,
    overwrite: bool = False,
):
    """
    Generates a function based on set of models and a set of variables that we want this
    functions to take.

    :param local_system: the system the function is going to be based one, can be either an
    OpenMDAO component (Implicit, Explicit or a Group), a registered FAST-OAD id, or the absolute
    path to a configuration file
    :param var_inputs: a list of variables name that we want the
    patched function to have as an input, it will be the list of keys expected as an input of the
    function
    :param xml_file_path: the path of the XML that contains the values of the variables
    necessary for the models but that are not inputs
    :param options: the options of the group, required if an id is provided
    :param overwrite: boolean to set whether or not the input XML file will be overwritten once
    the function runs

    :return patched_function: the function constructed based on the provided system which takes
    var_inputs as inputs under the form of a dictionary {"var_name": (var_value, var_units)}
    """

    # If a valid ID or a path to a configuration file is provided, build a system based on that ID
    if isinstance(local_system, str):
        if local_system.endswith(".yml"):
            configurator = oad.FASTOADProblemConfigurator(local_system)
            dummy_problem = configurator.get_problem(read_inputs=False)
            local_system = dummy_problem.model
        else:
            local_system = oad.RegisterOpenMDAOSystem.get_system(local_system, options=options)

    # Search what are the component/group outputs
    variables = list_variables(local_system)
    inputs_names = [var.name for var in variables if var.is_input]
    outputs_names = [var.name for var in variables if not var.is_input]

    # Check the sub-systems of the local_system in question, and if there are ivc, list the
    # outputs  of those ivc. We are gonna assume that ivc are only use in a situation similar to
    # the one for the ComputePropellerPerformance group, meaning if there is an ivc,
    # it will always start the group

    ivc_outputs_names = list_ivc_outputs_name(local_system)

    # Check that variable inputs are in the group/component list
    if not (set(var_inputs) == set(inputs_names).intersection(set(var_inputs))):
        raise Exception("The input list contains name(s) out of component/group input list!")

    # Perform some tests on the .xml availability and completeness
    if not (os.path.exists(xml_file_path)) and not (set(var_inputs) == set(inputs_names)):
        # If no input file and some inputs are missing, generate it and return None
        group = AutoUnitsDefaultGroup()
        group.add_subsystem("system", local_system, promotes=["*"])
        problem = oad.FASTOADProblem()
        problem.model = group
        problem.setup()
        write_needed_inputs(problem, xml_file_path, VariableXmlStandardFormatter())
        raise Exception(
            "Input .xml file not found, a default file has been created with default NaN values, "
            "but no function is returned!\nConsider defining proper values before second execution!"
        )

    else:
        if os.path.exists(xml_file_path):
            reader = VariableIO(xml_file_path, VariableXmlStandardFormatter()).read(
                ignore=(var_inputs + outputs_names + ivc_outputs_names)
            )
            xml_inputs = reader.names()
        else:
            xml_inputs = []
        if not (
            set(xml_inputs + var_inputs + ivc_outputs_names).intersection(set(inputs_names))
            == set(inputs_names)
        ):
            # If some inputs are missing write an error message and add them to the problem if
            # authorized
            missing_inputs = list(
                set(inputs_names).difference(
                    set(xml_inputs + var_inputs + ivc_outputs_names).intersection(set(inputs_names))
                )
            )
            message = "The following inputs are missing in .xml file:"
            for item in missing_inputs:
                message += " [" + item + "],"
            message = message[:-1] + ".\n"
            if overwrite:
                # noinspection PyUnboundLocalVariable
                reader.path_separator = ":"
                ivc = reader.to_ivc()
                group = AutoUnitsDefaultGroup()
                group.add_subsystem("system", local_system, promotes=["*"])
                group.add_subsystem("ivc", ivc, promotes=["*"])
                problem = oad.FASTOADProblem()
                problem.model = group
                problem.input_file_path = xml_file_path
                problem.output_file_path = xml_file_path
                problem.setup()
                problem.write_outputs()
                message += (
                    "Default values have been added to {} file. "
                    "Consider modifying them for a second run!".format(xml_file_path)
                )
                raise Exception(message)
            else:
                raise Exception(message)
        else:
            # If all inputs addressed either by .xml or var_inputs or in an IVC, construct the
            # function
            def patched_function(inputs_dict: dict) -> dict:
                """
                The patched function perform a run of an openmdao component or group applying
                FASTOAD formalism.

                @param inputs_dict: dictionary of input (values, units) saved with their key name,
                as an example: inputs_dict = {'in1': (3.0, "m")}.
                @return: dictionary of the component/group outputs saving names as keys and (value,
                units) as tuple.
                """

                # Read .xml file and construct Independent Variable Component excluding outputs
                if os.path.exists(xml_file_path):
                    reader.path_separator = ":"
                    ivc_local = reader.to_ivc()
                else:
                    ivc_local = IndepVarComp()
                for name, value in inputs_dict.items():
                    ivc_local.add_output(name, value[0], units=value[1])
                group_local = AutoUnitsDefaultGroup()
                group_local.add_subsystem("ivc", ivc_local, promotes=["*"])
                group_local.add_subsystem("system", local_system, promotes=["*"])
                problem_local = oad.FASTOADProblem()
                model_local = problem_local.model
                model_local.add_subsystem("local_system", group_local, promotes=["*"])
                problem_local.setup()
                problem_local.run_model()
                if overwrite:
                    problem_local.output_file_path = xml_file_path
                    problem_local.write_outputs()
                # Get output names from component/group and construct dictionary
                outputs_units = [var.units for var in variables if not var.is_input]
                outputs_dict = {}
                for idx, _ in enumerate(outputs_names):
                    value = problem_local.get_val(outputs_names[idx], outputs_units[idx])
                    outputs_dict[outputs_names[idx]] = (value, outputs_units[idx])
                return outputs_dict

            return patched_function


def list_all_subsystem(model, model_address, dict_subsystems):
    # noinspection PyBroadException
    try:
        # noinspection PyProtectedMember
        subsystem_list = model._proc_info.keys()
        for subsystem in subsystem_list:
            dict_subsystems = list_all_subsystem(
                getattr(model, subsystem), model_address + "." + subsystem, dict_subsystems
            )
    except AttributeError:
        dict_subsystems[model_address] = get_type(model)

    return dict_subsystems


def get_type(model):
    raw_type = model.msginfo.split("<")[-1]
    type_alone = raw_type.split(" ")[-1]
    model_type = type_alone[:-1]

    return model_type


class VariableListLocal(oad.VariableList):
    @classmethod
    def from_system(cls, local_system: System) -> "oad.VariableList":
        """
        Creates a VariableList instance containing inputs and outputs of a an OpenMDAO System.
        The inputs (is_input=True) correspond to the variables of IndepVarComp
        components and all the unconnected variables.

        Warning: setup() must NOT have been called.

        In the case of a group, if variables are promoted, the promoted name
        will be used. Otherwise, the absolute name will be used.

        :param local_system: OpenMDAO Component instance to inspect
        :return: VariableList instance.
        """

        problem = oad.FASTOADProblem()
        if isinstance(local_system, om.Group):
            problem.model = deepcopy(local_system)
        else:
            # problem.model has to be a group
            problem.model.add_subsystem("comp", deepcopy(local_system), promotes=["*"])
        try:
            problem.setup()
        except RuntimeError:
            _LOGGER.info(
                "Some problem occurred while setting-up the problem without input file probably "
                "because shape_by_conn variables exist!"
            )
        return VariableListLocal.from_problem(problem, use_initial_values=True)


def list_variables(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """Reads all variables from a component/problem and return as a list."""
    if isinstance(component, om.Group):
        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
    variables = VariableListLocal.from_system(component)

    return variables


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """Reads all variables from a component/problem and returns inputs as a list."""
    variables = list_variables(component)
    input_names = [var.name for var in variables if var.is_input]

    return input_names


def list_inputs_metadata(component: Union[om.ExplicitComponent, om.Group]) -> tuple:
    """
    Reads all variables from a component/problem and returns inputs name and metadata as a
    list.
    """

    prob = oad.FASTOADProblem()
    model = prob.model
    model.add_subsystem("component", component, promotes=["*"])

    prob_copy = deepcopy(prob)

    var_copy_shape_name_list = []
    var_copy_shape_list = []

    try:
        prob_copy.setup()
    except RuntimeError:
        # noinspection PyProtectedMember
        vars_metadata = oad.FASTOADProblem()._get_undetermined_dynamic_vars_metadata(prob_copy)
        if vars_metadata:
            # If vars_metadata is empty, it means the RuntimeError was not because
            # of dynamic shapes, and the incoming self.setup() will raise it.
            ivc = om.IndepVarComp()
            for name, meta in vars_metadata.items():
                # We use a (2,)-shaped array as value here. This way, it will be easier to
                # identify dynamic-shaped data in an input file generated from current problem.
                var_copy_shape_name_list.append(name)
                var_copy_shape_list.append(meta["copy_shape"])
                ivc.add_output(name, [np.nan, np.nan], units=meta["units"])
            prob.model.add_subsystem("temp_shaper", ivc, promotes=["*"])

    variables = prob_copy.model.get_io_metadata("input")

    var_inputs = []
    var_units = []
    var_shape = []
    var_shape_by_conn = []
    var_copy_shape = []

    for variable_name in variables:
        variable = variables[variable_name]
        var_prom_name = variable["prom_name"]

        # We check that it has not been added already
        if var_prom_name not in var_inputs:
            var_inputs.append(variable["prom_name"])
            var_units.append(variable["units"])
            var_shape.append(variable["shape"])
            var_shape_by_conn.append(variable["shape_by_conn"])
            var_copy_shape.append(variable["copy_shape"])

    return var_inputs, var_units, var_shape, var_shape_by_conn, var_copy_shape


def list_outputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """Reads all variables from a component/problem and returns outputs as a list."""
    variables = list_variables(component)
    output_names = [var.name for var in variables if not var.is_input]

    return output_names


def string_to_array(arr):
    """
    Convert a numpy array that was stored as a string back to a numpy array.

    "[-20.0, -19.5, -19.0, ..., 19.0, 19.5, 20.0]" --> [-20.0, -19.5, -19.0, ..., 19.0, 19.5, 20.0]
    """
    return np.array(arr.strip("[]").split(","), dtype=float)
