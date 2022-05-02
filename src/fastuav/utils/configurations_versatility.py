"""
Methods to ensure the OpenMDAO components' versatility for both fixed wing and multirotor configurations.
"""

import openmdao.api as om
import re
from typing import List
import warnings


def promote_and_rename(
        group: om.Group,
        subsys: om.Group,
        rename_inputs: bool=True,
        rename_outputs: bool=True,
        old_patterns_list: List[str] = None,
        new_patterns_list: List[str] = None,
):
    """
    Promote the inputs and outputs variables of the OpenMDAO subsystem,
    and rename the variables according the new pattern.

    Parameters
    ----------
    group : om.Group
            Parent group object.
    subsys : om.Group
            Subsystem whose variables are to be promoted.
    rename_inputs : bool
            whether to rename inputs or not.
    rename_outputs : bool
            whether to rename outputs or not.
    old_patterns_list : list[str]
            Old string pattern to be renamed.
    new_patterns_list : list[str]
            New string pattern.
    ----------

    Example
    >> promote_and_rename(parent_group, subsystem, old_pattern=":propulsion:", new_pattern=":propulsion:multirotor:")

    PLEASE NOTE:
    promote_and_rename() must be called in the configure method of the parent group instead of the setup method.
    This is because the information from the subsystems (i.e. the variables names) is not available until the full
    setup has be achieved.
    Visit https://openmdao.org/newdocs/versions/latest/theory_manual/setup_stack.html for more information.

    """

    # Get input and output variables names from subsystem
    # TODO: list only promoted variables from subsubsytems \
    #  (here '*uncertainty:*:mean' are non-promoted variables but still visible so have to be excluded by hand)
    var_in_names = [var[0].split(".")[-1] for var in
                    subsys.list_inputs(val=False, out_stream=None, excludes=['*uncertainty:*:mean'])]
    var_out_names = [var[0].split(".")[-1] for var in
                     subsys.list_outputs(val=False, out_stream=None, excludes=['*uncertainty:*:mean'])]

    # Keep only unique values
    var_in_names = list(set(var_in_names))
    var_out_names = list(set(var_out_names))

    # Create lists of new names
    var_in_names_new = var_in_names
    var_out_names_new = var_out_names
    for old_pattern, new_pattern in zip(old_patterns_list, new_patterns_list):
        var_in_names_new = [re.sub(old_pattern, new_pattern, name) for name in var_in_names_new]
        var_out_names_new = [re.sub(old_pattern, new_pattern, name) for name in var_out_names_new]

    # Promote variables with new name
    inputs = [(old_name, new_name) for old_name, new_name in
              zip(var_in_names, var_in_names_new)] if rename_inputs else var_in_names
    outputs = [(old_name, new_name) for old_name, new_name in
               zip(var_out_names, var_out_names_new)] if rename_outputs else var_out_names
    group.promotes(subsys.name,
                   inputs=inputs,
                   outputs=outputs)

    # Turn off warnings (calling list_inputs and list_outputs before final_setup will issue a warning
    # as only the default values of the variables will be displayed. This behaviour is not impacting the results here)
    warnings.filterwarnings('ignore', category=om.OpenMDAOWarning)