"""
Variable uncertainty component.
"""

import openmdao.api as om
from typing import List


def add_subsystem_with_deviation(
    group: om.Group,
    subsys_name: str,
    subsys: om.ExplicitComponent,
    uncertain_outputs: dict = None,
):
    """
    Add the model component as a subsystem to the group,
    then add an additional subsystem that modifies the uncertain outputs values according to the user-defined variations.

    Parameters
    ----------
    group : om.Group
            Parent group object.
    subsys_name : str
            Name of the subsystem being added.
    subsys : om.Performance
            An instantiated, but not-yet-set up system object.
    uncertain_outputs: dict
            Dictionary containing the names and units of the subsystem's outputs to be modified.
    ----------

    Example
        >> class Group(om.Group):
        >> def setup(self):
        >>      add_subsystem_with_deviation(self, "nominal_torque", NominalTorque(),
                                                uncertain_outputs={'data:propulsion:motor:torque:nominal:estimated': 'N*m'})
    """

    # add model component
    group.add_subsystem(subsys_name, subsys, promotes_inputs=["*"])

    # variables to be modified
    long_names = []
    short_names = []  # names without first filter (e.g. 'data:')
    units = []  # units
    for name, unit in uncertain_outputs.items():
        long_names.append(name)
        name_split = name.split(":")
        # if name_split[-1] == 'estimated':
        short_names.append(
            ":".join(name.split(":")[1:-1])
        )  # remove first and last ('data:' and ':estimated')
        # else:
        #    short_names.append(':'.join(name.split(':')[1:]))  # remove first ('data:')
        units.append(unit)

    # add 'deviation' component to modify the variables
    if uncertain_outputs is not None:
        subsys_uncertainty = ComponentWithDeviation()  # create component
        subsys_uncertainty.add_variables(
            long_names=long_names, short_names=short_names, units=units
        )  # add variables to be modified
        group.add_subsystem(
            subsys_name + "_deviation", subsys_uncertainty
        )  # add component to group

        # connect variables
        for i, long_name in enumerate(long_names):
            short_name = short_names[i]
            group.connect(
                subsys_name + "." + long_name,
                subsys_name + "_deviation.uncertainty:" + short_name + ":mean",
            )
            group.promotes(
                subsys_name + "_deviation",
                inputs=[
                    "uncertainty:" + short_name + ":rel",
                    "uncertainty:" + short_name + ":abs",
                ],
                outputs=[long_name],
            )

    return 0


class ComponentWithDeviation(om.ExplicitComponent):
    """
    Modify a mean value by adding a deviation:
        y = mean(y) * ( 1 + var)
    The deviation is provided by the user as an input ('uncertainty:parameter_name:var').
    """

    def __init__(self):
        super().__init__()
        self._long_names = []  # variables names
        self._short_names = []  # short variables names

    def add_variables(self, long_names=None, short_names=None, units=None):
        for i, long_name in enumerate(long_names):
            if long_names not in self._long_names:
                short_name = short_names[i]
                unit = units[i]
                self.add_input("uncertainty:" + short_name + ":mean", units=unit)
                self.add_input(
                    "uncertainty:" + short_name + ":rel", val=0.0, units=None
                )
                self.add_input(
                    "uncertainty:" + short_name + ":abs", val=0.0, units=unit
                )
                self.add_output(long_name, units=unit)
                self._long_names.append(long_name)
                self._short_names.append(short_name)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        for i, long_name in enumerate(self._long_names):
            short_name = self._short_names[i]
            mean = inputs["uncertainty:" + short_name + ":mean"]  # mean value
            eps = inputs["uncertainty:" + short_name + ":rel"]  # relative deviation
            delta = inputs["uncertainty:" + short_name + ":abs"]  # absolute deviation
            outputs[long_name] = mean * (1 + eps) + delta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for i, long_name in enumerate(self._long_names):
            short_name = self._short_names[i]
            mean = inputs["uncertainty:" + short_name + ":mean"]  # mean value
            eps = inputs["uncertainty:" + short_name + ":rel"]  # relative deviation
            partials[long_name, "uncertainty:" + short_name + ":mean"] = 1 + eps
            partials[long_name, "uncertainty:" + short_name + ":rel"] = mean
            partials[long_name, "uncertainty:" + short_name + ":abs"] = 1


def add_model_deviation(
    group: om.Group, subsys_name: str, model, uncertain_parameters: List[str] = None
):
    """
    Modify the uncertain parameters of the model.

    The model should be wrapped in a class with @staticmethods, e.g.:
        >> Class model:
                init_uncertain_parameters = {}  # initial model's parameters
                uncertain_parameters = init_uncertain_parameters.copy()  # model's parameters

                @staticmethod
                def method():
                    y = f(x, uncertain_parameters)
                    return y

    The deviation is provided by the user as an input. To do so, a new component is instantiated.

    Parameters
    ----------
    group : om.Group
            Parent group object.
    subsys_name : str
            Name of the openmdao component that will be added to represent the model deviation.
    model : a Class containing static methods and parameters.
    uncertain_parameters: List[str]
            List containing the names of the model's parameters to be modified.

    ----------
    Example
        >> class Group(om.Group):
        >> def setup(self):
        >>      add_model_deviation(self, "aerodynamics_model_deviation", PropellerAerodynamicsModel,
                                        uncertain_parameters=['Ct_axial_var', 'Cp_axial_var'])
    """

    # add 'deviation' component to modify the variables
    if uncertain_parameters is not None:
        subsys_uncertainty = ModelDeviation()  # create component
        subsys_uncertainty.add_variables(
            names=uncertain_parameters, model=model
        )  # add variables to be modified
        group.add_subsystem(
            subsys_name, subsys_uncertainty, promotes=["*"]
        )  # add component to group

    return 0


class ModelDeviation(om.ExplicitComponent):
    """
    Modify the parameters of a model with a relative error.
    The deviation is provided by the user as an input ('uncertainty:model:parameter_name:rel').
    """

    def __init__(self):
        super().__init__()
        self._model = None  # model to be modified
        self._names = []  # names of parameters to be modified
        # self._is_modified = False  # whether the model has been modified yet or not

    def add_variables(self, names=None, model=None):
        self._model = model
        for i, name in enumerate(names):
            if name not in self._names:
                self.add_input("uncertainty:" + name + ":rel", val=0.0, units=None)
                self.add_input("uncertainty:" + name + ":abs", val=0.0, units=None)
                self._names.append(name)

    def compute(self, inputs, outputs):
        # if self._is_modified:
        #    pass
        # else:
        model_attrs = self._model.__dict__  # get class attributes
        if (
            "uncertain_parameters" in model_attrs.keys()
        ):  # and 'init_uncertain_parameters' in model_attrs.keys():
            v = model_attrs[
                "uncertain_parameters"
            ]  # get the dictionary of uncertain parameters
            # v_init = model_attrs['init_uncertain_parameters']  # get the dictionary of initial uncertain parameters
            for i, name in enumerate(
                self._names
            ):  # loop through parameters to be modified
                name_rel = "uncertainty:" + name + ":rel"
                name_abs = "uncertainty:" + name + ":abs"
                if name_rel in v.keys():
                    v[name_rel] = inputs[name_rel]  # relative deviation
                if name_abs in v.keys():
                    v[name_abs] = inputs[name_abs]  # absolute deviation
        # self._is_modified = True
