"""
LCA models and calculations.
"""

# import time
import openmdao.api as om
import numpy as np
import brightway2 as bw
import lca_algebraic as lcalg
from lcav.io.configuration import LCAProblemConfigurator
import re
import sympy as sym
from fastuav.exceptions import FastLcaProjectDoesNotExist, \
    FastLcaDatabaseIsNotImported, \
    FastLcaMethodDoesNotExist, \
    FastLcaParameterNotDeclared
from fastuav.constants import LCA_DEFAULT_PROJECT, LCA_DEFAULT_ECOINVENT, LCA_USER_DB, LCA_MODEL_KEY, \
    LCA_PARAM_KEY, LCA_CHARACTERIZATION_KEY, LCA_DEFAULT_METHOD, SIZING_MISSION_TAG, LCA_FUNCTIONAL_UNITS_LIST, \
    LCA_DEFAULT_FUNCTIONAL_UNIT, LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY, LCA_FACTOR_KEY, LCA_WEIGHTED_SINGLE_SCORE_KEY, \
    LCA_AGGREGATION_KEY


class LCAcore(om.Group):
    """
    Group for LCA models and calculations.
    """

    def initialize(self):
        # Declare options
        self.options.declare("configuration_file", default=None, types=str)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)

        # Computation options for optimization
        self.options.declare("analytical_derivatives", default=True, types=bool)

        # FAST-UAV model specific parameters
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

        # FAST-UAV specific option for selecting mission to evaluate
        self.options.declare("mission", default=SIZING_MISSION_TAG, types=str)

    def setup(self):
        # LCA MODEL
        self.add_subsystem(
            "model",
            Model(
                configuration_file=self.options["configuration_file"],
                mission=self.options["mission"]
            ),
            promotes=["*"]
        )

        # CHARACTERIZATION
        #self.add_subsystem(
        #    "characterization",
        #    Characterization()
        #    # promote for this subsystem is done in configure() method
        #)

        # NORMALIZATION
        if self.options["weighting"] or self.options["normalization"]:
            self.add_subsystem("normalization", Normalization(), promotes=["*"])

        # WEIGHTING AND AGGREGATION
        if self.options["weighting"]:
            self.add_subsystem("weighting", Weighting(), promotes=["*"])
            self.add_subsystem("aggregation", Aggregation(), promotes=["*"])

    def configure(self):
        """
        Set inputs and options for characterization module by copying `model` outputs and options.
        Configure() method from the containing group is necessary to get access to `model` metadata after Setup().
        """

        # Add LCA parameters declared in the LCA model to the characterization module,
        # either as inputs (float parameters) or options (str parameters)
        # for name, object in self.model.parameters.items():
        #     if object.type == 'float':  # add float parameters as inputs to calculation module
        #         self.characterization.add_input(LCA_PARAM_KEY + name, val=np.nan, units=None)
        #     elif name in self.options["parameters"].keys():  # add non-float parameters as options
        #         self.characterization.options["parameters"][name] = self.options["parameters"][name]
        #
        # # Add LCIA methods retrieved from the lca configuration file
        # self.characterization.options["methods"] = self.model.methods
        # self.normalization.options["methods"] = self.model.methods
        # self.weighting.options["methods"] = self.model.methods
        #
        # # Promote variables and declare partials
        # self.promotes("characterization", any=['*'])
        # if self.options['analytical_derivatives']:
        #     self.characterization.declare_partials("*", "*", method="exact")
        # else:
        #     self.characterization.declare_partials("*", "*", method="fd")


class Model(om.ExplicitComponent):
    """
    This OpenMDAO component implements an LCA model using brightway2 and lca_algebraic libraries.
    It creates an LCA model and sets the LCA parameters for further parametric LCA calculation.
    """

    def initialize(self):
        # Attributes
        self.parameters = dict()  # dictionary of {parameter_name: parameter_object} to store all parameters
        self.model = None  # LCA model
        self.methods = list()  # list of methods for LCIA

        # Declare options
        self.options.declare("configuration_file", default=None, types=str)
        self.options.declare("mission", default=SIZING_MISSION_TAG, types=str)

    def setup(self):
        _, self.model, self.methods = LCAProblemConfigurator(self.options['configuration_file']).generate()
        self.parameters = lcalg.all_params().values()

        # Add inputs and outputs related to LCA parameters.
        # (Note: non-float parameters are to be declared through options)
        # Inputs: UAV parameters
        # NB: check that units are consistence with LCA parameters/activities!
        mission_name = self.options["mission"]
        self.add_input("mission:%s:distance" % mission_name, val=np.nan, units='m')
        self.add_input("mission:%s:duration" % mission_name, val=np.nan, units='min')
        self.add_input("mission:%s:energy" % mission_name, val=np.nan, units='kJ')
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')  # TODO: specify operational mission payload?
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:battery:efficiency", val=np.nan, units=None)

        # Outputs: LCA parameters calculated in compute() method
        self.add_output(LCA_PARAM_KEY + 'mission_distance', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mission_duration', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mission_energy', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_payload', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_motors', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_propellers', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_controllers', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_batteries', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_airframe', val=np.nan, units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # The compute method is used here to set lca parameters' values
        mission_name = self.options["mission"]
        mission_distance = inputs["mission:%s:distance" % mission_name] / 1000  # [km]  # TODO: select distance of a single route?
        mission_duration = inputs["mission:%s:duration" % mission_name] / 60  # [h]
        mass_payload = inputs["mission:sizing:payload:mass"]  # [kg]
        mission_energy = inputs["mission:%s:energy" % mission_name] / 3600 / inputs[
            "data:propulsion:multirotor:battery:efficiency"]  # power at grid [kWh]
        mass_batteries = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        mass_motors = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        mass_propellers = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        mass_controllers = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
        mass_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]

        # Data manipulation for normalized models (avoid division by zero)
        eps = np.array([1e-9])
        mission_distance = max(mission_distance, eps)
        mission_duration = max(mission_duration, eps)
        mass_payload = max(mass_payload, eps)

        # Outputs: LCA parameters (names should be the same as declared in the declare_parameters() function
        outputs[LCA_PARAM_KEY + 'mission_distance'] = mission_distance
        outputs[LCA_PARAM_KEY + 'mission_duration'] = mission_duration
        outputs[LCA_PARAM_KEY + 'mission_energy'] = mission_energy
        outputs[LCA_PARAM_KEY + 'mass_payload'] = mass_payload
        outputs[LCA_PARAM_KEY + 'mass_motors'] = mass_motors
        outputs[LCA_PARAM_KEY + 'mass_propellers'] = mass_propellers
        outputs[LCA_PARAM_KEY + 'mass_controllers'] = mass_controllers
        outputs[LCA_PARAM_KEY + 'mass_batteries'] = mass_batteries
        outputs[LCA_PARAM_KEY + 'mass_airframe'] = mass_airframe

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_name = self.options["mission"]

        partials[LCA_PARAM_KEY + 'mission_distance', "mission:%s:distance" % mission_name] = 1 / 1000
        partials[LCA_PARAM_KEY + 'mission_duration', "mission:%s:duration" % mission_name] = 1 / 60
        partials[LCA_PARAM_KEY + 'mission_energy',
                 "data:propulsion:multirotor:battery:efficiency"] = - inputs["mission:%s:energy" % mission_name] / 3600 / inputs["data:propulsion:multirotor:battery:efficiency"] ** 2
        partials[LCA_PARAM_KEY + 'mission_energy',
                 "mission:%s:energy" % mission_name] = 1 / 3600 / inputs["data:propulsion:multirotor:battery:efficiency"]
        partials[LCA_PARAM_KEY + 'mass_payload', "mission:sizing:payload:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_motors',
                 "data:weight:propulsion:multirotor:motor:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_motors',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:motor:mass"]
        partials[LCA_PARAM_KEY + 'mass_propellers',
                 "data:weight:propulsion:multirotor:propeller:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_propellers',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:propeller:mass"]
        partials[LCA_PARAM_KEY + 'mass_controllers',
                 "data:weight:propulsion:multirotor:esc:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_controllers',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:esc:mass"]
        partials[LCA_PARAM_KEY + 'mass_batteries', "data:weight:propulsion:multirotor:battery:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_airframe', "data:weight:airframe:body:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_airframe', "data:weight:airframe:arms:mass"] = 1.0


class Characterization(om.ExplicitComponent):
    """
    This OpenMDAO component achieves a life cycle inventory (LCI) and a life cycle impact assessment (LCIA)
    for the provided model, parameters and methods.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # symbolic expressions of LCA: used for providing analytical partials
        self.exprs_dict = dict()

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # methods
        self.methods = [eval(m) for m in self.options["methods"]]
        self.assert_methods(self.methods)  # check methods exist in brightway project
        self.set_method_labels(self.methods)  # some formatting for labels in lca_algebraic library

        # outputs
        for path in self.activities.keys():
            for m in self.methods:
                m_name = self.method_label_formatting(m)

                # characterized score
                self.add_output(LCA_CHARACTERIZATION_KEY + m_name + path,
                                units=None,  # LCA units not supported by OpenMDAO so set in description
                                desc=bw.Method(m).metadata['unit'] + "/FU")

    # def setup_partials(self):
    #     Declared in configure method of parent group

    def compute(self, inputs, outputs):

        #start_time = time.time()

        # parameters
        parameters = self.options["parameters"]  # initialized with non-float parameters provided as options
        for key, value in inputs.items():  # add float parameters provided as inputs
            if LCA_PARAM_KEY in key and not np.isnan(value):
                name = re.split(LCA_PARAM_KEY, key)[-1]
                parameters[name] = value

        # LCA calculations (first call may be time-consuming but next calls are faster thanks to cache)
        activities = self.activities  # get all activities and sub activities in model
        for path, act in activities.items():
            res = self.compute_lca(extract_activities=act,
                                   parameters=parameters)  # parametric LCA
            res.index.values[0] = act['name']

            # storing symbolic expression for future use (e.g. analytical derivatives calculation)
            if act not in self.exprs_dict:  # TODO: add option to disable this storage (e.g. only if analytical_derivatives)
                self.exprs_dict[act] = self.lca_model_expression(extract_activities=act)

            # Outputs
            for m in res:
                # get score for method m
                score = res[m][0]  # if a parameter is provided as a list of values, there will be several scores. For now we only get the first one.
                # results from lca_algebraic does not use the same names as the input methods list...
                end = m.find("[")  # TODO: mapping function to improve calculation time?
                m_name = m[:end]
                # set output value
                outputs[LCA_CHARACTERIZATION_KEY + m_name + path] = score

        #elapsed_time = time.time() - start_time
        #print(elapsed_time)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # Parameters dictionary
        parameters = self.options["parameters"]  # initialized with non-float parameters provided as options
        for key, value in inputs.items():  # add float parameters provided as inputs
            if LCA_PARAM_KEY in key and not np.isnan(value):
                name = re.split(LCA_PARAM_KEY, key)[-1]
                parameters[name] = value

        # partials
        for path, act in self.activities.items():
            for m in self.methods:
                m_name = self.method_label_formatting(m)
                output_name = LCA_CHARACTERIZATION_KEY + m_name + path
                for input_name in inputs.keys():
                    param_name = re.split(LCA_PARAM_KEY, input_name)[-1]
                    partials[output_name,
                             input_name] = self.partials_lca(param_name, m_name, parameters, act)

    def partials_lca(self, input_param, output_method, parameters, activity):
        """
        returns the partial derivative of a method's result with respect to a parameter.
        """
        # TODO: compute derivative expressions at initialization to avoid systematic differentiation of sympy expression.

        # Dictionary of algebraic LCA expressions
        exprs_dict = self.exprs_dict

        # Sub-dictionary of expressions for a given activity
        exprs = exprs_dict[activity]

        # Expression for a given method
        expr = exprs[output_method]

        # replace functions that has no well-defined derivatives
        expr_simplified = expr.replace(sym.ceiling, lambda x: x)  # here we replace ceiling function by identity

        # differentiate expression
        derivative = sym.diff(expr_simplified, input_param)

        # expand enumParams --> 'elec_switch_param': 'eu' becomes 'elec_switch_param_us' : 0, 'elec_switch_param_eu': 1
        new_params = {name: value for name, value in lcalg.params._completeParamValues(parameters).items()}
        for key, val in new_params.items():
            if isinstance(val, np.ndarray):
                new_params[key] = val[0]

        # evaluate derivative
        res = derivative.evalf(subs=new_params)

        return res

    def compute_lca(self, extract_activities, parameters=None):
        """
        Main LCIA calculation method.
        """
        if extract_activities == self.model:
            extract_activities = None
        else:
            extract_activities = [extract_activities]

        if parameters is None:
            parameters = {}

        res = lcalg.multiLCAAlgebric(
            self.model,  # The model
            self.methods,  # Impact categories / methods

            # List of sub activities to consider
            extract_activities=extract_activities,

            # Parameters of the model
            **parameters
        )
        return res

    def lca_model_expression(self, extract_activities=None):
        """
        computes algebraic expressions corresponding to a model for each method
        """
        exprs = dict()
        if extract_activities == self.model:
            extract_activities = None
        else:
            extract_activities = [extract_activities]
        with lcalg.params.DbContext(self.model):
            exprs_list, _ = lcalg.lca._modelToExpr(self.model, self.methods, extract_activities=extract_activities)
        for i, method in enumerate(self.methods):
            m_name = self.method_label_formatting(method)
            exprs[m_name] = exprs_list[i]
        return exprs

    @staticmethod
    def assert_methods(methods):
        """
        Check if methods exist in brightway.
        """
        for method in methods:
            if method not in bw.methods:
                raise FastLcaMethodDoesNotExist(method)

    @staticmethod
    def set_method_labels(methods):
        """
        Set custom method labels for lca_algebraic results.
        """
        dict_labels = {}
        for m in methods:
            dict_labels[m] = Characterization.method_label_formatting(m)
        lcalg.set_custom_impact_labels(dict_labels)

    @staticmethod
    def method_label_formatting(method_name):
        """
        Format method labels for fast-oad compatibility (handling of variables names).
        """
        new_name = [
            s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')',
                                                                                                               '').replace('/', '_')
            for s in method_name]  # replace invalid characters
        new_name = ':'.join(['%s'] * len(new_name)) % tuple(new_name)  # concatenate method specifications
        return new_name


class Normalization(om.ExplicitComponent):
    """
    Normalization of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for m_name in self.methods:
            self.add_input(LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY,  # normalization factors
                           val=np.nan,
                           units=None)

            for path in self.activities.keys():
                self.add_input(LCA_CHARACTERIZATION_KEY + m_name + path,  # characterized scores
                               val=np.nan,
                               units=None)
                self.add_output(LCA_NORMALIZATION_KEY + m_name + path,  # normalized scores for each activity
                                units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        for m_name in self.methods:
            method_factor = inputs[LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY]
            for path in self.activities.keys():
                score = inputs[LCA_CHARACTERIZATION_KEY + m_name + path]
                normalized_score = score / method_factor
                outputs[LCA_NORMALIZATION_KEY + m_name + path] = normalized_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for m_name in self.methods:
            method_factor = inputs[LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY]
            for path in self.activities.keys():
                score = inputs[LCA_CHARACTERIZATION_KEY + m_name + path]
                partials[LCA_NORMALIZATION_KEY + m_name + path,
                         LCA_CHARACTERIZATION_KEY + m_name + path] = 1 / method_factor
                partials[LCA_NORMALIZATION_KEY + m_name + path,
                         LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY] = - score / method_factor ** 2


class Weighting(om.ExplicitComponent):
    """
    Weighting of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # sub activities in model
        self.activities = Characterization.recursive_activities(self.model)

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for m_name in self.methods:
            self.add_input(LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY,  # weighting factors
                           val=np.nan,
                           units=None)

            for path in self.activities.keys():
                self.add_input(LCA_NORMALIZATION_KEY + m_name + path,  # normalized scores
                               val=np.nan,
                               units=None)
                self.add_output(LCA_WEIGHTING_KEY + m_name + path,  # weighted scores for each activity
                                units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        for path in self.activities.keys():
            # aggregated_score = 0.0
            for m_name in self.methods:
                normalized_score = inputs[LCA_NORMALIZATION_KEY + m_name + path]
                method_weight = inputs[LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY]
                weighted_score = normalized_score * method_weight
                outputs[LCA_WEIGHTING_KEY + m_name + path] = weighted_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for path in self.activities.keys():
            for m_name in self.methods:
                normalized_score = inputs[LCA_NORMALIZATION_KEY + m_name + path]
                method_weight = inputs[LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY]
                partials[LCA_WEIGHTING_KEY + m_name + path, LCA_NORMALIZATION_KEY + m_name + path] = method_weight
                partials[LCA_WEIGHTING_KEY + m_name + path, LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY] = normalized_score


class Aggregation(om.ExplicitComponent):
    """
    Aggregation of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for path in self.activities.keys():
            for m_name in self.methods:
                self.add_input(LCA_WEIGHTING_KEY + m_name + path,  # weighted scores for each activity
                               val=np.nan,
                               units=None)
            self.add_output(LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path,  # aggregated scores
                            units=None,
                            desc='points')

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # aggregated_score = 0.0
        for path in self.activities.keys():
            aggregated_score = sum([inputs[LCA_WEIGHTING_KEY + m_name + path] for m_name in self.methods])
            outputs[LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path] = aggregated_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for path in self.activities.keys():
            for m_name in self.methods:
                partials[LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path,
                         LCA_WEIGHTING_KEY + m_name + path] = 1.0