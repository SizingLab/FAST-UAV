"""
LCA models and calculations.
"""

# import time
import openmdao.api as om
import numpy as np
import brightway2 as bw
import pandas as pd
import lca_algebraic as agb
from lca_algebraic.axis_dict import AxisDict
from typing import Dict
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
        self.options.declare("axis", default=None, types=str)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)

        # Computation options for optimization
        self.options.declare("analytical_derivatives", default=True, types=bool)

    def setup(self):
        # LCA MODEL
        self.add_subsystem(
            "model",
            Model(
                configuration_file=self.options["configuration_file"],
                axis=self.options["axis"]
            ),
            promotes=["*"]
        )

        # NORMALIZATION
        if self.options["weighting"] or self.options["normalization"]:
            self.add_subsystem("normalization", Normalization(), promotes=["*"])

        # WEIGHTING AND AGGREGATION
        if self.options["weighting"]:
            self.add_subsystem("weighting", Weighting(), promotes=["*"])
            self.add_subsystem("aggregation", Aggregation(), promotes=["*"])

   #def configure(self):
   #     """
   #     Set inputs and options for characterization module by copying `model` outputs and options.
   #     Configure() method from the containing group is necessary to get access to `model` metadata after Setup().
   #     """

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
        # Declare options
        self.options.declare("configuration_file", default=None, types=str)
        self.options.declare("axis", default=None, types=str)

    def setup(self):
        # Read LCA configuration file, build model and get methods
        _, self.model, self.methods = LCAProblemConfigurator(self.options['configuration_file']).generate()

        # Retrieve LCA parameters declared in model
        self.parameters = agb.all_params().values()

        # Compile expressions for impacts
        self.lambdas = agb.lca._preMultiLCAAlgebric(self.model, self.methods, axis=self.options['axis'])

        # Compile expressions for partial derivatives of impacts w.r.t. parameters
        self.partial_lambdas_dict = _preMultiLCAAlgebricPartials(self.model, self.methods, axis=self.options['axis'])
        # self.partial_lambdas_dict = {
        #     param.name: [
        #         agb.lca.LambdaWithParamNames(  # lambdify expression for future evaluation by lca_algebraic
        #             lambd.expr.replace(sym.ceiling, lambda x: x).  # replace ceiling function by identity for better derivatives
        #             diff(param)  # differentiate expression w.r.t. parameter
        #         ) for lambd in self.lambdas  # for each LCIA method
        #     ] for param in self.parameters  # for each parameter
        # }

        # Get axis keys to ventilate results by e.g. life-cycle phase
        self.axis_keys = self.lambdas[0].axis_keys

        # Each LCA parameter is declared as an input
        for p in self.parameters:
            if p.type == 'float':
                p_name = p.name.replace('__', ':')  # refactor names (':' is not supported in LCA parameters)
                self.add_input(p_name, val=np.nan, units=None)
            # TODO: add non float parameters as options?

        # Declare outputs for each method and axis key
        for m in self.methods:
            m_name = re.sub(r': |/| ', '_', m[1])
            self.add_output(LCA_CHARACTERIZATION_KEY + m_name,
                            units=None,  # NB: LCA units not supported by OpenMDAO so set in description
                            desc=bw.Method(m).metadata['unit'] + "/FU")
            if self.axis_keys:
                for axis_key in self.axis_keys:
                    self.add_output(LCA_CHARACTERIZATION_KEY + m_name + ':' + axis_key,
                                    units=None,
                                    desc=bw.Method(m).metadata['unit'] + "/FU")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # Refactor input names
        parameters = {
            name.replace(':', '__'): value[0] for name, value in inputs.items()
        }

        # Compute impacts from pre-compiled expressions and current parameters values
        res = self.compute_impacts_from_lambdas(
            self.lambdas,
            **parameters
        )

        # Store results in outputs
        for m in res:  # for each LCIA method
            m_name = re.sub(r': |/| ', '_', m.split(' - ')[0])
            if self.axis_keys:  # results by phase/contributor
                s = 0
                for axis_key in self.axis_keys:
                    s_i = res[m][res.index.get_level_values(self.options['axis']) == axis_key].iloc[0]
                    outputs[LCA_CHARACTERIZATION_KEY + m_name + ':' + axis_key] = s_i
                    s += s_i
                outputs[LCA_CHARACTERIZATION_KEY + m_name] = s
            else:
                outputs[LCA_CHARACTERIZATION_KEY + m_name] = res[m].iloc[0]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # Refactor input names
        parameters = {
            name.replace(':', '__'): value[0] for name, value in inputs.items()
        }

        # Compute partials from pre-compiled expressions and current parameters values
        res = {param_name: self.compute_impacts_from_lambdas(partial_lambdas, **parameters) for param_name, partial_lambdas in
               self.partial_lambdas_dict.items()}

        # Compute partials
        for param_name, res_param in res.items():
            for m in res_param:
                m_name = re.sub(r': |/| ', '_', m.split(' - ')[0])
                input_name = param_name.replace('__', ':')
                if self.axis_keys:  # results by phase/contributor
                    s = 0
                    for axis_key in self.axis_keys:
                        s_i = res_param[m][res_param.index.get_level_values(self.options['axis']) == axis_key].iloc[0]
                        partials[LCA_CHARACTERIZATION_KEY + m_name + ':' + axis_key, input_name] = s_i
                        s += s_i
                    partials[LCA_CHARACTERIZATION_KEY + m_name, input_name] = s
                else:
                    partials[LCA_CHARACTERIZATION_KEY + m_name, input_name] = res_param[m].iloc[0]

    def compute_impacts_from_lambdas(
        self,
        lambdas,
        **params: Dict[str, agb.SingleOrMultipleFloat],
    ):
        """
        Modified version of compute_impacts from lca_algebraic.
        More like a wrapper of _postLCAAlgebraic, to avoid calling _preLCAAlgebraic which is unecessarily
        time consuming when lambdas have already been calculated and doesn't have to be updated.
        """
        dfs = dict()

        dbname = self.model.key[0]
        with agb.DbContext(dbname):
            # Check no params are passed for FixedParams
            for key in params:
                if key in agb.params._fixed_params():
                    print("Param '%s' is marked as FIXED, but passed in parameters : ignored" % key)

            #lambdas = _preMultiLCAAlgebric(model, methods, alpha=alpha, axis=axis)  # <-- this is the time-consuming part

            df = agb.lca._postMultiLCAAlgebric(self.methods, lambdas, **params)

            model_name = agb.base_utils._actName(self.model)
            while model_name in dfs:
                model_name += "'"

            # param with several values
            list_params = {k: vals for k, vals in params.items() if isinstance(vals, list)}

            # Shapes the output / index according to the axis or multi param entry
            if self.options['axis']:
                df[self.options['axis']] = lambdas[0].axis_keys
                df = df.set_index(self.options['axis'])
                df.index.set_names([self.options['axis']])

                # Filter out line with zero output
                df = df.loc[
                    df.apply(
                        lambda row: not (row.name is None and row.values[0] == 0.0),
                        axis=1,
                    )
                ]

                # Rename "None" to others
                df = df.rename(index={None: "other"})

                # Sort index
                df.sort_index(inplace=True)

                # Add "total" line
                df.loc["*sum*"] = df.sum(numeric_only=True)

            elif len(list_params) > 0:
                for k, vals in list_params.items():
                    df[k] = vals
                df = df.set_index(list(list_params.keys()))

            else:
                # Single output ? => give the single row the name of the model activity
                df = df.rename(index={0: model_name})

            dfs[model_name] = df

        if len(dfs) == 1:
            df = list(dfs.values())[0]
        else:
            # Concat several dataframes for several models
            df = pd.concat(list(dfs.values()))

        return df


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
        self.model = agb.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

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
        new_params = {name: value for name, value in agb.params._completeParamValues(parameters).items()}
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

        res = agb.multiLCAAlgebric(
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
        with agb.params.DbContext(self.model):
            exprs_list, _ = agb.lca._modelToExpr(self.model, self.methods, extract_activities=extract_activities)
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
        agb.set_custom_impact_labels(dict_labels)

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
        self.model = agb.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

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
        self.model = agb.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

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
        self.model = agb.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

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


def _preMultiLCAAlgebricPartials(model, methods, alpha=1, axis=None):
    """
    Modified version of _preMultiLCAAlgebric from lca_algebraic
    to compute partial derivatives of impacts w.r.t. parameters instead of expressions of impacts.
    """
    with agb.DbContext(model):
        exprs = agb.lca._modelToExpr(model, methods, alpha=alpha, axis=axis)

        # Replace ceiling function by identity for better derivatives
        exprs = [expr.replace(sym.ceiling, lambda x: x) for expr in exprs]

        # Lambdify (compile) expressions
        if isinstance(exprs[0], AxisDict):
            return {
                param.name: [
                    agb.lca.LambdaWithParamNames(
                        AxisDict({axis_tag: res.diff(param) for axis_tag, res in expr.items()})) for expr in exprs
                ] for param in agb.all_params().values()
            }
        else:
            return {
                param.name: [
                    agb.lca.LambdaWithParamNames(expr.diff(param)) for expr in exprs
                ]
                for param in agb.all_params().values()
            }