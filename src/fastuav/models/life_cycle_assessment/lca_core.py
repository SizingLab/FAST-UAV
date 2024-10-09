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
from fastuav.constants import LCA_CHARACTERIZATION_KEY, LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY, LCA_FACTOR_KEY, LCA_SINGLE_SCORE_KEY


class LifeCycleAssessment(om.Group):
    """
    Group for LCA models and calculations.
    """

    def initialize(self):
        # Declare options
        self.options.declare("configuration_file", default=None, types=str)
        self.options.declare("axis", default=None, types=str)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)

    def setup(self):
        # Core LCA (model + characterisation)
        self.add_subsystem(
            "core",
            LCAcore(
                configuration_file=self.options["configuration_file"],
                axis=self.options["axis"]
            ),
            promotes=["*"]
        )

        # NORMALIZATION
        if self.options["weighting"] or self.options["normalization"]:
            self.add_subsystem("normalization", Normalization())

        # WEIGHTING AND AGGREGATION
        if self.options["weighting"]:
            self.add_subsystem("weighting", Weighting())
            self.add_subsystem("aggregation", Aggregation())

    def configure(self):
        """
        Configure() method is called after setup() of the containing group to get access to `model` metadata and
        set the appropriate inputs/outputs of the normalization and weighting modules.
        """
        if self.options["weighting"] or self.options["normalization"]:
            methods = []
            axis_keys = []
            for var_in in self.core.list_outputs(return_format='dict', out_stream=None).keys():
                var_out_norm = var_in.replace(LCA_CHARACTERIZATION_KEY, LCA_NORMALIZATION_KEY)
                m_name = var_in.split(":")[2]  # get method name (not very generic, but works for now)
                norm_factor = LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY

                # Add outputs from core LCIA as inputs to normalization
                self.normalization.add_input(var_in, val=np.nan, units=None)
                self.normalization.add_output(var_out_norm, units=None)
                self.normalization.declare_partials(var_out_norm, var_in, method="exact")

                # Add normalization factor as input
                if m_name not in methods:
                    self.normalization.add_input(norm_factor, val=np.nan, units=None)
                    self.normalization.declare_partials(var_out_norm, norm_factor, method="exact")

                if self.options["weighting"]:
                    var_out_weight = var_out_norm.replace(LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY)
                    weight_factor = LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY

                    # Add outputs from normalization as inputs to weighting
                    self.weighting.add_input(var_out_norm, val=np.nan, units=None)
                    self.weighting.add_output(var_out_weight, units=None)
                    self.weighting.declare_partials(var_out_weight, var_out_norm, method="exact")

                    # Add weighting factor as input
                    if m_name not in methods:
                        self.weighting.add_input(weight_factor, val=1.0, units=None)
                        self.weighting.declare_partials(var_out_weight, weight_factor, method="exact")

                    # Add outputs from weighting as inputs to aggregation (single score)
                    self.aggregation.add_input(var_out_weight, val=np.nan, units=None)

                    # get axis key (not very generic, but works for now)
                    axis_key = ":" + ":".join(var_in.split(":")[3:]) if len(var_in.split(":")) > 3 else ""
                    var_single_score = LCA_SINGLE_SCORE_KEY + axis_key
                    if axis_key not in axis_keys:
                        self.aggregation.add_output(var_single_score, val=np.nan, units=None)
                        axis_keys.append(axis_key)
                    self.aggregation.declare_partials(var_single_score, var_out_weight, val=1.0)

                if m_name not in methods:
                    methods.append(m_name)

            # Promote variables
            self.promotes("normalization", any=['*'])
            if self.options["weighting"]:
                self.promotes("weighting", any=['*'])
                self.promotes("aggregation", any=['*'])


class LCAcore(om.ExplicitComponent):
    """
    This OpenMDAO component implements an LCA model using brightway2 and lca_algebraic libraries.
    It creates an LCA model from a configuration file, then compiles functions for the impacts and partial derivatives.
    The parametric functions (lambdas) are used to compute the impacts and partial derivatives of the impacts w.r.t.
    parameters in a very fast way compared to conventional LCA.
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
        # TODO: enable multiple axes to be declared, e.g. to ventilate impacts by phase and component.

        # Compile expressions for partial derivatives of impacts w.r.t. parameters
        self.partial_lambdas_dict = _preMultiLCAAlgebricPartials(self.model, self.methods, axis=self.options['axis'])

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
        # TODO: distinguish between constant and variable parameters to avoid unnecessary partials calculations

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


class Normalization(om.ExplicitComponent):
    """
    Normalization of the LCIA results.
    """

    def compute(self, inputs, outputs):
        for var_in in inputs:
            if LCA_FACTOR_KEY not in var_in:
                var_out = var_in.replace(LCA_CHARACTERIZATION_KEY, LCA_NORMALIZATION_KEY)
                m_name = var_in.split(":")[2]  # Not very generic, but works for now
                norm_factor = LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY
                outputs[var_out] = inputs[var_in] / inputs[norm_factor]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for var_in in inputs:
            if LCA_FACTOR_KEY not in var_in:
                var_out = var_in.replace(LCA_CHARACTERIZATION_KEY, LCA_NORMALIZATION_KEY)
                m_name = var_in.split(":")[2]  # Not very generic, but works for now
                norm_factor = LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY
                partials[var_out, var_in] = 1.0 / inputs[norm_factor]
                partials[var_out, norm_factor] = -inputs[var_in] / inputs[norm_factor] ** 2


class Weighting(om.ExplicitComponent):
    """
    Weighting of the normalised LCIA results.
    """

    def compute(self, inputs, outputs):
        for var_in in inputs:
            if LCA_FACTOR_KEY not in var_in:
                var_out = var_in.replace(LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY)
                m_name = var_in.split(":")[2]  # Not very generic, but works for now
                weight_factor = LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY
                outputs[var_out] = inputs[var_in] / inputs[weight_factor]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for var_in in inputs:
            if LCA_FACTOR_KEY not in var_in:
                var_out = var_in.replace(LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY)
                m_name = var_in.split(":")[2]  # Not very generic, but works for now
                weight_factor = LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY
                partials[var_out, var_in] = inputs[weight_factor]
                partials[var_out, weight_factor] = inputs[var_in]


class Aggregation(om.ExplicitComponent):
    """
    Aggregation of the weighted LCIA results into a single score.
    """

    def compute(self, inputs, outputs):
        axis_keys = []
        for var_in in inputs:
            axis_key = ":" + ":".join(var_in.split(":")[3:]) if len(var_in.split(":")) > 3 else ""
            if axis_key not in axis_keys:
                var_out = LCA_SINGLE_SCORE_KEY + axis_key
                if axis_key == "":  # single score for entire system
                    outputs[var_out] = sum(inputs[var_in] for var_in in inputs if len(var_in.split(":")) == 3)
                else:  # single scores by phase/contributor
                    outputs[var_out] = sum(inputs[var_axis] for var_axis in inputs if axis_key in var_axis)


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