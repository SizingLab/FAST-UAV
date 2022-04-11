"""
Driver that uses Covariance Matrix Adaptation Evolution Strategy (CMAES).
Based on pycma (https://github.com/CMA-ES/pycma)
Adapted from  OpenMDAO / RevHack2020 (https://github.com/OpenMDAO/RevHack2020/)
New in this version: use of the fmin_con method from pycma for handling constraints with an Augmented Lagrangian.
"""
import os
import copy
import time

import numpy as np

import openmdao
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.concurrent import concurrent_eval
from openmdao.utils.mpi import MPI
from openmdao.core.analysis_error import AnalysisError

import cma


class CMAESDriver(Driver):
    """
    Driver for a Covariance Matrix Adaptation Evolution Strategy (CMAES).
    This algorithm requires that inputs are floating point numbers.
    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _desvar_idx : dict
        Keeps track of the indices for each desvar.
    CMAOptions : CMAOptions
        Options for CMAES execution.
    _cmaes : <CMAES>
        CMAES object.
    _randomstate : np.random.RandomState, int
        Random state (or seed-number) which controls the seed.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CMAESDriver driver.
        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super().__init__(**kwargs)

        # What we support
        self.supports["inequality_constraints"] = True
        self.supports["equality_constraints"] = True
        self.supports["multiple_objectives"] = True

        # What we don't support yet
        self.supports["integer_design_vars"] = False
        self.supports["two_sided_constraints"] = False
        self.supports["linear_constraints"] = False
        self.supports["simultaneous_derivatives"] = False
        self.supports["active_set"] = False
        self.supports["distributed_design_vars"] = False
        self.supports._read_only = True

        self._desvar_idx = {}

        self.CMAOptions = cma.CMAOptions()

        # random state can be set for predictability during testing
        if "CMAESDriver_seed" in os.environ:
            self.CMAOptions["seed"] = int(os.environ["CMAESDriver_seed"])

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            "sigma0",
            default=0.1,
            types=float,
            desc="Initial standard deviation in each coordinate. "
            "sigma0 should be about 1/4th of the search domain width "
            "(where the optimum is to be expected).",
        )
        self.options.declare(
            "run_parallel",
            types=bool,
            default=False,
            desc="Set to True to execute the points in a generation in parallel.",
        )
        self.options.declare(
            "procs_per_model",
            default=1,
            lower=1,
            desc="Number of processors to give each model under MPI.",
        )
        self.options.declare(
            "penalty_parameter",
            default=10.0,
            lower=0.0,
            desc="Penalty function parameter.",
        )
        self.options.declare("penalty_exponent", default=1.0, desc="Penalty function exponent.")
        self.options.declare(
            "multi_obj_weights",
            default={},
            types=(dict),
            desc="Weights of objectives for multi-objective optimization."
            "Weights are specified as a dictionary with the absolute names "
            "of the objectives. The same weights for all objectives are "
            "assumed, if not given.",
        )
        self.options.declare(
            "multi_obj_exponent",
            default=1.0,
            lower=0.0,
            desc="Multi-objective weighting exponent.",
        )

        # Felix Pollet 01/2022
        self.options.declare(
            "augmented_lagrangian",
            types=bool,
            default=False,
            desc="Set to True to use augmented lagrangian to handle constraint. "
            " Else a penalty method will be used.",
        )
        self.options.declare(
            "restarts",
            types=int,
            default=0,
            lower=0,
            desc="number of restarts with increasing population size",
        )
        self.options.declare(
            "restart_from_best",
            types=bool,
            default=False,
            desc="which point to restart from",
        )
        self.options.declare(
            "bipop",
            types=bool,
            default=False,
            desc="if `True`, run as BIPOP-CMA-ES; BIPOP is a special restart"
            "strategy switching between two population sizings - small"
            "(like the default CMA, but with more focused search) and"
            "large (progressively increased as in IPOP). This makes the"
            "algorithm perform well both on functions with many regularly"
            "or irregularly arranged local optima (the latter by frequently"
            "restarting with small populations).",
        )

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.
        This is the final thing to run during setup.
        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        model_mpi = None
        comm = problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options["run_parallel"]:
            comm = None

        aug_lagrangian = self.options["augmented_lagrangian"]
        restarts = self.options["restarts"]
        restart_from_best = self.options["restart_from_best"]
        bipop = self.options["bipop"]

        if aug_lagrangian:
            self._cmaes = CMAES(
                objfun=self.objective_callback,
                gfun=self.g_constraints_callback,
                hfun=self.h_constraints_callback,
                aug_lagrangian=aug_lagrangian,
                restarts=restarts,
                restart_from_best=restart_from_best,
                bipop=bipop,
                comm=comm,
                model_mpi=model_mpi,
            )
        else:
            self._cmaes = CMAES(
                objfun=self.objective_penalty_callback,
                restarts=restarts,
                restart_from_best=restart_from_best,
                bipop=bipop,
                comm=comm,
                model_mpi=model_mpi,
            )

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.
        Here, we generate the model communicators.
        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.
        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        procs_per_model = self.options["procs_per_model"]
        if MPI and self.options["run_parallel"]:

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError(
                    "The total number of processors is not evenly divisible "
                    "by the specified number of processors per model.\n "
                    "Provide a number of processors that is a multiple of %d, "
                    "or specify a number of processors per model that divides "
                    "into %d." % (procs_per_model, full_size)
                )
            color = comm.rank % size
            model_comm = comm.Split(color)

            # Everything we need to figure out which case to run.
            self._concurrent_pop_size = size
            self._concurrent_color = color

            return model_comm

        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        return comm

    def _get_name(self):
        """
        Get name of current Driver.
        Returns
        -------
        str
            Name of current Driver.
        """
        return "CMAES"

    def run(self):
        """
        Execute the CMA-ES algorithm.
        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        self._check_for_missing_objective()

        # Size design variables.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()

        count = 0
        for name, meta in desvars.items():
            size = meta["size"]
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count,))
        upper_bound = np.empty((count,))
        x0 = np.empty(count)

        # Figure out bounds vectors and initial design vars
        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta["lower"]
            upper_bound[i:j] = meta["upper"]
            x0[i:j] = desvar_vals[name]

        self.CMAOptions["bounds"] = [lower_bound, upper_bound]

        desvar_new, obj = self._cmaes.execute(x0, self.options["sigma0"], self.CMAOptions)
        # desvar_new, obj = self._cmaes.execute(lambda: np.random.uniform(lower_bound, upper_bound), self.options['sigma0'], self.CMAOptions)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = desvar_new[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self._problem().model.run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_penalty_callback(self, x):
        r"""
        Evaluate problem objective at the requested point.
        In case of multi-objective optimization, a simple weighted sum method is used:
        .. math::
           f = (\sum_{k=1}^{N_f} w_k \cdot f_k)^a
        where :math:`N_f` is the number of objectives and :math:`a>0` is an exponential
        weight. Choosing :math:`a=1` is equivalent to the conventional weighted sum method.
        The weights given in the options are normalized, so:
        .. math::
            \sum_{k=1}^{N_f} w_k = 1
        If one of the objectives :math:`f_k` is not a scalar, its elements will have the same
        weights, and it will be normed with length of the vector.
        Takes into account constraints with a penalty function.
        All constraints are converted to the form of :math:`g_i(x) \leq 0` for
        inequality constraints and :math:`h_i(x) = 0` for equality constraints.
        The constraint vector for inequality constraints is the following:
        .. math::
           g = [g_1, g_2  \dots g_N], g_i \in R^{N_{g_i}}
           h = [h_1, h_2  \dots h_N], h_i \in R^{N_{h_i}}
        The number of all constraints:
        .. math::
           N_g = \sum_{i=1}^N N_{g_i},  N_h = \sum_{i=1}^N N_{h_i}
        The fitness function is constructed with the penalty parameter :math:`p`
        and the exponent :math:`\kappa`:
        .. math::
           \Phi(x) = f(x) + p \cdot \sum_{k=1}^{N^g}(\delta_k \cdot g_k)^{\kappa}
           + p \cdot \sum_{k=1}^{N^h}|h_k|^{\kappa}
        where :math:`\delta_k = 0` if :math:`g_k` is satisfied, 1 otherwise
        .. note::
            The values of :math:`\kappa` and :math:`p` can be defined as driver options.
        Parameters
        ----------
        x : ndarray
            Value of design variables.
        Returns
        -------
        float
            Objective value
        """
        model = self._problem().model

        objs = self.get_objective_values()
        nr_objectives = len(objs)

        # Single objective, if there is only one objective, which has only one element
        if nr_objectives > 1:
            is_single_objective = False
        else:
            for obj in objs.items():
                is_single_objective = len(obj) == 1
                break

        obj_exponent = self.options["multi_obj_exponent"]
        if self.options["multi_obj_weights"]:  # not empty
            obj_weights = self.options["multi_obj_weights"]
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1.0 for name in objs.keys()}
        sum_weights = sum(obj_weights.values())

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            obj_values = self.get_objective_values()
            if is_single_objective:  # Single objective optimization
                for i in obj_values.values():
                    obj = i  # First and only key in the dict
            else:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                for name, val in obj_values.items():
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = (
                            'Name "{}" in "multi_obj_weights" option '
                            "is not an absolute name of an objective."
                        )
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

                obj = sum(weighted_objectives / sum_weights) ** obj_exponent

            # Parameters of the penalty method
            penalty = self.options["penalty_parameter"]
            exponent = self.options["penalty_exponent"]

            if penalty == 0:
                fun = obj
            else:
                constraint_violations = np.array([])
                for name, val in self.get_constraint_values().items():
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    if (con["lower"] is not None) and np.any(con["lower"] > -almost_inf):
                        diff = val - con["lower"]
                        violation = np.array([0.0 if d >= 0 else abs(d) for d in diff])
                        constraint_violations = np.hstack((constraint_violations, violation))
                    if (con["upper"] is not None) and np.any(con["upper"] < almost_inf):
                        diff = val - con["upper"]
                        violation = np.array([0.0 if d <= 0 else abs(d) for d in diff])
                        constraint_violations = np.hstack((constraint_violations, violation))
                    if (con["equals"] is not None) and np.any(np.abs(con["equals"]) < almost_inf):
                        diff = val - con["equals"]
                        violation = np.absolute(diff)
                        constraint_violations = np.hstack((constraint_violations, violation))
                fun = obj + penalty * sum(np.power(constraint_violations, exponent))

            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        return fun

    def objective_callback(self, x):
        r"""
        Evaluate problem objective at the requested point.
        In case of multi-objective optimization, a simple weighted sum method is used:
        .. math::
           f = (\sum_{k=1}^{N_f} w_k \cdot f_k)^a
        where :math:`N_f` is the number of objectives and :math:`a>0` is an exponential
        weight. Choosing :math:`a=1` is equivalent to the conventional weighted sum method.
        The weights given in the options are normalized, so:
        .. math::
            \sum_{k=1}^{N_f} w_k = 1
        If one of the objectives :math:`f_k` is not a scalar, its elements will have the same
        weights, and it will be normed with length of the vector.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        Returns
        -------
        float
            Objective value
        """
        model = self._problem().model

        objs = self.get_objective_values()
        nr_objectives = len(objs)

        # Single objective, if there is only one objective, which has only one element
        if nr_objectives > 1:
            is_single_objective = False
        else:
            for obj in objs.items():
                is_single_objective = len(obj) == 1
                break

        obj_exponent = self.options["multi_obj_exponent"]
        if self.options["multi_obj_weights"]:  # not empty
            obj_weights = self.options["multi_obj_weights"]
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1.0 for name in objs.keys()}
        sum_weights = sum(obj_weights.values())

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            obj_values = self.get_objective_values()
            if is_single_objective:  # Single objective optimization
                for i in obj_values.values():
                    obj = i  # First and only key in the dict
            else:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                for name, val in obj_values.items():
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = (
                            'Name "{}" in "multi_obj_weights" option '
                            "is not an absolute name of an objective."
                        )
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

                obj = sum(weighted_objectives / sum_weights) ** obj_exponent

            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        return obj

    def g_constraints_callback(self, x):
        r"""
        Evaluate problem inequality constraints at the requested point.
        ----------
        x : ndarray
            Value of design variables.
        Returns
        -------
        ndarray
            Equality constraints values
        """
        model = self._problem().model

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            gfun = np.array([])
            for name, val in self.get_constraint_values().items():
                con = self._cons[name]
                if (con["lower"] is not None) and np.any(con["lower"] > -almost_inf):
                    diff = -(val - con["lower"])
                    violation = np.array([0.0 if d >= 0 else abs(d) for d in diff])
                    gfun = np.hstack((gfun, diff))
                if (con["upper"] is not None) and np.any(con["upper"] < almost_inf):
                    diff = val - con["upper"]
                    violation = np.array([0.0 if d <= 0 else abs(d) for d in diff])
                    gfun = np.hstack((gfun, diff))

            # Record after getting obj to assure they have
            # been gathered in MPI.
        # rec.abs = 0.0
        # rec.rel = 0.0

        return gfun

    def h_constraints_callback(self, x):
        r"""
        Evaluate problem inequality constraints at the requested point.
        ----------
        x : ndarray
            Value of design variables.
        Returns
        -------
        ndarray
            Equality constraints values
        """
        model = self._problem().model

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            hfun = np.array([])
            for name, val in self.get_constraint_values().items():
                con = self._cons[name]
                if (con["equals"] is not None) and np.any(np.abs(con["equals"]) < almost_inf):
                    diff = val - con["equals"]
                    violation = np.absolute(diff)
                    hfun = np.hstack((hfun, diff))

            # Record after getting obj to assure they have
            # been gathered in MPI.
        # rec.abs = 0.0
        # rec.rel = 0.0

        return hfun


class CMAES(object):
    """
    CMA Evolution Strategy.
    Attributes
    ----------
    comm : MPI communicator or None
        The MPI communicator that will be used objective evaluation for each generation.
    model_mpi : None or tuple
        If the model in objfun is also parallel, then this will contain a tuple with the the
        total number of population points to evaluate concurrently, and the color of the point
        to evaluate on this rank.
    objfun : function
        Objective function callback.
    """

    def __init__(
        self,
        objfun,
        gfun=[],
        hfun=[],
        aug_lagrangian=False,
        restarts=0,
        restart_from_best="False",
        bipop="False",
        comm=None,
        model_mpi=None,
    ):
        """
        Initialize CMA Evolution Strategy object.
        Parameters
        ----------
        objfun : function
            Objective callback function.
        gfun : functions
            Inequality Constraints callback functions.
        hfun : functions
            Equality Constraints callback functions.
        comm : MPI communicator or None
            The MPI communicator that will be used objective evaluation.
        model_mpi : None or tuple
            If the model in objfun is also parallel, then this will contain a tuple with the the
            total number of population points to evaluate concurrently, and the color of the point
            to evaluate on this rank.
        """
        self.comm = comm
        self.model_mpi = model_mpi
        self.objfun = objfun
        self.gfun = gfun
        self.hfun = hfun
        self.aug_lagrangian = aug_lagrangian
        self.restarts = restarts
        self.restart_from_best = restart_from_best
        self.bipop = bipop

    def execute(self, x0, sigma0, CMAOptions):
        """
        Execute the CMA Evolution Strategy.
        Parameters
        ----------
        x0 : ndarray
            Initial design values
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array.
        sigma0 : float
            Initial standard deviation in each coordinate.
        CMAOptions : CMAOptions
            Options for CMAES execution.
        Returns
        -------
        ndarray
            Best design point
        float
            Objective value at best design point.
        """
        comm = self.comm
        restarts = self.restarts
        restart_from_best = self.restart_from_best
        bipop = self.bipop

        if comm is None:
            # Running non-parallel, use functional interface

            if self.aug_lagrangian:
                xopt, es = cma.fmin_con(
                    self.objfun,
                    x0,
                    sigma0,
                    g=self.gfun,
                    h=self.hfun,
                    options=CMAOptions,
                    restarts=restarts,
                    restart_from_best=restart_from_best,
                    bipop=bipop,
                )
                fopt = es.best_feasible
            else:
                res = cma.fmin(
                    self.objfun,
                    x0,
                    sigma0,
                    options=CMAOptions,
                    restarts=restarts,
                    restart_from_best=restart_from_best,
                    bipop=bipop,
                )
                xopt = res[0]
                fopt = res[1]

            return xopt, fopt

        else:
            # Running parallel, use OO interface

            # make sure all procs have the same seed
            seed = CMAOptions["seed"]
            if comm.rank == 0 and (not isinstance(seed, int) or seed == 0):
                seed = int(time.time())
            CMAOptions["seed"] = comm.bcast(seed, root=0)

            optim = cma.CMAEvolutionStrategy(x0, sigma0, CMAOptions)

            stop = False

            while not stop:  # optim.stop():
                # get candidate solutions
                X = optim.ask()

                # pad candidates to make them divisible into procs.
                cases = [((item,), None) for ii, item in enumerate(X)]
                extra = len(cases) % comm.size
                if extra > 0:
                    for j in range(comm.size - extra):
                        cases.append(cases[-1])

                # evaluate candidate solutions concurrently
                results = concurrent_eval(
                    self.objfun, cases, comm, allgather=True, model_mpi=self.model_mpi
                )

                # assemble solutions corresponding to X
                f = []
                for i in range(len(X)):
                    returns, traceback = results[i]
                    if returns:
                        f.append(returns)
                    else:
                        # Print the traceback if it fails
                        print("A case failed:")
                        print(traceback)

                # do the "update", pass f-values and prepare for next iteration
                optim.tell(X, f)
                optim.disp(20)  # display info every 20th iteration
                optim.logger.add()  # log another "data line", non-standard

                # gather stop conditions, stop if any proc stops
                # (all procs should stop with same stop condition)
                stops = comm.allgather(optim.stop())
                for proc_stop in stops:
                    if len(proc_stop) > 0:
                        stop = True

            # final output
            # print('termination by', stops)
            # print('best f-value =', optim.result[1])
            # print('best solution =', optim.result[0])
            # optim.logger.plot()  # if matplotlib is available

            return optim.result[0], optim.result[1]
