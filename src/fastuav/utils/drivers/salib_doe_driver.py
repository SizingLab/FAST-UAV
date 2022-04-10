"""
Driver for running model on design of experiments cases using Salib sampling methods.
Original driver was implemented by Onera (openmdao_extensions library).
This version offers the capability of generating alternate sampling distributions (i.e. other than uniform).
"""
import numpy as np

from openmdao.api import DOEDriver, OptionsDictionary
from openmdao.drivers.doe_generators import DOEGenerator

SALIB_NOT_INSTALLED = False
try:
    from SALib.sample import morris as ms
    from SALib.sample import saltelli
except ImportError:
    SALIB_NOT_INSTALLED = True


class SalibDOEGenerator(DOEGenerator):
    def __init__(self, dist=None):
        if SALIB_NOT_INSTALLED:
            raise RuntimeError(
                "SALib library is not installed. \
                                cf. https://salib.readthedocs.io/en/latest/getting-started.html"
            )
        self.distribution = dist
        self._cases = np.array([])
        self._pb = None
        self.called = False

    def __call__(self, design_vars, model=None):
        bounds = []
        names = []
        for name, meta in design_vars.items():
            size = meta["size"]
            meta_low = meta["lower"]
            meta_high = meta["upper"]
            for j in range(size):
                if isinstance(meta_low, np.ndarray):
                    p_low = meta_low[j]
                else:
                    p_low = meta_low

                if isinstance(meta_high, np.ndarray):
                    p_high = meta_high[j]
                else:
                    p_high = meta_high

                display_name = name.split(".")[-1]
                if size > 1:
                    display_name += str(j)
                names.append(display_name)
                bounds.append((p_low, p_high))

        if self.distribution is None:
            self._pb = {
                "num_vars": len(names),
                "names": names,
                "bounds": bounds,
                "groups": None,
            }

        else:
            size = 0
            for name, meta in design_vars.items():
                size += meta["size"]
            if (size) != (len(self.distribution)):
                raise RuntimeError(
                    "Bad distribution dimension: should be equal to design variables size {} "
                    ", got {}".format(size, len(self.distribution))
                )
            self._pb = {
                "num_vars": len(names),
                "names": names,
                "bounds": bounds,
                "groups": None,
                "dists": self.distribution,
            }

        self._compute_cases()
        self.called = True
        sample = []
        for i in range(self._cases.shape[0]):
            j = 0
            for name, meta in design_vars.items():
                size = meta["size"]
                sample.append((name, self._cases[i, j : j + size]))
                j += size
            yield sample

    def _compute_cases(self):
        raise RuntimeError("Have to be implemented in subclass.")

    def get_cases(self):
        if not self.called:
            raise RuntimeError("Have to run the driver before getting cases")
        return self._cases

    def get_salib_problem(self):
        if not self.called:
            raise RuntimeError(
                "Have to run the driver before getting the SALib problem"
            )
        return self._pb


class SalibMorrisDOEGenerator(SalibDOEGenerator):
    def __init__(self, n_trajs=2, n_levels=4, dist=None):
        super(SalibMorrisDOEGenerator, self).__init__(dist)
        # number of trajectories to apply morris method
        self.n_trajs = n_trajs
        # number of grid levels
        self.n_levels = n_levels

    def _compute_cases(self):
        self._cases = ms.sample(self._pb, self.n_trajs, self.n_levels)


class SalibSobolDOEGenerator(SalibDOEGenerator):
    def __init__(self, n_samples=1000, calc_second_order=True, dist=None):
        super(SalibSobolDOEGenerator, self).__init__(dist)
        # number of samples to generate
        self.n_samples = n_samples
        # whether calculing second order indices
        self.calc_second_order = calc_second_order

    def _compute_cases(self):
        self._cases = saltelli.sample(self._pb, self.n_samples, self.calc_second_order)


class SalibDOEDriver(DOEDriver):
    """
    Baseclass for SALib design-of-experiments Drivers
    """

    def __init__(self, **kwargs):
        super(SalibDOEDriver, self).__init__()

        if SALIB_NOT_INSTALLED:
            raise RuntimeError(
                "SALib library is not installed. \
                                cf. https://salib.readthedocs.io/en/latest/getting-started.html"
            )
        self.options.declare(
            "distributions",
            types=list,
            default=None,
            allow_none=True,
            desc="Distributions of uncertain variables",
        )
        self.options.declare(
            "sa_method_name",
            default="Morris",
            values=["Morris", "Sobol"],
            desc="either Morris or Sobol",
        )
        self.options.declare(
            "sa_doe_options",
            types=dict,
            default={},
            desc="options for given SMT sensitivity analysis method",
        )
        self.options.update(kwargs)

        self.sa_settings = OptionsDictionary()
        if self.options["sa_method_name"] == "Morris":
            self.sa_settings.declare(
                "n_trajs",
                types=int,
                default=2,
                desc="number of trajectories to apply morris method",
            )
            self.sa_settings.declare(
                "n_levels", types=int, default=4, desc="number of grid levels"
            )
            self.sa_settings.update(self.options["sa_doe_options"])
            n_trajs = self.sa_settings["n_trajs"]
            n_levels = self.sa_settings["n_levels"]
            dist = self.options["distributions"]
            self.options["generator"] = SalibMorrisDOEGenerator(n_trajs, n_levels, dist)
        elif self.options["sa_method_name"] == "Sobol":
            self.sa_settings.declare(
                "n_samples",
                types=int,
                default=500,
                desc="number of samples to generate",
            )
            self.sa_settings.declare(
                "calc_second_order",
                types=bool,
                default=True,
                desc="calculate second-order sensitivities ",
            )
            self.sa_settings.update(self.options["sa_doe_options"])
            n_samples = self.sa_settings["n_samples"]
            calc_snd = self.sa_settings["calc_second_order"]
            dist = self.options["distributions"]
            self.options["generator"] = SalibSobolDOEGenerator(
                n_samples, calc_snd, dist
            )
        else:
            raise RuntimeError(
                "Bad sensitivity analysis method name '{}'".format(
                    self.options["sa_method_name"]
                )
            )

    def _set_name(self):
        self._name = "SALib_DOE_" + self.options["sa_method_name"]

    def get_cases(self):
        return self.options["generator"].get_cases()

    def get_salib_problem(self):
        return self.options["generator"].get_salib_problem()
