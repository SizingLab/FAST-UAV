"""
Non-regression test of the overall OAD process for the three UAV architectures.

For each architecture (multirotor, fixed-wing, hybrid VTOL) the full multidisciplinary
design optimization is run from the reference specification, and a set of characteristic
outputs (MTOW, propulsion and geometry sizing) is compared against committed reference
values. As in FAST-OAD-CS25's ``test_oad_process``, the goal is to detect any unintended
change in the design *results* introduced by a model or formulation change.

The convergence and feasibility of the optimization itself are covered by
``test_convergence_specs``; here we additionally pin the numerical results. The
configuration files, reference input files and the isolated-workdir machinery are shared
with that module.
"""

import shutil
import tempfile

import fastoad.api as oad
import pytest

from fastuav.tests.test_convergence_specs import ARCHITECTURES, _make_configuration

# Relative tolerance on the optimized outputs. Tight enough to catch a real model
# regression, loose enough to absorb optimizer/platform numerical noise: the SLSQP path
# is reproducible to ~1e-9 on a given machine but can drift slightly across scipy/BLAS
# versions. Tighten it (or switch to per-variable absolute tolerances) if the design
# point proves perfectly reproducible in CI.
REL_TOL = 1.0e-2

# Reference optimized outputs, generated with ``oad.optimize_problem`` on the reference
# specifications (default model units, as returned by ``problem.get_val``). Update these
# deliberately -- and only -- when a model change is expected to move the design point.
REFERENCE = {
    "multirotor": {
        "data:weight:mtow": 15.423052,
        "data:propulsion:multirotor:propeller:diameter": 0.488729,
        "data:propulsion:multirotor:battery:energy": 2626.90,
    },
    "fixedwing": {
        "data:weight:mtow": 15.000000,
        "data:propulsion:fixedwing:propeller:diameter": 0.591730,
        "data:propulsion:fixedwing:battery:energy": 3015.55,
        "data:geometry:wing:surface": 1.299489,
        "data:geometry:wing:span": 4.409024,
    },
    "hybrid": {
        "data:weight:mtow": 15.000000,
        "data:propulsion:multirotor:propeller:diameter": 0.479527,
        "data:propulsion:multirotor:battery:energy": 370.432,
        "data:propulsion:fixedwing:propeller:diameter": 0.660432,
        "data:propulsion:fixedwing:battery:energy": 2108.59,
        "data:geometry:wing:surface": 1.201450,
        "data:geometry:wing:span": 4.258786,
    },
}

# Guard against drift between the two modules: every pinned architecture must be a known
# one with a configuration + reference input file.
assert set(REFERENCE).issubset(ARCHITECTURES)


@pytest.mark.parametrize("architecture", list(REFERENCE))
def test_oad_process(architecture):
    """Run the full MDO and check the optimized outputs against the references."""
    workdir = tempfile.mkdtemp(prefix=f"fastuav_oad_{architecture}_")
    try:
        conf = _make_configuration(architecture, {}, workdir)
        problem = oad.optimize_problem(conf, overwrite=True)

        for name, expected in REFERENCE[architecture].items():
            value = float(problem.get_val(name).ravel()[0])
            assert value == pytest.approx(expected, rel=REL_TOL), (
                f"{architecture}: {name} = {value!r} (reference {expected!r})"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
