"""
Convergence / model-integrity tests across several specifications (cahiers des charges).

The goal of this module is to exercise the FAST-UAV models over a range of design
requirements and architectures (multirotor, fixed-wing, hybrid VTOL) in order to
surface model and formulation errors (NaN/inf, non-physical values, non-convergence
of the design optimization).

Two complementary checks are performed:

* ``test_model_integrity`` -- a single evaluation of the whole model for each
  architecture and each specification. It catches formulation errors that show up
  immediately (NaN/inf outputs, non-physical MTOW).

* ``test_mdo_convergence`` -- the full multidisciplinary design optimization is run
  with the standard ``oad.optimize_problem`` (SLSQP) for each architecture and each
  specification, and the design must converge to a feasible optimum (all constraints
  satisfied to a tight tolerance).

  Convergence relies on the OpenMDAO solver stack declared in the configuration files
  (``nonlinear_solver: om.NonlinearBlockGS`` + ``linear_solver: om.DirectSolver``):
  the disciplines are coupled (the sizing depends on the MTOW guess that depends on the
  sizing), so every evaluation must be converged for the function -- and therefore the
  analytic total derivatives -- to be consistent. Without the solver each evaluation is
  a single pass that lags by one iteration, the gradients are inconsistent and SLSQP
  stalls ("positive directional derivative").
"""

import os
import shutil
import tempfile
from pathlib import Path

import fastoad.api as oad
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]  # .../src/fastuav
CONF_DIR = PKG_ROOT / "configurations"
SRC_DIR = PKG_ROOT / "notebooks" / "data" / "source_files"

# Architecture -> (configuration file, reference source file)
ARCHITECTURES = {
    "multirotor": (CONF_DIR / "multirotor_mdo.yaml", SRC_DIR / "problem_inputs_DJI_M600.xml"),
    "fixedwing": (CONF_DIR / "fixedwing_mdo.yaml", SRC_DIR / "problem_inputs_FW.xml"),
    "hybrid": (CONF_DIR / "hybrid_mdo.yaml", SRC_DIR / "problem_inputs_hybrid.xml"),
}

# Cahiers des charges: a small set of representative specifications.
# Each entry is {variable_name: (value, units)}.
SPECS = {
    "nominal": {},
    "light_payload": {"mission:sizing:payload:mass": (2.0, "kg")},
    "heavy_payload": {"mission:sizing:payload:mass": (5.0, "kg")},
}

CONSTRAINT_TOL = 1e-4  # max allowed constraint violation at the converged optimum


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_configuration(architecture, overrides, workdir):
    """Write an isolated configuration + input file in ``workdir``.

    A copy of the configuration is written with absolute input/output and mission
    paths so the test never touches the repository working directory. ``overrides``
    (the cahier des charges) are applied to the generated input file.
    Returns the path to the temporary configuration file.
    """
    import yaml

    conf_path, src_path = ARCHITECTURES[architecture]
    conf = yaml.safe_load(conf_path.read_text())

    conf["input_file"] = os.path.join(workdir, "inputs.xml")
    conf["output_file"] = os.path.join(workdir, "outputs.xml")
    mission_rel = conf["model"]["performance"]["missions"]["file_path"]
    conf["model"]["performance"]["missions"]["file_path"] = str(
        (conf_path.parent / mission_rel).resolve()
    )

    new_conf = os.path.join(workdir, "configuration.yaml")
    with open(new_conf, "w") as f:
        yaml.safe_dump(conf, f)

    oad.generate_inputs(new_conf, str(src_path), overwrite=True)

    if overrides:
        data = oad.DataFile(conf["input_file"])
        for name, (value, units) in overrides.items():
            data[name].value = value
            if units is not None:
                data[name].units = units
        data.save()

    return new_conf


def _max_constraint_violation(problem):
    """Largest violation over all optimization constraints (0 if all satisfied)."""
    violation = 0.0
    worst = ""
    for name, meta in problem.driver._cons.items():
        value = float(problem.get_val(name).ravel()[0])
        lower, upper = meta.get("lower"), meta.get("upper")
        local = 0.0
        if lower is not None and np.isfinite(lower):
            local = max(local, lower - value)
        if upper is not None and np.isfinite(upper):
            local = max(local, value - upper)
        if local > violation:
            violation, worst = local, name
    return violation, worst


def _non_finite_outputs(problem):
    """Return the list of model output names holding a NaN or inf value."""
    bad = []
    for name, meta in problem.model.list_outputs(out_stream=None, val=True):
        if np.any(~np.isfinite(np.atleast_1d(meta["val"]))):
            bad.append(name)
    return bad


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("architecture", list(ARCHITECTURES))
@pytest.mark.parametrize("spec_name", list(SPECS))
def test_model_integrity(architecture, spec_name):
    """A single model evaluation must not produce NaN/inf or non-physical mass."""
    workdir = tempfile.mkdtemp(prefix=f"fastuav_{architecture}_{spec_name}_")
    try:
        conf = _make_configuration(architecture, SPECS[spec_name], workdir)
        problem = oad.evaluate_problem(conf, overwrite=True)

        bad = _non_finite_outputs(problem)
        assert not bad, f"{architecture}/{spec_name}: non-finite outputs: {bad[:15]}"

        mtow = float(problem.get_val("data:weight:mtow", units="kg")[0])
        assert np.isfinite(mtow) and mtow > 0.0, f"unphysical MTOW: {mtow}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.parametrize("architecture", list(ARCHITECTURES))
@pytest.mark.parametrize("spec_name", list(SPECS))
def test_mdo_convergence(architecture, spec_name):
    """The full design optimization must converge to a feasible optimum."""
    workdir = tempfile.mkdtemp(prefix=f"fastuav_mdo_{architecture}_{spec_name}_")
    try:
        conf = _make_configuration(architecture, SPECS[spec_name], workdir)
        problem = oad.optimize_problem(conf, overwrite=True)

        violation, worst = _max_constraint_violation(problem)
        assert violation < CONSTRAINT_TOL, (
            f"{architecture}/{spec_name}: optimization not feasible "
            f"(max violation {violation:.2e} on {worst})"
        )

        mtow = float(problem.get_val("data:weight:mtow", units="kg")[0])
        assert np.isfinite(mtow) and 0.0 < mtow < 1.0e4, f"unphysical MTOW: {mtow}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
