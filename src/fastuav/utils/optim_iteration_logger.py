"""
Optimization iteration logger for FAST-UAV fixed-wing MDO.

Logs all design variables, constraints, and the objective at every major
iteration of the optimizer into a CSV file.

This is a thin wrapper around ``fastoad.api.optimize_problem`` that attaches
an OpenMDAO ``SqliteRecorder`` to the driver before running it.

Usage
-----
Replace this in your notebook::

    optim_problem = oad.optimize_problem(CONFIGURATION_FILE, overwrite=True)

with::

    from optim_iteration_logger import (
        optimize_problem_with_logging,
        plot_iteration_log,
        plot_iteration_log_interactive,
        request_stop,
    )

    optim_problem, log_df = optimize_problem_with_logging(
        CONFIGURATION_FILE,
        log_path="./workdir/optim_log.csv",
        overwrite=True,
    )

    # Static plot (matplotlib):
    fig = plot_iteration_log("./workdir/optim_log.csv")

    # Interactive plot (plotly) - click legend entries to toggle,
    # double-click to isolate a single series, drag to zoom:
    fig = plot_iteration_log_interactive("./workdir/optim_log.csv")
    fig  # returns the figure; in Jupyter it renders inline

**Graceful stopping (Jupyter)**::

    While the optimiser is running in one cell, run this in another cell:

        from optim_iteration_logger import request_stop
        request_stop()

    The optimiser will finish the current iteration, save the XML and CSV,
    then stop. The output XML always reflects the last completed iteration.
"""

import os
import os.path as pth
import time
from typing import Dict, List, Tuple

import numpy as np
import openmdao.api as om
import pandas as pd
import yaml

# OpenMDAO renamed the recorder base class across versions:
#   ≤ 3.27:  openmdao.recorders.base_recorder.BaseRecorder
#   ≥ 3.28:  openmdao.recorders.case_recorder.CaseRecorder
#   Some versions expose om.BaseRecorder, others don't.
# We try each path and use whichever exists.
_RecorderBase = None
for _candidate in (
    lambda: om.BaseRecorder,
    lambda: __import__("openmdao.recorders.base_recorder", fromlist=["BaseRecorder"]).BaseRecorder,
    lambda: __import__("openmdao.recorders.case_recorder", fromlist=["CaseRecorder"]).CaseRecorder,
):
    try:
        _RecorderBase = _candidate()
        break
    except (AttributeError, ImportError, ModuleNotFoundError):
        continue
if _RecorderBase is None:
    # Last resort: fall back to SqliteRecorder's parent (whatever it is)
    _RecorderBase = om.SqliteRecorder.__bases__[0]

from fastoad.io.configuration import FASTOADProblemConfigurator  # noqa: E402

# ---------------------------------------------------------------------------
# Custom recorder: writes the FAST-OAD output XML after every driver iteration
# ---------------------------------------------------------------------------
_STOP_FILE_DEFAULT = "./workdir/STOP"


def request_stop(stop_file: str = _STOP_FILE_DEFAULT):
    """Create a stop file that tells a running optimisation to halt gracefully.

    Call this from a **separate Jupyter cell** while the optimiser is running::

        # Cell 1 (running):
        problem, df = optimize_problem_with_logging(...)

        # Cell 2 (run this to stop):
        from optim_iteration_logger import request_stop
        request_stop()

    The optimiser will finish the current iteration, write the output XML
    and CSV, then stop. Much more reliable than Jupyter's "Interrupt Kernel"
    button, which can get swallowed by C extensions.
    """
    stop_path = pth.abspath(stop_file)
    os.makedirs(pth.dirname(stop_path), exist_ok=True)
    with open(stop_path, "w") as f:
        f.write("stop requested\n")
    print(f"[stop] Created {stop_path} — optimisation will halt after the current iteration.")


class _SnapshotXmlRecorder(_RecorderBase):
    """Lightweight recorder that overwrites the output XML after each iteration.

    Attaches alongside the SqliteRecorder. After every driver iteration:
    1. Calls ``problem.write_outputs()`` so the XML file always reflects the
       latest evaluated state.
    2. Checks for a **stop file** (default: ``./workdir/STOP``). If present,
       raises ``KeyboardInterrupt`` to halt the optimiser gracefully. This is
       the recommended way to stop a run from Jupyter — call
       ``request_stop()`` from a separate cell.
    """

    def __init__(self, problem, output_path, stop_file=_STOP_FILE_DEFAULT, verbose=True):
        super().__init__()
        self._problem = problem
        self._output_path = output_path
        self._stop_file = pth.abspath(stop_file)
        self._verbose = verbose
        self._iter_count = 0

    # --- required overrides for BaseRecorder ---
    def startup(self, recording_requester, comm=None):
        super().startup(recording_requester, comm)

    def record_iteration_driver(self, recording_requester, data, metadata):
        """Called by the driver after each major iteration."""
        self._iter_count += 1

        # 1. Write the output XML
        try:
            self._problem.write_outputs()
            if self._verbose and (self._iter_count == 1 or self._iter_count % 10 == 0):
                print(f"[snapshot] iter {self._iter_count}: wrote {self._output_path}")
        except Exception as exc:
            if self._verbose and self._iter_count <= 3:
                print(f"[snapshot] iter {self._iter_count}: write_outputs failed: {exc!r}")

        # 2. Check for stop file
        if pth.isfile(self._stop_file):
            try:
                os.remove(self._stop_file)
            except OSError:
                pass
            if self._verbose:
                print(
                    f"\n[snapshot] iter {self._iter_count}: "
                    f"STOP file detected — halting optimisation"
                )
                print(f"[snapshot]   XML saved to: {self._output_path}")
            raise KeyboardInterrupt(
                f"Stop requested via {self._stop_file} at iteration {self._iter_count}"
            )

    def record_iteration_system(self, *args, **kwargs):
        pass  # not needed

    def record_iteration_solver(self, *args, **kwargs):
        pass  # not needed

    def record_metadata_system(self, *args, **kwargs):
        pass

    def record_metadata_solver(self, *args, **kwargs):
        pass

    def record_viewer_data(self, *args, **kwargs):
        pass

    def shutdown(self):
        if self._verbose:
            print(f"[snapshot] {self._iter_count} XML snapshots written to {self._output_path}")
        super().shutdown()


# ---------------------------------------------------------------------------
# Helper: parse the YAML to extract variable names and bounds automatically
# ---------------------------------------------------------------------------
def parse_optim_vars(
    configuration_file: str,
) -> Dict[str, List[Dict]]:
    """Read a FAST-OAD configuration YAML and return the optimisation section.

    Returns a dict with three keys: ``design_variables``, ``constraints``,
    ``objective``. Each value is a list of dicts with at minimum a ``name``
    field, plus any of ``lower``, ``upper``, ``ref``, ``scaler`` that were
    declared in the YAML.

    This is the single source of truth for what variables the optimisation
    cares about. The CSV writer uses it to decide which columns to extract
    from the SQL recorder file; the plotter uses it to decide which series
    to draw and what their bounds are.

    Robust to missing keys: if a section is absent, returns an empty list.
    """
    if not pth.isfile(configuration_file):
        raise FileNotFoundError(f"YAML configuration not found: {configuration_file}")

    with open(configuration_file) as f:
        cfg = yaml.safe_load(f) or {}

    opt = cfg.get("optimization", {}) or {}

    def _normalise(entries: List) -> List[Dict]:
        if entries is None:
            return []
        out = []
        for e in entries:
            if isinstance(e, str):
                out.append({"name": e})
            elif isinstance(e, dict) and "name" in e:
                out.append(dict(e))  # shallow copy so caller can't mutate cfg
            else:
                # Skip malformed entries silently
                continue
        return out

    return {
        "design_variables": _normalise(opt.get("design_variables", [])),
        "constraints": _normalise(opt.get("constraints", [])),
        "objective": _normalise(opt.get("objective", [])),
    }


def _parse_optim_vars(configuration_file: str) -> Tuple[List[str], List[str], List[str]]:
    """Legacy 3-tuple form. Kept for backward compatibility with the rest
    of this module. New code should use ``parse_optim_vars`` instead."""
    parsed = parse_optim_vars(configuration_file)
    return (
        [v["name"] for v in parsed["design_variables"]],
        [v["name"] for v in parsed["constraints"]],
        [v["name"] for v in parsed["objective"]],
    )


def _short_name(full_name: str, prefix: str = "") -> str:
    """Shorten a long variable name for CSV columns."""
    return prefix + ":".join(full_name.split(":")[-2:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def optimize_problem_with_logging(
    configuration_file: str,
    log_path: str = "./workdir/optim_log.csv",
    overwrite: bool = True,
    auto_scaling: bool = False,
    write_xml_every_iter: bool = True,
    stop_file: str = _STOP_FILE_DEFAULT,
    verbose: bool = True,
):
    """
    Run an optimisation while recording every driver iteration.

    Mirrors ``fastoad.api.optimize_problem`` but inserts an OpenMDAO
    ``SqliteRecorder`` on the driver. After the run completes (successfully
    or not), the recorded cases are exported to a tidy CSV.

    **Graceful stopping from Jupyter**: while the optimiser is running in
    one cell, execute ``request_stop()`` from another cell. The optimiser
    will finish the current iteration, save the XML and CSV, then stop.

    Parameters
    ----------
    configuration_file : str
        Path to the FAST-OAD YAML configuration file.
    log_path : str
        Output CSV path. Parent folder is created if missing.
    overwrite : bool
        If True, output XML file is overwritten if present.
    auto_scaling : bool
        Forwarded to FAST-OAD.
    write_xml_every_iter : bool
        If True (default), the output XML is overwritten after **every**
        driver iteration. This means that if you cancel the run (Ctrl+C,
        kernel interrupt, or ``request_stop()``), the XML on disk always
        reflects the last fully evaluated iteration.
    stop_file : str
        Path to the stop file. If this file exists at the end of any
        iteration, the optimiser halts gracefully. Default: ``./workdir/STOP``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    problem : FASTOADProblem
        The solved problem (same as ``oad.optimize_problem`` returns).
    df : pd.DataFrame
        One row per driver iteration.
    """
    # ---- 1. Read variable names from YAML ----
    dv_names, cn_names, obj_names = _parse_optim_vars(configuration_file)
    if verbose:
        print(
            f"[logger] {len(dv_names)} design vars, "
            f"{len(cn_names)} constraints, {len(obj_names)} objectives"
        )

    # Ensure log folder exists
    log_dir = pth.dirname(pth.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)
    recorder_path = pth.abspath(log_path).replace(".csv", "_cases.sql")
    # OpenMDAO appends to existing sqlite databases — wipe it first
    if pth.isfile(recorder_path):
        os.remove(recorder_path)

    # Clean up any leftover stop file from a previous run
    stop_path = pth.abspath(stop_file)
    if pth.isfile(stop_path):
        os.remove(stop_path)
        if verbose:
            print(f"[logger] Removed stale stop file: {stop_path}")

    # ---- 2. Build the problem (FAST-OAD does input reading + DV registration) ----
    conf = FASTOADProblemConfigurator(configuration_file)
    problem = conf.get_problem(read_inputs=True, auto_scaling=auto_scaling)

    # Handle overwrite the same way oad.optimize_problem does
    outputs_path = pth.abspath(problem.output_file_path)
    if (not overwrite) and pth.exists(outputs_path):
        raise FileExistsError(f"Output file {outputs_path} exists. Use overwrite=True to bypass.")

    # ---- 3. Attach the recorder BEFORE setup() ----
    # CRITICAL: recording options must be set before setup() finalises the
    # driver. Attaching the recorder after setup leads to empty cases — the
    # driver fires the recorder but never binds variables to it.
    recorder = om.SqliteRecorder(recorder_path)
    problem.driver.add_recorder(recorder)
    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True
    problem.driver.recording_options["record_constraints"] = True
    # CRITICAL: in OpenMDAO 3.33 (used by FAST-OAD 1.8.3), setting
    # record_outputs=False discards EVERYTHING — including design vars and
    # objectives. The `_get_vars_to_record` filtering pipeline only adds the
    # DVs/objs to the recording set if `record_outputs` is also True.
    # Leave it at True. Filtering happens via `includes` below.
    problem.driver.recording_options["record_outputs"] = True
    problem.driver.recording_options["record_inputs"] = False
    problem.driver.recording_options["record_residuals"] = False
    # Restrict to just the variables we care about. This keeps the SQL file small
    # while still capturing DVs / objectives / constraints (those are added
    # unconditionally by record_desvars/objectives/constraints).
    problem.driver.recording_options["includes"] = dv_names + cn_names + obj_names

    # ---- 3b. Attach the snapshot recorder (writes XML after each iter) ----
    # Attached BEFORE setup so OpenMDAO initialises it properly.
    if write_xml_every_iter:
        snapshot = _SnapshotXmlRecorder(problem, outputs_path, stop_file=stop_path, verbose=verbose)
        problem.driver.add_recorder(snapshot)

    problem.setup()

    if write_xml_every_iter and verbose:
        print(
            f"[logger] XML snapshot enabled — {outputs_path} "
            f"will be overwritten after every iteration"
        )

    # ---- 4. Run ----
    if verbose:
        print(f"[logger] Running optimisation, recording to {recorder_path}")
    start = time.time()
    was_interrupted = False
    try:
        result = problem.run_driver()
        if isinstance(result, bool):
            problem.optim_failed = result  # legacy OpenMDAO
        else:
            problem.optim_failed = not result.success
    except KeyboardInterrupt:
        was_interrupted = True
        problem.optim_failed = True
        if verbose:
            print("\n[logger] *** Interrupted by user (Ctrl+C) ***")
            print("[logger]   Saving last state to XML and CSV...")
    except Exception as exc:
        if verbose:
            print(f"[logger] run_driver raised: {exc!r}")
        problem.optim_failed = True
        # Continue to export whatever was recorded before the crash
    elapsed = round(time.time() - start, 2)
    if verbose:
        status = "INTERRUPTED" if was_interrupted else ("FAILED" if problem.optim_failed else "OK")
        print(f"[logger] Driver finished in {elapsed}s — status: {status}")

    # Write FAST-OAD outputs file BEFORE cleanup (cleanup may tear down
    # internal data structures that write_outputs needs).
    try:
        problem.write_outputs()
        if verbose:
            print(f"[logger] Wrote outputs to: {outputs_path}")
    except Exception as exc:
        # Always print this — if it fails silently the user gets no output XML
        print(f"[logger] WARNING: write_outputs() failed: {exc!r}")
        print(f"[logger]   Expected output path: {outputs_path}")
        print("[logger]   Trying fallback via oad.generate_inputs/problem API...")
        # Fallback: manually write using FAST-OAD's variable I/O
        try:
            from fastoad.io import VariableIO

            writer = VariableIO(outputs_path)
            variables = problem.model.get_io_metadata(iotypes=("output",), return_rel_names=False)
            from fastoad.openmdao.variables import VariableList

            var_list = VariableList()
            for abs_name, meta in variables.items():
                try:
                    val = problem.get_val(abs_name, units=meta.get("units"))
                    var_list.append(
                        VariableList.Variable(name=abs_name, value=val, units=meta.get("units", ""))
                    )
                except Exception:
                    pass
            writer.write(var_list)
            print(f"[logger] Fallback write succeeded: {outputs_path}")
        except Exception as exc2:
            print(f"[logger] Fallback also failed: {exc2!r}")
            print("[logger]   No output XML was saved. You may need to re-run")
            print("[logger]   with standard oad.optimize_problem() to get outputs.")

    # Now safe to flush the recorder
    try:
        problem.cleanup()
    except Exception:
        pass  # cleanup is best-effort

    # ---- 5. Convert recorded cases to a DataFrame ----
    df = _cases_to_dataframe(recorder_path, dv_names, cn_names, obj_names, verbose)

    # Always write the CSV, even if empty, so the path exists
    df.to_csv(log_path, index=False)
    if verbose:
        print(f"[logger] Wrote {len(df)} rows x {len(df.columns)} cols -> {log_path}")

    return problem, df


def _cases_to_dataframe(
    recorder_path: str,
    dv_names: List[str],
    cn_names: List[str],
    obj_names: List[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Read an SQLite recorder file and return a tidy DataFrame."""
    if not pth.isfile(recorder_path):
        if verbose:
            print(f"[logger] No recorder file at {recorder_path}")
        return pd.DataFrame()

    try:
        cr = om.CaseReader(recorder_path)
    except Exception as exc:
        if verbose:
            print(f"[logger] CaseReader failed: {exc!r}")
        return pd.DataFrame()

    try:
        case_ids = cr.list_cases("driver", out_stream=None)
    except Exception:
        case_ids = cr.list_cases(out_stream=None)

    if verbose:
        print(f"[logger] {len(case_ids)} cases recorded")

    records = []
    for i, cid in enumerate(case_ids):
        try:
            case = cr.get_case(cid)
        except Exception:
            continue
        row = {"iteration": i}

        for name in dv_names:
            row[_short_name(name)] = _safe_get(case, name)
        for name in cn_names:
            row[_short_name(name, "c:")] = _safe_get(case, name)
        for name in obj_names:
            row[_short_name(name, "obj:")] = _safe_get(case, name)

        records.append(row)

    return pd.DataFrame(records)


def _safe_get(case, name: str) -> float:
    """Get a scalar value from a recorded case, returning NaN on failure."""
    for getter in (
        lambda: case[name],
        lambda: case.get_design_vars()[name],
        lambda: case.get_constraints()[name],
        lambda: case.get_objectives()[name],
    ):
        try:
            v = getter()
            return float(np.asarray(v).ravel()[0])
        except Exception:
            continue
    return np.nan


# ---------------------------------------------------------------------------
# YAML-driven re-extraction: rebuild a CSV from an existing SQL using any YAML
# ---------------------------------------------------------------------------
def sql_to_csv_from_yaml(
    sql_path: str,
    configuration_file: str,
    csv_path: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-extract a CSV from an existing SQL recorder file, using the current
    YAML to decide which variables to extract.

    Use this when:
    - The YAML's design variables / constraints have changed between runs.
    - You want to re-process an old SQL file without re-running the optimisation.
    - You want a CSV schema that matches a specific YAML, independent of which
      YAML was used to produce the SQL.

    Variables present in the YAML but missing from the SQL are filled with NaN.
    Variables present in the SQL but not in the YAML are silently ignored.

    Parameters
    ----------
    sql_path : str
        Path to an existing ``.sql`` recorder file (e.g.
        ``./workdir/optim_log_cases.sql``).
    configuration_file : str
        Path to the YAML to use for column selection.
    csv_path : str, optional
        If given, write the extracted CSV to this path. Default: replace ``.sql``
        with ``.csv`` next to the input file.
    verbose : bool
        Print a summary of what was extracted vs. missing.

    Returns
    -------
    pd.DataFrame
        The extracted DataFrame with columns ``iteration``, design variables,
        ``c:*`` constraints, and ``obj:*`` objectives.
    """
    parsed = parse_optim_vars(configuration_file)
    dv_names = [v["name"] for v in parsed["design_variables"]]
    cn_names = [v["name"] for v in parsed["constraints"]]
    obj_names = [v["name"] for v in parsed["objective"]]

    if verbose:
        print(f"[reextract] YAML: {configuration_file}")
        print(f"[reextract]   {len(dv_names)} design variables")
        print(f"[reextract]   {len(cn_names)} constraints")
        print(f"[reextract]   {len(obj_names)} objective(s)")

    df = _cases_to_dataframe(sql_path, dv_names, cn_names, obj_names, verbose=verbose)

    # Report which YAML-declared variables had no SQL data
    if verbose and not df.empty:
        all_yaml = (
            [(_short_name(n), n) for n in dv_names]
            + [(_short_name(n, "c:"), n) for n in cn_names]
            + [(_short_name(n, "obj:"), n) for n in obj_names]
        )
        missing = [
            (short, full)
            for short, full in all_yaml
            if short in df.columns and df[short].isna().all()
        ]
        if missing:
            print(
                f"[reextract] {len(missing)} variable(s) in YAML are entirely "
                f"missing or NaN in the SQL (likely renamed or removed):"
            )
            for short, full in missing:
                print(f"            {short}    (full name: {full})")

    if csv_path is None:
        csv_path = sql_path.replace(".sql", ".csv")
        if not csv_path.endswith(".csv"):
            csv_path = sql_path + ".csv"

    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[reextract] Wrote {len(df)} rows x {len(df.columns)} cols -> {csv_path}")

    return df


def check_yaml_against_csv(
    configuration_file: str,
    csv_path: str,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Compare a YAML's optimisation section against an existing CSV.

    Returns a dict with keys:
    - ``added``    : variables in YAML but not in CSV (design var renamed/added)
    - ``removed``  : variables in CSV but not in YAML (design var removed)
    - ``matched``  : variables present in both

    Useful for catching schema drift between runs. Call this before re-running
    or re-plotting to detect whether the existing CSV is still valid.
    """
    if not pth.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    parsed = parse_optim_vars(configuration_file)
    yaml_dv = {_short_name(v["name"]) for v in parsed["design_variables"]}
    yaml_cn = {_short_name(v["name"], "c:") for v in parsed["constraints"]}
    yaml_obj = {_short_name(v["name"], "obj:") for v in parsed["objective"]}
    yaml_all = yaml_dv | yaml_cn | yaml_obj

    df_cols = set(pd.read_csv(csv_path, nrows=0).columns) - {"iteration"}

    added = sorted(yaml_all - df_cols)
    removed = sorted(df_cols - yaml_all)
    matched = sorted(yaml_all & df_cols)

    if verbose:
        print(f"[schema check] YAML: {configuration_file}")
        print(f"[schema check] CSV:  {csv_path}")
        print(f"[schema check]   {len(matched)} variables match")
        if added:
            print(f"[schema check]   {len(added)} variable(s) added in YAML, missing from CSV:")
            for n in added:
                print(f"                  + {n}")
        if removed:
            print(f"[schema check]   {len(removed)} variable(s) in CSV but no longer in YAML:")
            for n in removed:
                print(f"                  - {n}")
        if not added and not removed:
            print("[schema check]   Schemas match perfectly.")

    return {"added": added, "removed": removed, "matched": matched}


# ---------------------------------------------------------------------------
# SQL inspection — read everything the recorder stored, regardless of YAML
# ---------------------------------------------------------------------------
def inspect_sql(sql_path: str, verbose: bool = True) -> Dict[str, List[str]]:
    """List every variable recorded in an SQL file, independent of any YAML.

    Useful when you suspect the YAML and the SQL are out of sync, or when you
    want to look at variables that aren't currently in the YAML's optimisation
    section (e.g. an old run with extra design variables).

    Returns a dict with keys ``design_variables``, ``constraints``, ``objectives``,
    each holding the list of full variable names found in the first case.
    Also returns ``n_cases`` (the number of recorded iterations).
    """
    cr = om.CaseReader(sql_path)
    case_ids = cr.list_cases("driver", out_stream=None)
    if not case_ids:
        case_ids = cr.list_cases(out_stream=None)
    if not case_ids:
        raise ValueError(f"No driver cases found in {sql_path}")

    case0 = cr.get_case(case_ids[0])
    dvs = sorted(case0.get_design_vars().keys())
    cns = sorted(case0.get_constraints().keys())
    objs = sorted(case0.get_objectives().keys())

    if verbose:
        print(f"[sql] {sql_path}")
        print(f"[sql]   {len(case_ids)} cases recorded")
        print(f"[sql]   {len(dvs)} design variables")
        print(f"[sql]   {len(cns)} constraints")
        print(f"[sql]   {len(objs)} objectives")

    return {
        "n_cases": len(case_ids),
        "design_variables": dvs,
        "constraints": cns,
        "objectives": objs,
    }


def best_iteration(
    sql_path: str,
    metric: str = "n_violated",
    violation_tol: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """Find the iteration in the SQL that came closest to feasibility.

    Scores each iteration by either:
      - ``metric="n_violated"``: number of constraints with value < -tol
      - ``metric="sum_violation"``: sum of max(0, -c_i) across all constraints
      - ``metric="worst"``: the single largest constraint violation

    Ties are broken by ``sum_violation``, then by best objective value
    (smallest, since FAST-OAD minimises the objective).

    Returns a dict with the winning iteration index, the violation summary
    for every iteration, and the design-variable values at the winning point.
    Useful to identify a good warm-start point for the next run.
    """
    cr = om.CaseReader(sql_path)
    case_ids = cr.list_cases("driver", out_stream=None)
    if not case_ids:
        case_ids = cr.list_cases(out_stream=None)
    if not case_ids:
        raise ValueError(f"No driver cases in {sql_path}")

    summary = []  # one row per iteration
    for i, cid in enumerate(case_ids):
        case = cr.get_case(cid)
        cns = case.get_constraints()
        vios = [max(0.0, -float(np.asarray(v).ravel()[0])) for v in cns.values()]
        n_violated = sum(1 for v in vios if v > violation_tol)
        total = float(sum(vios))
        worst = float(max(vios)) if vios else 0.0
        # Objective (smaller is better, by FAST-OAD convention)
        objs = case.get_objectives()
        obj = float(np.asarray(list(objs.values())[0]).ravel()[0]) if objs else np.nan
        summary.append(
            {
                "iter": i,
                "n_violated": n_violated,
                "sum_violation": total,
                "worst": worst,
                "objective": obj,
            }
        )

    # Pick the best
    if metric not in ("n_violated", "sum_violation", "worst"):
        raise ValueError(f"Unknown metric: {metric}")

    best = min(summary, key=lambda r: (r[metric], r["sum_violation"], r["objective"]))
    best_idx = best["iter"]
    best_case = cr.get_case(case_ids[best_idx])

    # Extract DV values at the best iteration
    best_dvs = {
        name: float(np.asarray(v).ravel()[0]) for name, v in best_case.get_design_vars().items()
    }
    best_constraints = {
        name: float(np.asarray(v).ravel()[0]) for name, v in best_case.get_constraints().items()
    }

    if verbose:
        print(f"=== BEST ITERATION BY {metric.upper()} ===")
        print(f"Iteration: {best_idx}  /  {len(case_ids) - 1}")
        print(f"  Constraints violated:   {best['n_violated']}")
        print(f"  Sum of violations:      {best['sum_violation']:.4f}")
        print(f"  Worst violation:        {best['worst']:.4f}")
        print(f"  Objective:              {best['objective']:.6g}")
        print()
        print("Per-iteration summary:")
        print(f"  {'iter':>4} {'#viol':>6} {'sum':>10} {'worst':>10} {'obj':>10}")
        for r in summary:
            mark = "  ★" if r["iter"] == best_idx else ""
            print(
                f"  {r['iter']:>4} {r['n_violated']:>6d} {r['sum_violation']:>10.4f} "
                f"{r['worst']:>10.4f} {r['objective']:>10.4g}{mark}"
            )
        print()
        print("Violated constraints at this iteration:")
        for n, v in sorted(best_constraints.items(), key=lambda kv: kv[1]):
            if v < -violation_tol:
                short = ":".join(n.split(":")[-3:])
                print(f"  {short:<50} {v:+10.4f}")

    return {
        "best_iter": best_idx,
        "best_dvs": best_dvs,
        "best_constraints": best_constraints,
        "summary": summary,
    }


def export_iteration_as_inputs(
    sql_path: str,
    iteration: int,
    template_inputs_xml: str,
    output_xml: str,
    verbose: bool = True,
) -> str:
    """Write the design-variable values from a chosen iteration into a copy of
    the input XML, ready to use as the starting point for a fresh run.

    Reads the SQL at the chosen iteration, then for each design variable name
    found there, looks for the matching tag in ``template_inputs_xml`` and
    overwrites the value. Variables in the SQL but not in the XML are reported.
    Variables in the XML but not in the SQL are left untouched.

    Parameters
    ----------
    sql_path : str
        Path to the SQL recorder file.
    iteration : int
        Which iteration to extract (0-indexed). Use -1 for the final iteration.
        Use ``best_iteration(sql_path)["best_iter"]`` for the most feasible one.
    template_inputs_xml : str
        Path to a working ``problem_inputs.xml`` to use as the template.
    output_xml : str
        Where to write the modified XML.
    verbose : bool
    """
    import xml.etree.ElementTree as ET

    cr = om.CaseReader(sql_path)
    case_ids = cr.list_cases("driver", out_stream=None)
    if not case_ids:
        case_ids = cr.list_cases(out_stream=None)

    if iteration == -1:
        iteration = len(case_ids) - 1
    if iteration < 0 or iteration >= len(case_ids):
        raise IndexError(f"Iteration {iteration} out of range [0, {len(case_ids) - 1}]")

    case = cr.get_case(case_ids[iteration])
    dvs = {name: float(np.asarray(v).ravel()[0]) for name, v in case.get_design_vars().items()}

    tree = ET.parse(template_inputs_xml)
    root = tree.getroot()

    def _find_by_path(root, dotted_path: str):
        """Find an element by its colon-separated FAST-OAD name."""
        parts = dotted_path.split(":")
        node = root
        # The root tag is the aircraft id, skip the first lookup
        for part in parts:
            child = node.find(part)
            if child is None:
                return None
            node = child
        return node

    updated, missing = [], []
    for name, value in dvs.items():
        node = _find_by_path(root, name)
        if node is None:
            missing.append(name)
            continue
        node.text = repr(value)
        updated.append((name, value))

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    if verbose:
        print(f"[export] Iteration {iteration} written to {output_xml}")
        print(f"[export]   {len(updated)} DV(s) updated")
        for n, v in updated:
            short = ":".join(n.split(":")[-3:])
            print(f"           {short:<50} = {v:.5g}")
        if missing:
            print(f"[export]   {len(missing)} DV(s) not found in XML (probably need to be added):")
            for n in missing:
                print(f"           {n}")

    return output_xml


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_iteration_log(
    log_path: str = "./workdir/optim_log.csv",
    show_constraints: bool = True,
    figsize=(16, 10),
):
    """Plot history from a CSV produced by ``optimize_problem_with_logging``.

    Static matplotlib version. For an interactive version (clickable legend,
    double-click to isolate, zoom/pan), use ``plot_iteration_log_interactive``.
    """
    import matplotlib.pyplot as plt

    if not pth.isfile(log_path):
        raise FileNotFoundError(
            f"Log file not found: {log_path}\n"
            "Did optimize_problem_with_logging run successfully? "
            "Check its console output for [logger] messages."
        )

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(
            f"Log file {log_path} is empty - the recorder captured no cases. "
            "Check whether the driver actually ran any iterations."
        )

    iters = df["iteration"]
    dv_cols = [c for c in df.columns if not c.startswith(("c:", "obj:", "iteration"))]
    cn_cols = [c for c in df.columns if c.startswith("c:")]
    obj_cols = [c for c in df.columns if c.startswith("obj:")]

    n_plots = 1 + 1 + (1 if show_constraints and cn_cols else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    # Objective
    ax = axes[ax_idx]
    for col in obj_cols:
        ax.plot(iters, df[col], "o-", markersize=3, label=col)
    ax.set_ylabel("Objective")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Objective history")
    ax_idx += 1

    # Design variables - normalised to [0, 1] for visual comparison
    ax = axes[ax_idx]
    for col in dv_cols:
        vals = df[col]
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-12:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = vals * 0.0 + 0.5
        ax.plot(iters, normed, "-", linewidth=1, alpha=0.7, label=col)
    ax.set_ylabel("DVs (each normalised)")
    ax.legend(fontsize=6, ncol=3, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Design variable history")
    ax_idx += 1

    # Constraints
    if show_constraints and cn_cols:
        ax = axes[ax_idx]
        for col in cn_cols:
            ax.plot(iters, df[col], "-", linewidth=1, alpha=0.7, label=col)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Constraint value")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_title("Constraint history (dashed line = 0)")
    else:
        axes[-1].set_xlabel("Iteration")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly version
# ---------------------------------------------------------------------------
def plot_iteration_log_interactive(
    log_path: str = "./workdir/optim_log.csv",
    show_constraints: bool = True,
    height_per_panel: int = 380,
    clip_constraints: float = 5.0,
    output_html: str = None,
):
    """Interactive HTML plot of the optimisation history.

    Click a legend entry to toggle that series. **Double-click** a legend entry
    to isolate (hide all others). Drag to zoom, double-click the plot to reset.

    Parameters
    ----------
    log_path : str
        Path to the CSV produced by ``optimize_problem_with_logging``.
    show_constraints : bool
        Include the constraint panel.
    height_per_panel : int
        Pixel height of each of the three panels. Default 380.
    clip_constraints : float
        Clip constraint values to ``[-clip, +clip]`` for visualisation. Some
        constraints (e.g. ``c:fuselage:volume``) can take values of ~100+ which
        squash everything else flat. Set to ``None`` to disable clipping.
    output_html : str, optional
        If given, save the interactive plot to this file. Otherwise it renders
        inline in Jupyter (returns the figure object).

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure. In Jupyter, just return it from the cell to
        render. Outside Jupyter, pass ``output_html`` and open the file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Interactive plotting requires plotly. Install with:\n"
            "    pip install plotly\n"
            "Or use the static plot_iteration_log() instead."
        ) from exc

    if not pth.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(f"Log file {log_path} is empty.")

    iters = df["iteration"]
    dv_cols = [c for c in df.columns if not c.startswith(("c:", "obj:", "iteration"))]
    cn_cols = [c for c in df.columns if c.startswith("c:")]
    obj_cols = [c for c in df.columns if c.startswith("obj:")]

    n_plots = 1 + 1 + (1 if show_constraints and cn_cols else 0)
    subplot_titles = ["Objective history", "Design variable history (each normalised)"]
    if show_constraints and cn_cols:
        subplot_titles.append(
            "Constraint history (dashed = 0, shaded = infeasible)"
            + (f"  (clipped to ±{clip_constraints})" if clip_constraints else "")
        )

    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    # ----- Objective -----
    for col in obj_cols:
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=df[col],
                mode="lines+markers",
                name=col,
                legendgroup="obj",
                legendgrouptitle_text="Objective",
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

    # ----- Design variables (normalised) -----
    for col in dv_cols:
        vals = df[col]
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-12:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = pd.Series([0.5] * len(vals), index=vals.index)
        # Build a hover-text showing the actual (un-normalised) value
        hover = [f"{col}<br>iter {i}: {v:.5g}" for i, v in zip(iters, vals)]
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=normed,
                mode="lines+markers",
                name=col,
                legendgroup="dv",
                legendgrouptitle_text="Design variables",
                hovertext=hover,
                hoverinfo="text",
                marker=dict(size=4),
                line=dict(width=1.5),
            ),
            row=2,
            col=1,
        )

    # ----- Constraints -----
    if show_constraints and cn_cols:
        for col in cn_cols:
            vals = df[col]
            if clip_constraints is not None:
                vals_clipped = vals.clip(-clip_constraints, clip_constraints)
            else:
                vals_clipped = vals
            hover = [f"{col}<br>iter {i}: {v:+.5g}" for i, v in zip(iters, vals)]
            # Color final-iteration violators in red, others in default
            _final_val = vals.iloc[-1]
            line_dash = "solid"
            fig.add_trace(
                go.Scatter(
                    x=iters,
                    y=vals_clipped,
                    mode="lines+markers",
                    name=col,
                    legendgroup="cn",
                    legendgrouptitle_text="Constraints (need ≥ 0)",
                    hovertext=hover,
                    hoverinfo="text",
                    marker=dict(size=4),
                    line=dict(width=1.5, dash=line_dash),
                ),
                row=n_plots,
                col=1,
            )

        # Reference line at zero
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"), row=n_plots, col=1)
        # Shade the infeasible region
        if clip_constraints is not None:
            fig.add_hrect(
                y0=-clip_constraints,
                y1=0,
                fillcolor="red",
                opacity=0.05,
                line_width=0,
                row=n_plots,
                col=1,
            )

    # Layout
    fig.update_layout(
        height=height_per_panel * n_plots,
        hovermode="closest",
        legend=dict(
            groupclick="toggleitem",  # click whole group or individual
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Iteration", row=n_plots, col=1)
    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_yaxes(title_text="Normalised DV [0,1]", row=2, col=1)
    if show_constraints and cn_cols:
        fig.update_yaxes(title_text="Constraint value", row=n_plots, col=1)

    if output_html is not None:
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"[plot] Wrote interactive plot to: {output_html}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "plot":
        import matplotlib.pyplot as plt

        fig = plot_iteration_log(sys.argv[2])
        plt.show()
    elif len(sys.argv) >= 3 and sys.argv[1] == "iplot":
        out = sys.argv[3] if len(sys.argv) >= 4 else "optim_log_plot.html"
        plot_iteration_log_interactive(sys.argv[2], output_html=out)
        print(f"Open {out} in a browser.")
    elif len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        inspect_sql(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "best":
        metric = sys.argv[3] if len(sys.argv) >= 4 else "n_violated"
        best_iteration(sys.argv[2], metric=metric)
    elif len(sys.argv) >= 4 and sys.argv[1] == "reextract":
        sql_to_csv_from_yaml(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        print()
        print("CLI commands:")
        print("  python optim_iteration_logger.py plot <csv>           - static plot")
        print("  python optim_iteration_logger.py iplot <csv> [html]   - interactive plot")
        print("  python optim_iteration_logger.py inspect <sql>        - list all SQL vars")
        print("  python optim_iteration_logger.py best <sql> [metric]  - find best iteration")
        print("  python optim_iteration_logger.py reextract <sql> <yaml> - rebuild CSV from SQL")
