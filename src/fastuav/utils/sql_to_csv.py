"""
Extract a CSV iteration log from an OpenMDAO SqliteRecorder .sql file.

Useful when a run is in progress, was interrupted, or when the
optim_iteration_logger script didn't get a chance to write the CSV.

Usage (from a notebook cell or python prompt):

    from sql_to_csv import sql_to_csv
    df = sql_to_csv(
        "./workdir/optim_log_cases.sql",
        "./workdir/optim_log.csv",
        configuration_file="./data/configurations/fixedwing_mdo_FMUC.yaml",
    )

Or from the command line:

    python sql_to_csv.py ./workdir/optim_log_cases.sql ./workdir/optim_log.csv \\
        --config ./data/configurations/fixedwing_mdo_FMUC.yaml

Safe to run while the optimisation is still in progress: SQLite handles
concurrent readers fine. Just re-run it whenever you want a fresh snapshot.
"""

import os.path as pth
from typing import List, Optional, Tuple

import numpy as np
import openmdao.api as om
import pandas as pd
import yaml


def _parse_optim_vars(configuration_file: str) -> Tuple[List[str], List[str], List[str]]:
    with open(configuration_file) as f:
        cfg = yaml.safe_load(f)
    opt = cfg.get("optimization", {})
    dv = [d["name"] for d in opt.get("design_variables", [])]
    cn = [c["name"] for c in opt.get("constraints", [])]
    ob = [o["name"] for o in opt.get("objective", [])]
    return dv, cn, ob


def _short(name: str, prefix: str = "") -> str:
    return prefix + ":".join(name.split(":")[-2:])


def _safe_get(case, name: str) -> float:
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


def sql_to_csv(
    sql_path: str,
    csv_path: str,
    configuration_file: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read an SqliteRecorder file and write a tidy CSV.

    If ``configuration_file`` is provided, the CSV columns are restricted
    and renamed to the design variables, constraints and objectives listed
    in that YAML. Otherwise every recorded variable is dumped (wide CSV).

    Returns the DataFrame written.
    """
    if not pth.isfile(sql_path):
        raise FileNotFoundError(f"No such file: {sql_path}")

    cr = om.CaseReader(sql_path)
    try:
        case_ids = cr.list_cases("driver", out_stream=None)
    except Exception:
        case_ids = cr.list_cases(out_stream=None)

    if verbose:
        print(f"[sql->csv] {len(case_ids)} cases in {sql_path}")

    if len(case_ids) == 0:
        df = pd.DataFrame()
        df.to_csv(csv_path, index=False)
        print(f"[sql->csv] No cases — wrote empty file to {csv_path}")
        return df

    # Discover variable names: from YAML if given, otherwise from first case
    if configuration_file is not None:
        dv_names, cn_names, obj_names = _parse_optim_vars(configuration_file)
        if verbose:
            print(
                f"[sql->csv] YAML: {len(dv_names)} DVs, "
                f"{len(cn_names)} constraints, {len(obj_names)} obj"
            )
    else:
        first = cr.get_case(case_ids[0])
        dv_names = list(first.get_design_vars().keys())
        cn_names = list(first.get_constraints().keys())
        obj_names = list(first.get_objectives().keys())
        if verbose:
            print(
                f"[sql->csv] auto: {len(dv_names)} DVs, "
                f"{len(cn_names)} constraints, {len(obj_names)} obj"
            )

    records = []
    for i, cid in enumerate(case_ids):
        try:
            case = cr.get_case(cid)
        except Exception:
            continue
        row = {"iteration": i}
        for n in dv_names:
            row[_short(n)] = _safe_get(case, n)
        for n in cn_names:
            row[_short(n, "c:")] = _safe_get(case, n)
        for n in obj_names:
            row[_short(n, "obj:")] = _safe_get(case, n)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[sql->csv] Wrote {len(df)} rows x {len(df.columns)} cols -> {csv_path}")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sql", help="Path to the .sql recorder file")
    ap.add_argument("csv", help="Output CSV path")
    ap.add_argument("--config", default=None, help="Optional FAST-OAD YAML to restrict columns")
    args = ap.parse_args()
    sql_to_csv(args.sql, args.csv, configuration_file=args.config)
