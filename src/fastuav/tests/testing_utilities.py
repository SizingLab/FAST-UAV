"""
Shared helpers for the model unit tests, in the spirit of FAST-OAD-CS25's
``tests/testing_utilities``.

``run_system`` is re-exported from FAST-OAD (it wires an ``IndepVarComp`` of inputs to
the component under test, runs the model and returns the solved problem). ``get_indep_var_comp``
builds that input component from a plain mapping so each unit test can declare the exact
inputs of the component it exercises.
"""

import openmdao.api as om
from fastoad.testing import run_system  # noqa: F401  (re-exported for the test modules)

__all__ = ["run_system", "get_indep_var_comp"]


def get_indep_var_comp(values):
    """Build an ``om.IndepVarComp`` feeding the inputs of a component under test.

    ``values`` maps a variable name to either ``value`` or ``(value, units)``::

        get_indep_var_comp({
            "data:propulsion:propeller:diameter": (0.4, "m"),
            "data:propulsion:propeller:beta": 0.5,
        })
    """
    ivc = om.IndepVarComp()
    for name, spec in values.items():
        if isinstance(spec, tuple):
            value, units = spec
        else:
            value, units = spec, None
        ivc.add_output(name, val=value, units=units)
    return ivc
