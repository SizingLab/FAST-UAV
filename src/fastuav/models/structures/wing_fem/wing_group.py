"""
FEM wing-structure group: load distribution -> selected FEM model.

Wires the spanwise load distribution to one of the two FEM wing models and
promotes everything to the top-level FAST-UAV namespace so the registered
``Structures`` group can swap it in for the analytic ``WingStructuresFW``.
"""

from __future__ import annotations

import openmdao.api as om

from .load_distribution import WingLoadDistribution
from .tube_spar_foam import TubeSparFoamModel


class WingStructureFEM(om.Group):
    """FEM-based wing structure (``tube_spar_foam`` or ``wingbox_shell``)."""

    def initialize(self):
        self.options.declare("wing_model", default="tube_spar_foam",
                             values=["tube_spar_foam", "wingbox_shell"])
        self.options.declare("n_elements", types=int, default=20,
                             desc="Beam elements for the tube_spar_foam model.")
        self.options.declare("n_span", types=int, default=10,
                             desc="Spanwise shell stations for the wingbox_shell model.")
        self.options.declare("n_chord", types=int, default=6,
                             desc="Chordwise shell panels for the wingbox_shell model.")
        self.options.declare("ks_rho", types=float, default=100.0)
        self.options.declare("use_aero_vectors", types=bool, default=False)

    def setup(self):
        model = self.options["wing_model"]

        if model == "tube_spar_foam":
            n_load = self.options["n_elements"]
        else:
            n_load = self.options["n_span"]

        self.add_subsystem(
            "load_distribution",
            WingLoadDistribution(n_elements=n_load,
                                 use_aero_vectors=self.options["use_aero_vectors"]),
            promotes=["*"],
        )

        if model == "tube_spar_foam":
            self.add_subsystem(
                "fem",
                TubeSparFoamModel(n_elements=self.options["n_elements"],
                                  ks_rho=self.options["ks_rho"]),
                promotes=["*"],
            )
        else:  # wingbox_shell -- imported lazily so the rest works before Stage B lands
            from .wingbox.wingbox_model import WingboxModel

            self.add_subsystem(
                "fem",
                WingboxModel(n_span=self.options["n_span"],
                             n_chord=self.options["n_chord"],
                             ks_rho=self.options["ks_rho"]),
                promotes=["*"],
            )
