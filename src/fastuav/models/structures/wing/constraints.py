"""
Structural constraints for the wing
"""

import openmdao.api as om
import numpy as np


class SparsGeometricalConstraint(om.ExplicitComponent):
    """
    Geometrical constraint definition for the spars.
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        spar_model = self.options["spar_model"]
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")
        if spar_model == "pipe":
            self.add_input("data:structures:wing:spar:diameter:outer", val=np.nan, units="m")
            self.add_output("optimization:constraints:structures:wing:spar:diameter", units=None)
        else:
            self.add_input("data:structures:wing:spar:depth", val=np.nan, units="m")
            self.add_output("optimization:constraints:structures:wing:spar:depth", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        spar_model = self.options["spar_model"]
        t_wingtip = inputs["data:geometry:wing:tip:thickness"]
        if spar_model == "pipe":
            d_out = inputs["data:structures:wing:spar:diameter:outer"]
            spar_cnstr = (t_wingtip - d_out) / d_out  # constraint on spar external dimension [-]
            outputs["optimization:constraints:structures:wing:spar:diameter"] = spar_cnstr
        else:
            h_spar = inputs["data:structures:wing:spar:depth"]
            spar_cnstr = (t_wingtip - h_spar) / h_spar  # constraint on spar external dimension [-]
            outputs["optimization:constraints:structures:wing:spar:depth"] = spar_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        spar_model = self.options["spar_model"]
        t_wingtip = inputs["data:geometry:wing:tip:thickness"]
        if spar_model == "pipe":
            d_out = inputs["data:structures:wing:spar:diameter:outer"]
            partials["optimization:constraints:structures:wing:spar:diameter",
                     "data:geometry:wing:tip:thickness"] = 1 / d_out
            partials["optimization:constraints:structures:wing:spar:diameter",
                     "data:structures:wing:spar:diameter:outer"] = - t_wingtip / d_out ** 2
        else:
            h_spar = inputs["data:structures:wing:spar:depth"]
            partials["optimization:constraints:structures:wing:spar:depth",
                     "data:geometry:wing:tip:thickness"] = 1 / h_spar
            partials["optimization:constraints:structures:wing:spar:depth",
                     "data:structures:wing:spar:depth"] = - t_wingtip / h_spar ** 2


class SparsStressVTOLConstraint(om.ExplicitComponent):
    """
    Constraint on spar's stress during vertical takeoff scenario.
    """

    def setup(self):
        self.add_input("data:structures:wing:spar:stress:VTOL", val=np.nan, units="N/m**2")
        self.add_input("data:structures:wing:spar:stress:max", val=np.nan, units="N/m**2")
        self.add_output("optimization:constraints:structures:wing:spar:stress:VTOL", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        sig_root = inputs["data:structures:wing:spar:stress:VTOL"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]

        stress_cnstr = (sig_max - sig_root) / sig_root

        outputs["optimization:constraints:structures:wing:spar:stress:VTOL"] = stress_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sig_root = inputs["data:structures:wing:spar:stress:VTOL"]
        sig_max = inputs["data:structures:wing:spar:stress:max"]

        partials["optimization:constraints:structures:wing:spar:stress:VTOL",
                 "data:structures:wing:spar:stress:VTOL"] = - sig_max / sig_root ** 2

        partials["optimization:constraints:structures:wing:spar:stress:VTOL",
                 "data:structures:wing:spar:stress:max"] = 1 / sig_root



