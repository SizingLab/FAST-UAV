"""Estimation of the profile drag of miscellaneous items."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import fastoad.api as oad
from openmdao.core.explicitcomponent import ExplicitComponent

from ..constants import SUBMODEL_CD0_OTHER


@oad.RegisterSubmodel(SUBMODEL_CD0_OTHER, "fastga.submodel.aerodynamics.other.cd0.legacy")
class Cd0Other(ExplicitComponent):
    """
    Profile drag estimation for miscellaneous items such as cowling, cooling and various component

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        # self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")

        if self.options["low_speed_aero"]:
            # Gudmundsson p715. Assuming cx_cooling*wing area/MTOW value of the book is typical
            self.add_input("data:aerodynamics:cooling:low_speed:CD0", val=0.0005525)
            self.add_output("data:aerodynamics:other:low_speed:CD0")
            self.declare_partials(
                of="data:aerodynamics:other:low_speed:CD0",
                # wrt=["data:geometry:wing:area", "data:aerodynamics:cooling:low_speed:CD0"],
                wrt=["data:geometry:wing:surface", "data:aerodynamics:cooling:low_speed:CD0"],
                method="exact",
            )

        else:
            # Gudmundsson p715. Assuming cx_cooling*wing area/MTOW value of the book is typical
            self.add_input("data:aerodynamics:cooling:cruise:CD0", val=0.0005525)
            self.add_output("data:aerodynamics:other:cruise:CD0")
            self.declare_partials(
                of="data:aerodynamics:other:cruise:CD0",
                wrt=["data:geometry:wing:surface", "data:aerodynamics:cooling:cruise:CD0"],
                method="exact",
            )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]

        # COWLING (only if engine in fuselage): cx_cowl*wing_area assumed typical (Gudmundsson p739)
        if prop_layout == 3.0:
            cd0_cowling = 0.002*wing_area#0.0267 / wing_area 
        else:
            cd0_cowling = 0.0
        # Cooling (piston engine only)
        # Gudmundsson p739. Sum of other components (not calculated here), cx_other*wing_area
        # assumed typical
        cd0_components =  0.002*wing_area#0.0253 / wing_area

        if self.options["low_speed_aero"]:
            # FIXME: should come from propulsion model...
            cd0_cooling = inputs["data:aerodynamics:cooling:low_speed:CD0"]
            outputs["data:aerodynamics:other:low_speed:CD0"] = (
                cd0_cowling + cd0_cooling + cd0_components
            )
        else:
            cd0_cooling = inputs["data:aerodynamics:cooling:cruise:CD0"]
            outputs["data:aerodynamics:other:cruise:CD0"] = (
                cd0_cowling + cd0_cooling + cd0_components
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]

        if prop_layout == 3.0:
            d_cd0_cowl_d_wing_area = -0.0267 / wing_area**2.0
        else:
            d_cd0_cowl_d_wing_area = 0.0

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:other:low_speed:CD0", "data:aerodynamics:cooling:low_speed:CD0"
            ] = 1.0
            # partials["data:aerodynamics:other:low_speed:CD0", "data:geometry:wing:area"] = (
            partials["data:aerodynamics:other:low_speed:CD0", "data:geometry:wing:surface"] = (
                d_cd0_cowl_d_wing_area - 0.0253 / wing_area**2.0
            )
        else:
            partials[
                "data:aerodynamics:other:cruise:CD0", "data:aerodynamics:cooling:cruise:CD0"
            ] = 1.0
            # partials["data:aerodynamics:other:cruise:CD0", "data:geometry:wing:area"] = (
            partials["data:aerodynamics:other:cruise:CD0", "data:geometry:wing:surface"] = (
                d_cd0_cowl_d_wing_area - 0.0253 / wing_area**2.0
            )
