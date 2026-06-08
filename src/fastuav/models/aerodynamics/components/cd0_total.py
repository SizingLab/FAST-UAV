"""Estimation of total profile drag."""
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

from ..constants import SUBMODEL_CD0_SUM


@oad.RegisterSubmodel(SUBMODEL_CD0_SUM, "fastga.submodel.aerodynamics.sum.cd0.legacy")
class Cd0Total(ExplicitComponent):
    """
    Profile drag estimation for the whole aircraft. It is the simple sum of all the profile drag
    since every subpart was computed with the wing area as a reference and the interaction are
    taken into account with interference factors

    Based on : The drag build-up method in Gudmundsson, Snorri. General aviation aircraft design:
    Applied Methods and Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.add_input(
            "settings:aerodynamics:aircraft:undesirable_drag:k_factor",
            val=1.25,
            desc="Correction coefficient to take into account the other undesirable drag, "
            "default is 1.25 as suggested in Gudmundsson",
        )

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:vertical_tail:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:nacelles:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:landing_gear:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:other:low_speed:CD0", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:low_speed:CD0")

            self.declare_partials("*", "*", method="exact")
        else:
            self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:cruise:CD0", val=np.nan)
            # self.add_input("data:aerodynamics:horizontal_tail:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:tail:horizontal:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:vertical_tail:cruise:CD0", val=np.nan)
            # self.add_input("data:aerodynamics:nacelles:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:landing_gear:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:other:cruise:CD0", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:cruise:CD0")

            self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            cd0_wing = inputs["data:aerodynamics:wing:low_speed:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:low_speed:CD0"]
            cd0_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CD0"]
            # cd0_nac = inputs["data:aerodynamics:nacelles:low_speed:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:low_speed:CD0"]
            cd0_other = inputs["data:aerodynamics:other:low_speed:CD0"]
        else:
            cd0_wing = inputs["data:aerodynamics:wing:cruise:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:cruise:CD0"]
            # cd0_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CD0"]
            cd0_ht = inputs["data:aerodynamics:tail:horizontal:cruise:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:cruise:CD0"]
            # cd0_nac = inputs["data:aerodynamics:nacelles:cruise:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:cruise:CD0"]
            cd0_other = inputs["data:aerodynamics:other:cruise:CD0"]

        # CRUD (other undesirable drag). Factor from Gudmundsson book
        crud_factor = inputs["settings:aerodynamics:aircraft:undesirable_drag:k_factor"]

        cd0 = crud_factor * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_lg  + cd0_other) # + cd0_nac

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CD0"] = cd0
        else:
            outputs["data:aerodynamics:aircraft:cruise:CD0"] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        crud_factor = inputs["settings:aerodynamics:aircraft:undesirable_drag:k_factor"]

        if self.options["low_speed_aero"]:
            cd0_wing = inputs["data:aerodynamics:wing:low_speed:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:low_speed:CD0"]
            cd0_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CD0"]
            # cd0_nac = inputs["data:aerodynamics:nacelles:low_speed:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:low_speed:CD0"]
            cd0_other = inputs["data:aerodynamics:other:low_speed:CD0"]
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0", "data:aerodynamics:wing:low_speed:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:fuselage:low_speed:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:horizontal_tail:low_speed:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:vertical_tail:low_speed:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:nacelles:low_speed:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:landing_gear:low_speed:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0", "data:aerodynamics:other:low_speed:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "settings:aerodynamics:aircraft:undesirable_drag:k_factor",
            ] = cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_lg  + cd0_other #+ cd0_nac
        else:
            cd0_wing = inputs["data:aerodynamics:wing:cruise:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:cruise:CD0"]
            # cd0_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CD0"]
            cd0_ht = inputs["data:aerodynamics:tail:horizontal:cruise:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:cruise:CD0"]
            # cd0_nac = inputs["data:aerodynamics:nacelles:cruise:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:cruise:CD0"]
            cd0_other = inputs["data:aerodynamics:other:cruise:CD0"]
            partials[
                "data:aerodynamics:aircraft:cruise:CD0", "data:aerodynamics:wing:cruise:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0", "data:aerodynamics:fuselage:cruise:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:tail:horizontal:cruise:CD0",#"data:aerodynamics:horizontal_tail:cruise:CD0",
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:vertical_tail:cruise:CD0",
            ] = crud_factor
            # partials[
            #     "data:aerodynamics:aircraft:cruise:CD0", "data:aerodynamics:nacelles:cruise:CD0"
            # ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0", "data:aerodynamics:landing_gear:cruise:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0", "data:aerodynamics:other:cruise:CD0"
            ] = crud_factor
            partials[
                "data:aerodynamics:aircraft:cruise:CD0",
                "settings:aerodynamics:aircraft:undesirable_drag:k_factor",
            ] = cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_lg  + cd0_other #+ cd0_nac
