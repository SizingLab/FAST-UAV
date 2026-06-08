"""Estimation of the landing gear profile drag."""
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

from ..constants import SUBMODEL_CD0_LANDING_GEAR


@oad.RegisterSubmodel(
    SUBMODEL_CD0_LANDING_GEAR, "fastga.submodel.aerodynamics.landing_gear.cd0.legacy"
)
class Cd0LandingGear(ExplicitComponent):
    """
    Profile drag estimation for the landing gear

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # self.add_input("data:geometry:landing_gear:type", val=np.nan)
        # self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        # self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        # if self.options["low_speed_aero"]:
        #     self.add_output("data:aerodynamics:landing_gear:low_speed:CD0")
        #     self.declare_partials(
        #         "data:aerodynamics:landing_gear:low_speed:CD0",
        #         ["data:geometry:landing_gear:height", "data:geometry:wing:area"],
        #         method="exact",
        #     )
        # else:
        self.add_output("data:aerodynamics:landing_gear:cruise:CD0")
        self.declare_partials(
            "data:aerodynamics:landing_gear:cruise:CD0",
            # ["data:geometry:wing:area"], #"data:geometry:landing_gear:height",
            ["data:geometry:wing:surface"], #"data:geometry:landing_gear:height", 
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # lg_type = inputs["data:geometry:landing_gear:type"]
        # lg_height = inputs["data:geometry:landing_gear:height"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]
        # if lg_type == 0.0:  # non-retractable LG AC (ref: Cirrus SR22)
        #     # Gudmundsson example 15.12 (page 721)
        #     area_mlg = 15 * 6 * 0.0254**2  # Frontal area of wheel (data in inches)
        #     area_nlg = 14 * 5 * 0.0254**2
        #     # MLG
        #     cd_wheel = 0.484
        #     cd0_mlg = cd_wheel * area_mlg / wing_area
        #     # NLG
        #     cd_wheel = 0.484 / 2
        #     cd0_nlg = cd_wheel * area_nlg / wing_area
        #     cd0 = cd0_mlg + cd0_nlg

        #     if self.options["low_speed_aero"]:
        #         outputs["data:aerodynamics:landing_gear:low_speed:CD0"] = cd0
        #     else:
        #         outputs["data:aerodynamics:landing_gear:cruise:CD0"] = cd0

        # else:  # retractable LG AC
        #     tyre_width = 5 * 0.0254
        #     # MLG
        #     cd_mlg = 1.2
        #     area_mlg = tyre_width * 1.8 * lg_height
        #     # NLG
        #     cd_nlg = 0.65
        #     area_nlg = 14 * 5 * 0.0254**2
        #     cd0 = (cd_mlg * area_mlg + cd_nlg * area_nlg) / wing_area

        #     if self.options["low_speed_aero"]:
        #         outputs["data:aerodynamics:landing_gear:low_speed:CD0"] = cd0
        #     else:
        outputs["data:aerodynamics:landing_gear:cruise:CD0"] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # lg_type = inputs["data:geometry:landing_gear:type"]
        # lg_height = inputs["data:geometry:landing_gear:height"]
        # wing_area = inputs["data:geometry:wing:area"]

        # if lg_type == 0.0:
        #     area_mlg = 15 * 6 * 0.0254**2  # Frontal area of wheel (data in inches)
        #     area_nlg = 14 * 5 * 0.0254**2
        #     # MLG
        #     cd_wheel = 0.484
        #     cd0_mlg = cd_wheel * area_mlg / wing_area
        #     # NLG
        #     cd_wheel = 0.484 / 2
        #     cd0_nlg = cd_wheel * area_nlg / wing_area
        #     cd0 = cd0_mlg + cd0_nlg

        #     if self.options["low_speed_aero"]:
        #         partials[
        #             "data:aerodynamics:landing_gear:low_speed:CD0",
        #             "data:geometry:landing_gear:height",
        #         ] = 0
        #         partials[
        #             "data:aerodynamics:landing_gear:low_speed:CD0",
        #             "data:geometry:wing:area",
        #         ] = -cd0 / wing_area
        #     else:
        #         partials[
        #             "data:aerodynamics:landing_gear:cruise:CD0", "data:geometry:landing_gear:height"
        #         ] = 0
        #         partials[
        #             "data:aerodynamics:landing_gear:cruise:CD0",
        #             "data:geometry:wing:area",
        #         ] = -cd0 / wing_area
        # else:
        #     tyre_width = 5 * 0.0254
        #     # MLG
        #     cd_mlg = 1.2
        #     area_mlg = tyre_width * 1.8 * lg_height
        #     # NLG
        #     cd_nlg = 0.65
        #     area_nlg = 14 * 5 * 0.0254**2
        #     cd0 = (cd_mlg * area_mlg + cd_nlg * area_nlg) / wing_area

        #     if self.options["low_speed_aero"]:
        #         partials[
        #             "data:aerodynamics:landing_gear:low_speed:CD0",
        #             "data:geometry:landing_gear:height",
        #         ] = cd_mlg * tyre_width * 1.8 / wing_area
        #         partials[
        #             "data:aerodynamics:landing_gear:low_speed:CD0",
        #             "data:geometry:wing:area",
        #         ] = -cd0 / wing_area
        #     else:
        #         partials[
        #             "data:aerodynamics:landing_gear:cruise:CD0", "data:geometry:landing_gear:height"
        #         ] = 0.0
        partials[
            "data:aerodynamics:landing_gear:cruise:CD0",
            "data:geometry:wing:surface", #"data:geometry:wing:area",
        ] = 0.0