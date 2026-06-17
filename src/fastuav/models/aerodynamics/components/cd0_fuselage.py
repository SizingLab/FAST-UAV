"""Estimation of the fuselage profile drag."""
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

import fastoad.api as oad
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

from ..constants import SUBMODEL_CD0_FUSELAGE


@oad.RegisterSubmodel(SUBMODEL_CD0_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.cd0.legacy")
class Cd0Fuselage(ExplicitComponent):
    """
    Profile drag estimation for the fuselage

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.add_input("data:geometry:fuselage:fineness", val=np.nan, units=None)
        # self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        # self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        # self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        # self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:fuselage:low_speed:CD0", val=0.025)

            self.declare_partials(
                of="data:aerodynamics:fuselage:low_speed:CD0", wrt="*", method="exact"
            )
        else:
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:fuselage:cruise:CD0", val=0.025)

            self.declare_partials(
                of="data:aerodynamics:fuselage:cruise:CD0", wrt="*", method="exact"
            )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # height = inputs["data:geometry:fuselage:maximum_height"]
        # width = inputs["data:geometry:fuselage:maximum_width"]
        fineness_ratio = inputs["data:geometry:fuselage:fineness"]
        length = inputs["data:geometry:fuselage:length"]
        d_fus_mid = inputs["data:geometry:fuselage:diameter:mid"]
        # wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]
        if self.options["low_speed_aero"]:
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        # Local Reynolds:
        reynolds = unit_reynolds * length
        # 5% NLF
        x_trans = 0.05
        # Roots
        x0_turbulent = 36.9 * x_trans**0.625 * reynolds**-0.375
        cf_fus = 0.074 * reynolds**-0.2 * (1.0 - (x_trans - x0_turbulent)) ** 0.8
        # fineness_ratio = length / np.sqrt(4 * height * width / np.pi)
        form_factor_fus = 1.0 + 60.0 / (fineness_ratio**3.0) + fineness_ratio / 400.0
        # Fuselage
        S_wet_fus = (
            np.pi
            * d_fus_mid
            * length
            * (1 - 2 / fineness_ratio) ** (2 / 3)
            * (1 + 1 / (fineness_ratio**2))
        )
        cd0_fuselage = cf_fus * form_factor_fus * S_wet_fus / wing_area
        # Cockpit window (Gudmundsson p727)
        # cd0_window = 0.002 * (height * width) / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:fuselage:low_speed:CD0"] = cd0_fuselage  # + cd0_window
        else:
            outputs["data:aerodynamics:fuselage:cruise:CD0"] = cd0_fuselage  # + cd0_window

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # height = inputs["data:geometry:fuselage:maximum_height"]
        # width = inputs["data:geometry:fuselage:maximum_width"]
        d_fus_mid = inputs["data:geometry:fuselage:diameter:mid"]
        fineness_ratio = inputs["data:geometry:fuselage:fineness"]
        length = inputs["data:geometry:fuselage:length"]
        # wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]

        if self.options["low_speed_aero"]:
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        # Windows:
        # d_cd0_w_d_height = 0.002 * width / wing_area
        # d_cd0_w_d_width = 0.002 * height / wing_area
        # d_cd0_w_d_area = -0.002 * width * height / wing_area**2.0

        # Local Reynolds:
        reynolds = unit_reynolds * length
        x_trans = 0.05
        x0_turbulent = 36.9 * x_trans**0.625 * reynolds**-0.375
        S_wet_fus = (
            np.pi
            * d_fus_mid
            * length
            * (1 - 2 / fineness_ratio) ** (2 / 3)
            * (1 + 1 / (fineness_ratio**2))
        )
        form_factor_fus = 1.0 + 60.0 / (fineness_ratio**3.0) + fineness_ratio / 400.0

        d_x0_turb_d_unit_re = (
            -0.375 * 36.9 * x_trans**0.625 * length**-0.375 * unit_reynolds**-1.375
        )
        d_x0_turb_d_length = -0.375 * 36.9 * x_trans**0.625 * unit_reynolds**-0.375 * length**-1.375

        cf_fus = 0.074 * reynolds**-0.2 * (1.0 - (x_trans - x0_turbulent)) ** 0.8
        d_cf_fus_d_unit_re = (
            -0.2
            * 0.074
            * length**-0.2
            * unit_reynolds**-1.2
            * (1.0 - (x_trans - x0_turbulent)) ** 0.8
        ) + +0.074 * reynolds**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_turbulent)
        ) ** -0.2 * d_x0_turb_d_unit_re
        d_cf_fus_d_length = (
            -0.2
            * 0.074
            * unit_reynolds**-0.2
            * length**-1.2
            * (1.0 - (x_trans - x0_turbulent)) ** 0.8
        ) + +0.074 * reynolds**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_turbulent)
        ) ** -0.2 * d_x0_turb_d_length
        d_S_wet_fus_d_length = (
            np.pi * d_fus_mid * (1 - 2 / fineness_ratio) ** (2 / 3) * (1 + 1 / (fineness_ratio**2))
        )
        # fineness_ratio = length / np.sqrt(4 * height * width / np.pi)
        # d_f_d_length = 1.0 / np.sqrt(4 * height * width / np.pi)
        # d_f_d_height = -0.5 * length / np.sqrt(4 * width / np.pi) * height**-1.5
        # d_f_d_width = -0.5 * length / np.sqrt(4 * height / np.pi) * width**-1.5

        d_ff_d_f = -3.0 * 60 * fineness_ratio**-4.0 + 1.0 / 400.0
        d_S_wet_fus_d_f = (
            np.pi
            * d_fus_mid
            * length
            * (
                (1 + 1 / (fineness_ratio**2))
                * (2 / 3)
                * (1 - 2 / fineness_ratio) ** (-1 / 3)
                * (2 / fineness_ratio**2)
                + +((1 - 2 / fineness_ratio) ** (2 / 3)) * (-2 / fineness_ratio**3)
            )
        )
        # d_ff_d_length = d_ff_d_f * d_f_d_length
        # d_ff_d_height = d_ff_d_f * d_f_d_height
        # d_ff_d_width = d_ff_d_f * d_f_d_width

        d_cd0_fus_d_length = (
            form_factor_fus
            / wing_area
            * (d_cf_fus_d_length * S_wet_fus + d_S_wet_fus_d_length * cf_fus)
        )

        # d_cd0_fus_d_height = cf_fus * wet_area_fus / wing_area * d_ff_d_height
        # d_cd0_fus_d_width = cf_fus * wet_area_fus / wing_area * d_ff_d_width
        d_cd0_fus_d_fineness = (
            cf_fus / wing_area * (d_ff_d_f * S_wet_fus + form_factor_fus * d_S_wet_fus_d_f)
        )
        d_cd0_fus_d_diam_mid = (
            form_factor_fus
            * cf_fus
            / wing_area
            * np.pi
            * length
            * (1 - 2 / fineness_ratio) ** (2 / 3)
            * (1 + 1 / (fineness_ratio**2))
        )
        d_cd0_fus_d_unit_re = form_factor_fus * S_wet_fus / wing_area * d_cf_fus_d_unit_re
        # d_cd0_fus_d_wet_area = form_factor_fus * cf_fus / wing_area
        d_cd0_fus_d_area = -form_factor_fus * cf_fus * S_wet_fus / wing_area**2.0

        if self.options["low_speed_aero"]:
            # partials[
            #     "data:aerodynamics:fuselage:low_speed:CD0", "data:geometry:fuselage:maximum_height"
            # ] = d_cd0_fus_d_height + d_cd0_w_d_height
            # partials[
            #     "data:aerodynamics:fuselage:low_speed:CD0", "data:geometry:fuselage:maximum_width"
            # ] = d_cd0_fus_d_width + d_cd0_w_d_width
            partials[
                "data:aerodynamics:fuselage:low_speed:CD0", "data:geometry:fuselage:fineness"
            ] = d_cd0_fus_d_fineness
            partials[
                "data:aerodynamics:fuselage:low_speed:CD0", "data:geometry:fuselage:length"
            ] = d_cd0_fus_d_length
            partials[
                "data:aerodynamics:fuselage:low_speed:CD0",
                "data:geometry:fuselage:diameter:mid",  # ,"data:geometry:fuselage:wet_area"
            ] = d_cd0_fus_d_diam_mid
            partials[
                "data:aerodynamics:fuselage:low_speed:CD0",
                "data:geometry:wing:surface",  # "data:geometry:wing:area"
            ] = d_cd0_fus_d_area  # + d_cd0_w_d_area
            partials[
                "data:aerodynamics:fuselage:low_speed:CD0",
                "data:aerodynamics:low_speed:unit_reynolds",
            ] = d_cd0_fus_d_unit_re
        else:
            # partials[
            #     "data:aerodynamics:fuselage:cruise:CD0", "data:geometry:fuselage:maximum_height"
            # ] = d_cd0_fus_d_height + d_cd0_w_d_height
            # partials[
            #     "data:aerodynamics:fuselage:cruise:CD0", "data:geometry:fuselage:maximum_width"
            # ] = d_cd0_fus_d_width + d_cd0_w_d_width
            partials["data:aerodynamics:fuselage:cruise:CD0", "data:geometry:fuselage:fineness"] = (
                d_cd0_fus_d_fineness
            )
            partials["data:aerodynamics:fuselage:cruise:CD0", "data:geometry:fuselage:length"] = (
                d_cd0_fus_d_length
            )
            partials[
                "data:aerodynamics:fuselage:cruise:CD0",
                "data:geometry:fuselage:diameter:mid",  # ,"data:geometry:fuselage:wet_area"
            ] = d_cd0_fus_d_diam_mid

            partials[
                "data:aerodynamics:fuselage:cruise:CD0",
                "data:geometry:wing:surface",  # "data:geometry:wing:area"
            ] = d_cd0_fus_d_area  # + d_cd0_w_d_area
            partials[
                "data:aerodynamics:fuselage:cruise:CD0",
                "data:aerodynamics:cruise:unit_reynolds",
            ] = d_cd0_fus_d_unit_re
