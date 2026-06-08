"""Estimation of the vertical tail profile drag."""
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

from ..constants import SUBMODEL_CD0_VT


@oad.RegisterSubmodel(SUBMODEL_CD0_VT, "fastga.submodel.aerodynamics.vertical_tail.cd0.legacy")
class Cd0VerticalTail(ExplicitComponent):
    """
    Profile drag estimation for the vertical tail

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    And :
    Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of Aeronautics and
    Astronautics, Inc., 2012.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:tail:vertical:tip:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:tail:vertical:root:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:tail:vertical:sweep_25", val=np.nan, units="deg")
        # self.add_input("data:geometry:vertical_tail:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:tail:vertical:surface", val=np.nan, units="m**2")
        # self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        # self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        # self.add_input("data:geometry:vertical_tail:max_thickness:x_ratio", val=0.3)
        self.add_input("data:geometry:tail:vertical:max_thickness:x_ratio", val=0.3)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")

            self.add_output("data:aerodynamics:vertical_tail:low_speed:CD0")

            self.declare_partials("*", "*", method="exact")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")

            self.add_output("data:aerodynamics:vertical_tail:cruise:CD0")

            self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        tip_chord = inputs["data:geometry:tail:vertical:tip:chord"]
        # root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        root_chord = inputs["data:geometry:tail:vertical:root:chord"]
        # sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        sweep_25_vt = inputs["data:geometry:tail:vertical:sweep_25"]
        # wet_area_vt = inputs["data:geometry:vertical_tail:wet_area"]
        area_vt = inputs["data:geometry:tail:vertical:surface"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]
        # thickness = inputs["data:geometry:vertical_tail:thickness_ratio"]
        thickness = inputs["data:geometry:wing:tc"] # Assuming the same thickness ratio than the wing
        # x_t_max = inputs["data:geometry:vertical_tail:max_thickness:x_ratio"]
        x_t_max = inputs["data:geometry:tail:vertical:max_thickness:x_ratio"]
        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        # Root: 50% NLF
        x_trans = 0.5
        x0_turbulent = 36.9 * x_trans**0.625 * (1 / (unit_reynolds * root_chord)) ** 0.375
        cf_root = (
            0.074 / (unit_reynolds * root_chord) ** 0.2 * (1 - (x_trans - x0_turbulent)) ** 0.8
        )
        # Tip: 50% NLF
        x_trans = 0.5
        x0_turbulent = 36.9 * x_trans**0.625 * (1 / (unit_reynolds * tip_chord)) ** 0.375
        cf_tip = 0.074 / (unit_reynolds * tip_chord) ** 0.2 * (1 - (x_trans - x0_turbulent)) ** 0.8
        # Global
        cf_vt = (cf_root + cf_tip) * 0.5
        form_factor = 1 + 0.6 / x_t_max * thickness + 100 * thickness**4
        form_factor = form_factor * 1.05  # Due to hinged elevator (Raymer)
        if mach > 0.2:
            form_factor = form_factor * 1.34 * mach**0.18 * (np.cos(sweep_25_vt)) ** 0.28
        interference_factor = 1.05
        wet_area_vt = 1.05 * 2 * area_vt
        cd0 = form_factor * interference_factor * cf_vt * wet_area_vt / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:vertical_tail:low_speed:CD0"] = cd0
        else:
            outputs["data:aerodynamics:vertical_tail:cruise:CD0"] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        tip_chord = inputs["data:geometry:tail:vertical:tip:chord"]
        # root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        root_chord = inputs["data:geometry:tail:vertical:root:chord"]
        # sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        sweep_25_vt = inputs["data:geometry:tail:vertical:sweep_25"]
        # wet_area_vt = inputs["data:geometry:vertical_tail:wet_area"]
        area_vt = inputs["data:geometry:tail:vertical:surface"]
        # wing_area = inputs["data:geometry:wing:area"]
        wing_area = inputs["data:geometry:wing:surface"]
        # thickness = inputs["data:geometry:vertical_tail:thickness_ratio"]
        thickness = inputs["data:geometry:wing:tc"] 
        # x_t_max = inputs["data:geometry:vertical_tail:max_thickness:x_ratio"]
        x_t_max = inputs["data:geometry:tail:vertical:max_thickness:x_ratio"]

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]

        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]
        wet_area_vt = 1.05 * 2 * area_vt
        x_trans = 0.5

        # Tip
        reynolds_tip = unit_reynolds * tip_chord
        x0_tip = 36.9 * x_trans**0.625 * reynolds_tip**-0.375

        d_x0_tip_d_unit_re = (
            -0.375 * 36.9 * x_trans**0.625 * tip_chord**-0.375 * unit_reynolds**-1.375
        )
        d_x0_tip_d_tip_chord = (
            -0.375 * 36.9 * x_trans**0.625 * unit_reynolds**-0.375 * tip_chord**-1.375
        )

        cf_vt_tip = 0.074 * reynolds_tip**-0.2 * (1.0 - (x_trans - x0_tip)) ** 0.8

        d_cf_vt_tip_d_unit_re = (
            -0.2 * 0.074 * tip_chord**-0.2 * unit_reynolds**-1.2 * (1.0 - (x_trans - x0_tip)) ** 0.8
        ) + 0.074 * reynolds_tip**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_tip)
        ) ** -0.2 * d_x0_tip_d_unit_re
        d_cf_vt_tip_d_chord_tip = (
            -0.2 * 0.074 * unit_reynolds**-0.2 * tip_chord**-1.2 * (1.0 - (x_trans - x0_tip)) ** 0.8
        ) + 0.074 * reynolds_tip**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_tip)
        ) ** -0.2 * d_x0_tip_d_tip_chord

        # Root
        reynolds_root = unit_reynolds * root_chord
        x_trans = 0.5
        x0_root = 36.9 * x_trans**0.625 * reynolds_root**-0.375

        d_x0_root_d_unit_re = (
            -0.375 * 36.9 * x_trans**0.625 * root_chord**-0.375 * unit_reynolds**-1.375
        )
        d_x0_root_d_root_chord = (
            -0.375 * 36.9 * x_trans**0.625 * unit_reynolds**-0.375 * root_chord**-1.375
        )

        cf_vt_root = 0.074 * reynolds_root**-0.2 * (1.0 - (x_trans - x0_root)) ** 0.8

        d_cf_vt_root_d_unit_re = (
            -0.2
            * 0.074
            * root_chord**-0.2
            * unit_reynolds**-1.2
            * (1.0 - (x_trans - x0_root)) ** 0.8
        ) + 0.074 * reynolds_root**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_root)
        ) ** -0.2 * d_x0_root_d_unit_re
        d_cf_vt_root_d_chord_root = (
            -0.2
            * 0.074
            * unit_reynolds**-0.2
            * root_chord**-1.2
            * (1.0 - (x_trans - x0_root)) ** 0.8
        ) + 0.074 * reynolds_root**-0.2 * 0.8 * (
            1.0 - (x_trans - x0_root)
        ) ** -0.2 * d_x0_root_d_root_chord

        cf_vt = (cf_vt_root + cf_vt_tip) * 0.5
        d_cf_vt_d_unit_re = 0.5 * (d_cf_vt_root_d_unit_re + d_cf_vt_tip_d_unit_re)
        d_cf_vt_d_chord_tip = 0.5 * d_cf_vt_tip_d_chord_tip
        c_cf_vt_d_chord_root = 0.5 * d_cf_vt_root_d_chord_root

        form_factor = 1.05 * (1 + 0.6 / x_t_max * thickness + 100 * thickness**4.0)

        d_ff_d_location = -1.05 * 0.6 / x_t_max**2.0 * thickness
        d_ff_d_thickness = 1.05 * (0.6 / x_t_max + 4.0 * 100 * thickness**3.0)

        if mach > 0.2:
            mach_correction = 1.34 * mach**0.18 * (np.cos(sweep_25_vt)) ** 0.28
            d_mach_correction_d_mach = 0.18 * 1.34 * mach**-0.82 * (np.cos(sweep_25_vt)) ** 0.28
            d_mach_correction_d_sweep = (
                -0.28 * 1.34 * mach**0.18 * (np.cos(sweep_25_vt)) ** -0.72 * np.sin(sweep_25_vt)
            )
        else:
            mach_correction = 1.0
            d_mach_correction_d_mach = 0.0
            d_mach_correction_d_sweep = 0.0

        interference_factor = 1.05

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:geometry:tail:vertical:tip:chord",#"data:geometry:vertical_tail:tip:chord",
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * d_cf_vt_d_chord_tip
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:geometry:tail:vertical:root:chord",#"data:geometry:vertical_tail:root:chord"
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * c_cf_vt_d_chord_root
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:geometry:tail:vertical:sweep_25",#"data:geometry:vertical_tail:sweep_25"
            ] = (
                form_factor
                * d_mach_correction_d_sweep
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:surface",#"data:geometry:vertical_tail:wet_area"
            ] = form_factor * mach_correction * interference_factor * cf_vt * 1.05 * 2 / wing_area
            # partials["data:aerodynamics:vertical_tail:low_speed:CD0", "data:geometry:wing:area"] = (
            partials["data:aerodynamics:vertical_tail:low_speed:CD0", "data:geometry:wing:surface"] = (
                -form_factor
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:geometry:wing:tc",#"data:geometry:vertical_tail:thickness_ratio"
            ] = (
                d_ff_d_thickness
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:geometry:tail:vertical:max_thickness:x_ratio",#"data:geometry:vertical_tail:max_thickness:x_ratio",
            ] = (
                d_ff_d_location
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0", "data:aerodynamics:low_speed:mach"
            ] = (
                form_factor
                * d_mach_correction_d_mach
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:CD0",
                "data:aerodynamics:low_speed:unit_reynolds",
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * d_cf_vt_d_unit_re
                * wet_area_vt
                / wing_area
            )

        else:
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:tip:chord",#"data:geometry:vertical_tail:tip:chord"
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * d_cf_vt_d_chord_tip
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:root:chord",#"data:geometry:vertical_tail:root:chord",
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * c_cf_vt_d_chord_root
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:sweep_25",#"data:geometry:vertical_tail:sweep_25"
            ] = (
                form_factor
                * d_mach_correction_d_sweep
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:surface",#"data:geometry:vertical_tail:wet_area"
            ] = form_factor * mach_correction * interference_factor * cf_vt * 1.05 * 2 / wing_area
            # partials["data:aerodynamics:vertical_tail:cruise:CD0", "data:geometry:wing:area"] = (
            partials["data:aerodynamics:vertical_tail:cruise:CD0", "data:geometry:wing:surface"] = (
                -form_factor
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:wing:tc",#"data:geometry:vertical_tail:thickness_ratio"
            ] = (
                d_ff_d_thickness
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:geometry:tail:vertical:max_thickness:x_ratio",#"data:geometry:vertical_tail:max_thickness:x_ratio",
            ] = (
                d_ff_d_location
                * mach_correction
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0", "data:aerodynamics:cruise:mach"
            ] = (
                form_factor
                * d_mach_correction_d_mach
                * interference_factor
                * cf_vt
                * wet_area_vt
                / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:CD0",
                "data:aerodynamics:cruise:unit_reynolds",
            ] = (
                form_factor
                * mach_correction
                * interference_factor
                * d_cf_vt_d_unit_re
                * wet_area_vt
                / wing_area
            )
