"""
Group containing the computation of the aircraft profile drag based on the sum of the drag of
its components.
"""
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
from openmdao.core.group import Group

from ..constants import (
    SUBMODEL_CD0,
    SUBMODEL_CD0_FUSELAGE,
    SUBMODEL_CD0_HT,
    # SUBMODEL_CD0_NACELLE,
    SUBMODEL_CD0_LANDING_GEAR,
    SUBMODEL_CD0_OTHER,
    SUBMODEL_CD0_SUM,
    SUBMODEL_CD0_VT,
    SUBMODEL_CD0_WING,
)


@oad.RegisterSubmodel(SUBMODEL_CD0, "fastga.submodel.aerodynamics.aircraft.cd0.legacy")
class Cd0(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        # self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default="naca23012.af", types=str, allow_none=True
        )
        self.options.declare("htp_airfoil_file", default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        options_wing = {
            "airfoil_folder_path": self.options["airfoil_folder_path"],
            "low_speed_aero": self.options["low_speed_aero"],
            "wing_airfoil_file": self.options["wing_airfoil_file"],
        }
        self.add_subsystem(
            "cd0_wing",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_WING, options=options_wing),
            promotes=["*"],
        )

        low_speed_option = {"low_speed_aero": self.options["low_speed_aero"]}
        self.add_subsystem(
            "cd0_fuselage",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_FUSELAGE, options=low_speed_option),
            promotes=["*"],
        )

        options_htp = {
            "airfoil_folder_path": self.options["airfoil_folder_path"],
            "low_speed_aero": self.options["low_speed_aero"],
            "htp_airfoil_file": self.options["htp_airfoil_file"],
        }
        self.add_subsystem(
            "cd0_ht",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_HT, options=options_htp),
            promotes=["*"],
        )

        self.add_subsystem(
            "cd0_vt",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_VT, options=low_speed_option),
            promotes=["*"],
        )

        # options_nacelle = {
        #     "low_speed_aero": self.options["low_speed_aero"],
        #     "propulsion_id": self.options["propulsion_id"],
        # }
        # self.add_subsystem(
        #     "cd0_nacelle",
        #     oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_NACELLE, options=options_nacelle),
        #     promotes=["*"],
        # )

        self.add_subsystem(
            "cd0_l_gear",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_LANDING_GEAR, options=low_speed_option),
            promotes=["*"],
        )

        self.add_subsystem(
            "cd0_other",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_OTHER, options=low_speed_option),
            promotes=["*"],
        )

        self.add_subsystem(
            "cd0_total",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0_SUM, options=low_speed_option),
            promotes=["*"],
        )
