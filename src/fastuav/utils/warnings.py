"""A module for FAST-OAD-GA specific warnings."""
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


class FASTOADGAWarning(UserWarning):
    """Base class for FAST-OAD-GA warning."""

    name = "warn_fast_oad_ga"


class VariableDescriptionWarning(FASTOADGAWarning):
    """Warning class for warnings in the generation of variable_descriptions.txt"""

    name = "warn_variable_descriptions"
