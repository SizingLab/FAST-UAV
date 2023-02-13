"""
FAST - Copyright (c) 2016 ONERA ISAE.
"""

"""
DOC Controllability Computation Module - Concordia University, Robin Warren
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

import openmdao.api as om
import numpy as np
import matlab.engine
"requires matlab API installation from matlab 2022b"
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem

@RegisterOpenMDAOSystem("fastuav.plugin.DOC")
class SampleDiscipline(om.ExplicitComponent):
    """
    Sample discipline to give an example of how to register a custom module.
    """

    def setup(self):
        self.add_input("sample_input", val=np.nan, units="kg")

        self.add_output("sample_output", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        eng = matlab.engine.start_matlab()
        outputs["sample_output"] = eng.doc_multicopter(0,4,6,0.28,2,0.03,0.015,inputs["sample_input"],2)
        eng.quit()
