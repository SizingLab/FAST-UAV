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
#from fastuav.models.Control import doc_multicopter

@RegisterOpenMDAOSystem("fastuav.plugin.DOC")
class SampleDiscipline(om.ExplicitComponent):
    """
    Sample discipline to give an example of how to register a custom module.
    """

    def setup(self):
        self.add_input("Coaxiality", val=0, units="kg")
        self.add_input("RotorNumber", val=4, units="kg")
        self.add_input("Force", val=6, units="N")
        self.add_input("Distance", val=0.24, units="kg")
        self.add_input("TotalMass", val=2, units="kg")
        self.add_input("MotorMass", val=0.03, units="kg")
        self.add_input("PropellerMass", val=0.015, units="kg")
        self.add_input("Time", val=1, units="s")
        self.add_input("TimeSteps", val=2, units="s")

        self.add_output("sample_output", units="kg")

        # self.engine = matlab.engine.start_matlab()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        coaxiality = inputs["Coaxiality"]
        RotorNumber = inputs["RotorNumber"]
        Force = inputs["Force"]
        Distance = inputs["Distance"]
        TotalMass = inputs["TotalMass"]
        MotorMass = inputs["MotorMass"]
        PropellerMass = inputs["PropellerMass"]
        Time = inputs["Time"]
        TimeSteps = inputs["TimeSteps"]

        eng = matlab.engine.start_matlab()
        DOC = eng.doc_multicopter_2(coaxiality[0], RotorNumber[0], Force[0], Distance[0], TotalMass[0], MotorMass[0], PropellerMass[0], Time[0], TimeSteps[0])
        #eng.simple_script(nargout=0)
        outputs["sample_output"] = DOC
        eng.quit()