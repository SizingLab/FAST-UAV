"""
MR Drone problem
"""
import openmdao.api as om
from Propeller.propeller import PropellerMR
from Motor.motor import MotorMR
from Battery.battery import Battery
from ESC.esc import ESC
from Structure.structure import StructureMR

p = om.Problem()
model = p.model
model.add_subsystem('propeller', PropellerMR(),
                    promotes=["*"])
model.add_subsystem('motor', MotorMR(),
                    promotes=["*"])
model.add_subsystem('battery', Battery(),
                    promotes=["*"])
model.add_subsystem('esc', ESC(),
                    promotes=["*"])
model.add_subsystem('structure', StructureMR(),
                    promotes=["*"])
p.setup()
p.check_config(checks=['unconnected_inputs'], out_file=None)
p.run_model()