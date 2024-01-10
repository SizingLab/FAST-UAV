"""
Main module for propulsion system
"""
import fastoad.api as oad
import openmdao.api as om

from fastuav.models.propulsion.propeller.propeller import Propeller
from fastuav.models.propulsion.motor.motor import Motor
from fastuav.models.propulsion.gearbox.gearbox import Gearbox, NoGearbox
from fastuav.models.propulsion.energy.battery.battery import Battery
from fastuav.models.propulsion.esc.esc import ESC
from fastuav.constants import FW_PROPULSION, MR_PROPULSION
from fastuav.utils.configurations_versatility import promote_and_rename


@oad.RegisterOpenMDAOSystem("fastuav.propulsion")
class Propulsion(om.Group):
    """
    Group containing the propulsion system calculations
    """
    def initialize(self):
        self.options.declare("propulsion_id",
                             default=None,
                             values=[[MR_PROPULSION], [FW_PROPULSION], [MR_PROPULSION, FW_PROPULSION]])

        # TODO: declare the following options for each propulsion system (e.g. for hybrid UAVs with 2 propulsions)
        # TODO: add option to provide paths to catalogues
        self.options.declare("off_the_shelf_propeller", default=False, types=bool)
        self.options.declare("off_the_shelf_motor", default=False, types=bool)
        self.options.declare("off_the_shelf_battery", default=False, types=bool)
        self.options.declare("off_the_shelf_esc", default=False, types=bool)
        self.options.declare("gearbox", default=False, types=bool)

    def setup(self):
        off_the_shelf_propeller = self.options["off_the_shelf_propeller"]
        off_the_shelf_motor = self.options["off_the_shelf_motor"]
        off_the_shelf_battery = self.options["off_the_shelf_battery"]
        off_the_shelf_esc = self.options["off_the_shelf_esc"]
        gearbox = self.options["gearbox"]
        for propulsion_id in self.options["propulsion_id"]:
            propulsion = self.add_subsystem(propulsion_id,
                                            om.Group(),
                                            )
            propulsion.add_subsystem("propeller",
                                     Propeller(off_the_shelf=off_the_shelf_propeller),
                                     promotes=["*"])
            if gearbox:
                propulsion.add_subsystem("motor",
                                         Motor(off_the_shelf=off_the_shelf_motor),
                                         promotes=["*"])
                propulsion.add_subsystem("gearbox", Gearbox(), promotes=["*"])
            else:
                propulsion.add_subsystem("no_gearbox", NoGearbox(), promotes=["*"])
                propulsion.add_subsystem("motor",
                                         Motor(off_the_shelf=off_the_shelf_motor),
                                         promotes=["*"])
            propulsion.add_subsystem("battery",
                                     Battery(off_the_shelf=off_the_shelf_battery),
                                     promotes=["*"])
            propulsion.add_subsystem("esc",
                                     ESC(off_the_shelf=off_the_shelf_esc),
                                     promotes=["*"])

    def configure(self):
        for propulsion_id in self.options["propulsion_id"]:
            old_patterns_list = [":propulsion",
                                 "mission:sizing:main_route:climb:rate",
                                 "mission:sizing:main_route:climb:speed",
                                 "mission:sizing:main_route:cruise:speed",
                                 "mission:sizing:main_route:stall:speed",
                                 "mission:sizing:payload:power",
                                 ]
            new_patterns_list = [":propulsion:%s" % propulsion_id,
                                 "mission:sizing:main_route:climb:rate:%s" % propulsion_id,
                                 "mission:sizing:main_route:climb:speed:%s" % propulsion_id,
                                 "mission:sizing:main_route:cruise:speed:%s" % propulsion_id,
                                 "mission:sizing:main_route:stall:speed:%s" % propulsion_id,
                                 "mission:sizing:payload:power:%s" % propulsion_id,
                                 ]
            promote_and_rename(group=self,
                               subsys=getattr(self, propulsion_id),
                               old_patterns_list=old_patterns_list,
                               new_patterns_list=new_patterns_list)

