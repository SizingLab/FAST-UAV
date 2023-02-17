"""
Calculations of the maximum range and endurance based on the sizing scenarios parameters.
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.constants import MR_PROPULSION, FW_PROPULSION, PROPULSION_ID_LIST, HOVER_TAG, CRUISE_TAG


@oad.RegisterOpenMDAOSystem("fastuav.performance.endurance.multirotor")
class EnduranceMultirotor(om.Group):
    """
    Endurance and range calculations for multirotor UAVs
    """

    def setup(self):
        self.add_subsystem("hover",
                           Endurance(propulsion_id=MR_PROPULSION, phase_name=HOVER_TAG),
                           promotes=["*"])
        self.add_subsystem("cruise",
                           Endurance(propulsion_id=MR_PROPULSION, phase_name=CRUISE_TAG),
                           promotes=["*"])


@oad.RegisterOpenMDAOSystem("fastuav.performance.endurance.fixedwing")
class EnduranceFixedWing(om.Group):
    """
    Endurance and range calculations for multirotor UAVs
    """

    def setup(self):
        self.add_subsystem("cruise",
                           Endurance(propulsion_id=FW_PROPULSION, phase_name=CRUISE_TAG),
                           promotes=["*"])


@oad.RegisterOpenMDAOSystem("fastuav.performance.endurance.hybrid")
class EnduranceHybrid(om.Group):
    """
    Endurance and range calculations for hybrid (fixed wing VTOL) UAVs
    """

    def setup(self):
        self.add_subsystem("hover",
                           Endurance(propulsion_id=MR_PROPULSION, phase_name=HOVER_TAG),
                           promotes=["*"])
        self.add_subsystem("cruise",
                           Endurance(propulsion_id=FW_PROPULSION, phase_name=CRUISE_TAG),
                           promotes=["*"])


class Endurance(om.ExplicitComponent):
    """
    Endurance and range calculations for given flight scenario (at design payload).
    """

    def initialize(self):
        self.options.declare("propulsion_id",
                             default=FW_PROPULSION, values=PROPULSION_ID_LIST)
        self.options.declare("phase_name",
                             default=HOVER_TAG, values=[HOVER_TAG, CRUISE_TAG])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        phase_name = self.options["phase_name"]
        self.add_input("data:propulsion:%s:battery:capacity" % propulsion_id, val=np.nan, units="A*s")
        self.add_input("data:propulsion:%s:battery:DoD:max" % propulsion_id, val=0.8, units=None)
        self.add_input("data:propulsion:%s:battery:current:%s" % (propulsion_id, phase_name), val=np.nan,
                       units="A")
        if phase_name != HOVER_TAG:
            self.add_input("mission:sizing:main_route:%s:speed:%s" % (phase_name, propulsion_id), val=0.0, units="m/s")
            self.add_output("data:performance:range:%s" % phase_name, units="m")
        self.add_output("data:performance:endurance:%s" % phase_name, units="min")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        phase_name = self.options["phase_name"]
        C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
        C_bat = inputs["data:propulsion:%s:battery:capacity" % propulsion_id]
        I_bat = inputs["data:propulsion:%s:battery:current:%s" % (propulsion_id, phase_name)]

        # Endurance calculation
        t_max = C_ratio * C_bat / I_bat if I_bat > 0 else 0.0  # [s] Max. cruise flight time at design payload

        # Range calculation
        if phase_name != HOVER_TAG:
            V = inputs["mission:sizing:main_route:%s:speed:%s" % (phase_name, propulsion_id)]
            D_max = V * t_max  # [m] Max. Range at given cruise speed and design payload
            outputs["data:performance:range:%s" % phase_name] = D_max  # [m]

        outputs["data:performance:endurance:%s" % phase_name] = t_max / 60.0  # [min]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        phase_name = self.options["phase_name"]
        C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
        C_bat = inputs["data:propulsion:%s:battery:capacity" % propulsion_id]
        I_bat = inputs["data:propulsion:%s:battery:current:%s" % (propulsion_id, phase_name)]
        t_max = C_ratio * C_bat / I_bat if I_bat > 0 else 0.0

        partials["data:performance:endurance:%s" % phase_name,
                 "data:propulsion:%s:battery:DoD:max" % propulsion_id] = C_bat / I_bat / 60.0 if I_bat > 0 else 0.0
        partials["data:performance:endurance:%s" % phase_name,
                 "data:propulsion:%s:battery:capacity" % propulsion_id] = C_ratio / I_bat / 60.0 if I_bat > 0 else 0.0
        partials[
            "data:performance:endurance:%s" % phase_name,
            "data:propulsion:%s:battery:current:%s" % (propulsion_id, phase_name)
        ] = -C_ratio * C_bat / I_bat**2 / 60.0

        if phase_name != HOVER_TAG:
            V = inputs["mission:sizing:main_route:%s:speed:%s" % (phase_name, propulsion_id)]
            partials["data:performance:range:%s" % phase_name,
                     "mission:sizing:main_route:%s:speed:%s" % (phase_name, propulsion_id)] = t_max
            partials["data:performance:range:%s" % phase_name,
                     "data:propulsion:%s:battery:DoD:max" % propulsion_id] = C_bat / I_bat * V if I_bat > 0 else 0.0
            partials["data:performance:range:%s" % phase_name,
                     "data:propulsion:%s:battery:capacity" % propulsion_id] = C_ratio / I_bat * V if I_bat > 0 else 0.0
            partials["data:performance:range:%s" % phase_name,
                     "data:propulsion:%s:battery:current:%s" % (propulsion_id, phase_name)] = -V * C_ratio * C_bat / I_bat**2 if I_bat > 0 else 0.0


