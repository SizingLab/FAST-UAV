"""
Main module for wires component.
Wires are used to connect the battery to the ESCs, and the ESCs to the motors.
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.constants import FW_PROPULSION, MR_PROPULSION, PROPULSION_ID_LIST
from fastuav.utils.configurations_versatility import promote_and_rename


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.wires")
class Wires(om.Group):
    """
    Group containing the wires MDA
    """

    def initialize(self):
        self.options.declare("propulsion_id",
                             default=[MR_PROPULSION],
                             values=[[MR_PROPULSION], [FW_PROPULSION], [MR_PROPULSION, FW_PROPULSION]])

    def setup(self):
        for propulsion_id in self.options["propulsion_id"]:
            wires = self.add_subsystem(propulsion_id,
                                       om.Group(),
                                       )
            wires.add_subsystem("radius", Radius(), promotes=["*"])
            wires.add_subsystem("length", Length(propulsion_id=propulsion_id), promotes=["*"])
            wires.add_subsystem("weight", Weight(), promotes=["*"])

    def configure(self):
        for propulsion_id in self.options["propulsion_id"]:
            old_patterns_list = [":propeller", ":motor", ":battery", ":wires"]
            new_patterns_list = [":" + propulsion_id + varname for varname in old_patterns_list]
            promote_and_rename(group=self,
                               subsys=getattr(self, propulsion_id),
                               old_patterns_list=old_patterns_list,
                               new_patterns_list=new_patterns_list)


class Radius(om.ExplicitComponent):
    """
    Computes wires radius.
    The design driver for the wires is the maximal insulator temperature,
    such that the cross-section area is selected from the nominal current.
    By default, assumption is made that the ESCs are located as close as possible to the battery,
    such that only the wires connecting the ESCs to the motors are to be sized, from the motors' nominal current.
    """

    def initialize(self):
        # Choose which current is used for sizing the wire
        self.options.declare("sizing_component", default="motor", values=["motor", "battery"])

    def setup(self):
        sizing_component = self.options["sizing_component"]
        self.add_input("data:propulsion:wires:radius:reference", val=np.nan, units="m")
        self.add_input("data:propulsion:wires:current:reference", val=np.nan, units="A")
        self.add_input("data:propulsion:%s:current:cruise" % sizing_component, val=np.nan, units="A")
        self.add_output("data:propulsion:wires:radius", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        sizing_component = self.options["sizing_component"]
        r_ref = inputs["data:propulsion:wires:radius:reference"]  # [m] radius of reference wire
        I_ref = inputs["data:propulsion:wires:current:reference"]  # [A] nominal current of reference wire
        I = inputs["data:propulsion:%s:current:cruise" % sizing_component]  # [A] nominal current

        r = r_ref * (I / I_ref) ** (2 / 3)  # [m] radius of wire

        outputs["data:propulsion:wires:radius"] = r


class Length(om.ExplicitComponent):
    """
    Sets number of wires and their length.
    Assumption is made that the ESCs are located as close to the battery as possible.
    Consequently, only the wires connecting the ESCs to the motors are sized.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        if propulsion_id == MR_PROPULSION:
            self.add_input("data:geometry:arms:length", val=np.nan, units="m")
        elif propulsion_id == FW_PROPULSION:
            self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_output("data:propulsion:wires:number", units=None)
        self.add_output("data:propulsion:wires:length", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        L_wir = .0
        if propulsion_id == MR_PROPULSION:
            L_wir = inputs["data:geometry:arms:length"]
        elif propulsion_id == FW_PROPULSION:
            L_wir = inputs["data:geometry:fuselage:length"] / 2
        N_pro = inputs["data:propulsion:propeller:number"]
        N_wir = 3 * N_pro  # 3 wires are required to connect each motor to esc

        outputs["data:propulsion:wires:number"] = N_wir  # total number of wires
        outputs["data:propulsion:wires:length"] = L_wir  # length of a single wire

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        if propulsion_id == MR_PROPULSION:
            partials["data:propulsion:wires:length",
                     "data:geometry:arms:length"] = 1.0
        elif propulsion_id == FW_PROPULSION:
            partials["data:propulsion:wires:length",
                     "data:geometry:fuselage:length"] = 0.5
        partials["data:propulsion:wires:number",
                 "data:propulsion:propeller:number"] = 3.0


class Weight(om.ExplicitComponent):
    """
    Computes wires weight
    """

    def setup(self):
        self.add_input("data:weight:propulsion:wires:density:reference", val=np.nan, units="kg/m")
        self.add_input("data:propulsion:wires:radius:reference", val=np.nan, units="m")
        self.add_input("data:propulsion:wires:radius", val=np.nan, units="m")
        self.add_input("data:propulsion:wires:number", val=np.nan, units=None)
        self.add_input("data:propulsion:wires:length", val=np.nan, units="m")
        self.add_output("data:weight:propulsion:wires:density", units="kg/m")
        self.add_output("data:weight:propulsion:wires:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        mu_ref = inputs["data:weight:propulsion:wires:density:reference"]  # [kg/m] linear mass of reference cable
        r_ref = inputs["data:propulsion:wires:radius:reference"]  # [m] radius of reference wire
        r = inputs["data:propulsion:wires:radius"]
        N_wir = inputs["data:propulsion:wires:number"]
        L_wir = inputs["data:propulsion:wires:length"]

        mu = mu_ref * (r / r_ref) ** 2  # [kg/m] linear mass of cable
        m_wir = mu * L_wir * N_wir  # [kg] mass of wires

        outputs["data:weight:propulsion:wires:density"] = mu
        outputs["data:weight:propulsion:wires:mass"] = m_wir

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mu_ref = inputs["data:weight:propulsion:wires:density:reference"]  # [kg/m] linear mass of reference cable
        r_ref = inputs["data:propulsion:wires:radius:reference"]  # [m] radius of reference wire
        r = inputs["data:propulsion:wires:radius"]
        N_wir = inputs["data:propulsion:wires:number"]
        L_wir = inputs["data:propulsion:wires:length"]
        mu = mu_ref * (r / r_ref) ** 2

        partials["data:weight:propulsion:wires:density",
                 "data:propulsion:wires:radius"] = 2 * mu_ref / r_ref ** 2 * r
        partials["data:weight:propulsion:wires:density",
                 "data:weight:propulsion:wires:density:reference"] = (r / r_ref) ** 2
        partials["data:weight:propulsion:wires:density",
                 "data:propulsion:wires:radius:reference"] = -2 * mu_ref * r ** 2 / r_ref ** 3
        partials["data:weight:propulsion:wires:mass",
                 "data:propulsion:wires:radius"] = 2 * mu_ref / r_ref ** 2 * r * L_wir * N_wir
        partials["data:weight:propulsion:wires:mass",
                 "data:weight:propulsion:wires:density:reference"] = (r / r_ref) ** 2 * L_wir * N_wir
        partials["data:weight:propulsion:wires:mass",
                 "data:propulsion:wires:radius:reference"] = -2 * mu_ref * r ** 2 / r_ref ** 3 * L_wir * N_wir
        partials["data:weight:propulsion:wires:mass",
                 "data:propulsion:wires:number"] = mu * L_wir
        partials["data:weight:propulsion:wires:mass",
                 "data:propulsion:wires:length"] = mu * N_wir
