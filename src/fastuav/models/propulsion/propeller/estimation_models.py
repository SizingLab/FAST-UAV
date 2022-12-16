"""
Estimation models for the propeller
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import (
    add_subsystem_with_deviation,
    add_model_deviation,
)
from fastuav.models.propulsion.propeller.aerodynamics.surrogate_models import PropellerAerodynamicsModel
from stdatm import AtmosphereSI
import logging

_LOGGER = logging.getLogger(__name__)  # Logger for this module


class PropellerEstimationModels(om.Group):
    """
    Group containing the estimation models for the propeller.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "diameter",
            Diameter(),
            uncertain_outputs={"data:propulsion:propeller:diameter:estimated": "m"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weight:propulsion:propeller:mass:estimated": "kg"},
        )


class Diameter(om.ExplicitComponent):
    """
    Computes propeller diameter from the takeoff scenario.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("mission:sizing:main_route:takeoff:altitude", val=0.0, units="m")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:propulsion:propeller:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:model:static:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:model:static:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:ND:takeoff", val=np.nan, units="m/s")
        self.add_output("data:propulsion:propeller:diameter:estimated", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_to = inputs["data:propulsion:propeller:thrust:takeoff"]
        ND_to = inputs["data:propulsion:propeller:ND:takeoff"]
        beta = inputs["data:propulsion:propeller:beta:estimated"]
        ct_model = inputs["data:propulsion:propeller:Ct:model:static:estimated"]
        cp_model = inputs["data:propulsion:propeller:Cp:model:static:estimated"]

        altitude_takeoff = inputs["mission:sizing:main_route:takeoff:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        rho_air = AtmosphereSI(
            altitude_takeoff, dISA
        ).density  # [kg/m3] Air density at takeoff level

        c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_static(beta,
                                                                       ct_model=ct_model,
                                                                       cp_model=cp_model)

        Dpro = (F_pro_to / (c_t * rho_air * ND_to**2)) ** 0.5  # [m] Propeller diameter

        outputs["data:propulsion:propeller:diameter:estimated"] = Dpro


class Weight(om.ExplicitComponent):
    """
    Computes propeller weight
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:diameter:estimated", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:diameter:reference", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:propeller:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weight:propulsion:propeller:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propulsion:propeller:diameter:estimated"]
        Dpro_ref = inputs["data:propulsion:propeller:diameter:reference"]
        m_pro_ref = inputs["data:weight:propulsion:propeller:mass:reference"]

        m_pro = m_pro_ref * (Dpro / Dpro_ref) ** 3  # [kg] Propeller mass

        outputs["data:weight:propulsion:propeller:mass:estimated"] = m_pro


