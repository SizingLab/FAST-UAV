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
from fastuav.utils.constants import PROPULSION_ID_LIST
from stdatm import AtmosphereSI
import logging

_LOGGER = logging.getLogger(__name__)  # Logger for this module


class PropellerEstimationModels(om.Group):
    """
    Group containing the estimation models for the propeller.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        add_model_deviation(
            self,
            "aerodynamics_model_deviation",
            PropellerAerodynamicsModel,
            uncertain_parameters=[
                "propeller:aerodynamics:CT:static",
                "propeller:aerodynamics:CP:static",
                "propeller:aerodynamics:CT:axial",
                "propeller:aerodynamics:CP:axial",
                "propeller:aerodynamics:CT:incidence",
                "propeller:aerodynamics:CP:incidence",
            ],
        )
        # self.add_subsystem("aerodynamics", Aerodynamics(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "diameter",
            Diameter(propulsion_id=propulsion_id),
            uncertain_outputs={"data:propulsion:propeller:diameter:estimated": "m"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weights:propulsion:propeller:mass:estimated": "kg"},
        )


class Diameter(om.ExplicitComponent):
    """
    Computes propeller diameter from the takeoff scenario.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, values=PROPULSION_ID_LIST)

    def setup(self):
        self.add_input("data:propulsion:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("data:scenarios:takeoff:altitude", val=0.0, units="m")
        self.add_input("data:scenarios:dISA", val=0.0, units="K")
        self.add_input("data:propulsion:propeller:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:ND:takeoff", val=np.nan, units="m/s")
        self.add_output("data:propulsion:propeller:diameter:estimated", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        F_pro_to = inputs["data:propulsion:propeller:thrust:takeoff"]
        ND_to = inputs["data:propulsion:propeller:ND:takeoff"]
        beta = inputs["data:propulsion:propeller:beta:estimated"]

        altitude_takeoff = inputs["data:scenarios:takeoff:altitude"]
        dISA = inputs["data:scenarios:dISA"]
        rho_air = AtmosphereSI(
            altitude_takeoff, dISA
        ).density  # [kg/m3] Air density at takeoff level

        C_t, C_p = PropellerAerodynamicsModel.aero_coefficients_static(beta, propulsion_id=propulsion_id)

        Dpro = (F_pro_to / (C_t * rho_air * ND_to**2)) ** 0.5  # [m] Propeller diameter

        outputs["data:propulsion:propeller:diameter:estimated"] = Dpro


class Weight(om.ExplicitComponent):
    """
    Computes propeller weight
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:diameter:estimated", val=np.nan, units="m")
        self.add_input("data:propulsion:propeller:diameter:reference", val=np.nan, units="m")
        self.add_input("data:weights:propulsion:propeller:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weights:propulsion:propeller:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propulsion:propeller:diameter:estimated"]
        Dpro_ref = inputs["data:propulsion:propeller:diameter:reference"]
        Mpro_ref = inputs["data:weights:propulsion:propeller:mass:reference"]

        Mpro = Mpro_ref * (Dpro / Dpro_ref) ** 3  # [kg] Propeller mass

        outputs["data:weights:propulsion:propeller:mass:estimated"] = Mpro


