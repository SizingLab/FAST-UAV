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
from stdatm import AtmosphereSI, AtmosphereWithPartials
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

        self.add_subsystem("figure_of_merit", FigureOfMerit(), promotes=["*"])


class Diameter(om.ExplicitComponent):
    """
    Computes propeller diameter from the takeoff scenario.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("mission:sizing:main_route:takeoff:altitude", val=0.0, units="m")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:propulsion:propeller:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:static:polynomial:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:static:polynomial:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:ND:takeoff", val=np.nan, units="m/s")
        self.add_output("data:propulsion:propeller:diameter:estimated", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        F_pro_to = inputs["data:propulsion:propeller:thrust:takeoff"]
        ND_to = inputs["data:propulsion:propeller:ND:takeoff"]
        beta = inputs["data:propulsion:propeller:beta:estimated"]
        ct_model = inputs["data:propulsion:propeller:Ct:static:polynomial:estimated"]
        cp_model = inputs["data:propulsion:propeller:Cp:static:polynomial:estimated"]

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

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        out = "data:propulsion:propeller:diameter:estimated"
        F        = inputs["data:propulsion:propeller:thrust:takeoff"]
        ND       = inputs["data:propulsion:propeller:ND:takeoff"]
        beta     = inputs["data:propulsion:propeller:beta:estimated"]
        ct_model = inputs["data:propulsion:propeller:Ct:static:polynomial:estimated"]
        cp_model = inputs["data:propulsion:propeller:Cp:static:polynomial:estimated"]
        altitude = inputs["mission:sizing:main_route:takeoff:altitude"]
        dISA     = inputs["mission:sizing:dISA"]

        atm = AtmosphereSI(altitude, dISA)
        rho, T = atm.density, atm.temperature
        c_t, _ = PropellerAerodynamicsModel.aero_coefficients_static(beta, ct_model=ct_model, cp_model=cp_model)
        D = (F / (c_t * rho * ND**2)) ** 0.5

        # Power-law base partials (D is a monomial: exponent * D / variable)
        dD_dF   =  0.5 * D / F
        dD_dct  = -0.5 * D / c_t
        dD_drho = -0.5 * D / rho
        dD_dND  = -1.0 * D / ND

        # Coefficient derivatives from the model class
        dct, _ = PropellerAerodynamicsModel.aero_coefficients_static_derivatives(beta, ct_model, cp_model)

        # Density derivatives (stdatm; dISA held at fixed pressure -> -rho/T)
        drho_dh = AtmosphereWithPartials(altitude, dISA, altitude_in_feet=False).partial_density_altitude

        partials[out, "data:propulsion:propeller:thrust:takeoff"] = dD_dF
        partials[out, "data:propulsion:propeller:ND:takeoff"]     = dD_dND
        partials[out, "data:propulsion:propeller:beta:estimated"] = dD_dct * dct["dbeta"]
        partials[out, "data:propulsion:propeller:Ct:static:polynomial:estimated"] = dD_dct * dct["dmodel"]
        partials[out, "data:propulsion:propeller:Cp:static:polynomial:estimated"] = np.zeros((1, cp_model.size))
        partials[out, "mission:sizing:main_route:takeoff:altitude"] = dD_drho * drho_dh
        partials[out, "mission:sizing:dISA"]                        = dD_drho * (-rho / T)    


class Weight(om.ExplicitComponent):
    """
    Computes propeller weight
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:diameter:estimated", val=np.nan, units="m")
        self.add_input("models:propulsion:propeller:diameter:reference", val=np.nan, units="m")
        self.add_input("models:weight:propulsion:propeller:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weight:propulsion:propeller:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propulsion:propeller:diameter:estimated"]
        Dpro_ref = inputs["models:propulsion:propeller:diameter:reference"]
        m_pro_ref = inputs["models:weight:propulsion:propeller:mass:reference"]

        m_pro = m_pro_ref * (Dpro / Dpro_ref) ** 3  # [kg] Propeller mass

        outputs["data:weight:propulsion:propeller:mass:estimated"] = m_pro
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        out = "data:weight:propulsion:propeller:mass:estimated"
        Dpro = inputs["data:propulsion:propeller:diameter:estimated"]
        Dpro_ref = inputs["models:propulsion:propeller:diameter:reference"]
        m_pro_ref = inputs["models:weight:propulsion:propeller:mass:reference"]

        partials[out, "data:propulsion:propeller:diameter:estimated"] = 3 * m_pro_ref * Dpro** 2 / Dpro_ref ** 3
        partials[out, "models:propulsion:propeller:diameter:reference"] = -3 * m_pro_ref * Dpro** 3 / Dpro_ref ** 4
        partials[out, "models:weight:propulsion:propeller:mass:reference"] = Dpro** 3 / Dpro_ref ** 3


class FigureOfMerit(om.ExplicitComponent):
    """
    Computes figure of merit of propeller.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:beta:estimated", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Ct:static:polynomial:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:Cp:static:polynomial:estimated", shape_by_conn=True, val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:FoM:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        beta = inputs["data:propulsion:propeller:beta:estimated"]
        ct_model = inputs["data:propulsion:propeller:Ct:static:polynomial:estimated"]
        cp_model = inputs["data:propulsion:propeller:Cp:static:polynomial:estimated"]

        c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_static(beta,
                                                                       ct_model=ct_model,
                                                                       cp_model=cp_model)

        FoM = c_t ** (3/2) / c_p

        outputs["data:propulsion:propeller:FoM:estimated"] = FoM

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        out = "data:propulsion:propeller:FoM:estimated"
        beta     = inputs["data:propulsion:propeller:beta:estimated"]
        ct_model = inputs["data:propulsion:propeller:Ct:static:polynomial:estimated"]
        cp_model = inputs["data:propulsion:propeller:Cp:static:polynomial:estimated"]

        c_t, c_p = PropellerAerodynamicsModel.aero_coefficients_static(beta, ct_model=ct_model, cp_model=cp_model)
        FoM = c_t ** 1.5 / c_p

        # Monomial base partials: exponent * FoM / coefficient
        dFoM_dct = 1.5 * FoM / c_t
        dFoM_dcp = -1.0 * FoM / c_p

        # Coefficient derivatives from the model class (both c_t and c_p here)
        dct, dcp = PropellerAerodynamicsModel.aero_coefficients_static_derivatives(beta, ct_model, cp_model)

        partials[out, "data:propulsion:propeller:beta:estimated"] = (
            dFoM_dct * dct["dbeta"] + dFoM_dcp * dcp["dbeta"]
        )
        partials[out, "data:propulsion:propeller:Ct:static:polynomial:estimated"] = dFoM_dct * dct["dmodel"]
        partials[out, "data:propulsion:propeller:Cp:static:polynomial:estimated"] = dFoM_dcp * dcp["dmodel"]




