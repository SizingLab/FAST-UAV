"""
LCA postprocessing calculations
"""

import openmdao.api as om
import numpy as np
from scipy.constants import g
from stdatm import AtmosphereSI
from fastuav.constants import LCA_RESULT_KEY, LCA_POSTPROCESS_KEY, LCA_DEFAULT_METHOD
from fastuav.models.life_cycle_assessment.lca_core import LCAcalc


class SpecificComponentContributions(om.ExplicitComponent):
    """
    This function is specific to MULTIROTOR UAVs in CRUISE conditions.
    Computes the different pathways through which each component has an impact.
    The different contributions are:
        - The resources extractions and manufacturing processes
        - The energy consumption in operation, which is in turn split into terms attributed to
            - the components' masses
            - and the components' efficiencies.
    """

    def initialize(self):
        # model
        self.model_name = None

        # methods
        self.method_names = list()
        self.options.declare("methods", default=LCA_DEFAULT_METHOD, types=list)

        # Functional unit
        self.options.declare("model", default="kg.km", values=["kg.km", "kg.h", "lifetime"])

    def setup(self):
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')
        self.add_input("data:weight:misc:mass", val=0.0, units='kg')
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')

        self.add_input("data:propulsion:multirotor:propeller:efficiency:cruise", val=np.nan, units=None)
        self.add_input("data:propulsion:multirotor:motor:efficiency:cruise", val=np.nan, units=None)
        self.add_input("data:propulsion:multirotor:esc:efficiency", val=np.nan, units=None)
        self.add_input("data:propulsion:multirotor:battery:efficiency", val=np.nan, units=None)

        self.add_input("data:propulsion:multirotor:battery:power:cruise", val=np.nan, units="W")
        self.add_input("data:aerodynamics:multirotor:CD0", val=np.nan, units=None)
        self.add_input("data:propulsion:multirotor:propeller:AoA:cruise", val=np.nan, units='rad')
        self.add_input("data:geometry:projected_area:top", val=np.nan, units='m**2')
        self.add_input("data:geometry:projected_area:front", val=np.nan, units='m**2')
        self.add_input("data:propulsion:multirotor:propeller:diameter", val=np.nan, units='m')
        self.add_input("mission:sizing:main_route:cruise:speed:multirotor", val=np.nan, units='m/s')
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:dISA", val=np.nan, units='K')

        # impact assessment methods
        model_name = self.model_name = ":model_per_FU" if self.options["model"] == "lifetime" else ":model_per_FU:model" # TODO: find a generic way of doing this
        method_names = self.method_names = [LCAcalc.method_label_formatting(eval(m)) for m in self.options["methods"]]
        for m_name in method_names:
            # TODO: retrieve and add units?
            result_path = LCA_RESULT_KEY + m_name + model_name
            self.add_input(result_path + ":operation", val=np.nan, units=None)
            self.add_input(result_path + ":production:battery", val=np.nan, units=None)
            self.add_input(result_path + ":production:controller", val=np.nan, units=None)
            self.add_input(result_path + ":production:motor", val=np.nan, units=None)
            self.add_input(result_path + ":production:propeller", val=np.nan, units=None)
            self.add_input(result_path + ":production:airframe", val=np.nan, units=None)

            postprocessing_path = LCA_POSTPROCESS_KEY + m_name + model_name
            self.add_output(postprocessing_path + ":battery:production", units=None)
            self.add_output(postprocessing_path + ":controller:production", units=None)
            self.add_output(postprocessing_path + ":motor:production", units=None)
            self.add_output(postprocessing_path + ":propeller:production", units=None)
            self.add_output(postprocessing_path + ":airframe:production", units=None)

            self.add_output(postprocessing_path + ":battery:mass", units=None)
            self.add_output(postprocessing_path + ":controller:mass", units=None)
            self.add_output(postprocessing_path + ":motor:mass", units=None)
            self.add_output(postprocessing_path + ":propeller:mass", units=None)
            self.add_output(postprocessing_path + ":airframe:mass", units=None)
            self.add_output(postprocessing_path + ":payload:mass", units=None)
            self.add_output(postprocessing_path + ":misc:mass", units=None)

            self.add_output(postprocessing_path + ":battery:efficiency", units=None)
            self.add_output(postprocessing_path + ":controller:efficiency", units=None)
            self.add_output(postprocessing_path + ":motor:efficiency", units=None)
            self.add_output(postprocessing_path + ":propeller:efficiency", units=None)
            self.add_output(postprocessing_path + ":airframe:efficiency", units=None)

            self.add_output(postprocessing_path + ":battery", units=None)
            self.add_output(postprocessing_path + ":controller", units=None)
            self.add_output(postprocessing_path + ":motor", units=None)
            self.add_output(postprocessing_path + ":propeller", units=None)
            self.add_output(postprocessing_path + ":airframe", units=None)
            self.add_output(postprocessing_path + ":payload", units=None)
            self.add_output(postprocessing_path + ":misc", units=None)

            self.add_output(postprocessing_path, units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # INPUTS
        # Masses
        m_pay = inputs["mission:sizing:payload:mass"]  # [kg]
        m_misc = inputs["data:weight:misc:mass"]  # [kg]
        m_bat = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        m_mot = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        m_pro = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        m_esc = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
        m_frame = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]
        m_tot = m_pay + m_pro + m_mot + m_esc + m_bat + m_frame  # [kg] total mass of UAV

        # Efficiencies for propulsion system
        eta_pro = inputs["data:propulsion:multirotor:propeller:efficiency:cruise"]  # [-] propeller efficiency
        eta_mot = inputs["data:propulsion:multirotor:motor:efficiency:cruise"]  # [-] motor efficiency
        eta_esc = inputs["data:propulsion:multirotor:esc:efficiency"]  # [-] ESC efficiency
        eta_bat = inputs["data:propulsion:multirotor:battery:efficiency"]  # [-] battery efficiency
        eta_tot = eta_pro * eta_mot * eta_esc * eta_bat  # [-] total efficiency for propulsion system

        # Power at battery (before heat losses)
        P = inputs["data:propulsion:multirotor:battery:power:cruise"] / eta_bat  # [W]

        # Aerodynamics
        C_D = inputs["data:aerodynamics:multirotor:CD0"]  # [-] drag coefficient for the UAV

        # Geometry
        alpha = inputs["data:propulsion:multirotor:propeller:AoA:cruise"]  # [rad]
        S_top = inputs["data:geometry:projected_area:top"]  # [m]
        S_front = inputs["data:geometry:projected_area:front"]  # [m]
        S = S_top * np.sin(alpha) + S_front * np.cos(alpha)  # [m]
        d_pro = inputs["data:propulsion:multirotor:propeller:diameter"]  # [m] propellers diameter

        # Operating conditions
        v_inf = inputs["mission:sizing:main_route:cruise:speed:multirotor"]  # [m/s] forward flight velocity
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]  # [m]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = v_inf
        rho_air = atm.density  # [kg/m3] air density

        # INTERMEDIATE CALCULATIONS
        A = 2 * g ** 2 * m_tot / (np.pi * rho_air * v_inf * d_pro ** 2 * N_pro)

        B_1 = 0.5 * rho_air * S * v_inf ** 3
        B_2 = B_1 * S / (np.pi * d_pro ** 2 * N_pro)

        P_ideal = A * m_tot + B_1 * C_D + B_2 * C_D ** 2
        eta_array = [eta_pro, eta_mot, eta_esc, eta_bat]
        C = (1 / eta_tot - 1) * P_ideal / np.sum([(1 / eta - 1) for eta in eta_array])

        # CORRECTION FACTOR
        corr_fact = eta_tot * P / P_ideal
        A = A * corr_fact
        B_1 = B_1 * corr_fact
        B_2 = B_2 * corr_fact
        C = C * corr_fact

        # CONTRIBUTIONS (for each impact method)
        model_name = self.model_name
        for m_name in self.method_names:
            result_path = LCA_RESULT_KEY + m_name + model_name

            # CONTRIBUTIONS TO ENERGY CONSUMPTION
            EI_operation = inputs[result_path + ":operation"]

            # Masses
            EI_mass_bat = A / P * EI_operation * m_bat
            EI_mass_esc = A / P * EI_operation * m_esc
            EI_mass_mot = A / P * EI_operation * m_mot
            EI_mass_pro = A / P * EI_operation * m_pro
            EI_mass_frame = A / P * EI_operation * m_frame
            EI_mass_pay = A / P * EI_operation * m_pay  # the payload mass is also responsible for energy consumption!
            EI_mass_misc = A / P * EI_operation * m_misc

            # Aerodynamics
            EI_aero_frame = (B_1 * C_D + B_2 * C_D ** 2) / P * EI_operation

            # Efficiencies
            EI_eff_bat = C / P * EI_operation * (1 - eta_bat) / eta_bat
            EI_eff_esc = C / P * EI_operation * (1 - eta_esc) / eta_esc
            EI_eff_mot = C / P * EI_operation * (1 - eta_mot) / eta_mot
            EI_eff_pro = C / P * EI_operation * (1 - eta_pro) / eta_pro

            # MANUFACTURING
            EI_manuf_bat = inputs[result_path + ":production:battery"]
            EI_manuf_esc = inputs[result_path + ":production:controller"]
            EI_manuf_mot = inputs[result_path + ":production:motor"]
            EI_manuf_pro = inputs[result_path + ":production:propeller"]
            EI_manuf_frame = inputs[result_path + ":production:airframe"]

            # SUM CONTRIBUTIONS for each component
            EI_bat = EI_mass_bat + EI_eff_bat + EI_manuf_bat
            EI_esc = EI_mass_esc + EI_eff_esc + EI_manuf_esc
            EI_mot = EI_mass_mot + EI_eff_mot + EI_manuf_mot
            EI_pro = EI_mass_pro + EI_eff_pro + EI_manuf_pro
            EI_frame = EI_mass_frame + EI_aero_frame + EI_manuf_frame
            EI_pay = EI_mass_pay
            EI_misc = EI_mass_misc

            # TOTAL
            EI_tot = EI_bat + EI_esc + EI_mot + EI_pro + EI_frame + EI_pay + EI_misc

            # set output values
            postprocessing_path = LCA_POSTPROCESS_KEY + m_name + model_name
            outputs[postprocessing_path + ":battery:production"] = EI_manuf_bat
            outputs[postprocessing_path + ":controller:production"] = EI_manuf_esc
            outputs[postprocessing_path + ":motor:production"] = EI_manuf_mot
            outputs[postprocessing_path + ":propeller:production"] = EI_manuf_pro
            outputs[postprocessing_path + ":airframe:production"] = EI_manuf_frame

            outputs[postprocessing_path + ":battery:mass"] = EI_mass_bat
            outputs[postprocessing_path + ":controller:mass"] = EI_mass_esc
            outputs[postprocessing_path + ":motor:mass"] = EI_mass_mot
            outputs[postprocessing_path + ":propeller:mass"] = EI_mass_pro
            outputs[postprocessing_path + ":airframe:mass"] = EI_mass_frame
            outputs[postprocessing_path + ":payload:mass"] = EI_mass_pay
            outputs[postprocessing_path + ":misc:mass"] = EI_mass_misc

            outputs[postprocessing_path + ":battery:efficiency"] = EI_eff_bat
            outputs[postprocessing_path + ":controller:efficiency"] = EI_eff_esc
            outputs[postprocessing_path + ":motor:efficiency"] = EI_eff_mot
            outputs[postprocessing_path + ":propeller:efficiency"] = EI_eff_pro
            outputs[postprocessing_path + ":airframe:efficiency"] = EI_aero_frame

            outputs[postprocessing_path + ":battery"] = EI_bat
            outputs[postprocessing_path + ":controller"] = EI_esc
            outputs[postprocessing_path + ":motor"] = EI_mot
            outputs[postprocessing_path + ":propeller"] = EI_pro
            outputs[postprocessing_path + ":airframe"] = EI_frame
            outputs[postprocessing_path + ":payload"] = EI_pay
            outputs[postprocessing_path + ":misc"] = EI_misc

            outputs[postprocessing_path] = EI_tot
