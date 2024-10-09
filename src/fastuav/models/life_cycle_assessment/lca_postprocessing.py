"""
LCA postprocessing calculations.

TO BE UPDATED WITH NEW LCA MODULE
"""

import openmdao.api as om
import numpy as np
from scipy.constants import g
from stdatm import AtmosphereSI
from fastuav.constants import LCA_CHARACTERIZATION_KEY, LCA_POSTPROCESS_KEY, LCA_DEFAULT_METHOD, \
    LCA_DEFAULT_FUNCTIONAL_UNIT, LCA_FUNCTIONAL_UNITS_LIST, LCA_WEIGHTING_KEY, LCA_NORMALIZATION_KEY, LCA_SINGLE_SCORE_KEY,\
    LCA_WEIGHTED_SINGLE_SCORE_KEY
from fastuav.models.life_cycle_assessment.lca_core import CharacterizationDeprecated


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

        # functional unit
        self.options.declare("functional_unit", default=LCA_DEFAULT_FUNCTIONAL_UNIT, values=LCA_FUNCTIONAL_UNITS_LIST)

        # characterization, normalization and weighting
        self.results_dict = dict()
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)

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

        self.add_input("data:aerodynamics:multirotor:CD0", val=np.nan, units=None)
        self.add_input("data:propulsion:multirotor:propeller:AoA:cruise", val=np.nan, units='rad')
        self.add_input("data:geometry:projected_area:top", val=np.nan, units='m**2')
        self.add_input("data:geometry:projected_area:front", val=np.nan, units='m**2')
        self.add_input("data:propulsion:multirotor:propeller:diameter", val=np.nan, units='m')
        self.add_input("mission:sizing:main_route:cruise:speed:multirotor", val=np.nan, units='m/s')
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:dISA", val=np.nan, units='K')

        # Taylor approximation error
        self.add_output(LCA_POSTPROCESS_KEY + 'taylor_approx:rel_error', units=None)

        # impact assessment methods
        model_name = self.model_name = ":model_per_FU"  #if self.options["functional_unit"] == "lifetime" else ":model_per_FU:model" # FIXME: check no error for different functional units
        method_names = self.method_names = [CharacterizationDeprecated.method_label_formatting(eval(m)) for m in
                                            self.options["methods"]]

        results_dict = self.results_dict = {LCA_CHARACTERIZATION_KEY: method_names}  # characterized scores
        if self.options['normalization']:
            results_dict[LCA_NORMALIZATION_KEY] = method_names  # normalized scores
        if self.options['weighting']:
            results_dict[LCA_WEIGHTING_KEY] = method_names  # weighted scores
            results_dict[LCA_SINGLE_SCORE_KEY] = [LCA_WEIGHTED_SINGLE_SCORE_KEY]  # aggregated score

        for result_key, result_methods in results_dict.items():
            for m_name in result_methods:
                # TODO: retrieve and add units?
                result_path = result_key + m_name + model_name
                self.add_input(result_path + ":operation", val=np.nan, units=None)
                self.add_input(result_path + ":production:batteries", val=np.nan, units=None)
                self.add_input(result_path + ":production:controllers", val=np.nan, units=None)
                self.add_input(result_path + ":production:motors", val=np.nan, units=None)
                self.add_input(result_path + ":production:propellers", val=np.nan, units=None)
                self.add_input(result_path + ":production:airframe", val=np.nan, units=None)

                postprocessing_path = LCA_POSTPROCESS_KEY + result_key.split(':', 1)[-1] + m_name + model_name
                self.add_output(postprocessing_path + ":batteries:production", units=None)
                self.add_output(postprocessing_path + ":controllers:production", units=None)
                self.add_output(postprocessing_path + ":motors:production", units=None)
                self.add_output(postprocessing_path + ":propellers:production", units=None)
                self.add_output(postprocessing_path + ":airframe:production", units=None)

                self.add_output(postprocessing_path + ":batteries:mass", units=None)
                self.add_output(postprocessing_path + ":controllers:mass", units=None)
                self.add_output(postprocessing_path + ":motors:mass", units=None)
                self.add_output(postprocessing_path + ":propellers:mass", units=None)
                self.add_output(postprocessing_path + ":airframe:mass", units=None)
                self.add_output(postprocessing_path + ":payload:mass", units=None)

                self.add_output(postprocessing_path + ":batteries:efficiency", units=None)
                self.add_output(postprocessing_path + ":controllers:efficiency", units=None)
                self.add_output(postprocessing_path + ":motors:efficiency", units=None)
                self.add_output(postprocessing_path + ":propellers:efficiency", units=None)
                self.add_output(postprocessing_path + ":airframe:efficiency", units=None)

                self.add_output(postprocessing_path + ":batteries", units=None)
                self.add_output(postprocessing_path + ":controllers", units=None)
                self.add_output(postprocessing_path + ":motors", units=None)
                self.add_output(postprocessing_path + ":propellers", units=None)
                self.add_output(postprocessing_path + ":airframe", units=None)
                self.add_output(postprocessing_path + ":payload", units=None)

                self.add_output(postprocessing_path, units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # INPUTS
        # Masses
        m_pay = inputs["mission:sizing:payload:mass"] + inputs["data:weight:misc:mass"]  # [kg]
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
        eta_array = [eta_pro, eta_mot, eta_esc, eta_bat]
        eta_tot = eta_pro * eta_mot * eta_esc * eta_bat  # [-] total efficiency for propulsion system

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
        A = 2 * g ** 2 / (np.pi * rho_air * v_inf * d_pro ** 2 * N_pro)
        B_1 = 0.5 * rho_air * S * v_inf ** 3
        B_2 = B_1 * S / (np.pi * d_pro ** 2 * N_pro)
        P_ideal = A * m_tot ** 2 + B_1 * C_D + B_2 * C_D ** 2
        P = P_ideal / eta_tot

        # POWER DECOMPOSITION
        # Payload
        P_0 = A * m_pay ** 2

        # Aerodynamics
        P_aero_frame = B_1 * C_D + B_2 * C_D ** 2 + 0.5 * B_1 * C_D * np.sum([(1 - eta) for eta in eta_array])

        # Masses
        P_mass_bat = A * m_bat * (m_pay + m_tot + 0.5 * m_pay * np.sum([(1 - eta) for eta in eta_array]))
        P_mass_esc = A * m_esc * (m_pay + m_tot + 0.5 * m_pay * np.sum([(1 - eta) for eta in eta_array]))
        P_mass_mot = A * m_mot * (m_pay + m_tot + 0.5 * m_pay * np.sum([(1 - eta) for eta in eta_array]))
        P_mass_pro = A * m_pro * (m_pay + m_tot + 0.5 * m_pay * np.sum([(1 - eta) for eta in eta_array]))
        P_mass_frame = A * m_frame * (m_pay + m_tot + 0.5 * m_pay * np.sum([(1 - eta) for eta in eta_array]))

        # Efficiencies
        P_eff_bat = A * m_pay ** 2 * (1 - eta_bat) * (
                2 - eta_bat + np.sum([(1 - eta) for eta in eta_array]) + m_tot / m_pay) + 0.5 * B_1 * (
                            1 - eta_bat)
        P_eff_esc = A * m_pay ** 2 * (1 - eta_esc) * (
                2 - eta_esc + np.sum([(1 - eta) for eta in eta_array]) + m_tot / m_pay) + 0.5 * B_1 * (
                            1 - eta_esc)
        P_eff_mot = A * m_pay ** 2 * (1 - eta_mot) * (
                2 - eta_mot + np.sum([(1 - eta) for eta in eta_array]) + m_tot / m_pay) + 0.5 * B_1 * (
                            1 - eta_mot)
        P_eff_pro = A * m_pay ** 2 * (1 - eta_pro) * (
                2 - eta_pro + np.sum([(1 - eta) for eta in eta_array]) + m_tot / m_pay) + 0.5 * B_1 * (
                            1 - eta_pro)

        # Compare approximation with actual power
        P_mass = P_mass_bat + P_mass_esc + P_mass_mot + P_mass_pro + P_mass_frame
        P_eff = P_eff_bat + P_eff_esc + P_eff_mot + P_eff_pro
        P_approx = P_0 + P_mass + P_aero_frame + P_eff
        power_rel_error = (P - P_approx) / P  # [-]
        outputs[LCA_POSTPROCESS_KEY + 'taylor_approx:rel_error'] = power_rel_error

        # For each impact method and each step of the impact assessment: characterization, normalization, weighting
        results_dict = self.results_dict
        model_name = self.model_name
        for result_key, result_methods in results_dict.items():
            for m_name in result_methods:
                result_path = result_key + m_name + model_name

                # ENVIRONMENTAL IMPACTS - OPERATION
                EI_operation = inputs[result_path + ":operation"]
                EI_0 = P_0 / P_approx * EI_operation
                EI_mass_bat = P_mass_bat / P_approx * EI_operation
                EI_mass_esc = P_mass_esc / P_approx * EI_operation
                EI_mass_mot = P_mass_mot / P_approx * EI_operation
                EI_mass_pro = P_mass_pro / P_approx * EI_operation
                EI_mass_frame = P_mass_frame / P_approx * EI_operation
                EI_aero_frame = P_aero_frame / P_approx * EI_operation
                EI_eff_bat = P_eff_bat / P_approx * EI_operation
                EI_eff_esc = P_eff_esc / P_approx * EI_operation
                EI_eff_mot = P_eff_mot / P_approx * EI_operation
                EI_eff_pro = P_eff_pro / P_approx * EI_operation

                # ENVIRONMENTAL IMPACTS - MANUFACTURING
                EI_manuf_bat = inputs[result_path + ":production:batteries"]
                EI_manuf_esc = inputs[result_path + ":production:controllers"]
                EI_manuf_mot = inputs[result_path + ":production:motors"]
                EI_manuf_pro = inputs[result_path + ":production:propellers"]
                EI_manuf_frame = inputs[result_path + ":production:airframe"]

                # SUM CONTRIBUTIONS for each component
                EI_bat = EI_mass_bat + EI_eff_bat + EI_manuf_bat
                EI_esc = EI_mass_esc + EI_eff_esc + EI_manuf_esc
                EI_mot = EI_mass_mot + EI_eff_mot + EI_manuf_mot
                EI_pro = EI_mass_pro + EI_eff_pro + EI_manuf_pro
                EI_frame = EI_mass_frame + EI_aero_frame + EI_manuf_frame
                EI_pay = EI_0

                # TOTAL
                EI_tot = EI_bat + EI_esc + EI_mot + EI_pro + EI_frame + EI_pay

                # set output values
                postprocessing_path = LCA_POSTPROCESS_KEY + result_key.split(':', 1)[-1] + m_name + model_name

                outputs[postprocessing_path + ":batteries:production"] = EI_manuf_bat
                outputs[postprocessing_path + ":controllers:production"] = EI_manuf_esc
                outputs[postprocessing_path + ":motors:production"] = EI_manuf_mot
                outputs[postprocessing_path + ":propellers:production"] = EI_manuf_pro
                outputs[postprocessing_path + ":airframe:production"] = EI_manuf_frame

                outputs[postprocessing_path + ":batteries:mass"] = EI_mass_bat
                outputs[postprocessing_path + ":controllers:mass"] = EI_mass_esc
                outputs[postprocessing_path + ":motors:mass"] = EI_mass_mot
                outputs[postprocessing_path + ":propellers:mass"] = EI_mass_pro
                outputs[postprocessing_path + ":airframe:mass"] = EI_mass_frame
                outputs[postprocessing_path + ":payload:mass"] = EI_pay

                outputs[postprocessing_path + ":batteries:efficiency"] = EI_eff_bat
                outputs[postprocessing_path + ":controllers:efficiency"] = EI_eff_esc
                outputs[postprocessing_path + ":motors:efficiency"] = EI_eff_mot
                outputs[postprocessing_path + ":propellers:efficiency"] = EI_eff_pro
                outputs[postprocessing_path + ":airframe:efficiency"] = EI_aero_frame

                outputs[postprocessing_path + ":batteries"] = EI_bat
                outputs[postprocessing_path + ":controllers"] = EI_esc
                outputs[postprocessing_path + ":motors"] = EI_mot
                outputs[postprocessing_path + ":propellers"] = EI_pro
                outputs[postprocessing_path + ":airframe"] = EI_frame
                outputs[postprocessing_path + ":payload"] = EI_pay

                outputs[postprocessing_path] = EI_tot

