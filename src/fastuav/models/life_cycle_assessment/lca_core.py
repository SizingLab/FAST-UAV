"""
LCA models and calculations.
"""

import openmdao.api as om
import numpy as np
import brightway2 as bw
import lca_algebraic as lcalg
import re
import sympy as sym
from fastuav.exceptions import FastLcaProjectDoesNotExist, \
    FastLcaDatabaseIsNotImported, \
    FastLcaMethodDoesNotExist, \
    FastLcaParameterNotDeclared
from fastuav.constants import LCA_DEFAULT_PROJECT, LCA_DEFAULT_ECOINVENT, LCA_USER_DB, LCA_MODEL_KEY, \
    LCA_PARAM_KEY, LCA_CHARACTERIZATION_KEY, LCA_DEFAULT_METHOD, SIZING_MISSION_TAG, LCA_FUNCTIONAL_UNITS_LIST, \
    LCA_DEFAULT_FUNCTIONAL_UNIT, LCA_NORMALIZATION_KEY, LCA_WEIGHTING_KEY, LCA_FACTOR_KEY, LCA_WEIGHTED_SINGLE_SCORE_KEY, \
    LCA_AGGREGATION_KEY


class LCAcore(om.Group):
    """
    Group for LCA models and calculations.
    """

    def initialize(self):
        # Declare options
        self.options.declare("project", default=LCA_DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=LCA_DEFAULT_ECOINVENT, types=str)
        self.options.declare("functional_unit", default=LCA_DEFAULT_FUNCTIONAL_UNIT, types=str)
        self.options.declare("methods", default=LCA_DEFAULT_METHOD, types=list)
        self.options.declare("normalization", default=False, types=bool)
        self.options.declare("weighting", default=False, types=bool)
        self.options.declare("max_level_processes", default=10, types=int)

        # Computation options for optimization
        self.options.declare("analytical_derivatives", default=True, types=bool)

        # FAST-UAV model specific parameters
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

        # FAST-UAV specific option for selecting mission to evaluate
        self.options.declare("mission", default=SIZING_MISSION_TAG, types=str)

    def setup(self):
        # LCA MODEL
        self.add_subsystem("model",
                           Model(project=self.options["project"],
                                 database=self.options["database"],
                                 functional_unit=self.options["functional_unit"],
                                 mission=self.options["mission"]),
                           promotes=["*"])

        # CHARACTERIZATION
        self.add_subsystem("characterization",
                           Characterization(methods=self.options["methods"],
                                            max_level_processes=self.options["max_level_processes"])
                           # promote for these subsystems is done in configure() method
                           )

        # NORMALIZATION
        if self.options["weighting"] or self.options["normalization"]:
            self.add_subsystem("normalization",
                               Normalization(methods=self.options["methods"],
                                             max_level_processes=self.options["max_level_processes"]),
                               promotes=["*"])

        # WEIGHTING AND AGGREGATION
        if self.options["weighting"]:
            self.add_subsystem("weighting",
                               Weighting(methods=self.options["methods"],
                                         max_level_processes=self.options["max_level_processes"]),
                               promotes=["*"])
            self.add_subsystem("aggregation",
                               Aggregation(methods=self.options["methods"],
                                           max_level_processes=self.options["max_level_processes"]),
                               promotes=["*"])

    def configure(self):
        """
        Set inputs and options for characterization module by copying `model` outputs and options.
        Configure() method from the containing group is necessary to get access to `model` metadata after Setup().
        """

        # Add LCA parameters declared in the LCA model to the characterization module,
        # either as inputs (float parameters) or options (str parameters)
        for name, object in self.model.parameters.items():
            if object.type == 'float':  # add float parameters as inputs to calculation module
                self.characterization.add_input(LCA_PARAM_KEY + name, val=np.nan, units=None)
            elif name in self.options["parameters"].keys():  # add non-float parameters as options
                self.characterization.options["parameters"][name] = self.options["parameters"][name]

        # Promote variables and declare partials
        self.promotes("characterization", any=['*'])
        if self.options['analytical_derivatives']:
            self.characterization.declare_partials("*", "*", method="exact")
        else:
            self.characterization.declare_partials("*", "*", method="fd")


class Model(om.ExplicitComponent):
    """
    This OpenMDAO component implements an LCA model using brightway2 and lca_algebraic librairies.
    It creates an LCA model and sets the LCA parameters for further parametric LCA calculation.
    """

    def initialize(self):
        # Attributes
        self.parameters = dict()  # dictionary of {parameter_name: parameter_object} to store all parameters

        # Declare options
        self.options.declare("project", default=LCA_DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=LCA_DEFAULT_ECOINVENT, types=str)
        self.options.declare("functional_unit", default=LCA_DEFAULT_FUNCTIONAL_UNIT, values=LCA_FUNCTIONAL_UNITS_LIST)
        self.options.declare("mission", default=SIZING_MISSION_TAG, types=str)

    def setup(self):
        # Setup project
        self.setup_project()

        # Declare parameters for LCA
        self.declare_parameters()

        # Define foreground activities
        self.declare_foreground_activities()

        # Build model to be evaluated in LCA process
        self.build_model()

        # Add inputs and outputs related to LCA parameters.
        # (Note: non-float parameters are to be declared through options)
        # Inputs: UAV parameters
        # NB: check that units are consistence with LCA parameters/activities!
        mission_name = self.options["mission"]
        self.add_input("mission:%s:distance" % mission_name, val=np.nan, units='m')
        self.add_input("mission:%s:duration" % mission_name, val=np.nan, units='min')
        self.add_input("mission:%s:energy" % mission_name, val=np.nan, units='kJ')
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')  # TODO: specify operational mission payload?
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:battery:efficiency", val=np.nan, units=None)

        # Outputs: LCA parameters calculated in compute() method
        self.add_output(LCA_PARAM_KEY + 'mission_distance', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mission_duration', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mission_energy', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_payload', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_motors', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_propellers', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_controllers', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_batteries', val=np.nan, units=None)
        self.add_output(LCA_PARAM_KEY + 'mass_airframe', val=np.nan, units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # The compute method is used here to set lca parameters' values
        mission_name = self.options["mission"]
        mission_distance = inputs["mission:%s:distance" % mission_name] / 1000  # [km]
        mission_duration = inputs["mission:%s:duration" % mission_name] / 60  # [h]
        mass_payload = inputs["mission:sizing:payload:mass"]  # [kg]
        mission_energy = inputs["data:propulsion:multirotor:battery:efficiency"] * inputs[
            "mission:%s:energy" % mission_name] / 3600  # [kWh]
        mass_batteries = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        mass_motors = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        mass_propellers = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        mass_controllers = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
        mass_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]

        # Data manipulation for normalized models (avoid division by zero)
        eps = np.array([1e-9])
        mission_distance = max(mission_distance, eps)
        mission_duration = max(mission_duration, eps)
        mass_payload = max(mass_payload, eps)

        # Outputs: LCA parameters (names should be the same as declared in the declare_parameters() function
        outputs[LCA_PARAM_KEY + 'mission_distance'] = mission_distance
        outputs[LCA_PARAM_KEY + 'mission_duration'] = mission_duration
        outputs[LCA_PARAM_KEY + 'mission_energy'] = mission_energy
        outputs[LCA_PARAM_KEY + 'mass_payload'] = mass_payload
        outputs[LCA_PARAM_KEY + 'mass_motors'] = mass_motors
        outputs[LCA_PARAM_KEY + 'mass_propellers'] = mass_propellers
        outputs[LCA_PARAM_KEY + 'mass_controllers'] = mass_controllers
        outputs[LCA_PARAM_KEY + 'mass_batteries'] = mass_batteries
        outputs[LCA_PARAM_KEY + 'mass_airframe'] = mass_airframe

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_name = self.options["mission"]

        partials[LCA_PARAM_KEY + 'mission_distance', "mission:%s:distance" % mission_name] = 1 / 1000
        partials[LCA_PARAM_KEY + 'mission_duration', "mission:%s:duration" % mission_name] = 1 / 60
        partials[LCA_PARAM_KEY + 'mission_energy',
                 "data:propulsion:multirotor:battery:efficiency"] = inputs["mission:%s:energy" % mission_name] / 3600
        partials[LCA_PARAM_KEY + 'mission_energy',
                 "mission:%s:energy" % mission_name] = inputs["data:propulsion:multirotor:battery:efficiency"] / 3600
        partials[LCA_PARAM_KEY + 'mass_payload', "mission:sizing:payload:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_motors',
                 "data:weight:propulsion:multirotor:motor:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_motors',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:motor:mass"]
        partials[LCA_PARAM_KEY + 'mass_propellers',
                 "data:weight:propulsion:multirotor:propeller:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_propellers',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:propeller:mass"]
        partials[LCA_PARAM_KEY + 'mass_controllers',
                 "data:weight:propulsion:multirotor:esc:mass"] = inputs["data:propulsion:multirotor:propeller:number"]
        partials[LCA_PARAM_KEY + 'mass_controllers',
                 "data:propulsion:multirotor:propeller:number"] = inputs["data:weight:propulsion:multirotor:esc:mass"]
        partials[LCA_PARAM_KEY + 'mass_batteries', "data:weight:propulsion:multirotor:battery:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_airframe', "data:weight:airframe:body:mass"] = 1.0
        partials[LCA_PARAM_KEY + 'mass_airframe', "data:weight:airframe:arms:mass"] = 1.0

    def setup_project(self):
        """
        Set and initialize lca project.
        """
        project_name = self.options["project"]
        background_db_name = self.options["database"]

        # Check project already exists
        if project_name not in bw.projects:
            raise FastLcaProjectDoesNotExist(project_name)

        # Set current project
        bw.projects.set_current(project_name)

        # Check EcoInvent has been imported in project
        if background_db_name not in list(bw.databases):
            raise FastLcaDatabaseIsNotImported(project_name, background_db_name)

        # Import/create foreground database and reset for clean state
        lcalg.resetDb(LCA_USER_DB)
        lcalg.setForeground(LCA_USER_DB)

        # Reset project parameters for clean state
        lcalg.resetParams()

        return True

    def declare_parameters(self):
        """
        Declare parameters for the parametric LCA
        """

        # Float parameters are created with the 'newFloatParam' method
        self._add_param(lcalg.newFloatParam(
            'n_cycles_uav',  # name of the parameter
            default=1.0,  # default value
            min=1, max=10000,  # bounds (only for DoE purposes)
            description="number of cycles",  # description
            dbname=LCA_USER_DB  # we define the parameter in our own database
        ))

        self._add_param(lcalg.newFloatParam(
            'n_cycles_battery',
            default=1.0,
            min=0, max=10000,
            description="maximum number of cycles for battery technology",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'battery_recycling_share',
            default=0.0,
            min=0.0, max=1.0,
            description="share of battery that is recycled",
            dbname=LCA_USER_DB
        ))

        # Enum parameters are a facility to represent different options
        # and should be used with the 'newSwitchAct' method
        self._add_param(lcalg.newEnumParam(
            'battery_type',
            values=["nmc_811", "nmc_111", "nca", "lfp", "nimh", "si_nmc_811"],  # values this parameter can take
            default="nmc_811",
            description="battery technology",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newEnumParam(
            'elec_switch_param',
            values=["us", "eu", "fr"],  # values this parameter can take
            default="eu",
            description="switch on electricity mix",
            dbname=LCA_USER_DB
        ))

        # UAV specific parameters
        self._add_param(lcalg.newFloatParam(
            'mission_distance',
            default=1.0,
            min=1, max=1000,
            description="distance of sizing mission",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mission_duration',
            default=1.0,
            min=1, max=1000,
            description="duration of sizing mission",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mission_energy',
            default=1.0,
            min=0, max=1000,
            description="energy consumption for sizing mission",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_payload',
            default=1.0,
            min=0, max=1000,
            description="payload mass used for sizing mission",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_batteries',
            default=1.0,
            min=0, max=1000,
            description="batteries mass",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_motors',
            default=1.0,
            min=0, max=1000,
            description="motors mass",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_airframe',
            default=1.0,
            min=0, max=1000,
            description="airframe mass",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_propellers',
            default=1.0,
            min=0, max=1000,
            description="propellers mass",
            dbname=LCA_USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_controllers',
            default=1.0,
            min=0, max=1000,
            description="controllers mass",
            dbname=LCA_USER_DB
        ))

    def declare_foreground_activities(self):
        """
        Declare new foreground activities in our own database.
        The foreground activities are linked to the background activities
        with exchanges that can be parameterized.
        """

        db_ecoinvent = self.options["database"]

        ### References to some background activities
        motor_scooter = lcalg.findActivity(db_name=db_ecoinvent,
                                           code='a9f8412fe79b4fe74771ddfbeebb3f98')  # [kg] electric scooter motor
        controller_scooter = lcalg.findActivity(db_name=db_ecoinvent,
                                                code='9afe5ffc45f1b043596a7901a59c98eb')  # [kg] electric scooter controller
        transistor_igbt = lcalg.findActivity(db_name=db_ecoinvent,
                                             code='ec5357459a8277ad58908d6ceca6fee2')  # [kg] IGBT transistor
        printed_wiring_board = lcalg.findActivity(db_name=db_ecoinvent,
                                                  code='cc35723610fe4baece243288133a3ff1')  # [kg] printed wiring board, for power supply unit, desktop computer, Pb free
        composite = lcalg.findActivity(db_name=db_ecoinvent,
                                       code='11cd946a783d8de6f814fc2f5c3b4782')  # [kg] CFRP
        aluminium = lcalg.findActivity(db_name=db_ecoinvent,
                                       code='fa3d4c4f880c2f0240b557a3dac1f9d7')  # [kg] aluminium section bar extrusion
        electricity_eu = lcalg.findActivity(db_name=db_ecoinvent,
                                            code='da20ff6f4e46c6268b3017121bd0b2f4')  # [kWh] Europe w/o Switzerland, Low voltage
        electricity_us = lcalg.findActivity(db_name=db_ecoinvent,
                                            code='04ddd164cec6d9ed96cfc299cab21124')  # [kWh] United States, Low voltage
        electricity_fr = lcalg.findActivity(db_name=db_ecoinvent,
                                            code='a950938cf39595b8de0977d1b289d69a')  # [kWh] France, Low voltage
        battery_nmc_811 = lcalg.findActivity(db_name=db_ecoinvent,
                                             code='52e3cdd70890530eada4fbcef2741406')  # [kg] Li-ion NMC 811 battery cell
        battery_nmc_111 = lcalg.findActivity(db_name=db_ecoinvent,
                                             code='742db99703644938601390debe4d348e')  # [kg] Li-ion NMC 111 battery cell
        battery_nca = lcalg.findActivity(db_name=db_ecoinvent,
                                         code='7b1eff2765339e62ffb98ad1cadd2698')  # [kg] Li-ion NCA battery cell
        battery_lfp = lcalg.findActivity(db_name=db_ecoinvent,
                                         code='f6036ad86fb205d8712754f2fac10a16')  # [kg] Li-ion LFP battery cell
        battery_nimh = lcalg.findActivity(db_name=db_ecoinvent,
                                          code='604a095c71d418d248cf5f4bef12f5c4')  # [kg] NiMH battery pack
        silicon_powder = lcalg.findActivity(db_name=db_ecoinvent,
                                             code='83058091453a6152de2cbe7425e2cd4c')  # [kg] silicon, solar grade
        medium_voltage_electricity = lcalg.findActivity(db_name=db_ecoinvent,
                                                        code='48e589b2cea58ab594265ee58ce56015')  # [kWh] medium voltage China
        anode_graphite = lcalg.findActivity(db_name=db_ecoinvent,
                                            code='e13a71404f616d4166a76ecc603cc71c')  # [kg] anode, silicon coated graphite, NMC811
        battery_cell_g_nmc = lcalg.findActivity(db_name=db_ecoinvent,
                                                code='ecef71e2f1b5d874f97e3ee912abaf1b')  # [kg] G/NMC battery cell
        market_battery_cell_g_nmc = lcalg.findActivity(db_name=db_ecoinvent,
                                                       code='52e3cdd70890530eada4fbcef2741406')  # [kg] market for G/NMC battery cell
        used_battery = lcalg.findActivity(db_name=db_ecoinvent,
                                          code='82ebcdf42e8512cbe00151dda6210d29')  # [kg] Used li-ion battery


        ### Motor activity
        motor = lcalg.newActivity(
            LCA_USER_DB,  # we define foreground activities in our own database
            "motors",  # Name of the activity
            "kg",  # We define exchanges as a dictionary of 'activity : amount'
            exchanges={
                motor_scooter: 1.0,   # Amount can be a fixed value or a parameter
            }
        )

        ### ESC activity
        controller = lcalg.newActivity(
            LCA_USER_DB,
            "controllers",
            "kg",
            exchanges={
                controller_scooter: 1.0,
                # printed_wiring_board : 1.0
                # transistor_igbt: 1.0
            }
        )

        ### Propeller activity
        propeller = lcalg.newActivity(
            LCA_USER_DB,
            "propellers",
            "kg",
            exchanges={
                composite: 1.0,
            }
        )

        ### Airframe activity
        airframe = lcalg.newActivity(
            LCA_USER_DB,
            "airframe",
            "kg",
            exchanges={
                composite: 1.0,
                # aluminium: 0.8,  # one may also set a lower amount to account for recycled materials
            }
        )

        ### Battery activity

        # Create activity for Si/NMC battery
        # Reference:
        # Li, Bingbing, Xianfeng Gao, Jianyang Li, and Chris Yuan.
        # “Life Cycle Environmental Impact of High-Capacity Lithium Ion Battery
        # with Silicon Nanowires Anode for Electric Vehicles.”
        # Environmental Science & Technology 48, no. 5 (March 4, 2014): 3047–55. https://doi.org/10.1021/es4037786.
        silicon_nanowire = lcalg.newActivity(
            LCA_USER_DB,
            "silicon_nanowire",
            "kg",
            exchanges={
                silicon_powder: 5.0,  # kg
                medium_voltage_electricity: 1.3,  # [kWh]
            }
        )
        # Create new activity for silicon-based anode, derived from graphite-based anode
        anode_silicon = lcalg.copyActivity(
            LCA_USER_DB,
            anode_graphite,  # Initial activity : won't be altered
            "anode_silicon_nanowire")  # New name
        anode_silicon.updateExchanges({
            "synthetic graphite, battery grade": 0.0,  # remove graphite
        })
        anode_silicon.addExchanges({silicon_nanowire: 0.92})  # add silicon nanowire
        # Create new activity for Si/NMC811 battery cell, derived from G/NMC cell
        battery_cell_si_nmc = lcalg.copyActivity(
            LCA_USER_DB,
            battery_cell_g_nmc,
            "battery_cell_silicon_NMC811")
        battery_cell_si_nmc.updateExchanges({
            'anode, silicon coated graphite, for Li-ion battery': 0.0,  # remove graphite-based anode
        })
        battery_cell_si_nmc.addExchanges({anode_silicon: 0.2181})  # add silicon-based anode
        # Create new activity for MARKET for Si/NMC battery cell, derived from market for G/NMC cell
        market_battery_cell_si_nmc = lcalg.copyActivity(
            LCA_USER_DB,
            market_battery_cell_g_nmc,
            "market_battery_cell_silicon_NMC811")
        market_battery_cell_si_nmc.updateExchanges({
            'battery cell*#CN': 0.0,
            'battery cell*#RoW': 0.0,
        })
        market_battery_cell_si_nmc.addExchanges({battery_cell_si_nmc: 1.0})

        # Top battery activity (with option for selecting battery type)
        battery_production = lcalg.newSwitchAct(  # This is a switch activity to choose between different type of sub-activities (here, battery type)
            LCA_USER_DB,
            "battery_production",
            self._get_param('battery_type'),  # Switch parameter previously defined
            {  # Dictionary of enum values / activities : {"switch_option": (activity, amount)}
                "nmc_811": battery_nmc_811,
                "nmc_111": battery_nmc_111,
                "nca": battery_nca,
                "lfp": battery_lfp,
                "nimh": battery_nimh,
                "si_nmc_811": market_battery_cell_si_nmc,
            }
        )
        battery_recycling = lcalg.newActivity(
            LCA_USER_DB,
            "battery_recycling",
            "kg",
            exchanges={
                used_battery: self._get_param('battery_recycling_share')
            }
        )
        battery = lcalg.newActivity(
            LCA_USER_DB,
            "batteries",
            "kg",
            exchanges={
                battery_production: 1.0,
                battery_recycling: 1.0
            }
        )

        # Production activity: assembly of the previously defined components
        lcalg.newActivity(
            LCA_USER_DB,
            "production",
            "uav",  # unit is one uav
            exchanges={
                battery: self._get_param('mass_batteries') * sym.ceiling(self._get_param('n_cycles_uav') / self._get_param('n_cycles_battery')),  # Amount is a formula
                motor: self._get_param('mass_motors'),
                airframe: self._get_param('mass_airframe'),
                propeller: self._get_param('mass_propellers'),
                controller: self._get_param('mass_controllers'),
            }
        )

        # Operation activity: electricity used for flying.
        # This is a switch activity. One may choose between different type of sub-activities (here, electricity mix)
        lcalg.newSwitchAct(
            LCA_USER_DB,
            "operation",
            self._get_param('elec_switch_param'),  # Switch parameter previously defined
            {  # Dictionary of enum values / activities
                "us": electricity_us,
                "eu": electricity_eu,
                "fr": electricity_fr,
            }
        )

    def build_model(self):
        """
        Build the model that will be evaluated in the LCA process.
        """

        # Retrieve some previously defined foreground activities
        production = lcalg.getActByCode(LCA_USER_DB, "production")
        operation = lcalg.getActByCode(LCA_USER_DB, "operation")

        # Define model
        model_select = self.options["functional_unit"]

        if model_select == "kg.km":  # 1 kg payload on 1 km
            intermediate_model = lcalg.newActivity(  # Impacts over UAV's lifetime
                LCA_USER_DB,
                "functional_unit",
                "uav lifetime",
                exchanges={
                    production: 1.0,
                    operation: self._get_param('n_cycles_uav') * self._get_param('mission_energy'),
                })
            functional_value = self._get_param('n_cycles_uav') * self._get_param('mission_distance') * self._get_param(
                'mass_payload')
            lcalg.newActivity(
                LCA_USER_DB,
                LCA_MODEL_KEY,
                model_select,
                exchanges={
                    intermediate_model: 1 / functional_value  # normalize by functional value
                })

        elif model_select == "kg.h":  # 1 kg payload during 1 hour
            intermediate_model = lcalg.newActivity(  # Impacts over UAV's lifetime
                LCA_USER_DB,
                "functional_unit",
                "uav lifetime",
                exchanges={
                    production: 1.0,
                    operation: self._get_param('n_cycles_uav') * self._get_param('mission_energy'),
                })
            functional_value = self._get_param('n_cycles_uav') * self._get_param('mission_duration') * self._get_param(
                'mass_payload')
            lcalg.newActivity(
                LCA_USER_DB,
                LCA_MODEL_KEY,
                model_select,
                exchanges={
                    intermediate_model: 1 / functional_value  # normalize by functional value
                })

        elif model_select == "lifetime":
            lcalg.newActivity(  # Impacts over UAV's lifetime
                LCA_USER_DB,
                LCA_MODEL_KEY,
                "uav lifetime",
                exchanges={
                    production: 1.0,
                    operation: self._get_param('n_cycles_uav') * self._get_param('mission_energy'),
                })

    def _add_param(self, param):
        """Add a parameter to the parameters dictionary."""
        self.parameters[param.name] = param

    def _get_param(self, name):
        """Get a parameter from the parameters dictionary."""
        try:
            param = self.parameters[name]
        except KeyError:
            raise FastLcaParameterNotDeclared(name)
        return param

    def _set_parameters(self, parameters_dict: dict, outputs: dict):
        """
        Check if parameter exist, and sets the associated output value.
        Supports only float parameters (no enum or boolean).
        """
        for key, value in parameters_dict.items():
            if key not in self.parameters.keys():
                raise FastLcaParameterNotDeclared(key)
            elif all(isinstance(x, float) for x in value):  # check if parameter is a float
                # if key not in self.input_parameters:
                outputs[LCA_PARAM_KEY + key] = value


class Characterization(om.ExplicitComponent):
    """
    This OpenMDAO component achieves a life cycle inventory (LCI) and a life cycle impact assessment (LCIA)
    for the provided model, parameters and methods.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # parameters
        self.options.declare("parameters", default=dict(), types=dict)  # dictionary for storing non-float parameters

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # option for maximum level of activities to explore in the process tree
        self.options.declare("max_level_processes", default=10, types=int)

        # symbolic expressions of LCA: used for providing analytical partials
        self.exprs_dict = dict()

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # sub activities in model
        self.activities = self.recursive_activities(self.model, max_level=self.options['max_level_processes'])

        # parameters
        # list of parameters is retrieved from LCAmodel with configure() method of parent group.

        # methods
        self.methods = [eval(m) for m in self.options["methods"]]
        self.assert_methods(self.methods)  # check methods exist in brightway project
        self.set_method_labels(self.methods)  # some formatting for labels in lca_algebraic library

        # outputs
        for path in self.activities.keys():
            for m in self.methods:
                m_name = self.method_label_formatting(m)

                # characterized score
                self.add_output(LCA_CHARACTERIZATION_KEY + m_name + path,
                                units=None,  # LCA units not supported by OpenMDAO so set in description
                                desc=bw.Method(m).metadata['unit'] + "/FU")

    # def setup_partials(self):
    #     Declared in configure method of parent group

    def compute(self, inputs, outputs):
        # parameters
        parameters = self.options["parameters"]  # initialized with non-float parameters provided as options
        for key, value in inputs.items():  # add float parameters provided as inputs
            if LCA_PARAM_KEY in key and not np.isnan(value):
                name = re.split(LCA_PARAM_KEY, key)[-1]
                parameters[name] = value

        # LCA calculations (first call may be time-consuming but next calls are faster thanks to cache)
        activities = self.activities  # get all activities and sub activities in model
        for path, act in activities.items():
            res = self.compute_lca(extract_activities=act,
                                   parameters=parameters)  # parametric LCA
            res.index.values[0] = act['name']

            # storing symbolic expression for future use (e.g. analytical derivatives calculation)
            if act not in self.exprs_dict:  # TODO: add option to disable this storage (e.g. only if analytical_derivatives)
                self.exprs_dict[act] = self.lca_model_expression(extract_activities=act)

            # Outputs
            for m in res:
                # get score for method m
                score = res[m][0]  # if a parameter is provided as a list of values, there will be several scores. For now we only get the first one.
                # results from lca_algebraic does not use the same names as the input methods list...
                end = m.find("[")  # TODO: mapping function to improve calculation time?
                m_name = m[:end]
                # set output value
                outputs[LCA_CHARACTERIZATION_KEY + m_name + path] = score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # Parameters dictionary
        parameters = self.options["parameters"]  # initialized with non-float parameters provided as options
        for key, value in inputs.items():  # add float parameters provided as inputs
            if LCA_PARAM_KEY in key and not np.isnan(value):
                name = re.split(LCA_PARAM_KEY, key)[-1]
                parameters[name] = value

        # partials
        for path, act in self.activities.items():
            for m in self.methods:
                m_name = self.method_label_formatting(m)
                output_name = LCA_CHARACTERIZATION_KEY + m_name + path
                for input_name in inputs.keys():
                #for param_name in parameters.keys():
                #    if param_name in inputs.keys():
                #        input_name = LCA_PARAM_KEY + param_name
                    param_name = re.split(LCA_PARAM_KEY, input_name)[-1]
                    partials[output_name,
                             input_name] = self.partials_lca(param_name, m_name, parameters, act)

    def partials_lca(self, input_param, output_method, parameters, activity):
        """
        returns the partial derivative of a method's result with respect to a parameter.
        """
        # Dictionary of algebraic LCA expressions
        exprs_dict = self.exprs_dict

        # Sub-dictionary of expressions for a given activity
        exprs = exprs_dict[activity]

        # Expression for a given method
        expr = exprs[output_method]

        # replace functions that has no well-defined derivatives
        expr_simplified = expr.replace(sym.ceiling, lambda x: x)  # here we replace ceiling function by identity

        # differentiate expression
        derivative = sym.diff(expr_simplified, input_param)

        # expand enumParams --> 'elec_switch_param': 'eu' becomes 'elec_switch_param_us' : 0, 'elec_switch_param_eu': 1
        new_params = {name: value for name, value in lcalg.params._completeParamValues(parameters).items()}
        for key, val in new_params.items():
            if isinstance(val, np.ndarray):
                new_params[key] = val[0]

        #print(input_param, output_method)
        #print(new_params)
        #print(derivative)

        # evaluate derivative
        res = derivative.evalf(subs=new_params)

        #print(res)

        return res

    def compute_lca(self, extract_activities, parameters=None):
        """
        Main LCIA calculation method.
        """
        if extract_activities == self.model:
            extract_activities = None
        else:
            extract_activities = [extract_activities]

        if parameters is None:
            parameters = {}

        res = lcalg.multiLCAAlgebric(
            self.model,  # The model
            self.methods,  # Impact categories / methods

            # List of sub activities to consider
            extract_activities=extract_activities,

            # Parameters of the model
            **parameters
        )
        return res

    def lca_model_expression(self, extract_activities=None):
        """
        computes algebraic expressions corresponding to a model for each method
        """
        exprs = dict()
        if extract_activities == self.model:
            extract_activities = None
        else:
            extract_activities = [extract_activities]
        with lcalg.params.DbContext(self.model):
            exprs_list, _ = lcalg.lca._modelToExpr(self.model, self.methods, extract_activities=extract_activities)
        for i, method in enumerate(self.methods):
            m_name = self.method_label_formatting(method)
            exprs[m_name] = exprs_list[i]
        return exprs

    @staticmethod
    def assert_methods(methods):
        """
        Check if methods exist in brightway.
        """
        for method in methods:
            if method not in bw.methods:
                raise FastLcaMethodDoesNotExist(method)

    @staticmethod
    def set_method_labels(methods):
        """
        Set custom method labels for lca_algebraic results.
        """
        dict_labels = {}
        for m in methods:
            dict_labels[m] = Characterization.method_label_formatting(m)
        lcalg.set_custom_impact_labels(dict_labels)

    @staticmethod
    def method_label_formatting(method_name):
        """
        Format method labels for fast-oad compatibility (handling of variables names).
        """
        new_name = [
            s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')',
                                                                                                               '').replace('/', '_')
            for s in method_name]  # replace invalid characters
        new_name = ':'.join(['%s'] * len(new_name)) % tuple(new_name)  # concatenate method specifications
        return new_name

    @staticmethod
    def recursive_activities(act, max_level: int = 10):
        """Traverse tree of sub-activities of a given activity, until background database is reached."""
        act_dict = dict()

        def _recursive_activities(act, act_dict, act_path: str = "", level: int = 0, max_level: int = 10):
            if act.as_dict()['database'] != LCA_USER_DB or level > max_level:
                return
            act_path = act_path + ":" + act.as_dict()['name'].replace(" ", "_")
            act_dict[act_path] = act
            for exc in act.technosphere():
                _recursive_activities(exc.input, act_dict, act_path, level=level+1, max_level=max_level)
            return

        _recursive_activities(act, act_dict, level=0, max_level=max_level)
        return act_dict


class Normalization(om.ExplicitComponent):
    """
    Normalization of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # option for maximum level of activities to explore in the process tree
        self.options.declare("max_level_processes", default=10, types=int)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # sub activities in model
        self.activities = Characterization.recursive_activities(self.model, max_level=self.options['max_level_processes'])

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for m_name in self.methods:
            self.add_input(LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY,  # normalization factors
                           val=np.nan,
                           units=None)

            for path in self.activities.keys():
                self.add_input(LCA_CHARACTERIZATION_KEY + m_name + path,  # characterized scores
                               val=np.nan,
                               units=None)
                self.add_output(LCA_NORMALIZATION_KEY + m_name + path,  # normalized scores for each activity
                                units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        for m_name in self.methods:
            method_factor = inputs[LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY]
            for path in self.activities.keys():
                score = inputs[LCA_CHARACTERIZATION_KEY + m_name + path]
                normalized_score = score / method_factor
                outputs[LCA_NORMALIZATION_KEY + m_name + path] = normalized_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for m_name in self.methods:
            method_factor = inputs[LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY]
            for path in self.activities.keys():
                score = inputs[LCA_CHARACTERIZATION_KEY + m_name + path]
                partials[LCA_NORMALIZATION_KEY + m_name + path,
                         LCA_CHARACTERIZATION_KEY + m_name + path] = 1 / method_factor
                partials[LCA_NORMALIZATION_KEY + m_name + path,
                         LCA_NORMALIZATION_KEY + m_name + LCA_FACTOR_KEY] = - score / method_factor ** 2


class Weighting(om.ExplicitComponent):
    """
    Weighting of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # option for maximum level of activities to explore in the process tree
        self.options.declare("max_level_processes", default=10, types=int)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # sub activities in model
        self.activities = Characterization.recursive_activities(self.model, max_level=self.options['max_level_processes'])

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for m_name in self.methods:
            self.add_input(LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY,  # weighting factors
                           val=np.nan,
                           units=None)

            for path in self.activities.keys():
                self.add_input(LCA_NORMALIZATION_KEY + m_name + path,  # normalized scores
                               val=np.nan,
                               units=None)
                self.add_output(LCA_WEIGHTING_KEY + m_name + path,  # weighted scores for each activity
                                units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        for path in self.activities.keys():
            # aggregated_score = 0.0
            for m_name in self.methods:
                normalized_score = inputs[LCA_NORMALIZATION_KEY + m_name + path]
                method_weight = inputs[LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY]
                weighted_score = normalized_score * method_weight
                outputs[LCA_WEIGHTING_KEY + m_name + path] = weighted_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for path in self.activities.keys():
            for m_name in self.methods:
                normalized_score = inputs[LCA_NORMALIZATION_KEY + m_name + path]
                method_weight = inputs[LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY]
                partials[LCA_WEIGHTING_KEY + m_name + path, LCA_NORMALIZATION_KEY + m_name + path] = method_weight
                partials[LCA_WEIGHTING_KEY + m_name + path, LCA_WEIGHTING_KEY + m_name + LCA_FACTOR_KEY] = normalized_score


class Aggregation(om.ExplicitComponent):
    """
    Aggregation of the LCIA results.
    """

    def initialize(self):
        # model
        self.model = None

        # sub activities in model
        self.activities = dict()

        # methods
        self.methods = list()
        self.options.declare("methods",
                             default=LCA_DEFAULT_METHOD,
                             types=list)

        # option for maximum level of activities to explore in the process tree
        self.options.declare("max_level_processes", default=10, types=int)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(LCA_USER_DB, LCA_MODEL_KEY)

        # sub activities in model
        self.activities = Characterization.recursive_activities(self.model, max_level=self.options['max_level_processes'])

        # methods
        self.methods = [Characterization.method_label_formatting(eval(m)) for m in self.options["methods"]]

        # inputs
        for path in self.activities.keys():
            for m_name in self.methods:
                self.add_input(LCA_WEIGHTING_KEY + m_name + path,  # weighted scores for each activity
                               val=np.nan,
                               units=None)
            self.add_output(LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path,  # aggregated scores
                            units=None,
                            desc='points')

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # aggregated_score = 0.0
        for path in self.activities.keys():
            aggregated_score = sum([inputs[LCA_WEIGHTING_KEY + m_name + path] for m_name in self.methods])
            outputs[LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path] = aggregated_score

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for path in self.activities.keys():
            for m_name in self.methods:
                partials[LCA_AGGREGATION_KEY + LCA_WEIGHTED_SINGLE_SCORE_KEY + path,
                         LCA_WEIGHTING_KEY + m_name + path] = 1.0