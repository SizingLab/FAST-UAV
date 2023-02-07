"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
#import brightway2 as bw
import lca_algebraic as lcalg

ECOINVENT_PATH = r"D:\THESE\LCA_databases\ecoinvent 3.8_cutoff_ecoSpold02\datasets"
PROJECT_NAME = "bw2_uav"


@oad.RegisterOpenMDAOSystem("fastuav.plugin.lca")
class LCA(om.ExplicitComponent):
    """
    This OpenMDAO component implements an LCA object using brightway2 and lca_algebraic librairies.
    ONLY FOR MULTIROTORS FOR NOW.
    """

    def initialize(self):
        # Setup project
        self.setup_project()

        # Declare parameters for LCA
        self.declare_parameters()

        # Get background activities from EcoInvent
        self.declare_background_activities()

        # Define foreground activities
        self.declare_foreground_activities()

        # Build model to be evaluated in LCA process
        self.build_model()

        # Declare options
        self.options.declare("methods",
                             default=[('ReCiPe 2016 v1.03, midpoint (E)',
                                       'climate change',
                                       'global warming potential (GWP1000)')],
                             types=list)
        self.options.declare("elec_switch_param", default="eu", types=str)
        self.options.declare("functional_unit", default="kg.km", values=["kg.km", "lifetime"])

    def setup(self):
        # UAV parameters
        self.add_input("mission:sizing:main_route:cruise:distance",
                       val=np.nan,
                       units='m')  # check that units are consistence with LCA parameters/activities!!
        self.add_input("mission:sizing:energy", val=np.nan, units='kJ')
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')

        # LCA parameters
        self.add_input("lca:n_cycles", val=1000.0, units=None)
        methods = self.options["methods"]

        # output: LCA scores
        for m in methods:
            m_name = [s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')', '') for s in m]  # replace invalid characters
            m_name = ':'.join(['%s'] * len(m_name)) % tuple(m_name)  # concatenate method specifications
            # m_unit = bw.Method(m).metadata['unit']  # method unit
            m_unit = None  # methods units are not recognized by OpenMDAO (e.g. kg S04-eq)
            self.add_output(m_name, units=m_unit)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV parameters
        d_mission = inputs["mission:sizing:main_route:cruise:distance"] / 1000  # [km]
        e_mission = inputs["mission:sizing:energy"] / 3600  # [kWh]
        m_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]
        m_pay = inputs["mission:sizing:payload:mass"]  # [kg]
        m_bat = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        m_mot = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        m_pro = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        m_esc = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]

        # LCA model
        functional_unit = self.options["functional_unit"]
        if functional_unit == "kg.km" and d_mission > 0:  # TODO: set selection in setup? + Better discrimination
            model = self.normalized_model
        else:
            model = self.model

        # LCA methods and parameters
        methods = self.options["methods"]
        elec_switch_param = self.options["elec_switch_param"]
        n_cycles = inputs["lca:n_cycles"]

        # Set method labels
        dict_labels = {}
        for m in methods:
            m_name = [s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')', '') for s in m]  # replace invalid characters
            m_name = ':'.join(['%s'] * len(m_name)) % tuple(m_name)  # concatenate method specifications
            dict_labels[m] = m_name
        lcalg.set_custom_impact_labels(dict_labels)

        # LCA score
        # TODO: run first time in setup or init for creating cache before analysis (1st call is time consuming)
        df = lcalg.multiLCAAlgebric(
            model,  # The model
            methods,  # Impact categories / methods

            # Parameters of the model
            n_cycles=n_cycles,
            elec_switch_param=elec_switch_param,
            mission_distance=d_mission,
            mission_energy=e_mission,
            mass_payload=m_pay,
            mass_batteries=m_bat,
            mass_motors=m_mot,
            mass_propellers=m_pro,
            mass_airframe=m_airframe,
            mass_controllers=m_esc,
        )

        # Outputs
        for m in df:  # note that df does not use the exact same names as the input methods list...
            score = df[m][0]  # get score for method
            end = m.find("[")
            m_name = m[:end]
            outputs[m_name] = score

    def setup_project(self):
        """
        Create a new brightway2 project.
        """
        # Create/Select the brightway2 project
        lcalg.initProject(PROJECT_NAME)

        # Import background database --> Ecoinvent
        # TODO: set path as option for user
        lcalg.importDb('ecoinvent 3.8_cutoff_ecoSpold02', ECOINVENT_PATH)

        # Import/create foreground database and reset for clean state
        self.USER_DB = USER_DB = 'Foreground DB'
        lcalg.resetDb(USER_DB)
        lcalg.setForeground(USER_DB)

        # Reset project parameters for clean state
        lcalg.resetParams()

    def declare_parameters(self):
        """
        Declare parameters for the parametric LCA
        """

        # High level parameters
        self.param_n_cycles = lcalg.newFloatParam(
            'n_cycles',
            default=100.0,
            min=1, max=10000,
            description="number of cycles",
            dbname=self.USER_DB
        )

        self.param_elec_switch_param = lcalg.newEnumParam(
            'elec_switch_param',
            values=["us", "eu", "fr"],
            default="eu",
            description="Switch on electricty mix",
            dbname=self.USER_DB
        )

        # UAV specific parameters
        self.param_mission_distance = lcalg.newFloatParam(
            'mission_distance',
            default=10.0,
            min=1, max=1000,
            description="distance of sizing mission",
            dbname=self.USER_DB)

        self.param_mission_energy = lcalg.newFloatParam(
            'mission_energy',
            default=0.5,
            min=0, max=1000,
            description="energy consumption for sizing mission",
            dbname=self.USER_DB)

        self.param_mass_payload = lcalg.newFloatParam(
            'mass_payload',
            default=5.0,
            min=0, max=1000,
            description="payload mass used for sizing mission",
            dbname=self.USER_DB)

        self.param_mass_batteries = lcalg.newFloatParam(
            'mass_batteries',
            default=4.08,
            min=0, max=1000,
            description="batteries mass",
            dbname=self.USER_DB)

        self.param_mass_motors = lcalg.newFloatParam(
            'mass_motors',
            default=1.38,
            min=0, max=1000,
            description="motors mass",
            dbname=self.USER_DB)

        self.param_mass_airframe = lcalg.newFloatParam(
            'mass_airframe',
            default=3.50,
            min=0, max=1000,
            description="airframe mass",
            dbname=self.USER_DB)

        self.param_mass_propellers = lcalg.newFloatParam(
            'mass_propellers',
            default=0.35,
            min=0, max=1000,
            description="propellers mass",
            dbname=self.USER_DB)

        self.param_mass_controllers = lcalg.newFloatParam(
            'mass_controllers',
            default=0.54,
            min=0, max=1000,
            description="controllers mass",
            dbname=self.USER_DB)

    def declare_background_activities(self):
        """
        Get background activities from EcoInvent and copy them in our database.
        """

        self.act_battery = lcalg.copyActivity(
            self.USER_DB,
            lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='3cff7e6ccbeae483942dfa12a93a5aec'),
            # [kg] Li-ion NMC 811 battery
            "battery"
        )

        self.act_motor = lcalg.copyActivity(
            self.USER_DB,
            lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='910ad8e5f36aabe962d6bf1c07abff24'),
            # [kg] electric scooter motor
            "motor"
        )

        self.act_propeller = lcalg.copyActivity(
            self.USER_DB,
            lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5f83b772ba1476f12d0b3ef634d4409b'),
            # [kg] CFRP
            "composite"
        )

        self.act_airframe = lcalg.copyActivity(
            self.USER_DB,
            lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5f83b772ba1476f12d0b3ef634d4409b'),
            # [kg] CFRP
            "airframe"
        )

        self.act_controller = lcalg.copyActivity(
            self.USER_DB,
            lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='8c83fa62d7b2654a0bbc8313d13dc892'),
            # [kg] electric scooter controller
            "controller"
        )

        self.act_electricity_eu = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                               code='5915aad8afe41b757f731b8a5ec5d60e')  # [kWh] Europe w/o Switzerland, Low voltage
        self.act_electricity_us = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                               code='12e8a9953a2b09fa316106edc3b0e0da')  # [kWh] Europe w/o Switzerland, Low voltage
        self.act_electricity_fr = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                               code='ab9dc0c0cb4d12b5a1597fd4de0c88db')  # [kWh] Europe w/o Switzerland, Low voltage

        # TODO: check why copyActivity for electricity process returns error in multiLCAAlgebric calculations.
        # self.act_electricity_eu = copyActivity(
        #    self.USER_DB,
        #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5915aad8afe41b757f731b8a5ec5d60e'),  # [kWh] Europe w/o Switzerland, Low voltage
        #    "electricity_eu"
        # )

        # self.act_electricity_us = copyActivity(
        #    self.USER_DB,
        #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='12e8a9953a2b09fa316106edc3b0e0da'),  # [kWh] United States, Low voltage
        #    "electricity_us"
        # )

        # self.act_electricity_fr = copyActivity(
        #    self.USER_DB,
        #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='ab9dc0c0cb4d12b5a1597fd4de0c88db'),  # [kWh] France, Low voltage
        #    "electricity_fr"
        # )

    def declare_foreground_activities(self):
        """
        Declare new foreground activites in our own database.
        The foreground activities are linked to the background activities
        with exchanges that can be parameterized.
        """

        # Create new activites
        self.act_production = lcalg.newActivity(
            self.USER_DB,
            "production",  # Name of the activity
            "kg",  # Unit
            exchanges={  # We define exhanges as a dictionary of 'activity : amount'
                self.act_battery: self.param_mass_batteries,  # Amount can also be a fixed value
                self.act_motor: self.param_mass_motors,
                self.act_airframe: self.param_mass_airframe,
                self.act_propeller: self.param_mass_propellers,
                self.act_controller: self.param_mass_controllers,
            })

        # You can create a virtual "switch" activity combining several activities with a switch parameter
        self.act_operation = lcalg.newSwitchAct(
            self.USER_DB,
            "operation",
            self.param_elec_switch_param,  # Switch parameter
            {  # Dictionnary of enum values / activities
                "us": (self.act_electricity_us, self.param_n_cycles * self.param_mission_energy),
                # You can provide custom amout or formula with a tuple (By default associated amount is 1)
                "eu": (self.act_electricity_eu, self.param_n_cycles * self.param_mission_energy),
                "fr": (self.act_electricity_fr, self.param_n_cycles * self.param_mission_energy),
            })

    def build_model(self):
        """
        Build the model that will be evaluated in the LCA process.
        """

        # Define functional value
        functional_value = self.param_n_cycles * self.param_mission_distance * self.param_mass_payload

        self.model = model = lcalg.newActivity(
            self.USER_DB,  # We define foreground activities in our own DB
            "model",  # Name of the activity
            "uav",  # Functional Unit
            exchanges={
                self.act_production: 1.0,  # Reference the activity we just created
                self.act_operation: 1.0,
            })

        self.normalized_model = lcalg.newActivity(
            self.USER_DB,
            "normalized model",
            "kg.km",
            exchanges={
                model: 1 / functional_value
            })
