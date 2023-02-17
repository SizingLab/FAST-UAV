"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
import brightway2 as bw
from bw2data.parameters import DatabaseParameter
import lca_algebraic as lcalg
import re
from fastuav.exceptions import FastLcaProjectDoesNotExist, \
    FastLcaDatabaseIsNotImported, \
    FastLcaMethodDoesNotExist, \
    FastLcaParameterNotDeclared
from fastuav.constants import DEFAULT_PROJECT, DEFAULT_ECOINVENT, USER_DB, MODEL_KEY, NORM_MODEL_KEY, \
    PARAM_VARIABLE_KEY, RESULTS_VARIABLE_KEY, DEFAULT_METHOD


@oad.RegisterOpenMDAOSystem("fastuav.plugin.lca")
class LCA(om.Group):
    """
    LCA group.
    """

    def initialize(self):
        # Declare options
        self.options.declare("project", default=DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=DEFAULT_ECOINVENT, types=str)
        self.options.declare("model", default="model", values=["model", "normalized_model"])
        self.options.declare("methods",
                             default=DEFAULT_METHOD,
                             types=list)

        # model specific parameters
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

    def setup(self):
        self.add_subsystem("model",
                           LCAmodel(project=self.options["project"],
                                    database=self.options["database"],
                                    model=self.options["model"]),
                           promotes=["*"])
        self.add_subsystem("calculation", LCAcalc(methods=self.options["methods"]))
        # promote for 'calculation' component is done in configure() method

    def configure(self):
        """
        Set inputs and options for `calculation` component by copying `model` outputs and options.
        Configure() method from the containing group is necessary to get access to `model` metadata after Setup().
        """
        # Set outputs from 'model' as inputs for 'calculation' (for float LCA parameters)
        meta = self.model.get_io_metadata('output', includes=PARAM_VARIABLE_KEY + '*')
        for key in meta.keys():
            self.calculation.add_input(key, shape_by_conn=True, val=np.nan, units=None)

        # Copy options (for non-float LCA parameters)
        parameters = self.model.options["parameters"]
        for key in parameters.keys():
            if key in self.options["parameters"].keys():
                parameters[key] = self.options["parameters"][key]
        self.calculation.options["parameters"] = parameters

        # Promote variables and declare partials
        self.promotes('calculation', any=['*'])
        self.calculation.declare_partials("*", "*", method="fd")


class LCAmodel(om.ExplicitComponent):
    """
    This OpenMDAO component implements an LCA model using brightway2 and lca_algebraic librairies.
    It creates an LCA model and sets the LCA parameters for further parametric LCA calculation.
    """

    def initialize(self):
        # Attributes
        self.parameters = dict()  # dictionary of {parameter_name: parameter_object}

        # Declare options
        self.options.declare("project", default=DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=DEFAULT_ECOINVENT, types=str)
        self.options.declare("model", default="model", values=["model", "normalized_model"])
        self.options.declare("parameters", default=dict(), types=dict)

        # Setup project
        self.setup_project()

        # Declare parameters for LCA
        self.declare_parameters()

        # Non-float parameters are declared as options (cannot be passed as variables with add_output method)
        for name, parameter in self.parameters.items():
            if parameter.type != 'float':
                self.options.declare(name, default=parameter.default, values=parameter.values)

        # Define foreground activities
        self.declare_foreground_activities()

        # Build model to be evaluated in LCA process
        self.build_model()

    def setup(self):
        # UAV Parameters
        self.add_input("mission:sizing:main_route:cruise:distance",
                       val=np.nan,
                       units='m')  # check that units are consistence with LCA parameters/activities!
        self.add_input("mission:sizing:energy", val=np.nan, units='kJ')
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')
        self.add_input("lca:mission:n_cycles", val=1000.0, units=None)

        # LCA parameters
        for name, object in self.parameters.items():  # loop through parameters defined in initialize() method
            if object.type == 'float':  # get float parameters
                self.add_output(PARAM_VARIABLE_KEY + name, val=np.nan, units=None)  # add them as outputs
            else:
                self.options["parameters"][name] = object.default

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Parameters
        n_cycles = inputs["lca:mission:n_cycles"]
        mission_distance = inputs["mission:sizing:main_route:cruise:distance"] / 1000  # [km]
        mass_payload = inputs["mission:sizing:payload:mass"]  # [kg]
        mission_energy = inputs["mission:sizing:energy"] / 3600  # [kWh]
        mass_batteries = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        mass_motors = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        mass_propellers = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        mass_controllers = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
        mass_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"] # [kg]

        # special case for model normalized by n_cycles * distance * payload
        if self.options["model"] == "normalized_model":
            if n_cycles == 0:
                n_cycles = 1e-9
            if mission_distance == 0:
                mission_distance = 1e-9
            if mass_payload == 0:
                mass_payload = 1e-9

        # set values for lca parameters (only for float parameters;
        # non-float parameters are automatically declared as options)
        parameters_dict = {
            'n_cycles': n_cycles,
            'mission_distance': mission_distance,
            'mission_energy': mission_energy,
            'mass_payload': mass_payload,
            'mass_batteries': mass_batteries,
            'mass_motors': mass_motors,
            'mass_propellers': mass_propellers,
            'mass_controllers': mass_controllers,
            'mass_airframe': mass_airframe,
        }
        self._set_parameters(parameters_dict, outputs)

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
        lcalg.resetDb(USER_DB)
        lcalg.setForeground(USER_DB)

        # Reset project parameters for clean state
        lcalg.resetParams()

    def declare_parameters(self):
        """
        Declare parameters for the parametric LCA
        """

        # Float parameters are created with the 'newFloatParam' method
        self._add_param(lcalg.newFloatParam(
            'n_cycles',  # name of the parameter
            default=1.0,  # default value
            min=1, max=10000,  # bounds (only for DoE purposes)
            description="number of cycles",  # description
            dbname=USER_DB  # we define the parameter in our own database
        ))

        # Enum parameters are a facility to represent different options
        # and should be used with the 'newSwitchAct' method
        self._add_param(lcalg.newEnumParam(
            'elec_switch_param',
            values=["us", "eu", "fr"],  # values this parameter can take
            default="eu",
            description="switch on electricity mix",
            dbname=USER_DB
        ))

        # UAV specific parameters
        self._add_param(lcalg.newFloatParam(
            'mission_distance',
            default=1.0,
            min=1, max=1000,
            description="distance of sizing mission",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mission_energy',
            default=1.0,
            min=0, max=1000,
            description="energy consumption for sizing mission",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_payload',
            default=1.0,
            min=0, max=1000,
            description="payload mass used for sizing mission",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_batteries',
            default=1.0,
            min=0, max=1000,
            description="batteries mass",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_motors',
            default=1.0,
            min=0, max=1000,
            description="motors mass",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_airframe',
            default=1.0,
            min=0, max=1000,
            description="airframe mass",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_propellers',
            default=1.0,
            min=0, max=1000,
            description="propellers mass",
            dbname=USER_DB
        ))

        self._add_param(lcalg.newFloatParam(
            'mass_controllers',
            default=1.0,
            min=0, max=1000,
            description="controllers mass",
            dbname=USER_DB
        ))

    def declare_foreground_activities(self):
        """
        Declare new foreground activities in our own database.
        The foreground activities are linked to the background activities
        with exchanges that can be parameterized.
        """

        # References to some background activities
        battery_li_ion = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                            code='3cff7e6ccbeae483942dfa12a93a5aec')  # [kg] Li-ion NMC 811 battery
        motor_scooter = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                           code='910ad8e5f36aabe962d6bf1c07abff24')  # [kg] electric scooter motor
        controller_scooter = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                                code='8c83fa62d7b2654a0bbc8313d13dc892')  # [kg] electric scooter controller
        composite = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                       code='5f83b772ba1476f12d0b3ef634d4409b')  # [kg] CFRP
        aluminium = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                       code='fa1a2d6fc65234a5a873c6776d8fd6fb')  # [kg] aluminium
        electricity_eu = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                            code='5915aad8afe41b757f731b8a5ec5d60e')  # [kWh] Europe w/o Switzerland, Low voltage
        electricity_us = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                            code='12e8a9953a2b09fa316106edc3b0e0da')  # [kWh] United States, Low voltage
        electricity_fr = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
                                            code='ab9dc0c0cb4d12b5a1597fd4de0c88db')  # [kWh] France, Low voltage

        # Battery activity
        battery = lcalg.newActivity(
            USER_DB,  # we define foreground activities in our own database
            "battery",  # Name of the activity
            "kg",  # Unit
            exchanges={  # We define exchanges as a dictionary of 'activity : amount'
                battery_li_ion: 1.0,  # Amount can be a fixed value or a parameter (see higher-level activities)
                # battery_li_ion: 0.5,  # for instance, we can have half Li-Ion batteries...
                # battery_na_cl: 0.5,  # and half NaCl batteries
            }
        )

        # Motor activity
        motor = lcalg.newActivity(
            USER_DB,
            "motor",
            "kg",
            exchanges={
                motor_scooter: 1.0,
            }
        )

        # ESC activity
        controller = lcalg.newActivity(
            USER_DB,
            "controller",
            "kg",
            exchanges={
                controller_scooter: 1.0,
            }
        )

        # Propeller activity
        propeller = lcalg.newActivity(
            USER_DB,
            "propeller",
            "kg",
            exchanges={
                composite: 1.0,
            }
        )

        # Airframe activity
        airframe = lcalg.newActivity(
            USER_DB,
            "airframe",
            "kg",
            exchanges={
                composite: 1.0,
                # aluminium: 0.8,  # one may also set a lower amount to account for recycled materials
            }
        )

        # Production activity: assembly of the previously defined components
        lcalg.newActivity(
            USER_DB,
            "production",
            "uav",  # unit is one uav
            exchanges={  # we refer directly to the
                battery: self._get_param('mass_batteries'),  # Amount is a parameter
                motor: self._get_param('mass_motors'),
                airframe: self._get_param('mass_airframe'),
                propeller: self._get_param('mass_propellers'),
                controller: self._get_param('mass_controllers'),
            }
        )

        # Operation activity: electricity used for flying.
        # This is a switch activity. One may choose between different type of sub-activities (here, electricity mix)
        lcalg.newSwitchAct(
            USER_DB,
            "operation",
            self._get_param('elec_switch_param'),  # Switch parameter previously defined
            {  # Dictionary of enum values / activities
                "us": (electricity_us,
                       self._get_param('n_cycles') * self._get_param('mission_energy')),
                "eu": (electricity_eu,
                       self._get_param('n_cycles') * self._get_param('mission_energy')),
                "fr": (electricity_fr,
                       self._get_param('n_cycles') * self._get_param('mission_energy')),
            }
        )

    def build_model(self):
        """
        Build the model that will be evaluated in the LCA process.
        """

        # Retrieve some previously defined foreground activities
        production = lcalg.getActByCode(USER_DB, "production")
        operation = lcalg.getActByCode(USER_DB, "operation")

        # Define model
        model_select = self.options["model"]
        if model_select == "model":
            lcalg.newActivity(
                USER_DB,
                MODEL_KEY,  # Name of the model
                "uav lifetime",  # Functional Unit: one uav on its entire lifetime
                exchanges={
                    production: 1.0,  # Reference the activity we just created
                    operation: 1.0,
                })

        elif model_select == "normalized_model":
            intermediate_model = lcalg.newActivity(
                USER_DB,
                "intermediate_model",
                "uav lifetime",
                exchanges={
                    production: 1.0,
                    operation: 1.0,
                })

            # Normalized model (different functional unit)
            functional_value = self._get_param('n_cycles') * self._get_param('mission_distance') * self._get_param(
                'mass_payload')
            lcalg.newActivity(
                USER_DB,
                MODEL_KEY,
                "kg.km",  # function unit: one kg payload carried on one km
                exchanges={
                    intermediate_model: 1 / functional_value
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
                outputs[PARAM_VARIABLE_KEY + key] = value


class LCAcalc(om.ExplicitComponent):
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
                             default=DEFAULT_METHOD,
                             types=list)

    def setup(self):
        # model
        self.model = lcalg.getActByCode(USER_DB, MODEL_KEY)

        # sub activities in model
        self.activities = self.recursive_activities(self.model)

        # parameters
        # list of parameters is retrieved from LCAmodel with configure() method of parent group.

        # methods
        self.methods = [eval(m) for m in self.options["methods"]]
        self.assert_methods(self.methods)  # check methods exist in brightway project
        self.set_method_labels(self.methods)  # some formatting for labels

        # outputs: LCA scores
        for m in self.methods:
            m_name = self.method_label_formatting(m)
            # m_unit = bw.Method(m).metadata['unit']  # method unit
            m_unit = None  # methods units are not recognized by OpenMDAO (e.g. kg S04-eq)
            for path in self.activities.keys():
                self.add_output(RESULTS_VARIABLE_KEY + m_name + path, units=m_unit)

    # def setup_partials(self):
    #     Declared in configure method of parent group

    def compute(self, inputs, outputs):
        # model
        model = self.model

        # parameters
        parameters = self.options["parameters"]  # initialized with non-float parameters
        for key, value in inputs.items():  # add float parameters
            if PARAM_VARIABLE_KEY in key and not np.isnan(value):
                name = re.split(PARAM_VARIABLE_KEY, key)[-1]
                parameters[name] = value

        # methods
        methods = self.methods

        # LCA calculations (first call may be time-consuming but next calls are faster thanks to cache)
        activities = self.activities  # get all activities and sub activities in model
        for path, act in activities.items():
            if act == model:
                extract_activities = None
            else:
                extract_activities = [act]
            res = lcalg.multiLCAAlgebric(
                model,  # The model
                methods,  # Impact categories / methods

                # List of sub activities to consider
                extract_activities=extract_activities,

                # Parameters of the model
                **parameters
            )
            res.index.values[0] = act['name']

            # Outputs
            for m in res:
                # get score for method m
                score = res[m][0]
                # results from lca_algebraic does not use the same names as the input methods list...
                end = m.find("[")  # TODO: mapping function to improve calculation time?
                m_name = m[:end]
                # set output value
                outputs[RESULTS_VARIABLE_KEY + m_name + path] = score

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
            dict_labels[m] = LCAcalc.method_label_formatting(m)
        lcalg.set_custom_impact_labels(dict_labels)

    @staticmethod
    def method_label_formatting(method_name):
        """
        Format method labels for fast-oad compatibility (handling of variables names).
        """
        new_name = [
            s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')',
                                                                                                               '')
            for s in method_name]  # replace invalid characters
        new_name = ':'.join(['%s'] * len(new_name)) % tuple(new_name)  # concatenate method specifications
        return new_name

    @staticmethod
    def recursive_activities(act):
        """Traverse tree of sub-activities of a given activity, until background database is reached."""
        act_dict = dict()

        def _recursive_activities(act, act_dict, act_path: str = ""):
            if act.as_dict()['database'] != 'Foreground DB':
                return
            act_path = act_path + ":" + act.as_dict()['name'].replace(" ", "_")
            act_dict[act_path] = act
            for exc in act.technosphere():
                _recursive_activities(exc.input, act_dict, act_path)
            return

        _recursive_activities(act, act_dict)
        return act_dict


# @oad.RegisterOpenMDAOSystem("fastuav.plugin.lcatest")
# class LCAtest(om.ExplicitComponent):
#     """
#     This OpenMDAO component implements an LCA object using brightway2 and lca_algebraic librairies.
#     ONLY FOR MULTIROTORS FOR NOW.
#     """
#
#     def initialize(self):
#         # Declare options
#         self.options.declare("project", default=DEFAULT_PROJECT, types=str)
#         self.options.declare("database", default=DEFAULT_ECOINVENT, types=str)
#         self.options.declare("methods",
#                              default=[
#                                  "('ReCiPe 2016 v1.03, midpoint (E) no LT', "
#                                  "'climate change no LT', "
#                                  "'global warming potential (GWP1000) no LT')",
#                              ],
#                              types=list)
#         self.options.declare("elec_switch_param", default="eu", values=["eu", "fr", "us"])
#         self.options.declare("functional_unit", default="kg.km", values=["kg.km", "lifetime"])
#
#         # Setup project
#         self.setup_project()
#
#         # Declare parameters for LCA
#         self.declare_parameters()
#
#         # Get background activities from EcoInvent
#         self.declare_background_activities()
#
#         # Define foreground activities
#         self.declare_foreground_activities()
#
#         # Build model to be evaluated in LCA process
#         self.build_model()
#
#     def setup(self):
#         # UAV parameters
#         self.add_input("mission:sizing:main_route:cruise:distance",
#                        val=np.nan,
#                        units='m')  # check that units are consistence with LCA parameters/activities!
#         self.add_input("mission:sizing:energy", val=np.nan, units='kJ')
#         self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')
#         self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
#         self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
#         self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
#         self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
#         self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
#         self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
#         self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')
#
#         # LCA parameters
#         self.add_input("lca:n_cycles", val=1000.0, units=None)
#         methods = [eval(m) for m in self.options["methods"]]
#
#         # output: LCA scores
#         for m in methods:
#             m_name = [s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')', '') for s in m]  # replace invalid characters
#             m_name = ':'.join(['%s'] * len(m_name)) % tuple(m_name)  # concatenate method specifications
#             # m_unit = bw.Method(m).metadata['unit']  # method unit
#             m_unit = None  # methods units are not recognized by OpenMDAO (e.g. kg S04-eq)
#             self.add_output(m_name, units=m_unit)
#
#     def setup_partials(self):
#         self.declare_partials("*", "*", method="fd")
#
#     def compute(self, inputs, outputs):
#         # UAV parameters
#         d_mission = inputs["mission:sizing:main_route:cruise:distance"] / 1000  # [km]
#         e_mission = inputs["mission:sizing:energy"] / 3600  # [kWh]
#         m_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]
#         m_pay = inputs["mission:sizing:payload:mass"]  # [kg]
#         m_bat = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
#         N_pro = inputs["data:propulsion:multirotor:propeller:number"]
#         m_mot = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
#         m_pro = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
#         m_esc = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
#
#         # LCA model
#         functional_unit = self.options["functional_unit"]
#         if functional_unit == "kg.km" and d_mission > 0:  # TODO: set selection in setup? + Better discrimination
#             model = self.normalized_model
#         else:
#             model = self.model
#
#         # LCA methods and parameters
#         methods = [eval(m) for m in self.options["methods"]]
#         elec_switch_param = self.options["elec_switch_param"]
#         n_cycles = inputs["lca:n_cycles"]
#
#         # Set method labels
#         dict_labels = {}
#         for m in methods:
#             m_name = [s.replace(':', '-').replace('.', '_').replace(',', ':').replace(' ', '_').replace('(', '').replace(')', '') for s in m]  # replace invalid characters
#             m_name = ':'.join(['%s'] * len(m_name)) % tuple(m_name)  # concatenate method specifications
#             dict_labels[m] = m_name
#         lcalg.set_custom_impact_labels(dict_labels)
#
#         # LCA score
#         # TODO: run first time in setup or init for creating cache before analysis (1st call is time consuming)
#         df = lcalg.multiLCAAlgebric(
#             model,  # The model
#             methods,  # Impact categories / methods
#
#             # Parameters of the model
#             n_cycles=n_cycles,
#             elec_switch_param=elec_switch_param,
#             mission_distance=d_mission,
#             mission_energy=e_mission,
#             mass_payload=m_pay,
#             mass_batteries=m_bat,
#             mass_motors=m_mot,
#             mass_propellers=m_pro,
#             mass_airframe=m_airframe,
#             mass_controllers=m_esc,
#         )
#
#         # Outputs
#         for m in df:  # note that df does not use the exact same names as the input methods list...
#             score = df[m][0]  # get score for method
#             end = m.find("[")
#             m_name = m[:end]
#             outputs[m_name] = score
#
#     def setup_project(self):
#         """
#         Set and initialize lca project.
#         """
#         project_name = self.options["project"]
#         database_name = self.options["database"]
#         methods = [eval(m) for m in self.options["methods"]]
#
#         # Check project already exists
#         if project_name not in bw.projects:
#             raise FastLcaProjectDoesNotExist(project_name)
#
#         # Set current project
#         bw.projects.set_current(project_name)
#
#         # Check EcoInvent has been imported in project
#         if database_name not in list(bw.databases):
#             raise FastLcaDatabaseIsNotImported(project_name, database_name)
#
#         # Check if methods exist in brightway
#         for method in methods:
#             if method not in bw.methods:
#                 raise FastLcaMethodDoesNotExist(method)
#
#         # Import/create foreground database and reset for clean state
#         lcalg.resetDb(USER_DB)
#         lcalg.setForeground(USER_DB)
#
#         # Reset project parameters for clean state
#         lcalg.resetParams()
#
#     def declare_parameters(self):
#         """
#         Declare parameters for the parametric LCA
#         """
#
#         # High level parameters
#         self.param_n_cycles = lcalg.newFloatParam(
#             'n_cycles',
#             default=100.0,
#             min=1, max=10000,
#             description="number of cycles",
#             dbname=USER_DB
#         )
#
#         self.param_elec_switch_param = lcalg.newEnumParam(
#             'elec_switch_param',
#             values=["us", "eu", "fr"],
#             default="eu",
#             description="Switch on electricty mix",
#             dbname=USER_DB
#         )
#
#         # UAV specific parameters
#         self.param_mission_distance = lcalg.newFloatParam(
#             'mission_distance',
#             default=10.0,
#             min=1, max=1000,
#             description="distance of sizing mission",
#             dbname=USER_DB)
#
#         self.param_mission_energy = lcalg.newFloatParam(
#             'mission_energy',
#             default=0.5,
#             min=0, max=1000,
#             description="energy consumption for sizing mission",
#             dbname=USER_DB)
#
#         self.param_mass_payload = lcalg.newFloatParam(
#             'mass_payload',
#             default=5.0,
#             min=0, max=1000,
#             description="payload mass used for sizing mission",
#             dbname=USER_DB)
#
#         self.param_mass_batteries = lcalg.newFloatParam(
#             'mass_batteries',
#             default=4.08,
#             min=0, max=1000,
#             description="batteries mass",
#             dbname=USER_DB)
#
#         self.param_mass_motors = lcalg.newFloatParam(
#             'mass_motors',
#             default=1.38,
#             min=0, max=1000,
#             description="motors mass",
#             dbname=USER_DB)
#
#         self.param_mass_airframe = lcalg.newFloatParam(
#             'mass_airframe',
#             default=3.50,
#             min=0, max=1000,
#             description="airframe mass",
#             dbname=USER_DB)
#
#         self.param_mass_propellers = lcalg.newFloatParam(
#             'mass_propellers',
#             default=0.35,
#             min=0, max=1000,
#             description="propellers mass",
#             dbname=USER_DB)
#
#         self.param_mass_controllers = lcalg.newFloatParam(
#             'mass_controllers',
#             default=0.54,
#             min=0, max=1000,
#             description="controllers mass",
#             dbname=USER_DB)
#
#     def declare_background_activities(self):
#         """
#         Get background activities from EcoInvent and copy them in our database.
#         """
#
#         self.act_battery = lcalg.copyActivity(
#             USER_DB,
#             lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='3cff7e6ccbeae483942dfa12a93a5aec'),
#             # [kg] Li-ion NMC 811 battery
#             "battery"
#         )
#
#         self.act_motor = lcalg.copyActivity(
#             USER_DB,
#             lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='910ad8e5f36aabe962d6bf1c07abff24'),
#             # [kg] electric scooter motor
#             "motor"
#         )
#
#         self.act_propeller = lcalg.copyActivity(
#             USER_DB,
#             lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5f83b772ba1476f12d0b3ef634d4409b'),
#             # [kg] CFRP
#             "composite"
#         )
#
#         self.act_airframe = lcalg.copyActivity(
#             USER_DB,
#             lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5f83b772ba1476f12d0b3ef634d4409b'),
#             # [kg] CFRP
#             "airframe"
#         )
#
#         self.act_controller = lcalg.copyActivity(
#             USER_DB,
#             lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='8c83fa62d7b2654a0bbc8313d13dc892'),
#             # [kg] electric scooter controller
#             "controller"
#         )
#
#         self.act_electricity_eu = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
#                                                code='5915aad8afe41b757f731b8a5ec5d60e')  # [kWh] Europe w/o Switzerland, Low voltage
#         self.act_electricity_us = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
#                                                code='12e8a9953a2b09fa316106edc3b0e0da')  # [kWh] Europe w/o Switzerland, Low voltage
#         self.act_electricity_fr = lcalg.findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02',
#                                                code='ab9dc0c0cb4d12b5a1597fd4de0c88db')  # [kWh] Europe w/o Switzerland, Low voltage
#
#         # TODO: check why copyActivity for electricity process returns error in multiLCAAlgebric calculations.
#         # self.act_electricity_eu = copyActivity(
#         #    USER_DB,
#         #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='5915aad8afe41b757f731b8a5ec5d60e'),  # [kWh] Europe w/o Switzerland, Low voltage
#         #    "electricity_eu"
#         # )
#
#         # self.act_electricity_us = copyActivity(
#         #    USER_DB,
#         #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='12e8a9953a2b09fa316106edc3b0e0da'),  # [kWh] United States, Low voltage
#         #    "electricity_us"
#         # )
#
#         # self.act_electricity_fr = copyActivity(
#         #    USER_DB,
#         #    findActivity(db_name='ecoinvent 3.8_cutoff_ecoSpold02', code='ab9dc0c0cb4d12b5a1597fd4de0c88db'),  # [kWh] France, Low voltage
#         #    "electricity_fr"
#         # )
#
#     def declare_foreground_activities(self):
#         """
#         Declare new foreground activites in our own database.
#         The foreground activities are linked to the background activities
#         with exchanges that can be parameterized.
#         """
#
#         # Create new activites
#         self.act_production = lcalg.newActivity(
#             USER_DB,
#             "production",  # Name of the activity
#             "kg",  # Unit
#             exchanges={  # We define exhanges as a dictionary of 'activity : amount'
#                 self.act_battery: self.param_mass_batteries,  # Amount can also be a fixed value
#                 self.act_motor: self.param_mass_motors,
#                 self.act_airframe: self.param_mass_airframe,
#                 self.act_propeller: self.param_mass_propellers,
#                 self.act_controller: self.param_mass_controllers,
#             })
#
#         # You can create a virtual "switch" activity combining several activities with a switch parameter
#         self.act_operation = lcalg.newSwitchAct(
#             USER_DB,
#             "operation",
#             self.param_elec_switch_param,  # Switch parameter
#             {  # Dictionnary of enum values / activities
#                 "us": (self.act_electricity_us, self.param_n_cycles * self.param_mission_energy),
#                 # You can provide custom amout or formula with a tuple (By default associated amount is 1)
#                 "eu": (self.act_electricity_eu, self.param_n_cycles * self.param_mission_energy),
#                 "fr": (self.act_electricity_fr, self.param_n_cycles * self.param_mission_energy),
#             })
#
#     def build_model(self):
#         """
#         Build the model that will be evaluated in the LCA process.
#         """
#
#         # Define functional value
#         functional_value = self.param_n_cycles * self.param_mission_distance * self.param_mass_payload
#
#         self.model = model = lcalg.newActivity(
#             USER_DB,  # We define foreground activities in our own DB
#             "model",  # Name of the activity
#             "uav",  # Functional Unit
#             exchanges={
#                 self.act_production: 1.0,  # Reference the activity we just created
#                 self.act_operation: 1.0,
#             })
#
#         self.normalized_model = lcalg.newActivity(
#             USER_DB,
#             "normalized model",
#             "kg.km",
#             exchanges={
#                 model: 1 / functional_value
#             })
