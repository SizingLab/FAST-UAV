"""
Module for Life Cycle Assessment.
"""

import fastoad.api as oad
import openmdao.api as om
import numpy as np
import brightway2 as bw
import lca_algebraic as lcalg
import re
from sympy import ceiling
from fastuav.exceptions import FastLcaProjectDoesNotExist, \
    FastLcaDatabaseIsNotImported, \
    FastLcaMethodDoesNotExist, \
    FastLcaParameterNotDeclared
from fastuav.constants import DEFAULT_PROJECT, DEFAULT_ECOINVENT, USER_DB, MODEL_KEY, \
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
        self.options.declare("model", default="kg.km", types=str)
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
        self.parameters = dict()  # dictionary of {parameter_name: parameter_object} to store all parameters

        # Declare options
        self.options.declare("project", default=DEFAULT_PROJECT, types=str)
        self.options.declare("database", default=DEFAULT_ECOINVENT, types=str)
        self.options.declare("model", default="kg.km", values=["kg.km", "kg.h"])
        self.options.declare("parameters", default=dict(), types=dict)  # for storing non-float parameters

    def setup(self):
        # Setup project
        self.setup_project()

        # Declare parameters for LCA
        self.declare_parameters()

        # Define foreground activities
        self.declare_foreground_activities()

        # Build model to be evaluated in LCA process
        self.build_model()

        # UAV Parameters
        self.add_input("mission:sizing:main_route:cruise:distance",
                       val=np.nan,
                       units='m')  # NB: check that units are consistence with LCA parameters/activities!
        self.add_input("mission:sizing:duration", val=np.nan, units='min')
        self.add_input("mission:sizing:energy", val=np.nan, units='kJ')
        self.add_input("mission:sizing:payload:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:battery:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:arms:mass", val=np.nan, units='kg')
        self.add_input("data:weight:airframe:body:mass", val=np.nan, units='kg')
        self.add_input("data:propulsion:multirotor:propeller:number", val=np.nan, units=None)
        self.add_input("data:weight:propulsion:multirotor:motor:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:propeller:mass", val=np.nan, units='kg')
        self.add_input("data:weight:propulsion:multirotor:esc:mass", val=np.nan, units='kg')
        self.add_input("lca:mission:n_cycles", val=2000.0, units=None)
        self.add_input("lca:mission:n_cycles_battery", val=1000.0, units=None)

        # LCA parameters
        for name, object in self.parameters.items():  # loop through parameters defined in initialize() method
            if object.type == 'float':  # add float parameters as outputs
                self.add_output(PARAM_VARIABLE_KEY + name, val=np.nan, units=None)  # add them as outputs
            else:  # add non-float parameters as options
                self.options["parameters"][name] = object.default

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Parameters
        n_cycles = inputs["lca:mission:n_cycles"]
        n_cycles_battery = inputs["lca:mission:n_cycles_battery"]
        mission_distance = inputs["mission:sizing:main_route:cruise:distance"] / 1000  # [km]
        mission_duration = inputs["mission:sizing:duration"] / 60  # [h]
        mass_payload = inputs["mission:sizing:payload:mass"]  # [kg]
        mission_energy = inputs["mission:sizing:energy"] / 3600  # [kWh]
        mass_batteries = inputs["data:weight:propulsion:multirotor:battery:mass"]  # [kg]
        N_pro = inputs["data:propulsion:multirotor:propeller:number"]
        mass_motors = inputs["data:weight:propulsion:multirotor:motor:mass"] * N_pro  # [kg]
        mass_propellers = inputs["data:weight:propulsion:multirotor:propeller:mass"] * N_pro  # [kg]
        mass_controllers = inputs["data:weight:propulsion:multirotor:esc:mass"] * N_pro  # [kg]
        mass_airframe = inputs["data:weight:airframe:body:mass"] + inputs["data:weight:airframe:arms:mass"]  # [kg]

        # Data manipulation for normalized models (avoid division by zero)
        eps = np.array([1e-9])
        n_cycles = max(n_cycles, eps)
        mission_distance = max(mission_distance, eps)
        mission_duration = max(mission_duration, eps)
        mass_payload = max(mass_payload, eps)

        # set values for lca parameters (only for float parameters;
        # non-float parameters are automatically declared as options)
        parameters_dict = {
            'n_cycles': n_cycles,
            'n_cycles_battery': n_cycles_battery,
            'mission_distance': mission_distance,
            'mission_duration': mission_duration,
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

        self._add_param(lcalg.newFloatParam(
            'n_cycles_battery',
            default=1.0,
            min=0, max=10000,
            description="maximum number of cycles for battery technology",
            dbname=USER_DB
        ))

        # Enum parameters are a facility to represent different options
        # and should be used with the 'newSwitchAct' method
        self._add_param(lcalg.newEnumParam(
            'battery_type',
            values=["nmc_811", "nmc_111", "nca", "lfp", "nimh"],  # values this parameter can take
            default="nmc_811",
            description="battery technology",
            dbname=USER_DB
        ))

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
            'mission_duration',
            default=1.0,
            min=1, max=1000,
            description="duration of sizing mission",
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

        db_ecoinvent = self.options["database"]

        # References to some background activities
        # All activities here are of market type (i.e., transports are taken into account)
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
        motor_scooter = lcalg.findActivity(db_name=db_ecoinvent,
                                           code='a9f8412fe79b4fe74771ddfbeebb3f98')  # [kg] electric scooter motor
        controller_scooter = lcalg.findActivity(db_name=db_ecoinvent,
                                                code='9afe5ffc45f1b043596a7901a59c98eb')  # [kg] electric scooter controller
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

        # Motor activity
        motor = lcalg.newActivity(
            USER_DB,  # we define foreground activities in our own database
            "motor",  # Name of the activity
            "kg",  # We define exchanges as a dictionary of 'activity : amount'
            exchanges={
                motor_scooter: 1.0,   # Amount can be a fixed value or a parameter
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

        # Battery activity
        # battery = lcalg.newActivity(
        #     USER_DB,
        #     "battery",
        #     "kg",  # Unit
        #     exchanges={
        #         battery_nmc_811: 1.0,
        #         # battery_nmc_811: 0.5,  # for instance, we can have half NMC 811 batteries...
        #         # battery_nca: 0.5,  # and half NCA batteries
        #     }
        # )

        # This is a switch activity. One may choose between different type of sub-activities (here, electricity mix)
        battery = lcalg.newSwitchAct(
            USER_DB,
            "battery",
            self._get_param('battery_type'),  # Switch parameter previously defined
            {  # Dictionary of enum values / activities : {"switch_option": (activity, amount)}
                "nmc_811": (
                    battery_nmc_811,
                    ceiling(self._get_param('n_cycles') / self._get_param('n_cycles_battery'))
                ),
                "nmc_111": (
                    battery_nmc_111,
                    ceiling(self._get_param('n_cycles') / self._get_param('n_cycles_battery'))
                ),
                "nca": (
                    battery_nca,
                    ceiling(self._get_param('n_cycles') / self._get_param('n_cycles_battery'))
                ),
                "lfp": (
                    battery_lfp,
                    ceiling(self._get_param('n_cycles') / self._get_param('n_cycles_battery'))
                ),
                "nimh": (
                    battery_nimh,
                    ceiling(self._get_param('n_cycles') / self._get_param('n_cycles_battery'))
                ),
            }
        )

        # Production activity: assembly of the previously defined components
        lcalg.newActivity(
            USER_DB,
            "production",
            "uav",  # unit is one uav
            exchanges={  # we refer directly to the
                battery: self._get_param('mass_batteries'),  # Amount is a formula
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

        # Impacts over UAV's lifetime
        intermediate_model = lcalg.newActivity(
            USER_DB,
            "model",
            "uav lifetime",
            exchanges={
                production: 1.0,
                operation: 1.0,
            })

        # Normalize the impacts
        if model_select == "kg.km":  # 1 kg payload on 1 km
            # functional value to normalize
            functional_value = self._get_param('n_cycles') * self._get_param('mission_distance') * self._get_param(
                'mass_payload')
            lcalg.newActivity(
                USER_DB,
                MODEL_KEY,
                "kg.km",  # functional unit: one kg payload carried on one km
                exchanges={
                    intermediate_model: 1 / functional_value
                })

        elif model_select == "kg.h":  # 1 kg payload during 1 hour
            # functional value to normalize
            functional_value = self._get_param('n_cycles') * self._get_param('mission_duration') * self._get_param(
                'mass_payload')
            lcalg.newActivity(
                USER_DB,
                MODEL_KEY,
                "kg.h",  # functional unit: one kg payload carried during one hour
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
            # units.add_unit(m_unit, "kg")
            m_unit = None  # methods units are not recognized by OpenMDAO (e.g. kg S04-eq)
            for path in self.activities.keys():
                self.add_output(RESULTS_VARIABLE_KEY + m_name + path,
                                units=m_unit,
                                desc=bw.Method(m).metadata['unit'] + "/FU")

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
                score = res[m][0]  # if a parameter is provided as a list of values, there will be several scores. For now we only get the first one.
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
            if act.as_dict()['database'] != USER_DB:
                return
            act_path = act_path + ":" + act.as_dict()['name'].replace(" ", "_")
            act_dict[act_path] = act
            for exc in act.technosphere():
                _recursive_activities(exc.input, act_dict, act_path)
            return

        _recursive_activities(act, act_dict)
        return act_dict
