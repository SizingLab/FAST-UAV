"""
Constants for various purposes.
"""

# Concepts
FW_PROPULSION = "fixedwing"
MR_PROPULSION = "multirotor"
PROPULSION_ID_LIST = [MR_PROPULSION, FW_PROPULSION]

# Missions
PHASE_ID_TAG = "phase_id"
ROUTE_TAG = "route"
PARTS_TAG = "parts"
TAKEOFF_PART_TAG = "takeoff_part"
CLIMB_PART_TAG = "climb_part"
CRUISE_PART_TAG = "cruise_part"
DESCENT_PART_TAG = "descent_part"
HOVER_PART_TAG = "hover_part"
MISSION_DEFINITION_TAG = "missions"
ROUTE_DEFINITION_TAG = "routes"
SIZING_MISSION_TAG = "sizing"
TAKEOFF_TAG = "takeoff"
VERTICAL_TAKEOFF_TAG = "vertical_takeoff"
LAUNCHER_TAKEOFF_TAG = "launcher_takeoff"
STANDARD_TAKEOFF_TAG = "standard_takeoff"
CLIMB_TAG = "climb"
VERTICAL_CLIMB_TAG = "vertical_climb"
FW_CLIMB_TAG = "fixedwing_climb"
CRUISE_TAG = "cruise"
MR_CRUISE_TAG = "multirotor_cruise"
FW_CRUISE_TAG = "fixedwing_cruise"
HOVER_TAG = "hover"
PHASE_TAGS_LIST = [TAKEOFF_TAG, CLIMB_TAG, CRUISE_TAG, HOVER_TAG]

# Life Cycle Assessment
LCA_DEFAULT_PROJECT = 'fastuav'
LCA_DEFAULT_ECOINVENT = 'ecoinvent 3.9_cutoff_ecoSpold02'
LCA_USER_DB = 'Foreground DB'
LCA_MODEL_KEY = 'model per FU'
LCA_PARAM_KEY = 'lca:parameters:'
LCA_CHARACTERIZATION_KEY = 'lca:characterization:'
LCA_NORMALIZATION_KEY = 'lca:normalization:'
LCA_WEIGHTING_KEY = 'lca:weighting:'
LCA_AGGREGATION_KEY = 'lca:aggregation:'
LCA_WEIGHTED_SINGLE_SCORE_KEY = 'weighted_single_score'
LCA_POSTPROCESS_KEY = 'lca:postprocessing:'
LCA_FACTOR_KEY = ':factor'
LCA_DEFAULT_METHOD = [
    "('EF v3.1', 'acidification', 'accumulated exceedance (AE)')",
    "('EF v3.1', 'climate change', 'global warming potential (GWP100)')",
    "('EF v3.1', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe)')",
    "('EF v3.1', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels')",
    "('EF v3.1', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)')",
    "('EF v3.1', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)')",
    "('EF v3.1', 'eutrophication: terrestrial', 'accumulated exceedance (AE)')",
    "('EF v3.1', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)')",
    "('EF v3.1', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh)')",
    "('EF v3.1', 'ionising radiation: human health', 'human exposure efficiency relative to u235')",
    "('EF v3.1', 'land use', 'soil quality index')",
    "('EF v3.1', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)')",
    "('EF v3.1', 'ozone depletion', 'ozone depletion potential (ODP)')",
    "('EF v3.1', 'particulate matter formation', 'impact on human health')",
    "('EF v3.1', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase')",
    "('EF v3.1', 'water use', 'user deprivation potential (deprivation-weighted water consumption)')",
    ]
LCA_FUNCTIONAL_UNITS_LIST = ["kg.km", "kg.h", "lifetime"]
LCA_DEFAULT_FUNCTIONAL_UNIT = "lifetime"
