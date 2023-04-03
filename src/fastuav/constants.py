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
LCA_RESULT_KEY = 'lca:results:'
LCA_POSTPROCESS_KEY = 'lca:postprocessing:'
LCA_DEFAULT_METHOD = [
    "('ReCiPe 2016 v1.03, midpoint (E) no LT', "
    "'climate change no LT', "
    "'global warming potential (GWP1000) no LT')",
]
