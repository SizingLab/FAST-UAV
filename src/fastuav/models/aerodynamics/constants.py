#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

POLAR_POINT_COUNT = 150
SPAN_MESH_POINT = 50
MACH_NB_PTS = 5
ENGINE_COUNT = 10
FIRST_INVALID_COEFF = 100.0
DEFAULT_INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!

OPTION_ALPHA_START = "alpha_start"
OPTION_ALPHA_END = "alpha_end"
OPTION_COMP_NEG_AIR_SYM = "activate_negative_angle"
ALPHA_STEP = 0.5
_DEFAULT_AIRFOIL_FILE = "naca23012.af"
DEFAULT_2D_CL_MAX = 1.9
DEFAULT_2D_CL_MIN = -1.7

SUBMODEL_CD0 = "fastuav.submodel.aerodynamics.aircraft.cd0"
SUBMODEL_CD0_WING = "fastuav.submodel.aerodynamics.wing.cd0"
SUBMODEL_CD0_FUSELAGE = "fastuav.submodel.aerodynamics.fuselage.cd0"
SUBMODEL_CD0_HT = "fastuav.submodel.aerodynamics.horizontal_tail.cd0"
SUBMODEL_CD0_VT = "fastuav.submodel.aerodynamics.vertical_tail.cd0"
SUBMODEL_CD0_NACELLE = "fastuav.submodel.aerodynamics.nacelle.cd0"
SUBMODEL_CD0_LANDING_GEAR = "fastuav.submodel.aerodynamics.landing_gear.cd0"
SUBMODEL_CD0_OTHER = "fastuav.submodel.aerodynamics.other.cd0"
SUBMODEL_CD0_SUM = "fastuav.submodel.aerodynamics.sum.cd0"
SUBMODEL_AIRFOIL_LIFT_SLOPE = "fastuav.submodel.aerodynamics.airfoil.all.lift_curve_slope"
SUBMODEL_DELTA_HIGH_LIFT = "fastuav.submodel.aerodynamics.high_lift.delta"
SUBMODEL_DELTA_ELEVATOR = "fastuav.submodel.aerodynamics.elevator.delta"
SUBMODEL_CL_EXTREME = "fastuav.submodel.aerodynamics.aircraft.extreme_lift_coefficient"
SUBMODEL_CL_EXTREME_CLEAN_WING = "fastuav.submodel.aerodynamics.wing.extreme_lift_coefficient.clean"
SUBMODEL_CL_EXTREME_CLEAN_HT = (
    "fastuav.submodel.aerodynamics.horizontal_tail.extreme_lift_coefficient.clean"
)
SUBMODEL_CL_ALPHA_VT = "fastuav.submodel.aerodynamics.vertical_tail.lift_curve_slope"
SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER = (
    "fastuav.submodel.aerodynamics.propeller.effective_efficiency"
)
SUBMODEL_HINGE_MOMENTS_TAIL_2D = "fastuav.submodel.aerodynamics.tail.hinge_moments.2d"
SUBMODEL_HINGE_MOMENTS_TAIL_3D = "fastuav.submodel.aerodynamics.tail.hinge_moments.3d"
SUBMODEL_HINGE_MOMENTS_TAIL = "fastuav.submodel.aerodynamics.tail.hinge_moments"
SUBMODEL_MAX_L_D = "fastuav.submodel.aerodynamics.aircraft.l_d_max"
SUBMODEL_CM_ALPHA_FUSELAGE = "fastuav.submodel.aerodynamics.fuselage.pitching_moment_alpha"
SUBMODEL_VH = "fastuav.submodel.aerodynamics.aircraft.max_level_speed"
SUBMODEL_THRUST_POWER_SLIPSTREAM = (
    "fastuav.submodel.aerodynamics.wing.slipstream.thrust_power_computation"
)
SUBMODEL_CY_BETA_FUSELAGE = "fastuav.submodel.aerodynamics.fuselage.side_force_beta"
SUBMODEL_CY_BETA_WING = "fastuav.submodel.aerodynamics.wing.side_force_beta"
SUBMODEL_CY_BETA_VT = "fastuav.submodel.aerodynamics.vertical_tail.side_force_beta"
SUBMODEL_CY_BETA = "fastuav.submodel.aerodynamics.aircraft.side_force_beta"
SUBMODEL_DOWNWASH = "fastuav.submodel.aerodynamics.horizontal_tail.downwash"
SUBMODEL_CL_ALPHA_DOT = "fastuav.submodel.aerodynamics.aircraft.cl_rate_of_aoa_change"
SUBMODEL_CL_Q_HT = "fastuav.submodel.aerodynamics.horizontal_tail.cl_pitch_velocity"
SUBMODEL_CL_Q_WING = "fastuav.submodel.aerodynamics.wing.cl_pitch_velocity"
SUBMODEL_CL_Q = "fastuav.submodel.aerodynamics.aircraft.cl_pitch_velocity"
SUBMODEL_CY_R = "fastuav.submodel.aerodynamics.aircraft.cy_yaw_rate"
SUBMODEL_CY_P = "fastuav.submodel.aerodynamics.aircraft.cy_roll_rate"
SUBMODEL_CY_RUDDER = "fastuav.submodel.aerodynamics.rudder.yawing_moment"
SUBMODEL_CL_BETA_WING = "fastuav.submodel.aerodynamics.wing.roll_moment_beta"
SUBMODEL_CL_BETA_HT = "fastuav.submodel.aerodynamics.horizontal_tail.roll_moment_beta"
SUBMODEL_CL_BETA_VT = "fastuav.submodel.aerodynamics.vertical_tail.roll_moment_beta"
SUBMODEL_CL_BETA = "fastuav.submodel.aerodynamics.aircraft.roll_moment_beta"
SUBMODEL_CL_P_WING = "fastuav.submodel.aerodynamics.wing.roll_moment_roll_rate"
SUBMODEL_CL_P_HT = "fastuav.submodel.aerodynamics.horizontal_tail.roll_moment_roll_rate"
SUBMODEL_CL_P_VT = "fastuav.submodel.aerodynamics.vertical_tail.roll_moment_roll_rate"
SUBMODEL_CL_P = "fastuav.submodel.aerodynamics.aircraft.roll_moment_roll_rate"
SUBMODEL_CL_R_WING = "fastuav.submodel.aerodynamics.wing.roll_moment_yaw_rate"
SUBMODEL_CL_R_VT = "fastuav.submodel.aerodynamics.vertical_tail.roll_moment_yaw_rate"
SUBMODEL_CL_R = "fastuav.submodel.aerodynamics.aircraft.roll_moment_yaw_rate"
SUBMODEL_CL_AILERON = "fastuav.submodel.aerodynamics.aileron.roll_moment"
SUBMODEL_CL_RUDDER = "fastuav.submodel.aerodynamics.rudder.roll_moment"
SUBMODEL_CM_Q_WING = "fastuav.submodel.aerodynamics.wing.cm_pitch_velocity"
SUBMODEL_CM_Q_HT = "fastuav.submodel.aerodynamics.horizontal_tail.cm_pitch_velocity"
SUBMODEL_CM_Q = "fastuav.submodel.aerodynamics.aircraft.cm_pitch_velocity"
SUBMODEL_CM_ALPHA_DOT = "fastuav.submodel.aerodynamics.aircraft.cm_rate_of_aoa_change"
SUBMODEL_CN_BETA_FUSELAGE = "fastuav.submodel.aerodynamics.fuselage.yawing_moment_beta"
SUBMODEL_CN_BETA_VT = "fastuav.submodel.aerodynamics.vertical_tail.yawing_moment_beta"
SUBMODEL_CN_BETA = "fastuav.submodel.aerodynamics.aircraft.yawing_moment_beta"
SUBMODEL_CN_AILERON = "fastuav.submodel.aerodynamics.aileron.yaw_moment"
SUBMODEL_CN_RUDDER = "fastuav.submodel.aerodynamics.rudder.yaw_moment"
SUBMODEL_CN_P_WING = "fastuav.submodel.aerodynamics.wing.yaw_moment_roll_rate"
SUBMODEL_CN_P_VT = "fastuav.submodel.aerodynamics.vertical_tail.yaw_moment_roll_rate"
SUBMODEL_CN_P = "fastuav.submodel.aerodynamics.aircraft.yaw_moment_roll_rate"
SUBMODEL_CN_R_WING = "fastuav.submodel.aerodynamics.wing.yaw_moment_yaw_rate"
SUBMODEL_CN_R_VT = "fastuav.submodel.aerodynamics.vertical_tail.yaw_moment_yaw_rate"
SUBMODEL_CN_R = "fastuav.submodel.aerodynamics.aircraft.yaw_moment_yaw_rate"
