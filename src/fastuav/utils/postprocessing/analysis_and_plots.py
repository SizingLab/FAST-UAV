"""
Defines the analysis and plotting functions for postprocessing
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from typing import Dict
import numpy as np
import plotly
import plotly.graph_objects as go
from fastoad.io import VariableIO
from fastoad.openmdao.variables import VariableList
from openmdao.utils.units import convert_units
from fastuav.utils.constants import PROPULSION_ID_LIST, MR_PROPULSION

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def mass_breakdown_sun_plot_drone(drone_file_path: str, file_formatter=None):
    """
    Returns a figure sunburst plot of the drone mass breakdown.

    :param drone_file_path: path of data file
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: sunburst plot figure
    """
    variables = VariableIO(drone_file_path, file_formatter).read()

    # PROPULSION
    propellers = np.empty(len(PROPULSION_ID_LIST))
    motors = np.empty(len(PROPULSION_ID_LIST))
    gearboxes = np.empty(len(PROPULSION_ID_LIST))
    esc = np.empty(len(PROPULSION_ID_LIST))
    batteries = np.empty(len(PROPULSION_ID_LIST))
    wires = np.empty(len(PROPULSION_ID_LIST))
    propulsion = np.empty(len(PROPULSION_ID_LIST))
    propulsions = 0.0
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        propellers[i] = (
            variables["data:weights:propulsion:%s:propeller:mass" % propulsion_id].value[0]
            * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:propeller:mass" % propulsion_id in variables.names() else 0.0
        motors[i] = (
            variables["data:weights:propulsion:%s:motor:mass" % propulsion_id].value[0]
            * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:motor:mass" % propulsion_id in variables.names() else 0.0
        gearboxes[i] = (
                variables["data:weights:propulsion:%s:gearbox:mass" % propulsion_id].value[0]
                * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
            ) if "data:weights:propulsion:%s:gearbox:mass" % propulsion_id in variables.names() else 0.0
        esc[i] = (
            variables["data:weights:propulsion:%s:esc:mass" % propulsion_id].value[0]
            * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:esc:mass" % propulsion_id in variables.names() else 0.0
        batteries[i] = (
            variables["data:weights:propulsion:%s:battery:mass" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:battery:mass" % propulsion_id in variables.names() else 0.0
        wires[i] = (
            variables["data:weights:propulsion:%s:wires:mass" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:wires:mass" % propulsion_id in variables.names() else 0.0

        propulsion[i] = propellers[i] + motors[i] + gearboxes[i] + esc[i] + batteries[i] + wires[i]
        propulsions += propulsion[i]

    # AIRFRAME
    body = (
        variables["data:weights:airframe:body:mass"].value[0]
        if "data:weights:airframe:body:mass" in variables.names()
        else 0.0
    )
    arms = (
        variables["data:weights:airframe:arms:mass"].value[0]
        if "data:weights:airframe:arms:mass" in variables.names()
        else 0.0
    )
    wing = (
        variables["data:weights:airframe:wing:mass"].value[0]
        if "data:weights:airframe:wing:mass" in variables.names()
        else 0.0
    )
    fuselage = (
        variables["data:weights:airframe:fuselage:mass"].value[0]
        if "data:weights:airframe:fuselage:mass" in variables.names()
        else 0.0
    )
    htail = (
        variables["data:weights:airframe:tail:horizontal:mass"].value[0]
        if "data:weights:airframe:tail:horizontal:mass" in variables.names()
        else 0.0
    )
    vtail = (
        variables["data:weights:airframe:tail:vertical:mass"].value[0]
        if "data:weights:airframe:tail:vertical:mass" in variables.names()
        else 0.0
    )
    airframe = body + arms + wing + fuselage + htail + vtail

    # PAYLOAD
    payload = variables["data:scenarios:payload:mass"].value[0]

    # FUEL MISSION (not used yet. May be useful for hydrogen)
    fuel_mission = 0

    # MTOW
    MTOW = variables["data:weights:mtow"].value[0]

    if round(MTOW, 6) == round(propulsions + airframe + payload + fuel_mission, 6):
        MTOW = propulsions + airframe + payload + fuel_mission

    # DISPLAYED NAMES AND VALUES
    propellers_str = []
    motors_str = []
    gearboxes_str = []
    esc_str = []
    batteries_str = []
    wires_str = []
    propulsion_str = []
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        propellers_str.append(
            "Propellers"
            + "<br>"
            + str("{0:.2f}".format(propellers[i]))
            + " [kg] ("
            + str(round(propellers[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        motors_str.append(
            "Motors"
            + "<br>"
            + str("{0:.2f}".format(motors[i]))
            + " [kg] ("
            + str(round(motors[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        gearboxes_str.append(
            "Gearboxes"
            + "<br>"
            + str("{0:.2f}".format(gearboxes[i]))
            + " [kg] ("
            + str(round(gearboxes[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        esc_str.append(
            "esc"
            + "<br>"
            + str("{0:.2f}".format(esc[i]))
            + " [kg] ("
            + str(round(esc[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        batteries_str.append(
            "Battery"
            + "<br>"
            + str("{0:.2f}".format(batteries[i]))
            + " [kg] ("
            + str(round(batteries[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        wires_str.append(
            "Wires"
            + "<br>"
            + str("{0:.2f}".format(wires[i]))
            + " [kg] ("
            + str(round(wires[i] / propulsion[i] * 100, 1) if propulsion[i] > 0 else 0.0)
            + "%)"
        )
        propulsion_str.append(
            propulsion_id
            + "<br>"
            + str("{0:.2f}".format(propulsion[i]))
            + " [kg] ("
            + str(round(propulsion[i] / propulsions * 100, 1))
            + "%)"
        )

    propulsions_str = (
            "Propulsion"
            + "<br>"
            + str("{0:.2f}".format(propulsions))
            + " [kg] ("
            + str(round(propulsions / MTOW * 100, 1))
            + "%)"
    )

    body_str = (
        "Body"
        + "<br>"
        + str("{0:.2f}".format(body))
        + " [kg] ("
        + str(round(body / airframe * 100, 1))
        + "%)"
    )
    arms_str = (
        "Arms"
        + "<br>"
        + str("{0:.2f}".format(arms))
        + " [kg] ("
        + str(round(arms / airframe * 100, 1))
        + "%)"
    )
    wing_str = (
        "Wing"
        + "<br>"
        + str("{0:.2f}".format(wing))
        + " [kg] ("
        + str(round(wing / airframe * 100, 1))
        + "%)"
    )
    fuselage_str = (
        "Fuselage"
        + "<br>"
        + str("{0:.2f}".format(fuselage))
        + " [kg] ("
        + str(round(fuselage / airframe * 100, 1))
        + "%)"
    )
    htail_str = (
        "HTP"
        + "<br>"
        + str("{0:.2f}".format(htail))
        + " [kg] ("
        + str(round(htail / airframe * 100, 1))
        + "%)"
    )
    vtail_str = (
        "VTP"
        + "<br>"
        + str("{0:.2f}".format(vtail))
        + " [kg] ("
        + str(round(vtail / airframe * 100, 1))
        + "%)"
    )
    airframe_str = (
        "Airframe"
        + "<br>"
        + str("{0:.2f}".format(airframe))
        + " [kg] ("
        + str(round(airframe / MTOW * 100, 1))
        + "%)"
    )

    payload_str = (
        "Payload"
        + "<br>"
        + str("{0:.2f}".format(payload))
        + " [kg] ("
        + str(round(payload / MTOW * 100, 1))
        + "%)"
    )

    fuel_mission_str = (
        "Fuel mission"
        + "<br>"
        + str("{0:.2f}".format(fuel_mission))
        + " [kg] ("
        + str(round(fuel_mission / MTOW * 100, 1))
        + "%)"
    )

    MTOW_str = "MTOW" + "<br>" + str("{0:.2f}".format(MTOW)) + " [kg]"

    # CREATE SUNBURST FIGURE
    labels = [MTOW_str,
              payload_str,
              fuel_mission_str,
              propulsions_str,
              airframe_str,
              ]
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        labels.extend(
            [propulsion_str[i],
             propellers_str[i],
             motors_str[i],
             gearboxes_str[i],
             esc_str[i],
             batteries_str[i],
             wires_str[i]]
        )
    labels.extend(
        [body_str,
         arms_str,
         wing_str,
         fuselage_str,
         htail_str,
         vtail_str]
    )

    parents = ["",
               MTOW_str,
               MTOW_str,
               MTOW_str,
               MTOW_str,
               ]
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        parents.extend(
            [propulsions_str,
             propulsion_str[i],
             propulsion_str[i],
             propulsion_str[i],
             propulsion_str[i],
             propulsion_str[i],
             propulsion_str[i],
             ]
        )
    parents.extend(
        [airframe_str,
         airframe_str,
         airframe_str,
         airframe_str,
         airframe_str,
         airframe_str,
         ]
    )

    values = [MTOW,
              payload,
              fuel_mission,
              propulsions,
              airframe,
              ]
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        values.extend([
            propulsion[i],
            propellers[i],
            motors[i],
            gearboxes[i],
            esc[i],
            batteries[i],
            wires[i],
        ])
    values.extend([
        body,
        arms,
        wing,
        fuselage,
        htail,
        vtail,
    ])

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        ),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Mass Breakdown", title_x=0.5)

    return fig


def mass_breakdown_bar_plot_drone(
    drone_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    DEPRECATED FOR NOW.

    Returns a figure plot of the drone mass breakdown using bar plots.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param drone_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: bar plot figure
    """
    variables = VariableIO(drone_file_path, file_formatter).read()

    # PROPULSION
    propellers = np.empty(len(PROPULSION_ID_LIST))
    motors = np.empty(len(PROPULSION_ID_LIST))
    gearboxes = np.empty(len(PROPULSION_ID_LIST))
    esc = np.empty(len(PROPULSION_ID_LIST))
    batteries = np.empty(len(PROPULSION_ID_LIST))
    wires = np.empty(len(PROPULSION_ID_LIST))
    propulsion = np.empty(len(PROPULSION_ID_LIST))
    propulsions = 0.0
    for i, propulsion_id in enumerate(PROPULSION_ID_LIST):
        propellers[i] = (
                variables["data:weights:propulsion:%s:propeller:mass" % propulsion_id].value[0]
                * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:propeller:mass" % propulsion_id in variables.names() else 0.0
        motors[i] = (
                variables["data:weights:propulsion:%s:motor:mass" % propulsion_id].value[0]
                * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:motor:mass" % propulsion_id in variables.names() else 0.0
        gearboxes[i] = (
                variables["data:weights:propulsion:%s:gearbox:mass" % propulsion_id].value[0]
                * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:gearbox:mass" % propulsion_id in variables.names() else 0.0
        esc[i] = (
                variables["data:weights:propulsion:%s:esc:mass" % propulsion_id].value[0]
                * variables["data:propulsion:%s:propeller:number" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:esc:mass" % propulsion_id in variables.names() else 0.0
        batteries[i] = (
            variables["data:weights:propulsion:%s:battery:mass" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:battery:mass" % propulsion_id in variables.names() else 0.0
        wires[i] = (
            variables["data:weights:propulsion:%s:wires:mass" % propulsion_id].value[0]
        ) if "data:weights:propulsion:%s:wires:mass" % propulsion_id in variables.names() else 0.0

        propulsion[i] = propellers[i] + motors[i] + gearboxes[i] + esc[i] + batteries[i] + wires[i]
        propulsions += propulsion[i]

    # AIRFRAME
    body = (
        variables["data:weights:airframe:body:mass"].value[0]
        if "data:weights:airframe:body:mass" in variables.names()
        else 0.0
    )
    arms = (
        variables["data:weights:airframe:arms:mass"].value[0]
        if "data:weights:airframe:arms:mass" in variables.names()
        else 0.0
    )
    wing = (
        variables["data:weights:airframe:wing:mass"].value[0]
        if "data:weights:airframe:wing:mass" in variables.names()
        else 0.0
    )
    fuselage = (
        variables["data:weights:airframe:fuselage:mass"].value[0]
        if "data:weights:airframe:fuselage:mass" in variables.names()
        else 0.0
    )
    htail = (
        variables["data:weights:airframe:tail:horizontal:mass"].value[0]
        if "data:weights:airframe:tail:horizontal:mass" in variables.names()
        else 0.0
    )
    vtail = (
        variables["data:weights:airframe:tail:vertical:mass"].value[0]
        if "data:weights:airframe:tail:vertical:mass" in variables.names()
        else 0.0
    )
    airframe = body + arms + wing + fuselage + htail + vtail

    # PAYLOAD
    payload = variables["data:scenarios:payload:mass"].value[0]

    # FUEL MISSION (not used yet. May be useful for hydrogen)
    fuel_mission = 0

    # MTOW
    MTOW = variables["data:weights:mtow"].value[0]

    if round(MTOW, 6) == round(propulsions + airframe + payload + fuel_mission, 6):
        MTOW = propulsions + airframe + payload + fuel_mission

    # DISPLAYED NAMES AND VALUES
    if gearboxes == 0:
        weight_labels = [
            "MTOW",
            "Payload",
            "Battery",
            "esc",
            "Motors",
            "Propellers",
            "Cables",
            "Structure",
        ]
        weight_values = [
            MTOW,
            payload,
            battery,
            esc,
            motors,
            propellers,
            wires,
            airframe,
        ]
    else:
        weight_labels = [
            "MTOW",
            "Payload",
            "Battery",
            "esc",
            "Motors",
            "Gearboxes",
            "Propellers",
            "Cables",
            "Structure",
        ]
        weight_values = [
            MTOW,
            payload,
            battery,
            esc,
            motors,
            gearboxes,
            propellers,
            wires,
            structure,
        ]

    if fig is None:
        fig = go.Figure()

    # Same color for each drone configuration
    i = len(fig.data)
    fig.add_trace(
        go.Bar(name=name, x=weight_labels, y=weight_values, marker_color=COLS[i]),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Mass Breakdown", title_x=0.5)
    fig.update_layout(yaxis_title="Mass [kg]")

    return fig


def multirotor_geometry_plot(
    drone_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the drone.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param drone_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: drone plot figure
    """

    variables = VariableIO(drone_file_path, file_formatter).read()
    if fig is None:
        fig = go.Figure()
    k = len(fig.data)

    A_body = variables["data:geometry:body:surface:top"].value[0]  # [m**2]
    N_arms = variables["data:geometry:arms:number"].value[0]  # [-]
    arm_length = variables["data:geometry:arms:length"].value[0]  # [m]
    arm_diameter = variables["data:structures:arms:diameter:outer"].value[0]  # [m]
    N_pro_arm = variables["data:geometry:arms:prop_per_arm"].value[0]  # [-]
    D_pro = variables["data:propulsion:%s:propeller:diameter" % MR_PROPULSION].value[0]  # [m]
    Vol_bat = variables["data:propulsion:%s:battery:volume" % MR_PROPULSION].value[0] * 0.000001  # [m**3]
    Lmot = variables["data:propulsion:%s:motor:length:estimated" % MR_PROPULSION].value[
        0
    ]  # [m] TODO: get length from catalogues too

    # BATTERY
    Lbat = 3 * (Vol_bat / 6) ** (1 / 3)  # [m]
    Wbat = (Vol_bat / 6) ** (1 / 3)  # [m]
    # Hbat = 2 * (Vol_bat / 6) ** (1 / 3)  # [m]
    fig.add_shape(
        dict(
            type="rect",
            line=dict(color=COLS[k], width=3),
            fillcolor=COLS[k],
            x0=-Lbat / 2,
            y0=-Wbat / 2,
            x1=Lbat / 2,
            y1=Wbat / 2,
        )
    )

    # BODY - ARMS - PROPELLERS - MOTORS
    R_offset = (A_body / (N_arms * np.cos(np.pi / N_arms) * np.sin(np.pi / N_arms))) ** (
        1 / 2
    )  # body radius offset
    x_arms = []
    y_arms = []
    for i in range(int(N_arms)):
        sep_angle = -i * 2 * np.pi / N_arms

        # Arms
        y_arm = np.array(
            [
                -arm_diameter / 2 * np.sin(sep_angle) + R_offset * np.cos(sep_angle),
                -arm_diameter / 2 * np.sin(sep_angle) + arm_length * np.cos(sep_angle),
                arm_length * np.cos(sep_angle) + arm_diameter / 2 * np.sin(sep_angle),
                arm_diameter / 2 * np.sin(sep_angle) + R_offset * np.cos(sep_angle),
            ]
        )
        x_arm = np.array(
            [
                arm_diameter / 2 * np.cos(sep_angle) + R_offset * np.sin(sep_angle),
                arm_diameter / 2 * np.cos(sep_angle) + arm_length * np.sin(sep_angle),
                arm_length * np.sin(sep_angle) - arm_diameter / 2 * np.cos(sep_angle),
                -arm_diameter / 2 * np.cos(sep_angle) + R_offset * np.sin(sep_angle),
            ]
        )
        x_arms = np.concatenate((x_arms, x_arm))
        y_arms = np.concatenate((y_arms, y_arm))

        # Motors
        # Lhd = 0.25 * Lmot  # [m]
        # Dhd = 0.25 * 2.54  # [m] this shaft diameter is commonly used along the series of APC MR propellers
        Dmot = 0.7 * Lmot  # [m] geometric ratio used for AXI 2208, 2212, 2217
        fig.add_shape(
            dict(
                type="circle",
                line=dict(color=COLS[k], width=3),
                fillcolor=COLS[k],
                x0=arm_length * np.cos(sep_angle) - Dmot / 2,
                y0=arm_length * np.sin(sep_angle) - Dmot / 2,
                x1=arm_length * np.cos(sep_angle) + Dmot / 2,
                y1=arm_length * np.sin(sep_angle) + Dmot / 2,
            )
        )

        # Propellers
        for j in range(int(N_pro_arm)):
            prop_offset = j * D_pro / 15  # slightly shift the second propeller to make it visible
            fig.add_shape(
                dict(
                    type="circle",
                    line=dict(color=COLS[k], width=0),
                    fillcolor=COLS[k],
                    opacity=0.25,
                    x0=arm_length * np.cos(sep_angle) - D_pro / 2 + prop_offset,
                    y0=arm_length * np.sin(sep_angle) - D_pro / 2 + prop_offset,
                    x1=arm_length * np.cos(sep_angle) + D_pro / 2 + prop_offset,
                    y1=arm_length * np.sin(sep_angle) + D_pro / 2 + prop_offset,
                )
            )
    # Arms
    x_arms = np.concatenate((x_arms, [x_arms[0]]))  # to close the trace
    y_arms = np.concatenate((y_arms, [y_arms[0]]))  # to close the trace
    scatter = go.Scatter(x=y_arms, y=x_arms, mode="lines", line_color=COLS[k], name=name)
    fig.add_trace(scatter)

    # Push-pull configuration annotation
    if N_pro_arm == 2:
        config_text = "Coaxial propellers (push-pull)"
    else:
        config_text = "Single propellers"

    fig.add_annotation(
        xanchor="right",
        yanchor="top",
        x=1,
        y=1,
        xref="paper",
        yref="paper",
        text=config_text,
        showarrow=False,
        font=dict(color=COLS[k]),
        align="center",
        xshift=0,
        yshift=-k * 35,
        bordercolor=COLS[k],
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
    )

    fig = go.FigureWidget(fig)
    fig.update_shapes(xref="x", yref="y")
    fig.update_layout(
        title_text="Drone Geometry",
        title_x=0.5,
        xaxis_title="y",
        yaxis_title="x",
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    return fig


def fixedwing_geometry_plot(
    aircraft_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
                           default format will be assumed.
    :return: wing plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Wing parameters
    wing_tip_leading_edge_x = 0
    wing_root_y = 0  # variables["data:geometry:fuselage:diameter:mid"].value[0] / 2.0
    wing_tip_y = variables["data:geometry:wing:span"].value[0] / 2.0
    wing_root_chord = variables["data:geometry:wing:root:chord"].value[0]
    wing_tip_chord = variables["data:geometry:wing:tip:chord"].value[0]

    y_wing = np.array([0, wing_root_y, wing_tip_y, wing_tip_y, wing_root_y, 0, 0])

    x_wing = np.array(
        [
            0,
            0,
            wing_tip_leading_edge_x,
            wing_tip_leading_edge_x + wing_tip_chord,
            wing_root_chord,
            wing_root_chord,
            0,
        ]
    )

    # Horizontal Tail parameters
    ht_root_chord = variables["data:geometry:tail:horizontal:root:chord"].value[0]
    ht_tip_chord = variables["data:geometry:tail:horizontal:tip:chord"].value[0]
    ht_span = variables["data:geometry:tail:horizontal:span"].value[0]
    ht_sweep_0 = 0  # variables["data:geometry:horizontal_tail:sweep_0"].value[0]

    ht_tip_leading_edge_x = ht_span / 2.0 * np.tan(ht_sweep_0 * np.pi / 180.0)

    y_ht = np.array([0, ht_span / 2.0, ht_span / 2.0, 0.0, 0.0])

    x_ht = np.array(
        [0, ht_tip_leading_edge_x, ht_tip_leading_edge_x + ht_tip_chord, ht_root_chord, 0]
    )

    # Fuselage parameters
    fuselage_mid_width = variables["data:geometry:fuselage:diameter:mid"].value[0]
    fuselage_tip_width = variables["data:geometry:fuselage:diameter:tip"].value[0]
    fuselage_length = variables["data:geometry:fuselage:length"].value[0]
    fuselage_nose_length = variables["data:geometry:fuselage:length:nose"].value[0]
    fuselage_mid_length = variables["data:geometry:fuselage:length:mid"].value[0]
    fuselage_rear_length = variables["data:geometry:fuselage:length:rear"].value[0]
    fuselage_front_length = fuselage_nose_length + fuselage_mid_length

    x_fuselage = np.array(
        [
            0.0,
            fuselage_nose_length,
            fuselage_front_length,
            fuselage_length,
            fuselage_length,
        ]
    )

    y_fuselage = np.array(
        [
            0.0,
            fuselage_mid_width / 2.0,
            fuselage_mid_width / 2.0,
            fuselage_tip_width / 2.0,
            0.0,
        ]
    )

    # Absolute positions
    x_wing_root = variables["data:geometry:wing:root:LE:x"].value[0]
    x_ht_root = variables["data:geometry:tail:horizontal:root:LE:x"].value[0]

    x_wing = x_wing + x_wing_root
    x_ht = x_ht + x_ht_root

    # pylint: disable=invalid-name # that's a common naming
    x = np.concatenate((x_fuselage, x_wing, x_ht))
    # pylint: disable=invalid-name # that's a common naming
    y = np.concatenate((y_fuselage, y_wing, y_ht))

    # pylint: disable=invalid-name # that's a common naming
    y = np.concatenate((-y, y))
    # pylint: disable=invalid-name # that's a common naming
    x = np.concatenate((x, x))

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(x=y, y=x, mode="lines+markers", name=name)

    fig.add_trace(scatter)

    fig.layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    fig = go.FigureWidget(fig)

    fig.update_layout(title_text="Drone Geometry", title_x=0.5, xaxis_title="y", yaxis_title="x")

    return fig


def energy_breakdown_sun_plot_drone(
    drone_file_path: str,
    mission_name: str = "sizing",
    file_formatter=None,
    fig=None,
):
    """
    Returns a figure sunburst plot of the drone energy consumption breakdown.

    :param drone_file_path: path of data file
    :param mission_name: name of the mission to plot
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: sunburst plot figure
    """
    variables = VariableIO(drone_file_path, file_formatter).read()

    var_names_and_new_units = {
        "mission:%s:energy" % mission_name: "W*h",
    }

    total_energy = _get_variable_values_with_new_units(variables, var_names_and_new_units)[0]

    (
        categories_values,
        categories_names,
        categories_labels,
    ) = _data_mission_decomposition(variables, mission_name=mission_name)

    sub_categories_values = []
    sub_categories_labels = []
    sub_categories_parent = []
    for variable in variables.names():
        name_split = variable.split(":")
        if isinstance(name_split, list) and len(name_split) == 5:
            parent_name = name_split[2]
            if parent_name in categories_names and name_split[-1] == "energy":
                variable_name = name_split[3]
                # variable_name = "_".join(name_split[3:-1])
                sub_categories_values.append(
                    convert_units(variables[variable].value[0], variables[variable].units, "W*h")
                )
                sub_categories_parent.append(categories_labels[categories_names.index(parent_name)])
                # sub_categories_labels.append(variable_name)

                sub_categories_labels.append(
                    variable_name + "<br>" + str(int(sub_categories_values[-1])) + " [Wh] "
                )

    # Define figure data
    figure_labels = [mission_name + "<br>" + str(int(total_energy)) + " [Wh]"]
    figure_labels.extend(categories_labels)
    figure_labels.extend(sub_categories_labels)
    figure_parents = [""]
    for _ in categories_names:
        figure_parents.append(mission_name + "<br>" + str(int(total_energy)) + " [Wh]")
    figure_parents.extend(sub_categories_parent)
    figure_values = [total_energy]
    figure_values.extend(categories_values)
    figure_values.extend(sub_categories_values)

    # Plot figure
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Sunburst(
            labels=figure_labels,
            parents=figure_parents,
            values=figure_values,
            branchvalues="total",
            domain=dict(column=len(fig.data)),
        )
    )

    fig.update_layout(
        grid=dict(columns=len(fig.data), rows=1),
        margin=dict(t=80, l=0, r=0, b=0),
        title_text="Mission Energy Breakdown",
        title_x=0.5,
    )

    return fig


def _data_mission_decomposition(variables: VariableList, mission_name: str = "sizing"):
    """
    Returns the routes decomposition of mission.

    :param variables: instance containing variables information
    :return: route names
    """
    var_names_and_new_units = {
        "mission:%s:energy" % mission_name: "W*h",
    }
    total_energy = _get_variable_values_with_new_units(variables, var_names_and_new_units)[0]

    category_values = []
    category_names = []
    categories_labels = []
    for variable in variables.names():
        name_split = variable.split(":")
        if isinstance(name_split, list) and len(name_split) == 4:
            if (
                name_split[0] == "mission"
                and name_split[1] == mission_name
                and name_split[-1] == "energy"
            ):
                category_values.append(
                    convert_units(variables[variable].value[0], variables[variable].units, "W*h")
                )
                category_names.append(name_split[2])
                categories_labels.append(
                    name_split[2]
                    + "<br>"
                    + str(int(category_values[-1]))
                    + " [Wh] ("
                    + str(round(category_values[-1] / total_energy * 100, 1))
                    + "%)"
                )

    result = category_values, category_names, categories_labels
    return result


def _data_route_decomposition(variables: VariableList, mission_name: str = "sizing", route_name: str = None):
    """
    Returns the flight phases decomposition of route.

    :param variables: instance containing variables information
    :return: flight phases names
    """
    var_names_and_new_units = {
        "mission:%s:%s:energy" % (mission_name, route_name): "W*h",
    }
    total_energy = _get_variable_values_with_new_units(variables, var_names_and_new_units)[0]

    category_values = []
    category_names = []
    categories_labels = []
    for variable in variables.names():
        name_split = variable.split(":")
        if isinstance(name_split, list) and len(name_split) == 4:
            if (
                name_split[0] == "mission"
                and name_split[1] == mission_name
                and name_split[2] == route_name
                and name_split[-1] == "energy"
            ):
                category_values.append(
                    convert_units(variables[variable].value[0], variables[variable].units, "W*h")
                )
                category_names.append(name_split[2])
                categories_labels.append(
                    name_split[2]
                    + "<br>"
                    + str(int(category_values[-1]))
                    + " [Wh] ("
                    + str(round(category_values[-1] / total_energy * 100, 1))
                    + "%)"
                )

    result = category_values, category_names, categories_labels
    return result


def _get_variable_values_with_new_units(
    variables: VariableList, var_names_and_new_units: Dict[str, str]
):
    """
    Returns the value of the requested variable names with respect to their new units in the order
    in which their were given. This function works only for variable of value with shape=1 or float.

    :param variables: instance containing variables information
    :param var_names_and_new_units: dictionary of the variable names as keys and units as value
    :return: values of the requested variables with respect to their new units
    """
    new_values = []
    for variable_name, unit in var_names_and_new_units.items():
        new_values.append(
            convert_units(variables[variable_name].value[0], variables[variable_name].units, unit)
        )

    return new_values
