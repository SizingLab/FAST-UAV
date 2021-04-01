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
import numpy as np
import plotly
import plotly.graph_objects as go
from fastoad.io import VariableIO

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
    propellers = variables["data:propeller:mass"].value[0] * variables["data:propeller:number"].value[0]
    motors = variables["data:motor:mass"].value[0] * variables["data:propeller:number"].value[0]
    if "data:gearbox:mass" in variables.names():
        gearboxes = variables["data:gearbox:mass"].value[0] * variables["data:propeller:number"].value[0]
    else:
        gearboxes = 0
    ESC = variables["data:ESC:mass"].value[0] * variables["data:propeller:number"].value[0]
    battery = variables["data:battery:mass"].value[0]
    propulsion = propellers + motors + gearboxes + ESC + battery

    # STRUCTURE
    body = variables["data:structure:body:mass"].value[0]
    arms = variables["data:structure:arms:mass"].value[0]
    structure = body + arms

    # PAYLOAD
    payload = variables["data:payload:mass"].value[0]

    # FUEL MISSION (not used yet. May be useful for hydrogen)
    fuel_mission = 0

    # MTOW
    MTOW = variables["data:system:MTOW"].value[0]
    # TODO: Deal with this in a more generic manner ?
    if round(MTOW, 6) == round(propulsion + structure + payload + fuel_mission, 6):
        MTOW = propulsion + structure + payload + fuel_mission

    # DISPLAYED NAMES AND VALUES
    propellers_str = (
            "Propellers"
            + "<br>"
            + str("{0:.2f}".format(propellers))
            + " [kg] ("
            + str(round(propellers / propulsion * 100, 1))
            + "%)"
    )
    motors_str = (
            "Motors"
            + "<br>"
            + str("{0:.2f}".format(motors))
            + " [kg] ("
            + str(round(motors / propulsion * 100, 1))
            + "%)"
    )
    gearboxes_str = (
            "Gearboxes"
            + "<br>"
            + str("{0:.2f}".format(gearboxes))
            + " [kg] ("
            + str(round(gearboxes / propulsion * 100, 1))
            + "%)"
    )
    ESC_str = (
            "ESC"
            + "<br>"
            + str("{0:.2f}".format(ESC))
            + " [kg] ("
            + str(round(ESC / propulsion * 100, 1))
            + "%)"
    )
    battery_str = (
            "Battery"
            + "<br>"
            + str("{0:.2f}".format(battery))
            + " [kg] ("
            + str(round(battery / propulsion * 100, 1))
            + "%)"
    )
    propulsion_str = (
            "Propulsion"
            + "<br>"
            + str("{0:.2f}".format(propulsion))
            + " [kg] ("
            + str(round(propulsion / MTOW * 100, 1))
            + "%)"
    )

    arms_str = (
            "Arms"
            + "<br>"
            + str("{0:.2f}".format(arms))
            + " [kg] ("
            + str(round(arms / structure * 100, 1))
            + "%)"
    )
    body_str = (
            "Body"
            + "<br>"
            + str("{0:.2f}".format(body))
            + " [kg] ("
            + str(round(body / structure * 100, 1))
            + "%)"
    )
    structure_str = (
            "Structure"
            + "<br>"
            + str("{0:.2f}".format(structure))
            + " [kg] ("
            + str(round(structure / MTOW * 100, 1))
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

    MTOW_str = (
            "MTOW" + "<br>" + str("{0:.2f}".format(MTOW)) + " [kg]"
    )

    # CREATE SUNBURST FIGURE
    fig = go.Figure(
        go.Sunburst(
            labels=[
                MTOW_str,
                payload_str,
                fuel_mission_str,
                propulsion_str,
                structure_str,
                propellers_str,
                motors_str,
                gearboxes_str,
                ESC_str,
                battery_str,
                body_str,
                arms_str,
            ],
            parents=[
                "",
                MTOW_str,
                MTOW_str,
                MTOW_str,
                MTOW_str,
                propulsion_str,
                propulsion_str,
                propulsion_str,
                propulsion_str,
                propulsion_str,
                structure_str,
                structure_str,
            ],
            values=[
                MTOW,
                payload,
                fuel_mission,
                propulsion,
                structure,
                propellers,
                motors,
                gearboxes,
                ESC,
                battery,
                body,
                arms,
            ],
            branchvalues="total",
        ),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Mass Breakdown", title_x=0.5)

    return fig


def mass_breakdown_bar_plot_drone(
        drone_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
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
    propellers = variables["data:propeller:mass"].value[0] * variables["data:propeller:number"].value[0]
    motors = variables["data:motor:mass"].value[0] * variables["data:propeller:number"].value[0]
    if "data:gearbox:mass" in variables.names():
        gearboxes = variables["data:gearbox:mass"].value[0] * variables["data:propeller:number"].value[0]
    else:
        gearboxes = 0
    ESC = variables["data:ESC:mass"].value[0] * variables["data:propeller:number"].value[0]
    battery = variables["data:battery:mass"].value[0]
    propulsion = propellers + motors + gearboxes + ESC + battery

    # STRUCTURE
    body = variables["data:structure:body:mass"].value[0]
    arms = variables["data:structure:arms:mass"].value[0]
    structure = body + arms

    # PAYLOAD
    payload = variables["data:payload:mass"].value[0]

    # FUEL MISSION (not used yet. May be useful for hydrogen)
    fuel_mission = 0

    # MTOW
    MTOW = variables["data:system:MTOW"].value[0]
    # TODO: Deal with this in a more generic manner ?
    if round(MTOW, 6) == round(propulsion + structure + payload + fuel_mission, 6):
        MTOW = propulsion + structure + payload + fuel_mission

    # DISPLAYED NAMES AND VALUES
    weight_labels = ["MTOW", "Payload", "Structure", "Propellers", "Motors", "ESC", "Battery", "Gearboxes"]
    weight_values = [MTOW, payload, structure, propellers, motors, ESC, battery, gearboxes]

    if fig is None:
        fig = go.Figure()

    # Same color for each drone configuration
    i = len(fig.data)
    fig.add_trace(
        go.Bar(name=name, x=weight_labels, y=weight_values, marker_color=COLS[i]),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Mass Breakdown", title_x=0.5)
    fig.update_layout(yaxis_title="[kg]")

    return fig


def drone_geometry_plot(
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

    A_body = variables["data:structure:body:surface:top"].value[0]  # [m**2]
    N_arms = variables["data:structure:arms:number"].value[0]  # [-]
    arm_length = variables["data:structure:arms:length"].value[0]  # [m]
    arm_diameter = variables["data:structure:arms:diameter:outer"].value[0]  # [m]
    N_pro_arm = variables["data:structure:arms:prop_per_arm"].value[0]  # [-]
    D_pro = variables["data:propeller:geometry:diameter"].value[0]  # [m]
    Vol_bat = variables["data:battery:volume"].value[0] * 0.000001  # [m**3]
    Lmot = variables["data:motor:length:estimated"].value[0]  # [m] TODO: get length from catalogues too

    # BATTERY
    Lbat = 3 * (Vol_bat / 6) ** (1 / 3)  # [m]
    Wbat = (Vol_bat / 6) ** (1 / 3)  # [m]
    # Hbat = 2 * (Vol_bat / 6) ** (1 / 3)  # [m]
    fig.add_shape(
        dict(type="rect", line=dict(color=COLS[k], width=3), fillcolor=COLS[k],
             x0= - Lbat / 2,
             y0= - Wbat / 2,
             x1= Lbat / 2,
             y1= Wbat / 2,
             )
    )

    # BODY - ARMS - PROPELLERS - MOTORS
    R_offset = (A_body / (N_arms * np.cos(np.pi / N_arms) * np.sin(np.pi / N_arms))) ** (1/2)  # body radius offset
    x_arms = []
    y_arms = []
    for i in range(int(N_arms)):
        sep_angle = - i * 2 * np.pi / N_arms

        # Arms
        y_arm = np.array(
            [- arm_diameter / 2 * np.sin(sep_angle) + R_offset * np.cos(sep_angle),
             - arm_diameter / 2 * np.sin(sep_angle) + arm_length * np.cos(sep_angle),
             arm_length * np.cos(sep_angle) + arm_diameter / 2 * np.sin(sep_angle),
             arm_diameter / 2 * np.sin(sep_angle) + R_offset * np.cos(sep_angle)]
        )
        x_arm = np.array(
            [arm_diameter / 2 * np.cos(sep_angle) + R_offset * np.sin(sep_angle),
             arm_diameter / 2 * np.cos(sep_angle) + arm_length * np.sin(sep_angle),
             arm_length * np.sin(sep_angle) - arm_diameter / 2 * np.cos(sep_angle),
             - arm_diameter / 2 * np.cos(sep_angle) + R_offset * np.sin(sep_angle)]
        )
        x_arms = np.concatenate((x_arms, x_arm))
        y_arms = np.concatenate((y_arms, y_arm))

        # Motors
        # Lhd = 0.25 * Lmot  # [m]
        # Dhd = 0.25 * 2.54  # [m] this shaft diameter is commonly used along the series of APC MR propellers
        Dmot = 0.7 * Lmot  # [m] geometric ratio used for AXI 2208, 2212, 2217
        fig.add_shape(
            dict(type="circle", line=dict(color=COLS[k], width=3), fillcolor=COLS[k],
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
                dict(type="circle", line=dict(color=COLS[k], width=0), fillcolor=COLS[k], opacity=0.25,
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
        config_text='Coaxial propellers (push-pull)'
    else:
        config_text='Single propellers'

    fig.add_annotation(
        xanchor='right',
        yanchor='top',
        x=1, #x_arms[len(x_arms)-3],
        y=1, #x_arms[len(y_arms)-3],
        xref="paper",
        yref="paper",
        text=config_text,
        showarrow=False,
        font=dict(
            color=COLS[k]
        ),
        align="center",
        xshift=0,
        yshift=-k*35,
        bordercolor=COLS[k],
        borderwidth=2,
        borderpad=4,
        bgcolor='white'
    )

    fig = go.FigureWidget(fig)
    fig.update_shapes(xref="x", yref="y")
    fig.update_layout(
        title_text="Drone Geometry", title_x=0.5, xaxis_title="y", yaxis_title="x",
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    return fig



def energy_breakdown_sun_plot_drone(drone_file_path: str, file_formatter=None):
    """
    Returns a figure sunburst plot of the drone energy consumption breakdown.

    :param drone_file_path: path of data file
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: sunburst plot figure
    """
    variables = VariableIO(drone_file_path, file_formatter).read()

    # PROPULSION
    hover = variables['data:mission:energy:hover'].value[0]
    climb = variables['data:mission:energy:climb'].value[0]
    forward = variables['data:mission:energy:forward'].value[0]
    propulsion = hover + climb + forward

    # AVIONICS
    avionics = variables['data:mission:energy:avionics'].value[0]

    # PAYLOAD
    payload = variables['data:mission:energy:payload'].value[0]

    # TOTAL MISSION
    mission = variables['data:mission:energy'].value[0]

    # DISPLAYED NAMES AND VALUES
    propulsion_str = (
            "Propulsion"
            + "<br>"
            + str("{0:.2f}".format(propulsion))
            + " [kJ] ("
            + str(round(propulsion / mission * 100, 1))
            + "%)"
    )

    hover_str = (
            "Hover"
            + "<br>"
            + str("{0:.2f}".format(hover))
            + " [kJ] ("
            + str(round(hover / propulsion * 100, 1))
            + "%)"
    )

    climb_str = (
            "Climb"
            + "<br>"
            + str("{0:.2f}".format(climb))
            + " [kJ] ("
            + str(round(climb / propulsion * 100, 1))
            + "%)"
    )

    forward_str = (
            "Forward"
            + "<br>"
            + str("{0:.2f}".format(forward))
            + " [kJ] ("
            + str(round(forward / propulsion * 100, 1))
            + "%)"
    )

    avionics_str = (
            "Avionics"
            + "<br>"
            + str("{0:.2f}".format(avionics))
            + " [kJ] ("
            + str(round(avionics / mission * 100, 1))
            + "%)"
    )

    payload_str = (
            "Payload"
            + "<br>"
            + str("{0:.2f}".format(payload))
            + " [kJ] ("
            + str(round(payload / mission * 100, 1))
            + "%)"
    )

    mission_str = (
            "Mission" + "<br>" + str("{0:.2f}".format(mission)) + " [kJ]"
    )

    # CREATE SUNBURST FIGURE
    fig = go.Figure(
        go.Sunburst(
            labels=[
                mission_str,
                avionics_str,
                payload_str,
                propulsion_str,
                hover_str,
                climb_str,
                forward_str,
            ],
            parents=[
                "",
                mission_str,
                mission_str,
                mission_str,
                propulsion_str,
                propulsion_str,
                propulsion_str,
            ],
            values=[
                mission,
                avionics,
                payload,
                propulsion,
                hover,
                climb,
                forward,
            ],
            branchvalues="total",
        ),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Energy Breakdown", title_x=0.5)

    return fig