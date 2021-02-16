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
from plotly.subplots import make_subplots

from fastoad.io import VariableIO

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def mass_breakdown_drone(drone_file_path: str, file_formatter=None):
    """
    Returns a figure sunburst plot of the mass breakdown.

    :param drone_file_path: path of data file
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: sunburst plot figure
    """
    variables = VariableIO(drone_file_path, file_formatter).read()

    # PROPULSION
    propellers = variables["data:propeller:mass"].value[0] * variables["data:propeller:prop_number"].value[0]
    motors = variables["data:motor:mass"].value[0] * variables["data:propeller:prop_number"].value[0]
    if "data:gearbox:mass" in variables.names():
        gearboxes = variables["data:gearbox:mass"].value[0] * variables["data:propeller:prop_number"].value[0]
    else:
        gearboxes = 0
    ESC = variables["data:ESC:mass"].value[0] * variables["data:propeller:prop_number"].value[0]
    battery = variables["data:battery:mass"].value[0]
    propulsion = propellers + motors + gearboxes + ESC + battery

    # STRUCTURE
    frame = variables["data:structure:frame:mass"].value[0]
    arms = variables["data:structure:arms:mass"].value[0]
    structure = frame + arms

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
    frame_str = (
            "Frame"
            + "<br>"
            + str("{0:.2f}".format(frame))
            + " [kg] ("
            + str(round(frame / structure * 100, 1))
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
            "payload"
            + "<br>"
            + str("{0:.2f}".format(payload))
            + " [kg] ("
            + str(round(payload / MTOW * 100, 1))
            + "%)"
    )

    fuel_mission_str = (
            "fuel_mission"
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
                frame_str,
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
                frame,
                arms,
            ],
            branchvalues="total",
        ),
    )

    fig.update_layout(margin=dict(t=80, l=0, r=0, b=0), title_text="Mass Breakdown", title_x=0.5)

    return fig


def drone_geometry_plot(
        drone_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param drone_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """

    variables = VariableIO(drone_file_path, file_formatter).read()
    if fig is None:
        fig = go.Figure()
    k = len(fig.data)

    N_arms = variables["data:structure:arms:arm_number"].value[0]
    N_pro_arm = variables["data:propeller:prop_number_per_arm"].value[0]
    arm_length = variables["data:structure:arms:length"].value[0]
    arm_diameter = variables["data:structure:arms:diameter:outer"].value[0]
    D_pro = variables["data:propeller:geometry:diameter"].value[0]

    R = 0.1  # Frame radius #TODO: get frame radius

    x_arms = []
    y_arms = []

    for i in range(int(N_arms)):
        sep_angle = - i * 2 * np.pi / N_arms
        x_arm = np.array(
            [- arm_diameter / 2 * np.sin(sep_angle),
             - arm_diameter / 2 * np.sin(sep_angle) + arm_length * np.cos(sep_angle),
             arm_length * np.cos(sep_angle) + arm_diameter / 2 * np.sin(sep_angle),
             arm_diameter / 2 * np.sin(sep_angle)]
        )
        y_arm = np.array(
            [arm_diameter / 2 * np.cos(sep_angle),
             arm_diameter / 2 * np.cos(sep_angle) + arm_length * np.sin(sep_angle),
             arm_length * np.sin(sep_angle) - arm_diameter / 2 * np.cos(sep_angle),
             - arm_diameter / 2 * np.cos(sep_angle)]
        )

        x_arm = x_arm + R * np.cos(sep_angle)  # frame radius
        y_arm = y_arm + R * np.sin(sep_angle)  # frame radius

        x_arms = np.concatenate((x_arms, x_arm))
        y_arms = np.concatenate((y_arms, y_arm))

        # PROPELLERS
        for j in range(int(N_pro_arm)):  # TODO: distinguish the two propellers if push/pull configuration
            fig.add_shape(
                dict(type="circle", line=dict(color=COLS[k],width=0), fillcolor=COLS[k],
                     x0=arm_length * np.cos(sep_angle) - D_pro / 2 + R * np.cos(sep_angle),
                     y0=arm_length * np.sin(sep_angle) - D_pro / 2 + R * np.sin(sep_angle),
                     x1=arm_length * np.cos(sep_angle) + D_pro / 2 + R * np.cos(sep_angle),
                     y1=arm_length * np.sin(sep_angle) + D_pro / 2 + R * np.sin(sep_angle),
                     )
            )

    # ARMS
    x_arms = np.concatenate((x_arms, [x_arms[0]]))
    y_arms = np.concatenate((y_arms, [y_arms[0]]))
    scatter = go.Scatter(x=y_arms, y=x_arms, mode="lines", line_color=COLS[k], name=name)
    fig.add_trace(scatter)

    fig = go.FigureWidget(fig)
    fig.update_shapes(opacity=0.25, xref="x", yref="y")
    fig.update_layout(
        title_text="Drone Geometry", title_x=0.5, xaxis_title="y", yaxis_title="x",
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    return fig