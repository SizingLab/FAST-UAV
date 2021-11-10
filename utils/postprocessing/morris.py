"""
Morris Analysis Method
"""

from openmdao_extensions.salib_doe_driver import SalibDOEDriver
import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
import numpy as np
from ipywidgets import widgets, Layout
import plotly.graph_objects as go
from SALib.analyze import sobol, morris


def doe_morris(des_vars_dict: dict, obj_var: str, conf_file: str, nt: int = 1024) -> pd.DataFrame:
    """
    DoE for Morris Method

    :param des_vars_dict: inputs dictionnary {input_name: distribution_law}
    :param obj_var: output name
    :param conf_file: configuration file for the problem
    :param ns: number of samples for MC simulation

    :return: dataframe of the monte carlo simulation results
    """

    # Problem configuration
    conf = oad.FASTOADProblemConfigurator(conf_file)
    prob = conf.get_problem(read_inputs=True)

    # DoE parameters
    dists = []
    for des_var in list(des_vars_dict.items()):
        prob.model.add_design_var(des_var[0], lower=des_var[1][0],
                                  upper=des_var[1][1])  # add design variable to problem
        dist = des_var[1]  # add associated distribution law (not used here)
        dists.append(dist)
    prob.model.add_objective(obj_var)  # add objective

    # Setup driver (Morris method)
    prob.driver = SalibDOEDriver(
        sa_method_name="Morris",
        sa_doe_options={"n_trajs": nt},
    )

    # Run problem
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    prob.setup()
    prob.run_driver()
    prob.cleanup()

    # Get results
    cr = om.CaseReader("cases.sql")
    cases = cr.list_cases('driver', out_stream=None)

    df = pd.DataFrame()
    for case in cases:
        values = cr.get_case(case).outputs
        df = df.append(values, ignore_index=True)

    for i in df.columns:
        df[i] = df[i].apply(lambda x: x[0])

    return df


def morris_analysis(conf_file, output_file):
    """
    Interactive interface to define and simulate a Morris sensitivity analysis with Multiple Inputs and a Single Output.
    Plots the Morris results.

    :param conf_file: configuration file of the problem
    :param output_file: output file of the initial problem, to set up initial values.
    """

    # Get variables data from output file
    variables = DataFile(output_file)
    variables.sort(key=lambda var: var.name)
    table = variables.to_dataframe()[["name", "val", "units", "is_input", "desc"]].rename(
        columns={"name": "Name", "val": "Value", "units": "Unit", "desc": "Description"}
    )

    def input_box():
        # Input box
        inputbox = widgets.Dropdown(
            description='Input:   ',
            options=table.loc[table['is_input']]["Name"].unique().tolist(),
            value=None
        )
        # Values boxes
        value_box = widgets.Text(
            value='',
            description='',
            continuous_update=False,
            disabled=True
        )
        var_box = widgets.FloatSlider(
            value=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description='error interval:',
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format='.0%',
        )
        return widgets.HBox([inputbox, value_box, var_box])

    # "add input" button
    addinput_button = widgets.Button(description="add input")

    # Number of samples
    samples = widgets.FloatLogSlider(
        value=128,
        base=2,
        min=6,
        max=10,
        step=1,
        description='Samples:',
        continuous_update=False
    )

    # Output box
    outputbox = widgets.Dropdown(
        description='Output:   ',
        options=table.loc[~table['is_input']]["Name"].unique().tolist(),
        value=None
    )

    # "Update" button
    update_button = widgets.Button(description="update")

    # Assign empty figures
    # Bar plot
    fig1 = go.FigureWidget(layout=go.Layout(
        title=dict(text='Morris results'),
        yaxis=dict(title=r'$\mu^*$')
    ))
    fig1.add_trace(go.Bar(name='mu_star', x=[], y=[], error_y=dict(type='data', array=[])))

    # Scatter plot
    fig2 = go.FigureWidget(layout=go.Layout(
        xaxis=dict(title=r'$\mu^*$'),
        yaxis=dict(title=r'$\sigma$'))
    )
    fig2.add_trace(go.Scatter(x=[], y=[],
                              mode='markers+text',
                              name='inputs',
                              text=[],
                              textposition="top center"))
    fig2.add_trace(go.Scatter(x=[0, 1.0], y=[0, 1.0],
                              mode='lines',
                              line=dict(color='black'),
                              name=r'$\sigma/\mu^* = 1.0$'))
    fig2.add_trace(go.Scatter(x=[0, 2.0], y=[0, 1.0],
                              mode='lines',
                              line=dict(color='black', dash='dash'),
                              name=r'$\sigma/\mu^* = 0.5$'))
    fig2.add_trace(go.Scatter(x=[0, 10.0], y=[0, 1.0],
                              mode='lines',
                              line=dict(color='black', dash='dot'),
                              name=r'$\sigma/\mu^* = 0.1$'))

    def add_input(change):
        # add new input row
        new_input = input_box()
        inputs_array.append(new_input)
        widg.children = widg.children[:-1] + (new_input, addinput_button)
        n_input = len(inputs_array) - 1  # input row indice

        def variable_data(change):
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            x_data = table.loc[table['Name'] == inputbox.value]  # corresponding data from output file
            if x_data["Value"].unique().size != 0:  # check data exists
                x_value = x_data["Value"].unique()[0]  # get value
                x_unit = x_data["Unit"].unique()[0]
            else:
                # print("Please select value in dropwdown menu")
                return False
            value_box = inputs_array[n_input].children[1]  # value of the variable
            var_box = inputs_array[n_input].children[2]  # variation to apply for the DoE
            value_box.value = "{:10.3f} ".format(x_value) + (x_unit if x_unit is not None else '')
            var_box.value = 0.1

        # add an observe event to update value according to selected variable
        new_input.children[0].observe(variable_data, names="value")

    def validate(outputbox, x_dict):
        if outputbox.value is None or len(x_dict) == 0:
            return False
        else:
            return True

    def update_charts(change):
        # Get data from each input row
        x_dict = {}
        for inputrow in inputs_array:
            inputbox = inputrow.children[0]
            var_box = inputrow.children[2]
            x_data = table.loc[table['Name'] == inputbox.value]
            if x_data["Value"].unique().size != 0:  # not empty input
                x_value = x_data["Value"].unique()[0]
                a = x_value * (1. - var_box.value)
                b = x_value * (1. + var_box.value)
                bounds = [a, b]
                x_dict[inputbox.value] = bounds

        if not validate(outputbox, x_dict):
            return False

        # Design of experiments
        ns = int(samples.value)  # number of samples
        y = outputbox.value
        df = doe_morris(x_dict, y, conf_file, ns)

        # Screening method of Morris
        problem_morris = {
            'num_vars': len(x_dict),
            'names': list(x_dict.keys()),
            'bounds': list(x_dict.values())}
        X_morris = df[list(x_dict.keys())].to_numpy()
        Y_morris = df[y].to_numpy()
        Si = morris.analyze(problem_morris, X_morris, Y_morris, conf_level=0.95, num_resamples=100)

        # Update figures
        y_data = table.loc[table['Name'] == y]  # corresponding data from output file
        # y_value = y_data["Value"].unique()[0]  # get value
        y_unit = y_data["Unit"].unique()[0]  # get unit

        with fig1.batch_update():
            fig1.data[0].x = list(x_dict.keys())
            fig1.data[0].y = Si['mu_star']
            fig1.data[0].error_y = dict(type='data', array=Si['mu_star_conf'])
            fig1.update_layout(yaxis=dict(title='$\\mu^* \\text{ (' + (y_unit if y_unit is not None else '') + ')}$'))

        with fig2.batch_update():
            fig2.data[0].x = Si['mu_star']
            fig2.data[0].y = Si['sigma']
            fig2.data[0].text = list(x_dict.keys())

            fig2.data[1].x = [0, max(Si['mu_star'])]
            fig2.data[1].y = [0, max(Si['mu_star'])]
            fig2.data[2].x = [0, 2 * max(Si['mu_star'])]
            fig2.data[2].y = [0, max(Si['mu_star'])]
            fig2.data[3].x = [0, 10 * max(Si['mu_star'])]
            fig2.data[3].y = [0, max(Si['mu_star'])]

            fig2.update_layout(
                xaxis=dict(title='$\\mu^* \\text{ (' + (y_unit if y_unit is not None else '') + ')}$'),
                xaxis_range=[0, max(Si['mu_star'])],
                yaxis_range=[-0.05, max(0.1 * max(Si['mu_star']), max(Si['sigma']))]
            )

    # Set up Figure
    inputs_array = []
    options_panel = widgets.VBox([widgets.HBox([outputbox, samples, update_button])])
    widg = widgets.VBox([widgets.HBox([fig1, fig2]),
                         options_panel,
                         addinput_button
                         ],
                        layout=Layout(align_items="flex-start"))

    # add first input
    add_input(0)

    # add input and update buttons interactions
    addinput_button.on_click(add_input)
    update_button.on_click(update_charts)  # Run the Monte Carlo with provided parameters, and display the charts

    return widg