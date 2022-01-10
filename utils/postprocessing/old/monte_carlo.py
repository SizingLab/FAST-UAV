"""
Defines the Monte Carlo Simulation and post processing
"""

import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
from openmdao_extensions.openturns_doe_driver import OpenturnsDOEDriver
import openturns as ot
from ipywidgets import widgets, Layout
import plotly.graph_objects as go


def doe_montecarlo(des_vars_dict: dict, obj_var: str, conf_file: str, ns: int = 1000) -> pd.DataFrame:
    """
    Monte Carlo Simulation

    :param des_vars_dict: inputs dictionnary {input_name: distribution_law}
    :param obj_var: output name
    :param conf_file: configuration file for the problem
    :param ns: number of samples for MC simulation

    :return: dataframe of the monte carlo simulation results
    """
    # Problem configuration
    conf = oad.FASTOADProblemConfigurator(conf_file)
    prob = conf.get_problem(read_inputs=True)

    # DoE parameters (Monte Carlo simulation)
    dists = []

    for des_var in list(des_vars_dict.items()):
        prob.model.add_design_var(des_var[0])  # add design variable to problem
        dist = des_var[1]  # add associated distribution law
        dists.append(dist)
    prob.model.add_objective(obj_var)  # add objective

    # Setup driver
    prob.driver = OpenturnsDOEDriver(
        n_samples=ns, distribution=ot.ComposedDistribution(dists)
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


def montecarlo_siso(conf_file, output_file):
    """
    Interactive interface to define and simulate a Monte Carlo simulation with a Single Input and a Single Output.
    Plots the distribution of the input and output variables as well as the scatter plot and the parallel coordinates
    plot.

    :param conf_file: configuration file of the problem
    :param output_file: output file of the initial problem, to set up initial values.
    """
    # Get variables names
    variables = DataFile(output_file)
    variables.sort(key=lambda var: var.name)
    table = variables.to_dataframe()[["name", "val", "units", "is_input", "desc"]].rename(
        columns={"name": "Name", "val": "Value", "units": "Unit", "desc": "Description"}
    )

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
        description='std',
        tooltip='standard deviation',
        disabled=False,
        continuous_update=False,
        readout=True,
        readout_format='.0%',
    )

    # Distribution law boxes
    law_buttons = widgets.ToggleButtons(
        options=['Normal', 'Uniform'], #['Normal', 'Uniform', 'Exponential'],
        description='Distribution Law:',
        disabled=False,
        button_style='',
    )
    #law_param_box_1 = widgets.FloatText(
    #    value=None,
    #    description='mu:',
    #    continuous_update=False,
    #    step=0.01
    #)
    #law_param_box_2 = widgets.FloatText(
    #    value=None,
    #    description='sigma:',
    #    continuous_update=False
    #)

    # Monte Carlo n_samples
    samples = widgets.IntSlider(
        value=100,
        min=10,
        max=10000,
        step=10,
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
    fig1 = go.FigureWidget(data=[go.Histogram(histnorm='probability', autobinx=True)],
                           layout=go.Layout(
                               title=dict(
                                   text='Input Distribution'
                               )
                           ))
    fig2 = go.FigureWidget(data=[go.Histogram(histnorm='probability', autobinx=True)],
                           layout=go.Layout(
                               title=dict(
                                   text='Output Distribution'
                               )
                           ))
    fig3 = go.FigureWidget(data=[go.Scatter(mode='markers')],
                           layout=go.Layout(
                               title=dict(
                                   text='Scatter Plot'
                               )
                           ))
    fig4 = go.FigureWidget(data=go.Parcoords(),
                           layout=go.Layout(
                               title=dict(
                                   text='Parallel Coordinates Plot'
                               )
                           ))

    def variable_data(change):
        x_data = table.loc[table['Name'] == inputbox.value]
        if x_data["Value"].unique().size != 0:  # check data exists
            x_value = x_data["Value"].unique()[0]  # get value
            x_unit = x_data["Unit"].unique()[0]  # get unit
        else:
            # print("Please select value in dropwdown menu")
            return False
        value_box.value = "{:10.3f} ".format(x_value) + (x_unit if x_unit is not None else '')
        var_box.value = 0.1
        if law_buttons.value == "Normal":
            var_box.description = 'std'
            var_box.tooltip = 'standard deviation'
            #mu = x_value
            #sigma = 0.1 * mu if mu != 0 else 0.1
            #widg.children[0].children[0].children[2].description = 'mu'
            #widg.children[0].children[0].children[2].value = mu
            #widg.children[0].children[0].children[3].description = 'sigma'
            #widg.children[0].children[0].children[3].value = sigma

        if law_buttons.value == "Uniform":
            var_box.description = 'error interval'
            var_box.tooltip = 'error interval'
            #widg.children[0].children[0].children[2].description = 'a'
            #widg.children[0].children[0].children[2].value = x_value / 2 if x_value != 0 else 0.0
            #widg.children[0].children[0].children[3].description = 'b'
            #widg.children[0].children[0].children[3].value = x_value * 2

        #if law_buttons.value == "Exponential":
        #    widg.children[0].children[0].children[2].description = 'lambda'
        #    widg.children[0].children[0].children[2].value = 10.0
        #    widg.children[0].children[0].children[3].description = 'gamma'
        #    widg.children[0].children[0].children[3].value = x_value

    def validate():
        if outputbox.value in table['Name'].unique() and inputbox.value in table['Name'].unique():
            return True
        else:
            return False

    def update_charts(change):
        if validate():
            # Get data corresponding to user entries
            x_data = table.loc[table['Name'] == inputbox.value]
            y_data = table.loc[table['Name'] == outputbox.value]
            ns = int(samples.value)
            if x_data["Value"].unique().size != 0:  # check not empty input
                x_value = x_data["Value"].unique()[0]
                x_var = var_box.value
                if law_buttons.value == "Normal":
                    mu = x_value
                    sigma = x_var * x_value
                    #mu = law_param_box_1.value
                    #sigma = law_param_box_2.value
                    dist_law = ot.Normal(mu, sigma)
                elif law_buttons.value == "Uniform":
                    a = x_value * (1. - x_var)
                    b = x_value * (1. + x_var)
                    #a = law_param_box_1.value
                    #b = law_param_box_2.value
                    dist_law = ot.Uniform(a, b)
                #elif law_buttons.value == "Exponential":
                #    lmbda = law_param_box_1.value
                #    gamma = law_param_box_2.value
                #    dist_law = ot.Exponential(lmbda, gamma)
                else:
                    return 0
            else:
                return 0

            x_dict = {inputbox.value: dist_law}
            y = outputbox.value

            # Monte Carlo
            df = doe_montecarlo(x_dict, y, conf_file, ns)

            # Update figures
            with fig1.batch_update():  # Input Distribution
                fig1.data[0].x = df[inputbox.value]
                fig1.layout.xaxis.title = inputbox.value + ' [%s]' % (
                    x_data["Unit"].unique()[0] if x_data["Unit"].unique()[0] is not None else "-")
            with fig2.batch_update():  # Output Distribution
                fig2.data[0].x = df[outputbox.value]
                fig2.layout.xaxis.title = outputbox.value + ' [%s]' % (
                    y_data["Unit"].unique()[0] if y_data["Unit"].unique()[0] is not None else "-")
            with fig3.batch_update():  # Scatter plot
                fig3.data[0].x = df[inputbox.value]
                fig3.data[0].y = df[y]
                fig3.layout.xaxis.title = inputbox.value + ' [%s]' % (
                    x_data["Unit"].unique()[0] if x_data["Unit"].unique()[0] is not None else "-")
                fig3.layout.yaxis.title = outputbox.value + ' [%s]' % (
                    y_data["Unit"].unique()[0] if y_data["Unit"].unique()[0] is not None else "-")
            with fig4.batch_update():  # Parallel coordinate plot
                fig4.data[0].dimensions = [dict(label=inputbox.value, values=df[inputbox.value]),
                                           dict(label=outputbox.value, values=df[outputbox.value])]
                fig4.data[0].line = dict(color=df[outputbox.value], colorscale='Viridis')

    # Events
    law_buttons.observe(variable_data, names="value")  # update data when the user changes law
    inputbox.observe(variable_data, names="value")  # update data when the user changes input variable
    update_button.on_click(update_charts)  # Run the Monte Carlo with provided parameters, and display the charts

    # Set up Figure
    options_panel = widgets.VBox([widgets.HBox([inputbox, value_box, law_buttons, var_box]),
                                  widgets.HBox([outputbox, samples, update_button])])
    widg = widgets.VBox([options_panel,
                         widgets.HBox([fig1, fig2]),
                         widgets.HBox([fig3, fig4])
                         ],
                        layout=Layout(align_items="center"))

    return widg


def montecarlo_miso(conf_file, output_file):
    """
    Interactive interface to define and simulate a Monte Carlo simulation with Multiple Inputs and a Single Output.
    Plots the distribution of the input and output variables as well as the scatter plot and the parallel coordinates
    plot.

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
            description='std',
            tooltip='standard deviation',
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format='.0%',
        )

        # Input distribution parameters boxes
        law_buttons = widgets.ToggleButtons(
            options=['Normal', 'Uniform'],
            description='Distribution Law:',
            disabled=False,
            button_style='',
        )
        #law_param_box_1 = widgets.FloatText(
        #    value=None,
        #    description='mu:',
        #    continuous_update=False,
        #    step=0.01
        #)
        #law_param_box_2 = widgets.FloatText(
        #    value=None,
        #    description='sigma:',
        #    continuous_update=False
        #)
        return widgets.HBox([inputbox, value_box, law_buttons, var_box])

    # "add input" button
    addinput_button = widgets.Button(description="add input")

    # Monte Carlo n_samples
    samples = widgets.IntSlider(
        value=100,
        min=10,
        max=10000,
        step=10,
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
    fig1 = go.FigureWidget(data=go.Parcoords(),
                           layout=go.Layout(
                               title=dict(
                                   text='Parallel Coordinates Plot'
                               )
                           ))
    fig1.update_layout(width=1000)

    fig2 = go.FigureWidget(data=go.Splom(showupperhalf=False),
                           layout=go.Layout(
                               title=dict(
                                   text='Scatterplot Matrix'
                               )
                           ))

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
                x_unit = x_data["Unit"].unique()[0]  # get unit
            else:
                # print("Please select value in dropwdown menu")
                return False
            value_box = inputs_array[n_input].children[1]  # value of the variable
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            var_box = inputs_array[n_input].children[3]  # variation to apply for the DoE
            value_box.value = "{:10.3f} ".format(x_value) + (x_unit if x_unit is not None else '')
            var_box.value = 0.1

            if law_buttons.value == "Normal":
                var_box.description = 'std'
                var_box.tooltip = 'standard deviation'
            #mu = x_value
            #sigma = 0.1 * mu if mu != 0 else 0.1
            #widg.children[0].children[0].children[2].description = 'mu'
            #widg.children[0].children[0].children[2].value = mu
            #widg.children[0].children[0].children[3].description = 'sigma'
            #widg.children[0].children[0].children[3].value = sigma

            if law_buttons.value == "Uniform":
                var_box.description = 'error interval'
                var_box.tooltip = 'error interval'

            #f law_buttons.value == "Normal":  # Normal law
            #    mu = x_value
            #    sigma = 0.1 * mu if mu != 0.0 else 0.1
            #    law_param_box_1.description = 'mu'
            #    law_param_box_1.value = mu
            #    law_param_box_2.description = 'sigma'
            #    law_param_box_2.value = sigma

            #if law_buttons.value == "Uniform":  # Uniform law
            #    law_param_box_1.description = 'a'
            #    law_param_box_1.value = x_value / 2 if x_value != 0 else 0.0
            #    law_param_box_2.description = 'b'
            #    law_param_box_2.value = x_value * 2

            #if law_buttons.value == "Exponential":  # Log Uniform law
            #    law_param_box_1.description = 'lambda'
            #    law_param_box_1.value = 10.0
            #    law_param_box_2.description = 'gamma'
            #    law_param_box_2.value = x_value

        # add an observe event: probability law parameters update
        new_input.children[0].observe(variable_data, names="value")
        new_input.children[2].observe(variable_data, names="value")

        # add an observe event: if this row is filled, a new input row will be added
        # new_input.children[0].observe(add_input, names="value")
        # remove observe event from the other rows to avoid annoying creation of new rows.
        # print(inputs_array)
        # for previous_input in inputs_array[:-1]:
        #    previous_input.children[0].unobserve(add_input, names="value")

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
            x_data = table.loc[table['Name'] == inputbox.value]
            if x_data["Value"].unique().size != 0:  # not empty input
                law_buttons = inputrow.children[2]  # distribution law
                var_box = inputrow.children[3]  # variation to apply for the DoE
                x_value = x_data["Value"].unique()[0]
                x_var = var_box.value
                if law_buttons.value == "Normal":
                    mu = x_value
                    sigma = x_var
                    #mu = law_param_box_1.value
                    #sigma = law_param_box_2.value
                    dist_law = ot.Normal(mu, sigma)
                elif law_buttons.value == "Uniform":
                    a = x_value * (1. - x_var)
                    b = x_value * (1. + x_var)
                    #a = law_param_box_1.value
                    #b = law_param_box_2.value
                    dist_law = ot.Uniform(a, b)
                x_dict[inputbox.value] = dist_law
                #if law_buttons.value == "Normal":
                #    mu = law_param_box_1.value
                #    sigma = law_param_box_2.value
                #    dist_law = ot.Normal(mu, sigma)
                #elif law_buttons.value == "Uniform":
                #    a = law_param_box_1.value
                #    b = law_param_box_2.value
                #    dist_law = ot.Uniform(a, b)
                #elif law_buttons.value == "Exponential":
                #    lmbda = law_param_box_1.value
                #    gamma = law_param_box_2.value
                #    dist_law = ot.Exponential(lmbda, gamma)

        if not validate(outputbox, x_dict):
            return False

        # Design of experiments
        ns = int(samples.value)  # number of samples
        y = outputbox.value
        df = doe_montecarlo(x_dict, y, conf_file, ns)

        # Update figures
        with fig1.batch_update():  # Parallel coordinate plot
            dimensions = [dict(label=des_var, values=df[des_var]) for des_var in x_dict]
            dimensions.append(dict(label=outputbox.value, values=df[outputbox.value]))
            fig1.data[
                0].dimensions = dimensions
            fig1.data[0].line = dict(color=df[outputbox.value], colorscale='Viridis')

        with fig2.batch_update():  # Scatterplot matrix
            dimensions = [dict(label=des_var, values=df[des_var]) for des_var in x_dict]
            dimensions.append(dict(label=outputbox.value, values=df[outputbox.value]))
            fig2.data[
                0].dimensions = dimensions
            fig2.data[0].marker = dict(color=df[outputbox.value], colorscale='Viridis')

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
