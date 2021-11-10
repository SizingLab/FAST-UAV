"""
Sobol-Saltelli 2002 Analysis Method
"""

from openmdao_extensions.salib_doe_driver import SalibDOEDriver
import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
import numpy as np
from ipywidgets import widgets, Layout
import plotly.graph_objects as go
from SALib.analyze import sobol


def doe_sobol(des_vars_dict: dict, obj_var: str, conf_file: str, ns: int = 1024, calc_second_order: bool = True) -> pd.DataFrame:
    """
    DoE for Sobol-Saltelli 2002 Method

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

    # Setup driver (Sobol method with Saltelli sample)
    prob.driver = SalibDOEDriver(
        sa_method_name="Sobol",
        sa_doe_options={"n_samples": ns, "calc_second_order": calc_second_order},
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


def sobol_analysis(conf_file, output_file):
    """
    Interactive interface to define and simulate a Sobol' sensitivity analysis with Multiple Inputs and a Single Output.
    Plots the Sobol' indices.

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

    # Second order checkbox
    second_order_box = widgets.Checkbox(
        value=True,
        description='Second order indices',
        disabled=False,
        indent=False
    )

    # Assign empty figures
    # Bar plot
    fig1 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Contributions to the output standard deviation'
        ),
        yaxis=dict(title='Output standard deviation'),
        #yaxis2=dict(title='Output Variance', side='right',overlaying='y'),
    ))
    fig1.add_trace(go.Bar(name='ST', x=[], y=[], error_y=dict(type='data', array=[])))
    fig1.add_trace(go.Bar(name='S1', x=[], y=[], error_y=dict(type='data', array=[])))
    #fig1.add_trace(go.Bar(name='S2', x=[], y=[], error_y=dict(type='data', array=[])))
    fig1.update_layout(barmode='group')

    # Heat map
    fig2 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Interaction effects'
        )),
        data=go.Heatmap(
            z=[],
            x=[],
            y=[],
            hoverongaps=False,
            colorbar=dict(title='Sobol index', titleside="top")
        ))

    # Pie chart
    fig3 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Sobol indices'
        )),
        data=go.Sunburst(
            labels=[],
            parents=[],
            values=[],
            textinfo='label+percent entry',
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
                x_var = var_box.value
                a = x_value * (1. - x_var)
                b = x_value * (1. + x_var)
                bounds = [a, b]
                x_dict[inputbox.value] = bounds

        if not validate(outputbox, x_dict):
            return False

        # Design of experiments with Saltelli's sampling
        ns = int(samples.value)  # number of samples for MC simulation
        y = outputbox.value
        second_order = second_order_box.value  # boolean for second order Sobol' indices calculation
        df = doe_sobol(x_dict, y, conf_file, ns, second_order)

        # Additional data on the output variable of interest for plots
        y_data = table.loc[table['Name'] == y] # output variable of interest
        y_unit = y_data["Unit"].unique()[0]  # unit
        y_var = df[y].var()  # variance

        # Sobol' analysis
        problem_sobol = {
            'num_vars': len(x_dict),
            'names': list(x_dict.keys()),
            'bounds': list(x_dict.values())}
        Y_sobol = df[y].to_numpy()
        Si = sobol.analyze(problem_sobol, Y_sobol, calc_second_order=second_order)
        #S2 = np.nan_to_num(Si['S2']).sum(axis=1) # sum second-order effects
        #S2_conf = np.nan_to_num(Si['S2_conf']).sum(axis=1) # sum second-order confidence intervals

        # Update figures
        with fig1.batch_update():
            fig1.data[0].x = list(x_dict.keys())
            fig1.data[0].y = np.sqrt(Si['ST'] * y_var)
            fig1.data[0].error_y = dict(type='data', array=np.sqrt(Si['ST_conf'] * y_var))
            fig1.data[1].x = list(x_dict.keys())
            fig1.data[1].y = np.sqrt(Si['S1'] * y_var)
            fig1.data[1].error_y = dict(type='data', array=np.sqrt(Si['S1_conf'] * y_var))
            #if second_order == True:
            #    fig1.data[2].x = list(x_dict.keys())
            #    fig1.data[2].y = np.sqrt(S2 * y_var)
            #    fig1.data[2].error_y = dict(type='data', array=np.sqrt(S2_conf * y_var))
            #else:
            #    fig1.data[2].x = []
            #    fig1.data[2].y = []
            #    fig1.data[2].error_y = dict(type='data', array=[])
            fig1.update_yaxes(title='Standard deviation (' + (y_unit if y_unit is not None else '') + ') <br>' + str(y))
            # Si.plot()

        with fig2.batch_update():
            fig2.data[0].x = list(x_dict.keys())
            fig2.data[0].y = list(x_dict.keys())
            z_data = []
            if second_order == True:
                z_data = Si['S2']
            else:
                z_data = np.empty((len(x_dict), len(x_dict)))  # second-order indices
                z_data[:] = np.nan
            np.fill_diagonal(z_data, Si['ST'])  # add total order indices on diagonal
            fig2.data[0].z = z_data

        with fig3.batch_update():
            # sunburst title
            labels = ["Total-effect indices"]
            parents = [""]
            values = list([np.sum(Si['ST'])])
            values = list([""])
            # total-effect indices
            labels += list(x_dict.keys())
            parents += ["Total-effect indices" for i in range(len(x_dict))]
            values += list(Si['ST'])
            # First- and second-order indices
            #for i in range(len(x_dict)):
            #    key = list(x_dict.keys())[i]
            #    parents += [key]
            #    labels += ['S1']
            #    values += [Si['S1'][i]]
            #    values += [S2[i]]
            fig3.data[0].labels = labels
            fig3.data[0].parents = parents
            fig3.data[0].values = values

    # Set up Figure
    inputs_array = []
    options_panel = widgets.VBox([widgets.HBox([outputbox, samples, second_order_box, update_button])])
    widg = widgets.VBox([widgets.HBox([fig1, fig2]),
                         widgets.HBox([fig3]),
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