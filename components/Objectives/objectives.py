"""
Objectives definition
"""
import openmdao.api as om
import numpy as np


class Objective(om.Group):
    """
    Group containing the objective functions
    """

    def setup(self):
        self.add_subsystem("objective", WeightObjective(), promotes=["*"])


class WeightObjective(om.ExplicitComponent):
    """
    Weight objective and associated constraints definition
    """

    def setup(self):
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:structure:mass:frame', val=np.nan, units='kg')
        self.add_input('data:structure:mass:arms', val=np.nan, units='kg')
        self.add_input('specifications:load:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan)
        self.add_input('optimization:objectives:mass_total_estimated', val=np.nan, units='kg')
        self.add_input('optimization:objectives:hover_time', val=np.nan, units='min')
        self.add_input('specifications:hover_time', val=np.nan, units='min')
        self.add_output('optimization:objectives:mass_total', units='kg')
        self.add_output('optimization:constraints:mass_objective:cons_mass_convergence')
        self.add_output('optimization:constraints:mass_objective:cons_flight_autonomy')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mesc = inputs['data:ESC:mass']
        Mpro = inputs['data:propeller:mass']
        Mmot = inputs['data:motor:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['specifications:load:mass']
        Mbat = inputs['data:battery:mass']
        Mfra = inputs['data:structure:mass:frame']
        Marm = inputs['data:structure:mass:arms']
        Mtotal_estimated = inputs['optimization:objectives:mass_total_estimated']
        t_hf = inputs['optimization:objectives:hover_time']
        t_h = inputs['specifications:hover_time']

        # Objective : mass minimization
        Mtotal = (Mesc + Mpro + Mmot) * Npro + M_load + Mbat + Mfra + Marm  # total mass without reducer

        # Constraints
        mass_con1 = (Mtotal_estimated - Mtotal) / Mtotal_estimated  # mass convergence
        mass_con2 = (t_hf - t_h) / t_hf  # hover flight autonomy

        outputs['optimization:objectives:mass_total'] = Mtotal
        outputs['optimization:constraints:mass_objective:cons_mass_convergence'] = mass_con1
        outputs['optimization:constraints:mass_objective:cons_flight_autonomy'] = mass_con2

