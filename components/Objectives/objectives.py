"""
Objectives definition
"""
import openmdao.api as om
import numpy as np
from fastoad.models.options import OpenMdaoOptionDispatcherGroup


class Objective(OpenMdaoOptionDispatcherGroup):
    """
    Group containing the objective functions
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("define_objective", DefineObjectives(), promotes=["*"])


class DefineObjectives(om.ExplicitComponent):
    """
    Weight objective and flight autonomy objective definitions, with associated constraints
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:structure:mass:frame', val=np.nan, units='kg')
        self.add_input('data:structure:mass:arms', val=np.nan, units='kg')
        self.add_input('specifications:load:mass', val=np.nan, units='kg')
        self.add_input('data:gearbox:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan)
        self.add_input('data:battery:performances:current', val=np.nan, units='A')
        self.add_input('data:battery:performances:capacity', val=np.nan, units='A*s')
        self.add_input('optimization:settings:k_M', val=np.nan)
        #self.add_input('optimization:objectives:mass_total_estimated', val=np.nan, units='kg')
        self.add_input('specifications:hover_time', val=np.nan, units='min')
        self.add_input('specifications:MTOW', val=np.nan, units='kg')
        self.add_output('optimization:objectives:mass_total', units='kg')
        self.add_output('optimization:objectives:hover_time', units='min')
        self.add_output('optimization:constraints:cons_mass_convergence')
        self.add_output('optimization:constraints:cons_flight_autonomy')
        self.add_output('optimization:constraints:cons_MTOW')

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
        Mgear = inputs['data:gearbox:mass']
        k_M = inputs['optimization:settings:k_M']
        C_bat = inputs['data:battery:performances:capacity']
        I_bat = inputs['data:battery:performances:current']
        t_h = inputs['specifications:hover_time']
        MTOW = inputs['specifications:MTOW']

        # Objectives
        if self.options["use_gearbox"]:
            Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass with reducer
        else:
            Mtotal = (Mesc + Mpro + Mmot) * Npro + M_load + Mbat + Mfra + Marm  # total mass without reducer

        t_hf = .8 * C_bat / I_bat / 60  # [min] Hover time

        # Constraints
        mass_con = (t_hf - t_h) / t_hf  # Min. hover flight autonomy, for weight minimization
        time_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        # Global constraint : mass convergence
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        global_con = (Mtotal_estimated - Mtotal) / Mtotal  # mass convergence

        outputs['optimization:objectives:mass_total'] = Mtotal
        outputs['optimization:objectives:hover_time'] = t_hf
        outputs['optimization:constraints:cons_flight_autonomy'] = mass_con
        outputs['optimization:constraints:cons_MTOW'] = time_con
        outputs['optimization:constraints:cons_mass_convergence'] = global_con
