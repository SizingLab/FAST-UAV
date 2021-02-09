"""
Objectives definition
"""
import openmdao.api as om
import numpy as np

from models.Propeller.propeller import PropellerMR

class Objective(om.Group):
    """
    Group containing the objective functions
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        self.add_subsystem("define_objective", DefineObjectives(use_catalogues=self.options['use_catalogues'],
                                                                use_gearbox=self.options['use_gearbox']),
                           promotes=["*"])


class DefineObjectives(om.ExplicitComponent):
    """
    Weight objective and flight autonomy objective definitions, with associated constraints
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        if self.options["use_catalogues"]:
            self.add_input('data:ESC:catalogue:mass', val=np.nan, units='kg')
            self.add_input('data:motor:catalogue:mass', val=np.nan, units='kg')
            self.add_input('data:battery:catalogue:mass', val=np.nan, units='kg')
            self.add_input('data:battery:catalogue:capacity', val=np.nan, units='A*s')
            self.add_input('data:battery:catalogue:voltage', val=np.nan, units='V')
        else:
            self.add_input('data:ESC:mass', val=np.nan, units='kg')
            self.add_input('data:motor:mass', val=np.nan, units='kg')
            self.add_input('data:battery:mass', val=np.nan, units='kg')
            self.add_input('data:battery:capacity', val=np.nan, units='A*s')
            self.add_input('data:battery:voltage', val=np.nan, units='V')

        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:frame:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('specifications:load:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('optimization:settings:k_M', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_input('specifications:hover_time', val=np.nan, units='min')
        self.add_input('specifications:MTOW', val=np.nan, units='kg')
        self.add_output('optimization:objectives:mass_total', units='kg')
        self.add_output('optimization:objectives:hover_time', units='min')
        self.add_output('optimization:constraints:mass_convergence', units=None)
        self.add_output('optimization:constraints:flight_autonomy', units=None)
        self.add_output('optimization:constraints:MTOW', units=None)

        if self.options["use_gearbox"]:
            self.add_input('data:gearbox:mass', val=np.nan, units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if self.options["use_catalogues"]:
            Mmot = inputs['data:motor:catalogue:mass']
            Mesc = inputs['data:ESC:catalogue:mass']
            Mbat = inputs['data:battery:catalogue:mass']
            C_bat = inputs['data:battery:catalogue:capacity']
            V_bat = inputs['data:battery:catalogue:voltage']
        else:
            Mmot = inputs['data:motor:mass']
            Mesc = inputs['data:ESC:mass']
            Mbat = inputs['data:battery:mass']
            C_bat = inputs['data:battery:capacity']
            V_bat = inputs['data:battery:voltage']

        if self.options["use_gearbox"]:
            Mgear = inputs['data:gearbox:mass']
        else:
            Mgear = 0

        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['specifications:load:mass']
        Mfra = inputs['data:structure:frame:mass']
        Marm = inputs['data:structure:arms:mass']
        k_M = inputs['optimization:settings:k_M']
        t_h = inputs['specifications:hover_time']
        MTOW = inputs['specifications:MTOW']
        P_el_hover = inputs['data:motor:power:hover']
        eta_ESC = inputs['data:ESC:efficiency']

        # Objectives
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        I_bat = (P_el_hover * Npro) / eta_ESC / V_bat  # [I] Current of the battery
        t_hf = .8 * C_bat / I_bat / 60  # [min] Hover time

        # Constraints
        mass_con = (t_hf - t_h) / t_hf  # Min. hover flight autonomy, for weight minimization
        time_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        # Global constraint : mass convergence
        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        global_con = (Mtotal_estimated - Mtotal) / Mtotal  # mass convergence

        outputs['optimization:objectives:mass_total'] = Mtotal
        outputs['optimization:objectives:hover_time'] = t_hf
        outputs['optimization:constraints:flight_autonomy'] = mass_con
        outputs['optimization:constraints:MTOW'] = time_con
        outputs['optimization:constraints:mass_convergence'] = global_con
