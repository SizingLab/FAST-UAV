"""
System parameters
"""
import openmdao.api as om
import numpy as np


class System(om.Group):
    """
    Group containing the system parameters
    """
    def setup(self):
        self.add_subsystem("MTOW", MTOW(), promotes=['*'])
        self.add_subsystem("system_constraints", SystemConstraints(), promotes=['*'])


class MTOW(om.ExplicitComponent):
    """
    MTOW calculation
    """

    def setup(self):
        self.add_input('data:gearbox:mass', val=0.0, units='kg')
        self.add_input('data:ESC:mass', val=np.nan, units='kg')
        self.add_input('data:motor:mass', val=np.nan, units='kg')
        self.add_input('data:battery:mass', val=np.nan, units='kg')
        self.add_input('data:battery:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:voltage', val=np.nan, units='V')
        self.add_input('data:propeller:mass', val=np.nan, units='kg')
        self.add_input('data:structure:body:mass', val=np.nan, units='kg')
        self.add_input('data:structure:arms:mass', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:propeller:prop_number', val=np.nan, units=None)
        self.add_input('data:motor:power:hover', val=np.nan, units='W')
        self.add_input('data:ESC:efficiency', val=np.nan, units=None)
        self.add_output('data:system:MTOW', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Mgear = inputs['data:gearbox:mass']  # default value = .0 if use_gearbox = false
        Mmot = inputs['data:motor:mass']
        Mesc = inputs['data:ESC:mass']
        Mbat = inputs['data:battery:mass']
        Mpro = inputs['data:propeller:mass']
        Npro = inputs['data:propeller:prop_number']
        M_load = inputs['data:payload:mass']
        Mfra = inputs['data:structure:body:mass']
        Marm = inputs['data:structure:arms:mass']

        # System mass
        Mtotal = (Mesc + Mpro + Mmot + Mgear) * Npro + M_load + Mbat + Mfra + Marm  # total mass

        outputs['data:system:MTOW'] = Mtotal


class SystemConstraints(om.ExplicitComponent):
    """
    System constraints
    """

    def setup(self):
        self.add_input('specifications:system:MTOW', val=np.nan, units='kg')
        self.add_input('data:system:MTOW', val=np.nan, units='kg')
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:payload:settings:k_M', val=np.nan, units=None)
        self.add_output('data:system:constraints:mass:convergence', units=None)
        self.add_output('data:system:constraints:mass:MTOW', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        MTOW = inputs['specifications:system:MTOW']
        Mtotal = inputs['data:system:MTOW']
        M_load = inputs['data:payload:mass']
        k_M = inputs['data:payload:settings:k_M']

        Mtotal_estimated = k_M * M_load  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        mass_con = (Mtotal_estimated - Mtotal) / Mtotal  # mass convergence
        MTOW_con = (MTOW - Mtotal) / Mtotal  # Max. takeoff weight specification, for autonomy maximization

        outputs['data:system:constraints:mass:convergence'] = mass_con
        outputs['data:system:constraints:mass:MTOW'] = MTOW_con
