"""
Estimation models for the battery.
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import add_subsystem_with_deviation


class BatteryEstimationModels(om.Group):
    """
    Group containing the estimation models for the battery.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """
    def setup(self):
        add_subsystem_with_deviation(self, "max_current", MaxCurrent(),
                                     uncertain_outputs={'data:battery:current:max:estimated': 'A'})

        add_subsystem_with_deviation(self, "weight", Weight(),
                                     uncertain_outputs={'data:battery:mass:estimated': 'kg'})

        self.add_subsystem("geometry", Geometry(), promotes=["*"])

        add_subsystem_with_deviation(self, "max_DoD", MaxDepthOfDischarge(),
                                     uncertain_outputs={'data:battery:DoD:max:estimated': None})

        add_subsystem_with_deviation(self, "esc_efficiency", ESCEfficiency(),
                                     uncertain_outputs={'data:ESC:efficiency:estimated': None})


class MaxCurrent(om.ExplicitComponent):
    """
    Computes battery maximum current
    """

    def setup(self):
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:current:max', val=np.nan, units='A')
        self.add_input('data:battery:capacity:estimated', units='A*s')
        self.add_output('data:battery:current:max:estimated', units='A')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Cbat_ref = inputs['data:battery:reference:capacity']
        Imax_ref = inputs['data:battery:reference:current:max']
        C_bat = inputs['data:battery:capacity:estimated']

        Imax = Imax_ref * C_bat / Cbat_ref  # [A] max current battery

        outputs['data:battery:current:max:estimated'] = Imax


class Weight(om.ExplicitComponent):
    """
    Computes battery weight
    """

    def setup(self):
        self.add_input('data:battery:energy:estimated', val=np.nan, units='kJ')
        self.add_input('data:battery:reference:mass', val=np.nan, units='kg')
        self.add_input('data:battery:reference:energy', val=np.nan, units='kJ')
        self.add_output('data:battery:mass:estimated', units='kg')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        Ebat = inputs['data:battery:energy:estimated']
        Mbat_ref = inputs['data:battery:reference:mass']
        Ebat_ref = inputs['data:battery:reference:energy']

        Mbat = Mbat_ref * Ebat / Ebat_ref  # [kg] estimated battery mass

        outputs['data:battery:mass:estimated'] = Mbat

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mbat_ref = inputs['data:battery:reference:mass']
        Ebat_ref = inputs['data:battery:reference:energy']

        partials['data:battery:mass:estimated',
                 'data:battery:energy:estimated'] = Mbat_ref / Ebat_ref


class Geometry(om.ExplicitComponent):
    """
    Computes battery geometry
    """

    def setup(self):
        self.add_input('data:battery:voltage:estimated', val=np.nan, units='V')
        self.add_input('data:battery:capacity:estimated', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:voltage', val=np.nan, units='V')
        self.add_input('data:battery:reference:capacity', val=np.nan, units='A*s')
        self.add_input('data:battery:reference:volume', val=np.nan, units='cm**3')
        self.add_output('data:battery:volume:estimated', units='cm**3')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_bat = inputs['data:battery:capacity:estimated']
        V_bat = inputs['data:battery:voltage:estimated']
        Cbat_ref = inputs['data:battery:reference:capacity']
        Vbat_ref = inputs['data:battery:reference:voltage']
        Volbat_ref = inputs['data:battery:reference:volume']

        Vol_bat = Volbat_ref * (C_bat * V_bat / (Cbat_ref * Vbat_ref))  # [cm**3] Volume of the battery (estimated)

        outputs['data:battery:volume:estimated'] = Vol_bat


class MaxDepthOfDischarge(om.ExplicitComponent):
    """
    Computes max. depth of discharge of the battery  TODO: find a model
    """

    def setup(self):
        self.add_input('data:battery:reference:DoD:max', val=0.8, units=None)
        self.add_output('data:battery:DoD:max:estimated', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        C_ratio_ref = inputs['data:battery:reference:DoD:max']

        # Model to be defined
        C_ratio = C_ratio_ref

        outputs['data:battery:DoD:max:estimated'] = C_ratio


class ESCEfficiency(om.ExplicitComponent):
    """
    Computes efficiency of the ESC  TODO: find a model
    """

    def setup(self):
        self.add_input('data:ESC:reference:efficiency', val=0.95, units=None)
        self.add_output('data:ESC:efficiency:estimated', units=None)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        eta_ref = inputs['data:ESC:reference:efficiency']

        # Model to be defined
        eta = eta_ref

        outputs['data:ESC:efficiency:estimated'] = eta
