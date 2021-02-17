"""
Battery geometry
"""
import openmdao.api as om
import numpy as np

class ComputeBatteryGeometry(om.ExplicitComponent):
    """
    Geometry calculation of Battery
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