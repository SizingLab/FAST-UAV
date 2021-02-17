"""
Battery weight
"""
import openmdao.api as om
import numpy as np

class ComputeBatteryWeight(om.ExplicitComponent):
    """
    Weight calculation of Battery
    """

    def setup(self):
        self.add_input('data:payload:mass', val=np.nan, units='kg')
        self.add_input('data:battery:settings:k_Mb', val=np.nan)
        self.add_output('data:battery:mass:estimated', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        M_load = inputs['data:payload:mass']
        k_Mb = inputs['data:battery:settings:k_Mb']

        Mbat = k_Mb * M_load  # Battery mass (estimated)

        outputs['data:battery:mass:estimated'] = Mbat