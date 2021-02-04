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
        self.add_input('specifications:load:mass', val=np.nan, units='kg')
        self.add_input('optimization:settings:k_Mb', val=np.nan)
        self.add_output('data:battery:mass', units='kg')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        M_load = inputs['specifications:load:mass']
        k_Mb = inputs['optimization:settings:k_Mb']

        Mbat = k_Mb * M_load  # Battery mass

        outputs['data:battery:mass'] = Mbat