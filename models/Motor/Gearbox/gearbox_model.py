"""
Gearbox model
"""
import openmdao.api as om
import numpy as np


class ComputeGearboxCharacteristics(om.ExplicitComponent):
    """
    Simple Gearbox Model
    """

    def setup(self):
        self.add_input('data:gearbox:N_red', val=np.nan, units=None)
        self.add_input('data:motor:torque:hover', val=np.nan, units='N*m')
        self.add_output('data:gearbox:mass', units='kg')
        self.add_output('data:gearbox:gear_diameter', units='m')
        self.add_output('data:gearbox:pinion_diameter', units='m')
        self.add_output('data:gearbox:inner_diameter', units='m')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Nred = inputs['data:gearbox:N_red']
        Tmot_hover = inputs['data:motor:torque:hover']

        mg1=0.0309*Nred**2+0.1944*Nred+0.6389       # Ratio input pinion to mating gear
        WF=1+1/mg1+mg1+mg1**2+Nred**2/mg1+Nred**2   # Weight Factor (ƩFd2/C) [-]
        k_sd=1000                                   # Surface durability factor [lb/in]
        C=2*8.85*Tmot_hover/k_sd                    # Coefficient (C=2T/K) [in3]
        Fd2=WF*C                                    # Solid rotor volume [in3]
        Mgear=Fd2*0.3*0.4535                        # Mass reducer [kg] (0.3 is a coefficient evaluated for aircraft application and 0.4535 to pass from lb to kg)
        Fdp2=C*(Nred+1)/Nred                        # Solid rotor pinion volume [in3]
        dp=(Fdp2/0.7)**(1/3)*0.0254                 # Pinion diameter [m] (0.0254 to pass from in to m)
        dg=Nred*dp                                  # Gear diameter [m]
        di=mg1*dp                                   # Inner diameter [m]

        outputs['data:gearbox:mass'] = Mgear
        outputs['data:gearbox:gear_diameter'] = dg
        outputs['data:gearbox:pinion_diameter'] = dp
        outputs['data:gearbox:inner_diameter'] = di


