"""
FAST - Copyright (c) 2016 ONERA ISAE.
"""

"""
DOC Controllability Computation Module - Concordia University, Robin Warren
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import openmdao.api as om
import numpy as np
import matlab.engine
import math
import time
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from sympy import symbols, integrate, exp, Matrix
from scipy.linalg import expm
from itertools import combinations
#from fastuav.models.Control import doc_multicopter

@RegisterOpenMDAOSystem("fastuav.plugin.DOC")
class SampleDiscipline(om.ExplicitComponent):
    """
    Sample discipline to give an example of how to register a custom module.
    """

    def setup(self):
        self.add_input("Coaxiality", val=0, units="kg")
        self.add_input("RotorNumber", val=4, units="kg")
        self.add_input("Force", val=6, units="N")
        self.add_input("Distance", val=0.24, units="kg")
        self.add_input("TotalMass", val=2, units="kg")
        self.add_input("MotorMass", val=0.03, units="kg")
        self.add_input("PropellerMass", val=0.015, units="kg")
        self.add_input("Time", val=1, units="s")
        self.add_input("TimeSteps", val=2, units="s")

        self.add_output("sample_output", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        "Initialization"
        lapse1 = time.time()
        coaxial=inputs["Coaxiality"]
        rotors=inputs["RotorNumber"]
        fmax=inputs["Force"]
        d=inputs["Distance"]
        M=inputs["TotalMass"]
        M_motor=inputs["MotorMass"]
        M_prop=inputs["PropellerMass"]
        T=inputs["Time"]
        N=int(inputs["TimeSteps"])
        j = 0.1
        Grav = np.array([-M * 9.81, 0, 0, 0])
        M_center = M - rotors * (M_motor + M_prop)
        "Time Variables"
        # T=1
        # N=2
        dT = T / N

        "Determination of Multirotor Configuration"
        if coaxial == 1:
            if rotors == 8:
                config = 2
            elif rotors == 12:
                config = 4
        elif coaxial == 0:
            if rotors == 4:
                config = 1
            elif rotors == 6:
                config = 3
            elif rotors == 8:
                config = 5

        if config == 1:
            Bf = np.array([[1, 1, 1, 1],
                           [np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, np.sqrt(2) * d / 2],
                           [np.sqrt(2) * d / 2, np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2],
                           [j, -j, j, -j]])
            maxinput = np.array([1, 1, 1, 1])
            mininput = np.array([0, 0, 0, 0])
            Ixx = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 4 * (
                        (M_prop + M_motor) * (np.sqrt(2) * d / 2) ** 2)
            Iyy = Ixx
            Izz = M_center * 0.5 ** 2 / 2 + 4 * ((M_prop + M_motor) * (d ** 2))
            print("Simple Quadcopter")
        elif config == 2:
            Bf = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                           [np.sqrt(2) * d / 2, np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2,
                            -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, np.sqrt(2) * d / 2, np.sqrt(2) * d / 2],
                           [np.sqrt(2) * d / 2, np.sqrt(2) * d / 2, np.sqrt(2) * d / 2, np.sqrt(2) * d / 2,
                            -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2, -np.sqrt(2) * d / 2],
                           [j, -j, j, -j, -j, j, j, -j]])
            maxinput = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            mininput = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            Ixx = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 8 * (
                        (M_prop + M_motor) * (np.sqrt(2) * d / 2) ** 2)
            Iyy = Ixx
            Izz = M_center * 0.5 ** 2 / 2 + 8 * ((M_prop + M_motor) * (d ** 2))
            print("Coaxial Quadcopter")
        elif config == 3:
            Bf = np.array([[1, 1, 1, 1, 1, 1],
                           [0, -np.sqrt(3) * d / 2, -np.sqrt(3) * d / 2, 0, np.sqrt(3) * d / 2, np.sqrt(3) * d / 2],
                           [d, d / 2, -d / 2, -d, -d / 2, d / 2],
                           [j, -j, j, -j, j, -j]])
            maxinput = np.array([1, 1, 1, 1, 1, 1])
            mininput = np.array([0, 0, 0, 0, 0, 0])
            Ixx = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 4 * (
                        (M_prop + M_motor) * (np.sqrt(3) * d / 2) ** 2)
            Iyy = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 2 * ((M_prop + M_motor) * (d ** 2)) + 4 * (
                        (M_prop + M_motor) * ((d / 2) ** 2))
            Izz = M_center * 0.5 ** 2 / 2 + 8 * ((M_prop + M_motor) * (d ** 2))
            print("Simple Hexacopter")
        elif config == 4:
            Bf = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 0, -np.sqrt(3) * d / 2, -np.sqrt(3) * d / 2, -np.sqrt(3) * d / 2, -np.sqrt(3) * d / 2, 0,
                            0, np.sqrt(3) * d / 2, np.sqrt(3) * d / 2, np.sqrt(3) * d / 2, np.sqrt(3) * d / 2],
                           [d, d, d / 2, d / 2, -d / 2, -d / 2, -d, -d, -d / 2, -d / 2, d / 2, d / 2],
                           [j, -j, -j, j, j, -j, -j, j, j, -j, -j, j]])
            maxinput = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            mininput = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            Ixx = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 8 * (
                        (M_prop + M_motor) * (np.sqrt(3) * d / 2) ** 2)
            Iyy = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 4 * ((M_prop + M_motor) * (d ** 2)) + 8 * (
                        (M_prop + M_motor) * ((d / 2) ** 2))
            Izz = M_center * 0.5 ** 2 / 2 + 12 * ((M_prop + M_motor) * (d ** 2))
            print("Coaxial Hexacopter")
        elif config == 5:
            Bf = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                           [0, -np.sqrt(2) * d / 2, -d, -np.sqrt(2) * d / 2, 0, np.sqrt(2) * d / 2, d,
                            np.sqrt(2) * d / 2],
                           [d, np.sqrt(2) * d / 2, 0, -np.sqrt(2) * d / 2, -d, -np.sqrt(2) * d / 2, 0,
                            np.sqrt(2) * d / 2],
                           [-j, j, -j, j, -j, j, -j, j]])
            maxinput = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            mininput = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            Ixx = (M_center * 0.5 ** 2 / 4) + (M_center * 0.25 ** 2 / 12) + 2 * ((M_prop + M_motor) * (d ** 2)) + 4 * (
                        (M_prop + M_motor) * ((np.sqrt(2) * d / 2) ** 2))
            Iyy = Ixx
            Izz = M_center * 0.5 ** 2 / 2 + 8 * ((M_prop + M_motor) * (d ** 2))
            print("Simple Octocopter")
        "Matrix Initialization"
        A = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
        B_i = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1 / M, 0, 0, 0],
                        [0, 1 / Ixx, 0, 0],
                        [0, 0, 1 / Iyy, 0],
                        [0, 0, 0, 1 / Izz]])

        B = B_i @ Bf
        Bf = Bf.astype(float)

        "Discretization"
        G = expm(A * dT)
        # print(G)
        tau = symbols('tau')
        A_int = Matrix(A)
        integration = integrate(exp(-A_int * tau), (tau, 0, dT))
        # print(integration)
        integration_double = integration.evalf().tolist()
        integration_array = np.array(integration_double)
        H = expm(A * dT) @ (integration_array) @ B
        F = np.matmul(np.linalg.matrix_power(G, N - 1), H)
        for i in range(1, N):
            F = np.concatenate((F, np.matmul(np.linalg.matrix_power(G, (N - 1) - i), H)), axis=1)
        # print(F)
        K = -np.linalg.inv(np.linalg.matrix_power(G, N)) @ F
        "Failure Injection"
        Nbofrotors = mininput.size
        for j in range(1, Nbofrotors + 1):
            print("Failure of rotor ", j, " out of ", Nbofrotors)
            if j != 1:
                maxinput[j - 2] = 1
            maxinput[j - 1] = 0
            a = mininput + np.linalg.pinv(Bf) @ Grav
            b = fmax * maxinput + np.linalg.pinv(Bf) @ Grav
            average = (a + b) / 2
            delta = (b - a) / 2
            average_0 = average
            delta_0 = delta
            for i in range(1, N):
                average = np.concatenate((average, average_0))
                delta = np.concatenate((delta, delta_0))

            "DOC Computation"
            xp = K @ average
            sizeu = average_0.size
            F_uav = np.arange(1, N * sizeu + 1)

            # Combinations of (n_A - 1) effector actions out-of N * m_u
            S1 = list(combinations(F_uav, A.shape[0] - 1))
            # print(S1)
            # Compute DOC for each hyperplane segment
            n_S1 = len(S1)
            dL = np.zeros(n_S1)

            for i in range(1, n_S1 + 1):
                choose = np.subtract(S1[i - 1], 1)
                K1 = K[:, choose]
                K1 = K1.astype(float)
                K2 = np.delete(K, choose, axis=1)
                u2 = delta
                u2 = np.delete(u2, choose)
                U, s, Vt = np.linalg.svd(K1.T)
                xi = Vt[-1]
                # xi = xi[:, 0]
                xi = xi / np.linalg.norm(xi)
                d = abs(xi.T @ K2) @ u2
                L = abs(xi.T @ xp)
                dL[i - 1] = d - L
                # print(dL)
        print("Minimal Controllability is ", min(dL))
        DOC = min(dL)
        lapse2 = time.time()
        print("Computation Time is ", (lapse2 - lapse1))
        outputs["sample_output"]=DOC