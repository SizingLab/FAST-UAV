"""Vortex Lattice Method implementation."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import copy
import logging
import os
import os.path as pth
import warnings
from typing import Optional

import numpy as np
import openmdao.api as om
import pandas as pd
from stdatm import Atmosphere

from fastuav.command.api import string_to_array
from fastuav.models.geometry.profiles.get_profile import get_profile

from ...constants import MACH_NB_PTS, POLAR_POINT_COUNT, SPAN_MESH_POINT

DEFAULT_NX = 19
DEFAULT_NY1 = 3
DEFAULT_NY2 = 14

_LOGGER = logging.getLogger(__name__)


def _scalarize(value):
    """Return the scalar inside a one-element array/sequence (numpy>=2 safe).

    numpy>=2 (required by OpenMDAO) raises on ``float(np.array([x]))``, so this
    mirrors fast-oad-core's ``scalarize`` and is used instead of bare ``float``.
    """
    return float(np.asarray(value).item())


class VLMSimpleGeometry(om.ExplicitComponent):
    """Computation of the aerodynamics properties using the in-house VLM code."""

    def __init__(self, **kwargs):
        """Initializing parameters used in VLM computation."""
        super().__init__(**kwargs)
        self.wing = None
        self.htp = None
        self.n_x = None  # number of chord wise panel
        self.ny1 = None
        self.ny2 = None
        self.ny3 = None
        self.n_y = None  # number of semi-span wise panel

        # The computation of the aircraft aerodynamics relies on the computation of the wing
        # based on which a downwash angle is computed and used to compute the HTP. This
        # assumption means that the wing is undisturbed by the HTP. This means the wing alone and
        # the wing as part of the aircraft will give the same results. We can thus spare some
        # computation by saving the results of the wing alone and reusing it in the aircraft
        # instead of recomputing it. A proper cache should be used but I don't see any easy way
        # to do it right now because one of the input of the compute_ac function is th inputs in
        # the OpenMDAO format which seems hard to compare. Instead we will save the results and
        # clear them as soon as we are done with the aircraft computation.
        self.saved_wing_0 = None
        self.saved_wing_aoa = None

        # In-memory cache of VLM results keyed by rounded planform geometry + Mach
        # (geometry_set minus area_ratio). Replaces the old on-disk cache, whose folder
        # scan grew every iteration and slowed the optimization down.
        self._aero_cache = {}

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default="naca23012.af", types=str, allow_none=True
        )
        self.options.declare("htp_airfoil_file", default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        # self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("optimization:variables:geometry:wing:lambda", val=np.nan)
        # self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan)
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        # self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:tail:horizontal:sweep_25", val=np.nan, units="deg")
        # self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:tail:horizontal:lambda", val=0.9, units=None)
        # self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("optimization:variables:geometry:tail:horizontal:AR", val=np.nan)
        # self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:span", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:MAC:length", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:root:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:tip:chord", val=np.nan, units="m")
        # self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:tail:horizontal:surface", val=np.nan, units="m**2")
        # self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input(
            "data:geometry:wing:twist",
            val=np.nan,
            units="rad",
            desc="Negative twist means tip AOA is smaller than root",
        )

        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:low_speed:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CDp", val=nans_array)
        else:
            self.add_input("data:aerodynamics:wing:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:cruise:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CDp", val=nans_array)

    def compute_cl_alpha_aircraft(self, inputs, altitude, mach, aoa_angle):
        """
        Function that perform a complete calculation of aerodynamic parameters under VLM and
        returns only the cl_alpha_aircraft parameter.

        :param inputs: input necessary to compute the aircraft aerodynamics.
        :param altitude: altitude at which the aerodynamic properties are computed, in m.
        :param mach: mach number used for the computation
        :param aoa_angle: angle of attack used in the aoa derivative computation, in deg.
        """
        (
            _,
            _,
            cl_alpha_wing,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            cl_alpha_htp,
            _,
            _,
            _,
            _,
        ) = self.compute_aero_coeff(inputs, altitude, mach, aoa_angle)
        return _scalarize(cl_alpha_wing + cl_alpha_htp)

    def compute_cl_alpha_mach(self, inputs, aoa_angle, altitude, cruise_mach):
        """
        Function that performs multiple run of OpenVSP to get an interpolation of Cl_alpha as a
        function of Mach for later use in the computation of the V-n diagram.
        """
        mach_interp = np.log(np.linspace(np.exp(0.15), np.exp(1.55 * cruise_mach), MACH_NB_PTS))
        cl_alpha_interp = np.zeros(np.size(mach_interp))
        for idx, _ in enumerate(mach_interp):
            cl_alpha_interp[idx] = self.compute_cl_alpha_aircraft(
                inputs, altitude, mach_interp[idx], aoa_angle
            )

        # We add the case were M=0, for thoroughness and since we are in an incompressible flow,
        # the Cl_alpha is approximately the same as for the first Mach of the interpolation
        mach_interp = np.insert(mach_interp, 0, 0.0)
        cl_alpha_inc = cl_alpha_interp[0]
        cl_alpha_interp = np.insert(cl_alpha_interp, 0, cl_alpha_inc)

        return mach_interp, cl_alpha_interp

    def compute_aero_coeff(self, inputs, altitude, mach, aoa_angle):
        """
        Function that computes in VLM environment all the aerodynamic parameters @0° and
        aoa_angle and calculate the associated derivatives.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing,
        cl_0_htp, cl_X_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp,
        coef_k_htp parameters.

                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->
        |   |         |     x
        |   |         |
        P2--A--------P1
        """

        # Fix mach number of digits to consider similar results
        mach = round(_scalarize(mach) * 1e3) / 1e3

        # Get inputs necessary to define global geometry
        if self.options["low_speed_aero"]:
            cl_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CL"]
            cdp_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CDp"]
            cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CL"]
            cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CDp"]
        else:
            cl_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CL"]
            cdp_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CDp"]
            cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CL"]
            cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CDp"]
        # width_max = inputs["data:geometry:fuselage:maximum_width"]
        width_max = inputs["data:geometry:fuselage:diameter:mid"]
        span_wing = inputs["data:geometry:wing:span"]
        # sref_wing = _scalarize(inputs["data:geometry:wing:area"])
        sref_wing = _scalarize(inputs["data:geometry:wing:surface"])
        # sref_htp = _scalarize(inputs["data:geometry:horizontal_tail:area"])
        sref_htp = _scalarize(inputs["data:geometry:tail:horizontal:surface"])

        area_ratio = sref_htp / sref_wing
        sweep25_wing = _scalarize(inputs["data:geometry:wing:sweep_25"])
        taper_ratio_wing = _scalarize(inputs["optimization:variables:geometry:wing:lambda"])
        # aspect_ratio_wing = _scalarize(inputs["data:geometry:wing:aspect_ratio"])
        aspect_ratio_wing = _scalarize(inputs["optimization:variables:geometry:wing:AR"])
        # sweep25_htp = _scalarize(inputs["data:geometry:horizontal_tail:sweep_25"])
        sweep25_htp = _scalarize(inputs["data:geometry:tail:horizontal:sweep_25"])
        # aspect_ratio_htp = _scalarize(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        aspect_ratio_htp = _scalarize(inputs["optimization:variables:geometry:tail:horizontal:AR"])
        # taper_ratio_htp = _scalarize(inputs["data:geometry:horizontal_tail:taper_ratio"])
        taper_ratio_htp = _scalarize(
            inputs["data:geometry:tail:horizontal:lambda"]
        )  # assume same profile as wing
        dihedral_angle = _scalarize(inputs["data:geometry:wing:dihedral"])
        twist_angle = _scalarize(inputs["data:geometry:wing:twist"])
        geometry_set = np.around(
            np.array(
                [
                    sweep25_wing,
                    taper_ratio_wing,
                    aspect_ratio_wing,
                    dihedral_angle,
                    twist_angle,
                    sweep25_htp,
                    taper_ratio_htp,
                    aspect_ratio_htp,
                    mach,
                    area_ratio,
                ]
            ),
            decimals=6,
        )

        # Cache lookup. Key omits area_ratio (last element): the raw VLM result is
        # independent of it, so one run is reused for any area_ratio via _rescale_results().
        # Miss -> run the full VLM and store; hit -> no VLM, just rescale the stored result.
        geometry_key = tuple(geometry_set[:-1].tolist())
        cached = self._aero_cache.get(geometry_key)

        # Cache miss: run the VLM
        if cached is None:
            # Compute wing alone @ 0°/X° angle of attack
            wing_0 = self.compute_wing(
                inputs, altitude, mach, 0.0, flaps_angle=0.0, use_airfoil=True
            )
            self.saved_wing_0 = wing_0
            wing_aoa = self.compute_wing(
                inputs, altitude, mach, aoa_angle, flaps_angle=0.0, use_airfoil=True
            )
            self.saved_wing_aoa = wing_aoa

            # Compute complete aircraft @ 0°/X° angle of attack
            _, htp_0, _ = self.compute_aircraft(
                inputs,
                altitude,
                mach,
                0.0,
                flaps_angle=0.0,
                use_airfoil=True,
                saved_wing_result=self.saved_wing_0,
            )
            self.saved_wing_0 = None
            _, htp_aoa, _ = self.compute_aircraft(
                inputs,
                altitude,
                mach,
                aoa_angle,
                flaps_angle=0.0,
                use_airfoil=True,
                saved_wing_result=self.saved_wing_aoa,
            )
            self.saved_wing_aoa = None

            # Compute isolated HTP @ 0°/X° angle of attack
            htp_0_isolated = self.compute_htp(inputs, altitude, mach, 0.0, use_airfoil=True)
            htp_aoa_isolated = self.compute_htp(inputs, altitude, mach, aoa_angle, use_airfoil=True)

            # Post-process wing data ---------------------------------------------------------------
            (
                beta,
                cl_0_wing,
                cl_x_wing,
                cm_0_wing,
                cl_alpha_wing,
                y_vector_wing,
                cl_vector_wing,
                chord_vector_wing,
                coef_k_wing,
            ) = self.post_processing_wing(
                width_max,
                span_wing,
                mach,
                dihedral_angle,
                wing_0,
                wing_aoa,
                aoa_angle,
                aspect_ratio_wing,
                cl_wing_airfoil,
                cdp_wing_airfoil,
            )

            # Post-process HTP-aircraft data -------------------------------------------------------
            (
                cl_0_htp,
                cl_aoa_htp,
                cl_alpha_htp,
                coef_k_htp,
                y_vector_htp,
                cl_vector_htp,
            ) = self.post_processing_htp_ac(
                beta,
                aspect_ratio_htp,
                area_ratio,
                mach,
                aoa_angle,
                cl_htp_airfoil,
                cdp_htp_airfoil,
                htp_0,
                htp_aoa,
            )

            # Post-process HTP-isolated data -------------------------------------------------------
            cl_alpha_htp_isolated = (
                _scalarize(htp_aoa_isolated["cl"] - htp_0_isolated["cl"])
                / beta
                * area_ratio
                / (aoa_angle * np.pi / 180)
            )

            # Resize vectors -----------------------------------------------------------------------

            y_vector_wing, cl_vector_wing, chord_vector_wing = self.resize_vector(
                [y_vector_wing, cl_vector_wing, chord_vector_wing]
            )
            y_vector_htp, cl_vector_htp = self.resize_vector([y_vector_htp, cl_vector_htp])

            # Store raw results for reuse at any area_ratio
            results = [
                _scalarize(cl_0_wing),
                _scalarize(cl_x_wing),
                _scalarize(cl_alpha_wing),
                _scalarize(cm_0_wing),
                np.array(y_vector_wing).tolist(),
                np.array(cl_vector_wing).tolist(),
                np.array(chord_vector_wing).tolist(),
                _scalarize(coef_k_wing),
                _scalarize(cl_0_htp),
                _scalarize(cl_aoa_htp),
                _scalarize(cl_alpha_htp),
                _scalarize(cl_alpha_htp_isolated),
                np.array(y_vector_htp).tolist(),
                np.array(cl_vector_htp).tolist(),
                _scalarize(coef_k_htp),
                _scalarize(sref_wing),
            ]
            self._aero_cache[geometry_key] = (results, area_ratio)

        # Cache hit: reuse, rescaled to current area_ratio / reference area (no VLM run)
        else:
            stored_results, saved_area_ratio = cached
            (
                cl_0_wing,
                cl_x_wing,
                cl_alpha_wing,
                cm_0_wing,
                y_vector_wing,
                cl_vector_wing,
                chord_vector_wing,
                coef_k_wing,
                cl_0_htp,
                cl_aoa_htp,
                cl_alpha_htp,
                cl_alpha_htp_isolated,
                y_vector_htp,
                cl_vector_htp,
                coef_k_htp,
            ) = self._rescale_results(stored_results, sref_wing, area_ratio, saved_area_ratio)

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    def _rescale_results(self, results, sref_wing, area_ratio, saved_area_ratio):
        """
        Adapt a cached VLM result to the current geometry WITHOUT re-running the VLM.

        The cached coefficients are scale-invariant for a fixed planform, so only the
        dimensional quantities change: HTP terms scale linearly with the area ratio,
        wing span/chord vectors with sqrt of the reference-area ratio.

        Note: this assumes the key (planform + Mach) captures everything the VLM depends
        on. Fuselage-width effects (k_fus, root junction) are NOT in the key, so a hit is
        slightly stale if the fuselage diameter moves — inherited from the old cache.

        :param results: raw cached results (16 entries, last = saved reference area)
        :param sref_wing: current wing reference area
        :param area_ratio: current tail/wing area ratio
        :param saved_area_ratio: area ratio at which the cached result was computed
        :return: 15-element aerodynamic results tuple, rescaled to current geometry
        """
        saved_area_wing = _scalarize(results[15])
        area_scale = area_ratio / saved_area_ratio
        span_scale = np.sqrt(sref_wing / saved_area_wing)

        cl_0_wing = _scalarize(results[0])
        cl_x_wing = _scalarize(results[1])
        cl_alpha_wing = _scalarize(results[2])
        cm_0_wing = _scalarize(results[3])
        y_vector_wing = np.array(results[4]) * span_scale
        cl_vector_wing = np.array(results[5])
        chord_vector_wing = np.array(results[6]) * span_scale
        coef_k_wing = _scalarize(results[7])
        cl_0_htp = _scalarize(results[8]) * area_scale
        cl_aoa_htp = _scalarize(results[9]) * area_scale
        cl_alpha_htp = _scalarize(results[10]) * area_scale
        cl_alpha_htp_isolated = _scalarize(results[11]) * area_scale
        y_vector_htp = np.array(results[12])
        cl_vector_htp = np.array(results[13])
        coef_k_htp = _scalarize(results[14]) * area_scale

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    def compute_wing(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        flaps_angle: Optional[float] = 0.0,
        use_airfoil: Optional[bool] = True,
    ):
        """
        VLM computations for the wing alone.

        :param inputs: inputs parameters defined within FAST-OAD-GA
        :param altitude: altitude for aerodynamic calculation in meters
        :param mach: air speed expressed in mach
        :param aoa_angle: air speed angle of attack with respect to aircraft (degree)
        :param flaps_angle: flaps angle in Deg (default=0.0: i.e. no deflection)
        :param use_airfoil: adds the camber line coordinates of the selected airfoil (default=True)
        :return: wing dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e
        """

        # Generate geometries
        self._run(inputs, run_opt="wing")

        # Get inputs
        # aspect_ratio = _scalarize(inputs["data:geometry:wing:aspect_ratio"])
        aspect_ratio = _scalarize(inputs["optimization:variables:geometry:wing:AR"])
        l0_wing = inputs["data:geometry:wing:MAC:length"]

        # y2_wing = _scalarize(inputs["data:geometry:wing:root:y"])
        y2_wing = _scalarize(inputs["data:geometry:fuselage:diameter:mid"]) / 2.0
        semi_span = _scalarize(inputs["data:geometry:wing:span"]) / 2.0
        wing_twist = _scalarize(inputs["data:geometry:wing:twist"])

        # Initialization
        x_c = self.wing["x_c"]
        panel_chord = self.wing["panel_chord"]
        panel_surf = self.wing["panel_surf"]
        if use_airfoil:
            self.generate_curvature(self.wing, self.options["wing_airfoil_file"])
        self.apply_deflection(inputs, flaps_angle)
        self.generate_twist(self.wing, wing_twist, y2_wing, semi_span)
        panel_angle_vect = self.wing["panel_angle_vect"]
        aic = self.wing["aic"]
        aic_inv = np.linalg.inv(aic)
        aic_wake = self.wing["aic_wake"]

        # Compute air speed
        v_inf = max(
            Atmosphere(altitude, altitude_in_feet=False).speed_of_sound * mach, 0.01
        )  # avoid V=0 m/s crashes

        # Calculate all the aerodynamic parameters
        panel_surf_sum = np.sum(panel_surf)
        aoa_angle = np.deg2rad(aoa_angle)
        alpha = np.add(panel_angle_vect, aoa_angle)
        gamma = -np.dot(aic_inv, alpha) * v_inf
        c_p = -2 / v_inf * np.divide(gamma, panel_chord)
        cl_wing = -np.sum(c_p * panel_surf) / panel_surf_sum
        alpha_ind = np.dot(aic_wake, gamma) / v_inf
        cd_ind_panel = c_p * alpha_ind
        cdi_wing = np.sum(cd_ind_panel * panel_surf) / panel_surf_sum
        wing_e = cl_wing**2 / (np.pi * aspect_ratio * cdi_wing) * 0.955
        cm_panel = np.multiply(c_p, (x_c[: self.n_x * self.n_y] - l0_wing / 4))
        cm_wing = np.sum(cm_panel * panel_surf) / panel_surf_sum

        wing_y_vect = list(self.wing["yc"][: self.n_y])
        wing_chord_vect = list(
            (self.wing["chord"][: self.n_y] + self.wing["chord"][1 : self.n_y + 1]) / 2.0
        )

        wing_cl_vect = []
        for j in range(self.n_y):
            cl_span = np.sum(-c_p[j :: self.n_y] * panel_chord[j :: self.n_y] / wing_chord_vect[j])
            wing_cl_vect.append(cl_span)

        # Return values
        wing = {
            "y_vector": wing_y_vect,
            "cl_vector": wing_cl_vect,
            "chord_vector": wing_chord_vect,
            "cd_vector": [],
            "cm_vector": [],
            "cl": cl_wing,
            "cdi": cdi_wing,
            "cm": cm_wing,
            "coef_e": wing_e,
        }

        return wing

    def compute_htp(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        use_airfoil: Optional[bool] = True,
    ):
        """
        VLM computation for the horizontal tail alone.

        :param inputs: inputs parameters defined within FAST-OAD-GA.
        :param altitude: altitude for aerodynamic calculation in meters.
        :param mach: air speed expressed in mach.
        :param aoa_angle: air speed angle of attack with respect to aircraft (degree).
        :param use_airfoil: adds the camber line coordinates of the selected airfoil (default=True).
        :return: htp dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e.
        """

        # Generate geometries
        self._run(inputs, run_opt="htp")

        # Get inputs
        # aspect_ratio = _scalarize(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        aspect_ratio = _scalarize(inputs["optimization:variables:geometry:tail:horizontal:AR"])
        # l0_wing = inputs["data:geometry:horizontal_tail:MAC:length"]
        l0_wing = inputs["data:geometry:tail:horizontal:MAC:length"]
        # Initialization
        x_c = self.htp["x_c"]
        panel_chord = self.htp["panel_chord"]
        panel_surf = self.htp["panel_surf"]
        if use_airfoil:
            self.generate_curvature(self.htp, self.options["htp_airfoil_file"])
        panel_angle_vect = self.htp["panel_angle_vect"]
        aic = self.htp["aic"]
        aic_inv = np.linalg.inv(aic)
        aic_wake = self.htp["aic_wake"]

        # Compute air speed
        v_inf = max(
            Atmosphere(altitude, altitude_in_feet=False).speed_of_sound * mach, 0.01
        )  # avoid V=0 m/s crashes

        # Calculate all the aerodynamic parameters
        panel_surf_sum = np.sum(panel_surf)
        aoa_angle = np.deg2rad(aoa_angle)
        alpha = np.add(panel_angle_vect, aoa_angle)
        gamma = -np.dot(aic_inv, alpha) * v_inf
        c_p = -2 / v_inf * np.divide(gamma, panel_chord)
        cl_htp = -np.sum(c_p * panel_surf) / panel_surf_sum
        alpha_ind = np.dot(aic_wake, gamma) / v_inf
        cd_ind_panel = c_p * alpha_ind
        cdi_htp = np.sum(cd_ind_panel * panel_surf) / panel_surf_sum
        htp_e = cl_htp**2 / (np.pi * aspect_ratio * max(cdi_htp, 1e-12))  # avoid 0.0 division
        cm_panel = np.multiply(c_p, (x_c[: self.n_x * self.n_y] - l0_wing / 4))
        cm_htp = np.sum(cm_panel * panel_surf) / panel_surf_sum

        htp_y_vect = list(self.htp["yc"][: self.n_y])
        htp_chord_vect = list(
            (self.htp["chord"][: self.n_y] + self.htp["chord"][1 : self.n_y + 1]) / 2.0
        )

        htp_cl_vect = []
        for j in range(self.n_y):
            cl_span = np.sum(-c_p[j :: self.n_y] * panel_chord[j :: self.n_y] / htp_chord_vect[j])
            htp_cl_vect.append(cl_span)

        # Return values
        htp = {
            "y_vector": htp_y_vect,
            "cl_vector": htp_cl_vect,
            "cd_vector": [],
            "cm_vector": [],
            "cl": cl_htp,
            "cdi": cdi_htp,
            "cm": cm_htp,
            "coef_e": htp_e,
        }

        return htp

    def compute_aircraft(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        flaps_angle: Optional[float] = 0.0,
        use_airfoil: Optional[bool] = True,
        saved_wing_result: Optional[dict] = None,
    ):
        """
        VLM computation for the complete aircraft.

        :param inputs: inputs parameters defined within FAST-OAD-GA.
        :param altitude: altitude for aerodynamic calculation in meters.
        :param mach: air speed expressed in mach.
        :param aoa_angle: air speed angle of attack with respect to aircraft (degree).
        :param use_airfoil: adds the camber line coordinates of the selected airfoil (default=True).
        :param flaps_angle: flaps angle in Deg (default=0.0: i.e. no deflection).
        :param saved_wing_result: if existing result are available, used them instead of
        recomputing.
        :return: wing/htp and aircraft dictionaries including their respective aerodynamic
        coefficients.
        """

        # Get inputs
        # aspect_ratio_wing = _scalarize(inputs["data:geometry:wing:aspect_ratio"])
        aspect_ratio_wing = _scalarize(inputs["optimization:variables:geometry:wing:AR"])

        # Compute wing
        if saved_wing_result is None:
            wing = self.compute_wing(
                inputs, altitude, mach, aoa_angle, flaps_angle=flaps_angle, use_airfoil=use_airfoil
            )
        else:
            wing = saved_wing_result

        # Calculate downwash angle based on Gudmundsson model (p.467)
        cl_wing = wing["cl"]
        beta = np.sqrt(1 - mach**2)  # Prandtl-Glauert
        downwash_angle = 2.0 * np.array(cl_wing) / beta * 180.0 / (aspect_ratio_wing * np.pi**2)
        aoa_angle_corrected = aoa_angle - downwash_angle

        # Compute htp
        htp = self.compute_htp(inputs, altitude, mach, aoa_angle_corrected, use_airfoil=True)

        # Save results at aircraft level
        aircraft = {"cl": wing["cl"] + htp["cl"], "cd0": None, "cdi": None, "coef_e": None}

        return wing, htp, aircraft

    def _run(self, inputs, run_opt="wing"):
        wing_break = _scalarize(inputs["data:geometry:wing:kink:span_ratio"])

        # Define mesh size
        self.n_x = int(DEFAULT_NX)
        if wing_break > 0.0:
            self.ny1 = int(DEFAULT_NY1 + 5)  # n° of panels in the straight section of the wing
            self.ny2 = int(
                (DEFAULT_NY2 - 5) / 2
            )  # n° of panels in in the flapped portion of the wing
        else:
            self.ny1 = int(DEFAULT_NY1)  # n° of panels in the straight section of the wing
            self.ny2 = int(DEFAULT_NY2 / 2)  # n° of panels in in the flapped portion of the wing
        self.ny3 = self.ny2  # n° of panels in the un-flapped exterior portion of the wing

        self.n_y = int(self.ny1 + self.ny2 + self.ny3)

        # Generate WING
        if run_opt == "wing":
            self.wing = {
                "x_panel": np.zeros((self.n_x + 1, 2 * self.n_y + 1)),
                "y_panel": np.zeros(2 * self.n_y + 1),
                "z": np.zeros(self.n_x + 1),
                "x_le": np.zeros(2 * self.n_y + 1),
                "chord": np.zeros(2 * self.n_y + 1),
                "panel_span": np.zeros(2 * self.n_y),
                "panel_chord": np.zeros(self.n_x * self.n_y),
                "panel_surf": np.zeros(self.n_x * self.n_y),
                "x_c": np.zeros(self.n_x * 2 * self.n_y),
                "yc": np.zeros(self.n_x * 2 * self.n_y),
                "x1": np.zeros(self.n_x * 2 * self.n_y),
                "x2": np.zeros(self.n_x * 2 * self.n_y),
                "y1": np.zeros(self.n_x * 2 * self.n_y),
                "y2": np.zeros(self.n_x * 2 * self.n_y),
                "panel_angle": np.zeros(self.n_x),
                "panel_angle_vect": np.zeros(self.n_x * self.n_y),
                "aic": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
                "aic_wake": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
            }
            self._generate_wing(inputs)

        # Generate HTP
        if run_opt == "htp":
            # Define elements
            self.htp = {
                "x_panel": np.zeros((self.n_x + 1, 2 * self.n_y + 1)),
                "y_panel": np.zeros(2 * self.n_y + 1),
                "z": np.zeros(self.n_x + 1),
                "x_le": np.zeros(2 * self.n_y + 1),
                "chord": np.zeros(2 * self.n_y + 1),
                "panel_span": np.zeros(2 * self.n_y),
                "panel_chord": np.zeros(self.n_x * self.n_y),
                "panel_surf": np.zeros(self.n_x * self.n_y),
                "x_c": np.zeros(self.n_x * 2 * self.n_y),
                "yc": np.zeros(self.n_x * 2 * self.n_y),
                "x1": np.zeros(self.n_x * 2 * self.n_y),
                "x2": np.zeros(self.n_x * 2 * self.n_y),
                "y1": np.zeros(self.n_x * 2 * self.n_y),
                "y2": np.zeros(self.n_x * 2 * self.n_y),
                "panel_angle": np.zeros(self.n_x),
                "panel_angle_vect": np.zeros(self.n_x * self.n_y),
                "aic": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
                "aic_wake": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
            }
            self._generate_htp(inputs)

    def _generate_wing(self, inputs):
        """
        Generates the coordinates for VLM calculations and aic matrix of the wing.
           Pi +......> y     Given a trapezoid defined by vertices Pi and Pf
              | \            and chords 1 and 2 representing a wing segment
              |  \           that complies with the VLM theory, returns the
              |   + Pf       points and panels of the mesh:
        chord1|   |
              |   |             - Points are given as a list of Np elements,
              |   |chord2         being Np the number of points of the mesh.
              +---+
              |                 - Panels are given as a list of list of NP
              x 				      elements, each element composed of 4 points,
                                  where NP is the number of panels of the mesh.
              x - chord wise direction
              y - span wise direction
          x_le array identifies the x-coordinate of Pi and Pf along the span-direction
        """

        # y2_wing = inputs["data:geometry:wing:root:y"]
        y2_wing = inputs["data:geometry:fuselage:diameter:mid"] / 2.0
        semi_span = inputs["data:geometry:wing:span"] / 2.0
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]

        # Initial data (zero matrix/array)
        y_panel = self.wing["y_panel"]
        x_le = self.wing["x_le"]
        y_end_flaps = y2_wing + flap_span_ratio * (semi_span - y2_wing)
        # Definition of x_panel, y_panel, x_le and chord (Right side)

        indices = np.indices(y_panel.shape)[0].astype(float)

        y_panel = y2_wing * indices / self.ny1

        y_panel = np.where(
            indices >= self.ny1,
            y2_wing + (y_end_flaps - y2_wing) * (indices - self.ny1) / self.ny2,
            y_panel,
        )
        y_tapered_section = y_panel - y2_wing
        chord = np.where(
            indices >= self.ny1,
            root_chord + (tip_chord - root_chord) * y_tapered_section / (semi_span - y2_wing),
            np.full_like(indices, root_chord),
        )
        x_le = np.where(
            indices >= self.ny1,
            y_tapered_section * (root_chord - tip_chord) / (4 * (semi_span - y2_wing)),
            x_le,
        )

        y_panel = np.where(
            indices >= self.ny1 + self.ny2,
            y_end_flaps + (semi_span - y_end_flaps) * (indices - (self.ny1 + self.ny2)) / self.ny3,
            y_panel,
        )
        y_tapered_section = y_panel - y2_wing
        chord = np.where(
            indices >= self.ny1 + self.ny2,
            root_chord + (tip_chord - root_chord) * y_tapered_section / (semi_span - y2_wing),
            chord,
        )
        x_le = np.where(
            indices >= self.ny1 + self.ny2,
            y_tapered_section * (root_chord - tip_chord) / (4 * (semi_span - y2_wing)),
            x_le,
        )

        # Mirror the right side to define the left side for y_panel, x_le and chord
        y_panel[self.n_y + 1 :] = -y_panel[1 : self.n_y + 1]
        chord[self.n_y + 1 :] = chord[1 : self.n_y + 1]
        x_le[self.n_y + 1 :] = x_le[1 : self.n_y + 1]
        # Save data
        self.wing["y_panel"] = y_panel
        self.wing["chord"] = chord
        self.wing["x_le"] = x_le
        # Launch common code
        self._generate_common(self.wing)

    def _generate_htp(self, inputs):
        """
        Generates the coordinates for VLM calculations and AIC matrix of the htp.
           Pi +......> y     Given a trapezoid defined by vertices Pi and Pf
              | \            and chords 1 and 2 representing a wing segment
              |  \           that complies with the VLM theory, returns the
              |   + Pf       points and panels of the mesh:
        chord1|   |
              |   |             - Points are given as a list of Np elements,
              |   |chord2         being Np the number of points of the mesh.
              +---+
              |                 - Panels are given as a list of list of NP
              x 				      elements, each element composed of 4 points,
                                  where NP is the number of panels of the mesh.
              x - chord wise direction
              y - span wise direction
          x_le array identifies the x-coordinate of Pi and Pf along the span-direction
        """
        # semi_span = inputs["data:geometry:horizontal_tail:span"] / 2.0
        semi_span = inputs["data:geometry:tail:horizontal:span"] / 2.0
        # root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        root_chord = inputs["data:geometry:tail:horizontal:root:chord"]
        # tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        tip_chord = inputs["data:geometry:tail:horizontal:tip:chord"]

        # Initial data (zero matrix/array)
        y_panel = self.htp["y_panel"]
        # Definition of x_panel, y_panel, x_le and chord (Right side)

        indices = np.indices(y_panel.shape)[0].astype(float)
        y_panel = semi_span * indices / self.n_y
        chord = root_chord + (tip_chord - root_chord) * y_panel / semi_span
        x_le = y_panel * (root_chord - tip_chord) / (4 * semi_span)

        # Mirror the right side to define the left side for  y_panel, x_le and chord
        y_panel[self.n_y + 1 :] = -y_panel[1 : self.n_y + 1]
        chord[self.n_y + 1 :] = chord[1 : self.n_y + 1]
        x_le[self.n_y + 1 :] = x_le[1 : self.n_y + 1]

        # Save data
        self.htp["y_panel"] = y_panel
        self.htp["chord"] = chord
        self.htp["x_le"] = x_le
        # Launch common code
        self._generate_common(self.htp)

    def _generate_common(self, dictionary):
        """Common code shared between wing and htp to calculate geometry/aero parameters.

                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->
        |   |         |     x
        |   |         |
        P2--A--------P1

        """
        # Initial data (zero matrix/array)
        x_le = dictionary["x_le"]
        chord = dictionary["chord"]
        x_panel = dictionary["x_panel"]
        y_panel = dictionary["y_panel"]
        panel_chord = dictionary["panel_chord"]
        panel_surf = dictionary["panel_surf"]
        x_c = dictionary["x_c"]
        y_c = dictionary["yc"]
        x_1 = dictionary["x1"]
        y_1 = dictionary["y1"]
        x_2 = dictionary["x2"]
        y_2 = dictionary["y2"]

        n_x = self.n_x
        n_y = self.n_y
        # calculate panel geometry and coordinate value
        (
            x_panel,
            y_panel,
            x_le,
            chord,
            panel_chord,
            panel_span,
            panel_surf,
            x_1,
            y_1,
            x_2,
            y_2,
            x_c,
            y_c,
        ) = self.panel_point_calculation(
            x_panel,
            y_panel,
            x_le,
            chord,
            panel_chord,
            panel_surf,
            x_1,
            y_1,
            x_2,
            y_2,
            x_c,
            y_c,
            n_x,
            n_y,
        )

        aic, aic_wake = self.aic_computation(x_1, y_1, x_2, y_2, x_c, y_c, n_x, n_y)

        # Save data
        dictionary["x_panel"] = x_panel
        dictionary["panel_span"] = panel_span
        dictionary["panel_chord"] = panel_chord
        dictionary["panel_surf"] = panel_surf
        dictionary["x_c"] = x_c
        dictionary["yc"] = y_c
        dictionary["x1"] = x_1
        dictionary["y1"] = y_1
        dictionary["x2"] = x_2
        dictionary["y2"] = y_2
        dictionary["aic"] = aic
        dictionary["aic_wake"] = aic_wake

    def generate_twist(self, dictionary, twist, y_start, y_end):
        """
        Add the twist on the lifting surface assuming a linear variation between y_start and y_end.

        :param dictionary: dictionary which contains the point coordinates of the lifting surface
        :param twist: wing twist variation between root and tip, <0 when tip is pointed downwards
        :param y_start: y coordinate of the start of linear twist
        :param y_end:y coordinate of the end of linear twist
        """

        y_panel = dictionary["y_panel"][: self.n_y]
        twist = np.interp(y_panel, [y_start, y_end], [0.0, twist])
        twist_tile = np.tile(twist, self.n_x)

        panel_angle_vect = dictionary["panel_angle_vect"]

        dictionary["panel_angle_vect"] = panel_angle_vect + twist_tile

    def generate_curvature(self, dictionary, file_name):
        """Generates curvature corresponding to the airfoil contained in .af file."""
        x_panel = dictionary["x_panel"]

        root_chord = x_panel[self.n_x, 0] - x_panel[0, 0]
        # Calculation of panel_angle_vect
        profile = get_profile(
            airfoil_folder_path=self.options["airfoil_folder_path"], file_name=file_name
        )
        mean_line = profile.get_mean_line()
        x_red = (x_panel[: self.n_x + 1, 0] - x_panel[0, 0]) / root_chord
        x_red_clipped = np.clip(x_red, min(mean_line["x"]), max(mean_line["x"]))
        z_panel = np.interp(x_red_clipped, mean_line["x"], mean_line["z"]) * root_chord

        x_coord_chord = x_panel[:, 0]

        panel_angle = (z_panel[: self.n_x] - z_panel[1 : self.n_x + 1]) / (
            x_coord_chord[1 : self.n_x + 1] - x_coord_chord[: self.n_x]
        )
        panel_angle_vect = np.repeat(panel_angle, self.n_y)

        # Save results
        dictionary["panel_angle_vect"] = panel_angle_vect
        dictionary["panel_angle"] = panel_angle
        dictionary["z"] = z_panel

    def apply_deflection(self, inputs, deflection_angle):
        """Apply panel angle deflection due to flaps angle [UNUSED: deflection_angle=0.0]."""

        root_chord = inputs["data:geometry:wing:root:chord"]
        x_start = (1.0 - inputs["data:geometry:flap:chord_ratio"]) * root_chord

        deflection_angle *= np.pi / 180  # converted to radian
        x_panel = self.wing["x_panel"]
        panel_angle_vect = self.wing["panel_angle_vect"]

        z_panel_no_flaps = copy.deepcopy(self.wing["z"])
        mask = x_panel[:, 0] <= x_start
        z_panel = z_panel_no_flaps - np.sin(deflection_angle) * (x_panel[:, 0] - x_start)
        z_panel[mask] = z_panel_no_flaps[mask]

        # At this point z_panel contains the z coordinate of the flapped part of the wing,
        # panel angle vector should not change in the un-flapped part
        panel_angle = (z_panel[: self.n_x] - z_panel[1 : self.n_x + 1]) / (
            x_panel[1 : self.n_x + 1, 0] - x_panel[: self.n_x, 0]
        )

        # At this point panel_angle contains the panel angle of the flapped part of the wing,
        # panel angle vector should not change in the un-flapped part

        # If we are in the flapped portion of the wing
        for j in range(self.ny1, self.ny1 + self.ny2):
            for i in range(self.n_x):
                panel_angle_vect[i * self.n_y + j] = panel_angle[i]

        # Save results
        self.wing["panel_angle_vect"] = panel_angle_vect
        self.wing["panel_angle"] = panel_angle
        self.wing["z"] = z_panel

    @staticmethod
    def _interpolate_cdp(lift_coeff: np.ndarray, drag_coeff: np.ndarray, objective: float) -> float:
        """

        :param lift_coeff: CL array
        :param drag_coeff: CDp array
        :param objective: CL_ref objective value
        :return: CD_ref if CL_ref encountered, or default value otherwise.
        """
        # Reduce vectors for interpolation
        for idx in range(len(lift_coeff)):
            if np.sum(lift_coeff[idx : len(lift_coeff)] == 0) == (len(lift_coeff) - idx):
                lift_coeff = lift_coeff[0:idx]
                drag_coeff = drag_coeff[0:idx]
                break

        # Interpolate value if within the interpolation range
        if min(lift_coeff) <= objective <= max(lift_coeff):
            idx_max = int(_scalarize(np.where(lift_coeff == max(lift_coeff))[0]))
            return np.interp(objective, lift_coeff[0 : idx_max + 1], drag_coeff[0 : idx_max + 1])
        if objective < lift_coeff[0]:
            cdp = drag_coeff[0] + (objective - lift_coeff[0]) * (drag_coeff[1] - drag_coeff[0]) / (
                lift_coeff[1] - lift_coeff[0]
            )
        else:
            cdp = drag_coeff[-1] + (objective - lift_coeff[-1]) * (
                drag_coeff[-1] - drag_coeff[-2]
            ) / (lift_coeff[-1] - lift_coeff[-2])
        _LOGGER.warning("CL not in range. Linear extrapolation of CDp value %f", cdp)
        return cdp

    @staticmethod
    def search_results(result_folder_path, geometry_set):
        """Search the results folder to see if the geometry has already been calculated."""
        # default value
        result_file_path = None
        saved_area_ratio = 1.0
        if os.path.exists(result_folder_path):
            geometry_set_labels = [
                "sweep25_wing",
                "taper_ratio_wing",
                "aspect_ratio_wing",
                "dihedral_angle_wing",
                "twist_angle_wing",
                "sweep25_htp",
                "taper_ratio_htp",
                "aspect_ratio_htp",
                "mach",
                "area_ratio",
            ]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")):
                        data = pd.read_csv(
                            pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")
                        )
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if (
                                np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy())
                                == len(geometry_set_labels) - 1
                            ):
                                saved_set = np.around(
                                    data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6
                                )
                                if (
                                    np.sum(saved_set == geometry_set[0:-1])
                                    == len(geometry_set_labels) - 1
                                ):
                                    result_file_path = pth.join(
                                        result_folder_path, "vlm_" + str(idx) + ".csv"
                                    )
                                    saved_area_ratio = data.loc["area_ratio", 0]
                                    return result_file_path, saved_area_ratio
                        except Exception:
                            break
                    idx += 1

        return result_file_path, saved_area_ratio

    @staticmethod
    def save_geometry(result_folder_path, geometry_set):
        """Save geometry if not already computed by finding first available index."""
        geometry_set_labels = [
            "sweep25_wing",
            "taper_ratio_wing",
            "aspect_ratio_wing",
            "dihedral_angle_wing",
            "twist_angle_wing",
            "sweep25_htp",
            "taper_ratio_htp",
            "aspect_ratio_htp",
            "mach",
            "area_ratio",
        ]
        data = pd.DataFrame(geometry_set, index=geometry_set_labels)
        idx = 0
        while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
            idx += 1
        data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
        result_file_path = pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")

        return result_file_path

    @staticmethod
    def save_results(result_file_path, results):
        """Reads saved results."""
        labels = [
            "cl_0_wing",
            "cl_X_wing",
            "cl_alpha_wing",
            "cm_0_wing",
            "y_vector_wing",
            "cl_vector_wing",
            "chord_vector_wing",
            "coef_k_wing",
            "cl_0_htp",
            "cl_X_htp",
            "cl_alpha_htp",
            "cl_alpha_htp_isolated",
            "y_vector_htp",
            "cl_vector_htp",
            "coef_k_htp",
            "saved_ref_area",
        ]
        data = pd.DataFrame(results, index=labels)
        data.to_csv(result_file_path)

    @staticmethod
    def read_results(result_file_path):
        data = pd.read_csv(result_file_path)
        values = data.to_numpy()[:, 1].tolist()
        labels = data.to_numpy()[:, 0].tolist()

        return pd.DataFrame(values, index=labels)

    def post_processing_wing(
        self,
        width_max,
        span_wing,
        mach,
        dihedral_angle,
        wing_0,
        wing_aoa,
        aoa_angle,
        aspect_ratio_wing,
        cl_wing_airfoil,
        cdp_wing_airfoil,
    ):
        """
        Calculate aerodynamic characteristic with VLM results
        :param width_max: maximum fuselage width
        :param span_wing: wing span
        :param mach: Mach number
        :param dihedral_angle: wing dihedral angle
        :param wing_0: wing aerodynamic characteristics with zero angle of attack
        :param wing_aoa: wing aerodynamic characteristics with input angle of attack
        :param aoa_angle: input angle of attack
        :param aspect_ratio_wing: wing aspect ratio
        :param cl_wing_airfoil: airfoil lift coefficient curve
        :param cdp_wing_airfoil: airfoil pressure drag coefficient curve

        :return: post-processed data for other use
        """
        k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
        beta = np.sqrt(1 - mach**2)  # Prandtl-Glauert
        cl_0_wing = _scalarize((wing_0["cl"] * k_fus / beta) * np.cos(dihedral_angle) ** 2.0)
        cl_x_wing = _scalarize(wing_aoa["cl"] * k_fus / beta)
        cm_0_wing = _scalarize(wing_0["cm"] * k_fus / beta)
        cl_alpha_wing = ((cl_x_wing - cl_0_wing) / (aoa_angle * np.pi / 180)) * np.cos(
            dihedral_angle
        ) ** 2.0

        # The correction on the effect of dihedral angle cannot be taken into account by
        # modifying the generation of panel since only the surfaces of the panel and their
        # aoa impact the computation, so we will stick with this "post processing" of the
        # dihedral

        y_vector_wing = wing_aoa["y_vector"]
        cl_vector_wing = (np.array(wing_aoa["cl_vector"]) * k_fus / beta).tolist()
        chord_vector_wing = wing_aoa["chord_vector"]
        _cdp_foil = self._interpolate_cdp(cl_wing_airfoil, cdp_wing_airfoil, cl_x_wing)
        # Mach correction
        if mach <= 0.4:
            coef_e = wing_aoa["coef_e"]
        else:
            coef_e = wing_aoa["coef_e"] * (-0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1)
        cdi = cl_x_wing**2 / (np.pi * aspect_ratio_wing * coef_e)  # + cdp_foil
        coef_e = wing_aoa["cl"] ** 2 / (np.pi * aspect_ratio_wing * cdi)
        # Fuselage correction
        k_fus = 1 - 2 * (width_max / span_wing) ** 2
        coef_e = _scalarize(coef_e * k_fus)
        coef_k_wing = _scalarize(1.0 / (np.pi * aspect_ratio_wing * coef_e))

        return (
            beta,
            cl_0_wing,
            cl_x_wing,
            cm_0_wing,
            cl_alpha_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
        )

    def post_processing_htp_ac(
        self,
        beta,
        aspect_ratio_htp,
        area_ratio,
        mach,
        aoa_angle,
        cl_htp_airfoil,
        cdp_htp_airfoil,
        htp_0,
        htp_aoa,
    ):
        """
        Calculate aerodynamic characteristic with VLM results

        :param beta: sweep angle of the htp
        :param aspect_ratio_htp: htp aspect ratio
        :param area_ratio: area ratio between wing and htp
        :param mach: Mach number
        :param aoa_angle: input angle of attack
        :param cl_htp_airfoil: airfoil lift coefficient curve
        :param cdp_htp_airfoil: airfoil pressure drag coefficient curve
        :param htp_0: htp aerodynamic characteristics data list with zero angle of attack
        :param htp_aoa: htp aerodynamic characteristics data list with input angle of attack

        :return: post-processed data for other use
        """
        cl_0_htp = _scalarize(htp_0["cl"]) / beta * area_ratio
        cl_aoa_htp = _scalarize(htp_aoa["cl"]) / beta * area_ratio
        cl_alpha_htp = _scalarize((cl_aoa_htp - cl_0_htp) / (aoa_angle * np.pi / 180))
        cdp_foil = self._interpolate_cdp(cl_htp_airfoil, cdp_htp_airfoil, htp_aoa["cl"] / beta)
        # Mach correction
        if mach <= 0.4:
            coef_e = htp_aoa["coef_e"]
        else:
            coef_e = htp_aoa["coef_e"] * (-0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1)
        cdi = (htp_aoa["cl"] / beta) ** 2 / (np.pi * aspect_ratio_htp * coef_e) + cdp_foil
        coef_k_htp = _scalarize(cdi / cl_aoa_htp**2 * area_ratio)
        y_vector_htp = htp_aoa["y_vector"]
        cl_vector_htp = (np.array(htp_aoa["cl_vector"]) / beta * area_ratio).tolist()

        return (
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            coef_k_htp,
            y_vector_htp,
            cl_vector_htp,
        )

    def read_value_from_data(self, result_file_path, sref_wing, area_ratio, saved_area_ratio):
        """
        Read existed data to speed up the process

        :param result_file_path: path to the file with existing data
        :param sref_wing: wing reference area
        :param area_ratio: area ratio between wing and horizontal stabilizer
        :param saved_area_ratio:  area ratio between wing and horizontal stabilizer in existing data

        :return: existing data
        """
        data = self.read_results(result_file_path)
        saved_area_wing = _scalarize(data.loc["saved_ref_area", 0])
        cl_0_wing = _scalarize(data.loc["cl_0_wing", 0])
        cl_x_wing = _scalarize(data.loc["cl_X_wing", 0])
        cl_alpha_wing = _scalarize(data.loc["cl_alpha_wing", 0])
        cm_0_wing = _scalarize(data.loc["cm_0_wing", 0])
        y_vector_wing = string_to_array(data.loc["y_vector_wing", 0][1:-2]) * np.sqrt(
            sref_wing / saved_area_wing
        )
        cl_vector_wing = string_to_array(data.loc["cl_vector_wing", 0][1:-2])
        chord_vector_wing = string_to_array(data.loc["chord_vector_wing", 0][1:-2]) * np.sqrt(
            sref_wing / saved_area_wing
        )
        coef_k_wing = _scalarize(data.loc["coef_k_wing", 0])
        cl_0_htp = _scalarize(data.loc["cl_0_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_aoa_htp = _scalarize(data.loc["cl_X_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_alpha_htp = _scalarize(data.loc["cl_alpha_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_alpha_htp_isolated = _scalarize(data.loc["cl_alpha_htp_isolated", 0]) * (
            area_ratio / saved_area_ratio
        )
        y_vector_htp = string_to_array(data.loc["y_vector_htp", 0][1:-2])
        cl_vector_htp = string_to_array(data.loc["cl_vector_htp", 0][1:-2])
        coef_k_htp = _scalarize(data.loc["coef_k_htp", 0]) * (area_ratio / saved_area_ratio)

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    @staticmethod
    def resize_vector(vectors):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise we complete with zeros.
        First vector always has to be the y vector, any other vector can be a quantity expressed
        as that vector.

        :param vectors: a tuple with wing aerodynamic results

        :return: length-modified aerodynamic results
        """
        y_vector = vectors[0]

        resized_vectors = []

        # shorter
        if SPAN_MESH_POINT < len(y_vector):
            y_interp = np.linspace(y_vector[0], y_vector[-1], SPAN_MESH_POINT)
            warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")

            resized_vectors.append(y_interp)

            for vector in vectors[1:]:
                resized_vector = np.interp(y_interp, y_vector, vector)
                resized_vectors.append(resized_vector)

        # longer
        else:
            additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector)))
            y_vector = y_vector + additional_zeros

            resized_vectors.append(y_vector)

            for vector in vectors[1:]:
                resized_vector = vector + additional_zeros
                resized_vectors.append(resized_vector)

        return resized_vectors

    @staticmethod
    def aic_computation(x_1, y_1, x_2, y_2, x_c, y_c, n_x, n_y):
        """
                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->
        |   |         |     x
        |   |         |
        P2--A--------P1

        The P point is the center point of all the panel in AIC matrix construction. For each
        linear equations in the AIC linear system, each P point of each panel is  considered
        (double for loop).
        """
        midpoint = len(x_c) // 2

        x_c = x_c[:midpoint]
        x_c = np.repeat(x_c[:, np.newaxis], midpoint, axis=1).ravel()
        y_c = y_c[:midpoint]
        y_c = np.repeat(y_c[:, np.newaxis], midpoint, axis=1).ravel()
        x_1_r = x_1[:midpoint]
        x_1_r = np.tile(x_1_r, len(x_1_r))
        y_1_r = y_1[:midpoint]
        y_1_r = np.tile(y_1_r, len(y_1_r))
        x_2_r = x_2[:midpoint]
        x_2_r = np.tile(x_2_r, len(x_2_r))
        y_2_r = y_2[:midpoint]
        y_2_r = np.tile(y_2_r, len(y_2_r))
        x_1_l = x_1[midpoint:]
        x_1_l = np.tile(x_1_l, len(x_1_l))
        y_1_l = y_1[midpoint:]
        y_1_l = np.tile(y_1_l, len(y_1_l))
        x_2_l = x_2[midpoint:]
        x_2_l = np.tile(x_2_l, len(x_2_l))
        y_2_l = y_2[midpoint:]
        y_2_l = np.tile(y_2_l, len(y_2_l))

        coeff_1_r = x_c - x_1_r
        coeff_2_r = y_c - y_1_r
        coeff_3_r = x_c - x_2_r
        coeff_4_r = y_c - y_2_r
        coeff_5_r = np.sqrt(coeff_1_r**2 + coeff_2_r**2)
        coeff_6_r = np.sqrt(coeff_3_r**2 + coeff_4_r**2)
        coeff_7_r = x_2_r - x_1_r
        coeff_8_r = y_2_r - y_1_r
        coeff_9_r = (coeff_7_r * coeff_1_r + coeff_8_r * coeff_2_r) / coeff_5_r - (
            coeff_7_r * coeff_3_r + coeff_8_r * coeff_4_r
        ) / coeff_6_r
        coeff_10_r = (1 + coeff_3_r / coeff_6_r) / coeff_4_r - (
            1 + coeff_1_r / coeff_5_r
        ) / coeff_2_r
        coeff_1_l = x_c - x_1_l
        coeff_2_l = y_c - y_1_l
        coeff_3_l = x_c - x_2_l
        coeff_4_l = y_c - y_2_l
        coeff_5_l = np.sqrt(coeff_1_l**2 + coeff_2_l**2)
        coeff_6_l = np.sqrt(coeff_3_l**2 + coeff_4_l**2)
        coeff_7_l = x_2_l - x_1_l
        coeff_8_l = y_2_l - y_1_l
        coeff_9_l = (coeff_7_l * coeff_1_l + coeff_8_l * coeff_2_l) / coeff_5_l - (
            coeff_7_l * coeff_3_l + coeff_8_l * coeff_4_l
        ) / coeff_6_l
        coeff_10_l = (1 + coeff_3_l / coeff_6_l) / coeff_4_l - (
            1 + coeff_1_l / coeff_5_l
        ) / coeff_2_l
        aic_wake = coeff_10_r / (4 * np.pi) + coeff_10_l / (4 * np.pi)
        aic = coeff_10_l / (4 * np.pi) + coeff_10_r / (4 * np.pi)

        # Calculate cross products
        den_r = coeff_1_r * coeff_4_r - coeff_2_r * coeff_3_r
        den_l = coeff_1_l * coeff_4_l - coeff_2_l * coeff_3_l

        # Create masks
        mask_r = den_r != 0
        mask_l = den_l != 0
        # Update aic
        aic[mask_r] = aic[mask_r] + coeff_9_r[mask_r] / den_r[mask_r] / (4 * np.pi)
        aic[mask_l] = aic[mask_l] + coeff_9_l[mask_l] / den_l[mask_l] / (4 * np.pi)
        # reshape into 2D array for later matrix computation
        aic = aic.reshape(int(n_x * n_y), int(n_x * n_y), order="C")
        aic_wake = aic_wake.reshape(int(n_x * n_y), int(n_x * n_y), order="C")

        return aic, aic_wake

    @staticmethod
    def panel_point_calculation(
        x_panel,
        y_panel,
        x_le,
        chord,
        panel_chord,
        panel_surf,
        x_1,
        y_1,
        x_2,
        y_2,
        x_c,
        y_c,
        n_x,
        n_y,
    ):
        """
        Calculate the coordinate value of each geometry points in each panel.
        Store this value into position arrays for later AIC matrix computation.
        """
        for i in range(n_x + 1):
            x_panel[i, :] = x_le + chord * i / n_x

        # Calculate panel span with symmetry
        panel_span = y_panel[1:] - y_panel[:-1]
        panel_span[n_y:] = panel_span[:n_y]

        # Calculate characteristic points (Right and left side)
        for i in range(n_x):
            panel_chord[i * n_y : (i + 1) * n_y] = 0.5 * (
                (x_panel[i + 1, :n_y] - x_panel[i, :n_y])
                + (x_panel[i + 1, 1 : n_y + 1] - x_panel[i, 1 : n_y + 1])
            )
            panel_surf[i * n_y : (i + 1) * n_y] = (
                panel_span[:n_y] * panel_chord[i * n_y : (i + 1) * n_y]
            )
            # right side

            # point A for the right-half
            x_1[i * n_y : (i + 1) * n_y] = x_panel[i, :n_y] + 0.25 * (
                x_panel[i + 1, :n_y] - x_panel[i, :n_y]
            )

            y_1[i * n_y : (i + 1) * n_y] = y_panel[:n_y]

            # point B for the right-half
            x_2[i * n_y : (i + 1) * n_y] = x_panel[i, 1 : n_y + 1] + 0.25 * (
                x_panel[i + 1, 1 : n_y + 1] - x_panel[i, 1 : n_y + 1]
            )

            y_2[i * n_y : (i + 1) * n_y] = y_panel[1 : n_y + 1]

            # point P for the right-half
            x_c[i * n_y : (i + 1) * n_y] = (
                x_panel[i, :n_y] + x_panel[i, 1 : n_y + 1]
            ) * 0.5 + 0.75 * panel_chord[i * n_y : (i + 1) * n_y]

            y_c[i * n_y : (i + 1) * n_y] = (y_panel[:n_y] + y_panel[1 : n_y + 1]) * 0.5

            # left side

            # point A for the left-half
            x_1[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = x_panel[
                i, n_y + 1 : 2 * n_y + 1
            ] + 0.25 * (x_panel[i + 1, n_y + 1 : 2 * n_y + 1] - x_panel[i, n_y + 1 : 2 * n_y + 1])

            y_1[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = y_panel[n_y + 1 : 2 * n_y + 1]

            # point B for the left-half
            x_2[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = np.concatenate(
                (
                    np.array([x_panel[i, 0] + 0.25 * (x_panel[i + 1, 0] - x_panel[i, 0])]),
                    (
                        x_panel[i, n_y : 2 * n_y]
                        + 0.25 * (x_panel[i + 1, n_y : 2 * n_y] - x_panel[i, n_y : 2 * n_y])
                    )[1:],
                )
            )
            y_2[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = np.concatenate(
                (
                    np.array([0.0]),
                    y_panel[n_y + 1 : 2 * n_y],
                )
            )
        # Calculate remaining characteristic points (Left side)
        # point P for the left-half
        x_c[n_x * n_y :] = x_c[: n_x * n_y]
        y_c[n_x * n_y :] = -y_c[: n_x * n_y]

        return (
            x_panel,
            y_panel,
            x_le,
            chord,
            panel_chord,
            panel_span,
            panel_surf,
            x_1,
            y_1,
            x_2,
            y_2,
            x_c,
            y_c,
        )
