"""
Fixed Wing Structures
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.aerodynamics.constants import SPAN_MESH_POINT
from fastuav.models.structures.wing.wing import WingStructuresFW
from fastuav.models.structures.wing.fem import WingStructure as WingStructuresFEM
from fastuav.models.structures.tails import HorizontalTailStructures, VerticalTailStructures
from fastuav.models.structures.fuselage import FuselageStructures


@oad.RegisterOpenMDAOSystem("fastuav.structures.fixedwing")
class Structures(om.Group):
    """
    Group containing the airframe structural analysis and weights calculation
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])

    def setup(self):
        self.add_subsystem(
            "wing", WingStructuresFW(spar_model=self.options["spar_model"]), promotes=["*"]
        )
        self.add_subsystem("horizontal_tail", HorizontalTailStructures(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailStructures(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageStructures(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("fastuav.structures_fem.fixedwing")
class StructuresFEM(om.Group):
    """
    Variant of the airframe structural analysis where the wing spar is sized
    with a finite-element beam model (vendored from the fast-uav-fem archive)
    instead of the analytical/estimation models.

    The wing subsystem solves a clamped-root cantilever spar FEM (BEAM3 / 6-DOF
    spatial frame element by default) under the VLM-derived spanwise load
    distribution, recovers the bending stress with a KS-aggregated failure
    constraint, and integrates the spar mass. The spar cross-section is
    selectable via ``spar_model`` ("pipe" tube or "I_beam"). The tail and
    fuselage structures are unchanged with respect to :class:`Structures`.
    """

    def initialize(self):
        self.options.declare(
            "spar_model",
            default="pipe",
            values=["pipe", "I_beam"],
            desc="Spar cross-section configuration: circular tube ('pipe') or I-section.",
        )
        self.options.declare(
            "n_elements",
            default=20,
            types=int,
            desc="Number of beam elements along the wing half-span.",
        )
        self.options.declare(
            "n_vlm",
            default=SPAN_MESH_POINT,
            types=int,
            desc="Number of VLM spanwise stations feeding the load distribution "
            "(must match the aerodynamics Y_vector/CL_vector/chord_vector size).",
        )
        self.options.declare(
            "n_point_mass",
            default=0,
            types=int,
            desc="Number of wing-mounted point masses providing inertial relief.",
        )
        self.options.declare(
            "ks_rho",
            default=100.0,
            types=float,
            desc="Kreisselmeier-Steinhauser aggregation parameter for the stress constraint.",
        )

    def setup(self):
        self.add_subsystem(
            "wing",
            WingStructuresFEM(
                spar_model=self.options["spar_model"],
                n_elements=self.options["n_elements"],
                n_vlm=self.options["n_vlm"],
                n_point_mass=self.options["n_point_mass"],
                ks_rho=self.options["ks_rho"],
            ),
            promotes=["*"],
        )
        self.add_subsystem("horizontal_tail", HorizontalTailStructures(), promotes=["*"])
        self.add_subsystem("vertical_tail", VerticalTailStructures(), promotes=["*"])
        self.add_subsystem("fuselage", FuselageStructures(), promotes=["*"])
