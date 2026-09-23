# FEM wing structures (`fastuav.structures_fem.fixedwing`)

A finite-element alternative to the analytic wing sizing in
`fastuav.models.structures.wing`. The registered group
`fastuav.structures_fem.fixedwing` reuses the analytic tail and fuselage models
unchanged and swaps **only the wing** for an FE model that sizes it for **minimum
structural mass** under a **bending-stress** constraint, driven by the top-level
MDO.

Seeded from the standalone `fast_uav_fem` tube-spar prototype, re-namespaced to
FAST-UAV conventions and extended with the missing shell wingbox and
foam-sandwich models.

## Two wing models (`wing_model` option)

### `tube_spar_foam`
Tapered tubular spar (Euler-Bernoulli beam FEM) **plus** a foam-cored sandwich
skin that *shares* bending: the effective section stiffness is
`EI = E_spar·I_tube + E_skin·I_faces + E_foam·I_foam`, solved on the shared 1-D
bending backbone (`fe_core/beam_solver.py`). Outer-fibre stresses are recovered
for the spar and the faces, normalised by their allowables, and KS-aggregated.

### `wingbox_shell`
A true **shell + beam** wingbox: upper/lower skins and chordwise ribs are
4-node flat shell elements (`fe_core/shell_element.py`), and the four box
corners are spanwise beam booms (`fe_core/beam_element.py::BeamElement3D`) — the
"spars as beams" idealisation. Assembled, clamped at the root, and loaded by the
spanwise lift (`fe_core/assembly.py`, `wingbox/`). Skin von-Mises and cap axial
stresses feed the KS constraint.

> **Shell formulation note.** The shells use a Q4 Mindlin–Reissner element with
> **selective reduced integration** (2×2 bending, 1-point shear) rather than the
> MITC4 named in the original plan: it is materially simpler to implement
> correctly and validate, locking-free for thin panels, and adequate for the
> bending-stress objective.

## Options (`fastuav.structures_fem.fixedwing`)

| option | default | meaning |
|---|---|---|
| `wing_model` | `tube_spar_foam` | `tube_spar_foam` or `wingbox_shell` |
| `n_elements` | 20 | beam elements (tube_spar_foam) |
| `n_span` | 10 | spanwise stations / **ribs** (wingbox_shell) |
| `n_chord` | 6 | chordwise shell panels (wingbox_shell) |
| `ks_rho` | 100 | KS aggregation sharpness |
| `use_aero_vectors` | False | drive lift shape from VLM `low_speed:{Y,CL,chord}_vector`; else elliptical from geometry |

## Variables

**Shared inputs (already in FAST-UAV):** `data:geometry:wing:{span,root:chord,
tip:chord,root:thickness,tip:thickness}`, `mission:sizing:load_factor:ultimate`,
`optimization:variables:weight:mtow:guess`,
`data:weight:airframe:wing:{spar,skin,ribs}:density`.

**New inputs to add to the source XML** (example values):

| variable | example | units |
|---|---|---|
| `data:structures:wing:spar:material:E` | 70e9 | Pa |
| `data:structures:wing:spar:material:stress:max` | 600e6 | Pa |
| `data:structures:wing:skin:material:E` | 70e9 | Pa |
| `data:structures:wing:skin:material:stress:max` | 300e6 | Pa |
| `data:structures:wing:foam:E`  (tube_spar_foam) | 30e6 | Pa |
| `data:structures:wing:foam:density`  (tube_spar_foam) | 60 | kg/m³ |

**Design variables (top-level MDO):**
- `wingbox_shell`: `data:structures:wing:skin:thickness:{root,tip}`,
  `data:structures:wing:spar:cap_area:{root,tip}`,
  `data:structures:wing:ribs:thickness`.
- `tube_spar_foam`: `data:structures:wing:spar:diameter:outer:{root,tip}`,
  `data:structures:wing:spar:wall:thickness:{root,tip}`,
  `data:structures:wing:skin:face:thickness`.

**Constraint:** `data:constraints:structures:wing:failure_margin <= 0`
(KS stress utilisation − 1). **Objective:** `data:weight:airframe:wing:mass`
(minimised directly, or indirectly via MTOW/endurance).

**Diagnostic outputs:** `data:weight:airframe:wing:{mass,spar:mass,skin:mass,
ribs:mass}`, `data:loads:wing:{sigma_max,w_tip}`.

## Selecting it in a config

```yaml
structures:
    id: fastuav.structures_fem.fixedwing
    wing_model: "wingbox_shell"   # or "tube_spar_foam"
    n_span: 10
    n_chord: 6
```

See `notebooks/data/configurations/fixedwing_mdo_VLM_fem.yaml` for a full MDO
example (structural design variables / failure-margin constraint already wired).
Running the full MDO requires the source XML to include the new material inputs
above.

## Verification status

Validated against analytical/closed-form references (pure-NumPy cores):

- 2-D beam: cantilever `w_tip = qL⁴/8EI` to ~1e-12.
- `BeamElement3D`: `PL³/3EI` and torsion `TL/GJ` exact; transform orthonormal.
- Shell: 6 rigid-body modes; cantilever bending vs `PL³/3EI` to 0.1%; membrane
  tension to 1% with exact centroid stress.
- **Wingbox vs equivalent-EI beam: tip deflection within 1%** (prismatic box).
- Load model: half-span lift integral equals `n_ult·MTOW·g/2` exactly.
- FAST-OAD registry discovery + config build/run confirmed for both models.

## Known limitations / scope

- **Bending stress only** (KS); **no buckling** — thin skins/foam faces may be
  buckling-critical, so sized masses are non-conservative until buckling is
  added.
- `wingbox_shell` places a rib at **every** spanwise station, so the rib count
  equals `n_span + 1`; choose `n_span` to match the intended number of ribs.
- Linear small-deflection static analysis; FD partials (gradients are FD in the
  optimiser).
- Shells are SRI-Q4 (see note above); single-element spurious shear modes are
  benign in constrained meshes (confirmed by the cantilever/wingbox checks).
