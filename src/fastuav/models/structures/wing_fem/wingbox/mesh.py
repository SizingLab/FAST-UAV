"""
Wingbox mesh generator.

Builds a half-span structural box from planform geometry. The box spans the
chord from the front spar to the rear spar, with the upper and lower skins as
shell panels, vertical chordwise ribs as shell panels at every spanwise
station, and the four corner spar caps as spanwise beam "booms".

Topology
--------
At each spanwise station ``j`` (0..n_span) and chordwise index ``i`` (0..n_chord)
there are two nodes: upper (``z = +h/2``) and lower (``z = -h/2``). Node id::

    block = 2 * (n_chord + 1)
    id(j, i, surf) = j * block + surf * (n_chord + 1) + i      # surf: 0 upper, 1 lower

Elements
--------
* ``skin_quads`` -- upper and lower skin shells (n_span x n_chord each).
* ``rib_quads``  -- vertical chordwise rib shells closing the section at each
  station (n_chord per station). Ribs at every station keep the box closed and
  well-conditioned; the rib count therefore equals ``n_span + 1`` (choose
  ``n_span`` to match the intended number of ribs).
* ``web_quads`` -- vertical spanwise spar-web shells along the front (i=0) and
  rear (i=n_chord) spar lines, one per bay (n_span each). Together with the caps
  they complete the spar (caps = flanges, web = shear panel) and close the
  torsion cell with the skins.
* ``cap_beams`` -- spanwise beams along the 4 box corners (front/rear x
  upper/lower). These realise the spar-flange "booms" idealisation.

Each element record carries the mean spanwise ``eta = y / semi_span`` so the
caller can taper thickness / cap area along the span.
"""

from __future__ import annotations

import numpy as np


def build_wingbox_mesh(semi_span, c_root, c_tip, h_root, h_tip,
                       fs_ratio, rs_ratio, n_span, n_chord):
    """
    Returns a dict describing the half-span wingbox mesh:

    ``nodes`` (N,3); ``skin_quads``/``rib_quads``/``web_quads`` lists of
    (4 node-ids, eta); ``cap_beams`` list of (2 node-ids, eta);
    ``root_nodes`` (clamped); ``station_nodes`` (list per station, for load
    application); ``y_stations``.
    """
    n_nodes = (n_span + 1) * 2 * (n_chord + 1)
    block = 2 * (n_chord + 1)

    def nid(j, i, surf):
        return j * block + surf * (n_chord + 1) + i

    nodes = np.zeros((n_nodes, 3))
    y_stations = np.linspace(0.0, semi_span, n_span + 1)
    for j, y in enumerate(y_stations):
        eta = y / semi_span if semi_span > 0 else 0.0
        c = c_root + (c_tip - c_root) * eta
        h = h_root + (h_tip - h_root) * eta
        x_front, x_rear = fs_ratio * c, rs_ratio * c
        x_chord = np.linspace(x_front, x_rear, n_chord + 1)
        for i, x in enumerate(x_chord):
            nodes[nid(j, i, 0)] = (x, y, +0.5 * h)
            nodes[nid(j, i, 1)] = (x, y, -0.5 * h)

    def eta_of(ids):
        return float(np.mean(nodes[list(ids), 1])) / semi_span if semi_span > 0 else 0.0

    skin_quads, rib_quads, web_quads, cap_beams = [], [], [], []

    # Skins (upper surf=0, lower surf=1).
    for surf in (0, 1):
        for j in range(n_span):
            for i in range(n_chord):
                q = (nid(j, i, surf), nid(j + 1, i, surf),
                     nid(j + 1, i + 1, surf), nid(j, i + 1, surf))
                skin_quads.append((q, eta_of(q)))

    # Ribs (vertical, chordwise) at every station.
    for j in range(n_span + 1):
        for i in range(n_chord):
            q = (nid(j, i, 0), nid(j, i + 1, 0),
                 nid(j, i + 1, 1), nid(j, i, 1))
            rib_quads.append((q, eta_of(q)))

    # Spar webs: vertical shear panels along the front (i=0) and rear
    # (i=n_chord) spar lines, one quad per bay connecting upper and lower
    # surfaces between adjacent stations.
    for i_corner in (0, n_chord):
        for j in range(n_span):
            q = (nid(j, i_corner, 0), nid(j + 1, i_corner, 0),
                 nid(j + 1, i_corner, 1), nid(j, i_corner, 1))
            web_quads.append((q, eta_of(q)))

    # Spar-cap beams at the 4 corners (front i=0, rear i=n_chord; upper/lower).
    for i_corner in (0, n_chord):
        for surf in (0, 1):
            for j in range(n_span):
                e = (nid(j, i_corner, surf), nid(j + 1, i_corner, surf))
                cap_beams.append((e, eta_of(e)))

    root_nodes = [nid(0, i, s) for s in (0, 1) for i in range(n_chord + 1)]
    station_nodes = [[nid(j, i, s) for s in (0, 1) for i in range(n_chord + 1)]
                     for j in range(n_span + 1)]

    return {
        "nodes": nodes,
        "skin_quads": skin_quads,
        "rib_quads": rib_quads,
        "web_quads": web_quads,
        "cap_beams": cap_beams,
        "root_nodes": root_nodes,
        "station_nodes": station_nodes,
        "y_stations": y_stations,
    }
