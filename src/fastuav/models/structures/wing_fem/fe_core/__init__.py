"""
Pure-NumPy finite-element cores (no OpenMDAO dependency).

* :mod:`.beam_element`  -- 2-node beam elements (2D Euler-Bernoulli, 3D frame)
                           and tube cross-section properties.
* :mod:`.shell_element` -- 4-node MITC4 flat shell element.
* :mod:`.assembly`      -- generic DOF mapping, assembly, boundary conditions
                           and linear solve, shared by the beam and shell models.
"""
