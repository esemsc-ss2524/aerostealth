# case_template

Frozen OpenFOAM case for the NACA 0012 baseline: reference C-mesh, `system/`,
`constant/`, and adjoint dictionaries. `apply` copies this tree, morphs the
surface patch from `x_surf`, and runs the solver in a scratch directory.
