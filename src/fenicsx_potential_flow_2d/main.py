import numpy as np
import ufl
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
import gmsh
import meshio
import pyvista as pv
from pathlib import Path

def solve_potential_flow(cylinders, Lx, Ly, mesh_size, U_inf=1.0):
    # ==========================================
    # 1. Geometry and Mesh Generation (Gmsh)
    # ==========================================
    gmsh.initialize()
    gmsh.model.add("fluid_domain")

    rect = gmsh.model.occ.addRectangle(-Lx, -Ly, 0, 2 * Lx, 2 * Ly)

    disks = []
    for (cx, cy, r) in cylinders:
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        disks.append(disk)

    if disks:
        domain_tags, _ = gmsh.model.occ.cut([(2, rect)], [(2, d) for d in disks])
        domain_tag = domain_tags[0][1]
    else:
        domain_tag = rect

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setRecombine(2, domain_tag)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.model.mesh.generate(2)

    gmsh.write("temp_mesh.msh")
    gmsh.finalize()

    msh = meshio.read("temp_mesh.msh")

    quad_cells = None
    for cell in msh.cells:
        if cell.type == "quad":
            quad_cells = cell.data
            break

    if quad_cells is None:
        raise ValueError("No quadrilateral cells found in the mesh!")

    quad_mesh = meshio.Mesh(points=msh.points[:, :2], cells=[("quad", quad_cells)])
    meshio.write("temp_mesh.xdmf", quad_mesh)

    with XDMFFile(MPI.COMM_WORLD, "temp_mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # ==========================================
    # 2. Function Spaces & Variational Form
    # ==========================================
    V = fem.functionspace(domain, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # --- We define a and L AFTER we set up the boundary measures ---
    # (We will define L later in the BC section, but we keep a here)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # ==========================================
    # 3. Boundary Conditions (MODIFIED)
    # ==========================================
    tol = 1e-10

    # 3a. LEFT BOUNDARY (x = -Lx) : Dirichlet BC  phi = U_inf * x
    def left_boundary(x):
        return np.isclose(x[0], -Lx, atol=tol)

    left_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_boundary)
    left_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, left_facets)

    phi_bc_func = fem.Function(V)
    phi_bc_func.interpolate(lambda x: U_inf * x[0])
    bc = fem.dirichletbc(phi_bc_func, left_dofs) # velocity imposed on left boundary

    # 3b. RIGHT BOUNDARY (x = Lx) : Neumann BC  dphi/dn = U_inf
    #     Mathematically: ∫_Γ_right v * (dphi/dn) ds = ∫_Γ_right v * U_inf ds
    def right_boundary(x): #
        return np.isclose(x[0], Lx, atol=tol) #

    right_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right_boundary)

    # Mark the right facets with tag "1" so we can integrate over them
    indices = np.full(len(right_facets), 1, dtype=np.int32)
    right_mesh_tags = mesh.meshtags(domain, domain.topology.dim - 1, right_facets, indices)

    # Create the exterior facet measure 'ds' associated with these tags
    ds = ufl.Measure('ds', domain=domain, subdomain_data=right_mesh_tags)

    # Right-hand side now includes the Neumann integral on the right boundary
    L = fem.Constant(domain, default_scalar_type(U_inf)) * v * ds(1) # imposing dphi/dn = Uinf at the right boundary, i.e, normal component of velocity equal to infinite velocity

    # 3c. TOP and BOTTOM boundaries: NO explicit BC.
    #     The natural Neumann condition (dphi/dn = 0) applies automatically
    #     because we did not add any boundary integral there.
    #     This is exactly the free-slip far-field condition.

    # ==========================================
    # 4. Solve Problem
    # ==========================================
    problem = LinearProblem(
        a, L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="potential_flow_"
    )
    phi_sol = problem.solve()

    # ==========================================
    # 5. Compute Velocity and Pressure
    # ==========================================
    W = fem.functionspace(domain, ("DG", 1, (domain.geometry.dim,)))
    velocity_expr = fem.Expression(ufl.grad(phi_sol), W.element.interpolation_points)
    velocity = fem.Function(W)
    velocity.interpolate(velocity_expr)

    V_mag = fem.functionspace(domain, ("CG", 1))
    mag_expr = fem.Expression(ufl.sqrt(ufl.inner(velocity, velocity)), V_mag.element.interpolation_points)
    velocity_mag = fem.Function(V_mag)
    velocity_mag.interpolate(mag_expr)

    rho = 1.0
    pressure_expr = fem.Expression(0.5 * rho * (U_inf ** 2 - ufl.inner(velocity, velocity)),
                                   V_mag.element.interpolation_points)
    pressure_field = fem.Function(V_mag)
    pressure_field.interpolate(pressure_expr)

    # ==========================================
    # 6. Extract Mesh Data for PyVista
    # ==========================================
    topology_W, cell_types_W, geometry_W = dolfinx.plot.vtk_mesh(W)
    grid_vectors = pv.UnstructuredGrid(topology_W, cell_types_W, geometry_W)

    num_dofs_W = geometry_W.shape[0]
    u_2d = velocity.x.array.reshape(num_dofs_W, W.dofmap.index_map_bs)
    u_3d = np.pad(u_2d, ((0, 0), (0, 1)), mode='constant')
    grid_vectors["Velocity Vectors [m/s]"] = u_3d.real
    grid_vectors.set_active_vectors("Velocity Vectors [m/s]")

    topology_V, cell_types_V, geometry_V = dolfinx.plot.vtk_mesh(V_mag)
    grid_scalars = pv.UnstructuredGrid(topology_V, cell_types_V, geometry_V)

    grid_scalars["Velocity Magnitude [m/s]"] = velocity_mag.x.array.real
    grid_scalars["Relative Pressure [Pa]"] = pressure_field.x.array.real

    # ==========================================
    # 7. 2D Plotting
    # ==========================================
    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 500], off_screen=True)

    plotter.subplot(0, 0)
    grid_scalars.set_active_scalars("Relative Pressure [Pa]")
    plotter.add_mesh(grid_scalars, cmap="coolwarm", show_edges=False)
    plotter.view_xy()
    #plotter.show_grid(xtitle='X [m]', ytitle='Y [m]',n_xlabels=10, n_ylabels=10)
    plotter.add_text("Pressure Field", font_size=10)

    plotter.subplot(0, 1)
    streamlines = grid_vectors.streamlines_evenly_spaced_2D(
        vectors="Velocity Vectors [m/s]",
        start_position=(-Lx + 0.1, 0.0, 0.0),
        separating_distance=0.2,
        separating_distance_ratio=0.1,
        max_steps=2000
    )
    plotter.add_mesh(grid_vectors.outline(), color="black")
    plotter.add_mesh(streamlines.tube(radius=0.015), cmap="turbo")
    plotter.view_xy()
    #plotter.show_grid(xtitle='X [m]', ytitle='Y [m]', n_xlabels=10, n_ylabels=10)
    plotter.add_text("Streamlines", font_size=10)

    plotter.subplot(0, 2)
    grid_vectors = grid_vectors.sample(grid_scalars)
    arrows = grid_vectors.glyph(orient="Velocity Vectors [m/s]", scale="Velocity Magnitude [m/s]", factor=0.6, tolerance=0.005)
    plotter.add_mesh(grid_vectors.outline(), color="black")
    plotter.add_mesh(arrows, cmap="turbo")
    plotter.view_xy()
    #plotter.show_grid(xtitle='X [m]', ytitle='Y [m]', n_xlabels=10, n_ylabels=10)
    plotter.add_text("Velocity Vectors", font_size=10)

    file_path = Path.cwd().parent.parent / "output/potential_flow_results.png"
    plotter.screenshot(file_path)
    print("Plot successfully saved to 'potential_flow_results.png'")

if __name__ == "__main__":
    input_cylinders = [
        (0.0, 0.0, 1.0),
        (3.0, 1.5, 0.6),
        (3.0, -1.5, 0.6)
    ]

    domain_X_length = 10.0
    domain_Y_length = 5.0
    quad_mesh_size = 0.25

    print("Starting solver with mixed Dirichlet/Neumann BCs...")
    solve_potential_flow(
        cylinders=input_cylinders,
        Lx=domain_X_length,
        Ly=domain_Y_length,
        mesh_size=quad_mesh_size
    )