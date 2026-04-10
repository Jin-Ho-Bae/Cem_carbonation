"""
FEM Engine -- FEniCS Structural Analysis with Carbonation-Partitioned Properties.

Supports civil engineering structural elements:
- Beam (simply supported, cantilever)
- Column (axially loaded, with eccentricity)
- Beam-Column (combined bending + axial)
- Slab (2D plate)

Load types: tension, compression, shear, bending, pressure, combined.

Mesh partitioned by carbonation depth: carbonated zones get different
material properties from neat cement zones.

FEniCS installation for Google Colab:
    try:
        import dolfin
    except ImportError:
        !wget "https://fem-on-colab.github.io/releases/fenics-install-release-real.sh" \\
              -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"
        import dolfin
"""

import numpy as np
from typing import Optional, Dict, List


def install_fenics_colab():
    try:
        import dolfin
        return True
    except ImportError:
        import subprocess
        subprocess.run([
            "wget",
            "https://fem-on-colab.github.io/releases/fenics-install-release-real.sh",
            "-O", "/tmp/fenics-install.sh"
        ], check=True)
        subprocess.run(["bash", "/tmp/fenics-install.sh"], check=True)
        try:
            import dolfin
            return True
        except ImportError:
            return False


# Supported structural elements and load types
STRUCTURE_TYPES = {
    "beam": "Simply supported or cantilever beam (2D)",
    "column": "Axially loaded column (2D)",
    "beam_column": "Combined bending + axial load (2D)",
    "slab": "2D plate / slab element",
    "wall": "Shear wall (2D)",
}

LOAD_TYPES = {
    "tension": "Uniaxial tensile load",
    "compression": "Uniaxial compressive load",
    "shear": "Shear load (transverse)",
    "bending": "Pure bending moment (3-point or 4-point)",
    "pressure": "Distributed pressure on surface",
    "uniaxial": "Legacy: uniaxial load (same as tension)",
    "combined": "Combined axial + bending + shear",
}


class CarbonationFEMModel:
    """
    FEniCS FEM model for concrete structural analysis with carbonation.

    The mesh is partitioned based on carbonation depth:
    - Carbonated zone (near surface): modified E, nu from micromechanics
    - Neat cement zone (core): original E, nu
    """

    def __init__(self, geometry: dict, carbonation_depth: float,
                 E_carbonated: float, nu_carbonated: float,
                 E_neat: float, nu_neat: float,
                 shrinkage_carbonated: float = -0.5e-3):
        """
        geometry: {
            "type": "beam"/"column"/"beam_column"/"slab"/"wall",
            "length": m,
            "height": m,
            "width": m (optional, for 3D or out-of-plane),
            "support": "simply_supported"/"cantilever"/"fixed_fixed"/"pinned_roller"
        }
        """
        self.geometry = geometry
        self.x_carb = carbonation_depth
        self.E_carb = E_carbonated
        self.nu_carb = nu_carbonated
        self.E_neat = E_neat
        self.nu_neat = nu_neat
        self.shrinkage = shrinkage_carbonated

    def _create_mesh(self, df, mesh_density):
        geo = self.geometry
        geo_type = geo.get("type", "beam")
        L = geo.get("length", 0.5)
        H = geo.get("height", 0.1)

        if geo_type in ("beam", "beam_column", "wall"):
            nx = mesh_density * 5
            ny = mesh_density
        elif geo_type == "column":
            nx = mesh_density
            ny = mesh_density * 5
            L, H = H, L  # columns: height is the long dimension
        elif geo_type == "slab":
            nx = mesh_density * 3
            ny = mesh_density * 3
        else:
            nx = mesh_density * 5
            ny = mesh_density

        mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), nx, ny)
        return mesh, L, H

    def _partition_mesh(self, df, mesh, L, H):
        """Assign carbonated (1) vs neat (0) material regions."""
        materials = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        geo_type = self.geometry.get("type", "beam")

        for cell in df.cells(mesh):
            mp = cell.midpoint()

            if geo_type == "column":
                # Column: carbonation from all 4 sides
                dist = min(mp.x(), L - mp.x(), mp.y(), H - mp.y())
            elif geo_type == "slab":
                # Slab: carbonation from top and bottom
                dist = min(mp.y(), H - mp.y())
            else:
                # Beam/wall: carbonation from all exposed surfaces
                dist = min(mp.y(), H - mp.y(), mp.x(), L - mp.x())

            materials[cell] = 1 if dist < self.x_carb else 0

        return materials

    def _apply_boundary_conditions(self, df, V, L, H):
        """Apply BCs based on support type."""
        support = self.geometry.get("support", "cantilever")
        geo_type = self.geometry.get("type", "beam")

        bcs = []

        if support == "cantilever":
            def left(x, on_boundary):
                return on_boundary and df.near(x[0], 0)
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left))

        elif support == "simply_supported":
            # Pin at left (fix both), roller at right (fix y only)
            def left(x, on_boundary):
                return on_boundary and df.near(x[0], 0) and df.near(x[1], 0, 0.01*H)
            def right(x, on_boundary):
                return on_boundary and df.near(x[0], L) and df.near(x[1], 0, 0.01*H)
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left, method="pointwise"))
            bcs.append(df.DirichletBC(V.sub(1), df.Constant(0), right, method="pointwise"))

        elif support == "fixed_fixed":
            def left(x, on_boundary):
                return on_boundary and df.near(x[0], 0)
            def right(x, on_boundary):
                return on_boundary and df.near(x[0], L)
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left))
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), right))

        elif support == "pinned_roller":
            def left(x, on_boundary):
                return on_boundary and df.near(x[0], 0)
            def right_y(x, on_boundary):
                return on_boundary and df.near(x[0], L) and df.near(x[1], 0, 0.01*H)
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left))
            bcs.append(df.DirichletBC(V.sub(1), df.Constant(0), right_y, method="pointwise"))

        else:
            # Default: fix left
            def left(x, on_boundary):
                return on_boundary and df.near(x[0], 0)
            bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left))

        # Column base: always fix bottom
        if geo_type == "column":
            def bottom(x, on_boundary):
                return on_boundary and df.near(x[1], 0)
            bcs = [df.DirichletBC(V, df.Constant((0, 0)), bottom)]

        return bcs

    def _build_load(self, df, V, mesh, L, H, load):
        """Build load vector for the given load type."""
        load = load or {"type": "tension", "magnitude": 1e6}
        load_type = load.get("type", "tension")
        mag = load.get("magnitude", 1e6)
        geo_type = self.geometry.get("type", "beam")

        v = df.TestFunction(V)

        if load_type in ("tension", "uniaxial"):
            if geo_type == "column":
                # Axial tension at top
                class TopBoundary(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary and df.near(x[1], H)
                boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
                TopBoundary().mark(boundaries, 1)
                ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
                return df.dot(df.Constant((0, mag)), v) * ds(1)
            else:
                class RightBoundary(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary and df.near(x[0], L)
                boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
                RightBoundary().mark(boundaries, 1)
                ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
                return df.dot(df.Constant((mag, 0)), v) * ds(1)

        elif load_type == "compression":
            if geo_type == "column":
                class TopBoundary(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary and df.near(x[1], H)
                boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
                TopBoundary().mark(boundaries, 1)
                ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
                return df.dot(df.Constant((0, -mag)), v) * ds(1)
            else:
                class RightBoundary(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary and df.near(x[0], L)
                boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
                RightBoundary().mark(boundaries, 1)
                ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
                return df.dot(df.Constant((-mag, 0)), v) * ds(1)

        elif load_type == "shear":
            class RightBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[0], L)
            boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
            RightBoundary().mark(boundaries, 1)
            ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
            return df.dot(df.Constant((0, -mag)), v) * ds(1)

        elif load_type == "bending":
            # 3-point bending: point load at midspan (approximated as distributed over small region)
            class MidTopBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return (on_boundary and df.near(x[1], H) and
                            abs(x[0] - L/2) < L*0.05)
            boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
            MidTopBoundary().mark(boundaries, 1)
            ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
            return df.dot(df.Constant((0, -mag)), v) * ds(1)

        elif load_type == "pressure":
            # Uniform pressure on top surface
            class TopBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[1], H)
            boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
            TopBoundary().mark(boundaries, 1)
            ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
            return df.dot(df.Constant((0, -mag)), v) * ds(1)

        elif load_type == "combined":
            # Combined axial + bending
            axial = load.get("axial", mag)
            transverse = load.get("transverse", mag * 0.1)
            class RightBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[0], L)
            class TopBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[1], H)
            boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
            RightBoundary().mark(boundaries, 1)
            TopBoundary().mark(boundaries, 2)
            ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
            return (df.dot(df.Constant((axial, 0)), v) * ds(1) +
                    df.dot(df.Constant((0, -transverse)), v) * ds(2))

        # Fallback
        return df.dot(df.Constant((mag, 0)), v) * df.ds

    def solve(self, load=None, mesh_density=40):
        """
        Run FEniCS FEM analysis with carbonation-partitioned properties.

        load: {
            "type": "tension"/"compression"/"shear"/"bending"/"pressure"/"combined",
            "magnitude": Pa or N/m^2,
            "axial": Pa (for combined),
            "transverse": Pa (for combined)
        }
        """
        import dolfin as df

        mesh, L, H = self._create_mesh(df, mesh_density)
        materials = self._partition_mesh(df, mesh, L, H)

        # Material properties as DG0 fields
        V0 = df.FunctionSpace(mesh, "DG", 0)
        E_field = df.Function(V0, name="E")
        nu_field = df.Function(V0, name="nu")
        shrinkage_field = df.Function(V0, name="eps_carb")

        for cell in df.cells(mesh):
            idx = cell.index()
            if materials[cell] == 1:
                E_field.vector()[idx] = self.E_carb
                nu_field.vector()[idx] = self.nu_carb
                shrinkage_field.vector()[idx] = self.shrinkage
            else:
                E_field.vector()[idx] = self.E_neat
                nu_field.vector()[idx] = self.nu_neat
                shrinkage_field.vector()[idx] = 0.0

        # Lame parameters
        lam = E_field * nu_field / ((1 + nu_field) * (1 - 2 * nu_field))
        mu = E_field / (2 * (1 + nu_field))

        # Function space
        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

        # Boundary conditions
        bcs = self._apply_boundary_conditions(df, V, L, H)

        # Variational formulation
        u = df.TrialFunction(V)
        v = df.TestFunction(V)

        def epsilon(u):
            return df.sym(df.grad(u))

        def sigma(u):
            return lam * df.div(u) * df.Identity(2) + 2 * mu * epsilon(u)

        # Carbonation shrinkage
        eps_carb = df.as_tensor([[shrinkage_field, 0], [0, shrinkage_field]])
        sigma_carb = lam * 2 * shrinkage_field * df.Identity(2) + 2 * mu * eps_carb

        # Bilinear form
        a = df.inner(sigma(u), epsilon(v)) * df.dx

        # Load
        L_ext = self._build_load(df, V, mesh, L, H, load)
        L_form = L_ext + df.inner(sigma_carb, epsilon(v)) * df.dx

        # Solve
        u_sol = df.Function(V, name="displacement")
        df.solve(a == L_form, u_sol, bcs)

        # Post-processing
        V_stress = df.TensorFunctionSpace(mesh, "DG", 0)
        stress = df.project(sigma(u_sol), V_stress)

        s = sigma(u_sol) - (1.0/3.0) * df.tr(sigma(u_sol)) * df.Identity(2)
        von_mises = df.project(df.sqrt(3.0/2.0 * df.inner(s, s)), V0)

        u_array = u_sol.compute_vertex_values(mesh)
        n_vertices = mesh.num_vertices()
        coords = mesh.coordinates()

        results = {
            "mesh_coords": coords,
            "displacement": u_array.reshape(2, n_vertices).T,
            "max_displacement": float(np.max(np.abs(u_array))),
            "von_mises_max": float(von_mises.vector().max()),
            "von_mises_min": float(von_mises.vector().min()),
            "von_mises_mean": float(np.mean(von_mises.vector().get_local())),
            "carbonation_depth": self.x_carb,
            "E_carbonated": self.E_carb,
            "E_neat": self.E_neat,
            "n_carbonated_cells": int(np.sum(materials.array() == 1)),
            "n_neat_cells": int(np.sum(materials.array() == 0)),
            "geometry": self.geometry,
            "load": load or {"type": "tension", "magnitude": 1e6},
            "dolfin_objects": {
                "mesh": mesh,
                "u_sol": u_sol,
                "stress": stress,
                "von_mises": von_mises,
                "materials": materials,
            },
        }

        return results

    def solve_with_depth_profile(self, depth_profile, E_profile, nu_profile,
                                  mesh_density=40, load=None):
        """Apply continuous depth-dependent properties."""
        import dolfin as df
        from scipy.interpolate import interp1d

        mesh, L, H = self._create_mesh(df, mesh_density)

        E_interp = interp1d(depth_profile, E_profile,
                            bounds_error=False, fill_value=(E_profile[-1], E_profile[0]))
        nu_interp = interp1d(depth_profile, nu_profile,
                             bounds_error=False, fill_value=(nu_profile[-1], nu_profile[0]))

        V0 = df.FunctionSpace(mesh, "DG", 0)
        E_field = df.Function(V0)
        nu_field = df.Function(V0)

        for cell in df.cells(mesh):
            mp = cell.midpoint()
            min_dist = min(mp.y(), H - mp.y(), mp.x(), L - mp.x())
            E_field.vector()[cell.index()] = float(E_interp(min_dist))
            nu_field.vector()[cell.index()] = float(nu_interp(min_dist))

        lam = E_field * nu_field / ((1 + nu_field) * (1 - 2 * nu_field))
        mu = E_field / (2 * (1 + nu_field))

        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        bcs = self._apply_boundary_conditions(df, V, L, H)

        u = df.TrialFunction(V)
        v = df.TestFunction(V)

        def epsilon(u):
            return df.sym(df.grad(u))

        def sigma(u):
            return lam * df.div(u) * df.Identity(2) + 2 * mu * epsilon(u)

        a = df.inner(sigma(u), epsilon(v)) * df.dx
        L_form = self._build_load(df, V, mesh, L, H, load)

        u_sol = df.Function(V)
        df.solve(a == L_form, u_sol, bcs)

        s = sigma(u_sol) - (1./3.)*df.tr(sigma(u_sol))*df.Identity(2)
        vm = df.project(df.sqrt(1.5*df.inner(s,s)), V0)

        return {
            "max_displacement": float(np.max(np.abs(u_sol.compute_vertex_values(mesh)))),
            "von_mises_max": float(vm.vector().max()),
            "von_mises_mean": float(np.mean(vm.vector().get_local())),
            "mesh_coords": mesh.coordinates(),
            "dolfin_objects": {"mesh": mesh, "u_sol": u_sol, "von_mises": vm},
        }


def list_structure_types() -> dict:
    return STRUCTURE_TYPES.copy()


def list_load_types() -> dict:
    return LOAD_TYPES.copy()
