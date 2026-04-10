"""
Coupled Chemo-Transport-Micromechanical Carbonation Driver
===========================================================

Implements the full workflow described by the project:

    XRF raw material
        |
        v
    GEMS full hydration (Gibbs minimization, cemdata18)
        |  initial phase assemblage, porosity, CO2 capacity
        v
    1D transport (CO2 diffusion) over t in [0, t_final]
        |  c(x, t) profiles
        v
    Local GEMS equilibrium at each (x, t)
        |  phase assemblage, pH, porosity at each depth and time
        v
    6-level Eshelby-Mori-Tanaka homogenization per (x, t)
        |
        v
    E(x, t), nu(x, t), porosity(x, t), phases(x, t)
    Carbonation depth x_c(t), mid-to-long term durability

This module is the INTERNAL backend that the thermodynamics expert agent
uses through LangChain tools; GEMS is not a stand-alone application here.

Author: Cement Carbonation Modeling Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .gems_engine import (
    GEMSCarbonationEngine,
    compute_initial_hydration_from_xrf,
    REFERENCE_XRF,
)
from .transport_engine import CarbonationTransportModel
from .micromechanics_engine import compute_effective_properties


# ----------------------------------------------------------------------
#  Phase name mapping : GEMS (cemdata18) -> micromechanics ELASTIC_PROPERTIES
# ----------------------------------------------------------------------
GEMS_TO_MICRO_MAP: Dict[str, str] = {
    "Portlandite":     "CH",
    "CSH":             "CSH",
    "Calcite":         "CaCO3",
    "Vaterite":        "CaCO3",
    "Aragonite":       "CaCO3",
    "Ettringite":      "ettringite",
    "Tricarboaluminate": "ettringite",
    "Monosulfate":     "monosulfate",
    "Monocarbonate":   "monocarboaluminate",
    "Hemicarbonate":   "monocarboaluminate",
    "OH_AFm":          "monocarboaluminate",
    "C3AH6":           "AH3",
    "Hydrotalcite_OH": "hydrotalcite",
    "SiO2_am":         "SiO2_gel",
    "Gypsum":          "gypsum",
    "Stratlingite":    "stratlingite",
    "Brucite":         "hydrotalcite",
    "Thaumasite":      "thaumasite",
}


def gems_state_to_micro_phi(state, clinker_fraction: float = 0.0) -> Dict[str, float]:
    """Convert a GEMSState snapshot to the phi-dict used by compute_effective_properties."""
    phi: Dict[str, float] = {
        "CH": 0.0, "CSH": 0.0, "CaCO3": 0.0, "SiO2_gel": 0.0,
        "ettringite": 0.0, "monosulfate": 0.0, "monocarboaluminate": 0.0,
        "AH3": 0.0, "hydrotalcite": 0.0, "gypsum": 0.0,
        "stratlingite": 0.0, "thaumasite": 0.0,
    }
    for gems_name, vf in state.phase_volume_fracs.items():
        micro_name = GEMS_TO_MICRO_MAP.get(gems_name)
        if micro_name is None:
            continue
        phi[micro_name] = phi.get(micro_name, 0.0) + float(vf)
    phi["porosity"] = float(state.porosity)
    if clinker_fraction > 0:
        phi["clinker"] = clinker_fraction
    return phi


# ----------------------------------------------------------------------
#  Result container
# ----------------------------------------------------------------------
@dataclass
class CoupledProfileResult:
    """
    Full spatio-temporal result of the coupled chemo-transport-mechanical run.
    Arrays are shape (nt, nx) unless noted; x goes from exposed surface to depth.
    """
    x_grid: np.ndarray                       # (nx,) depth coordinates [m]
    times_sec: np.ndarray                    # (nt,) output times [s]
    times_years: np.ndarray                  # (nt,)
    c_CO2: np.ndarray                        # (nt, nx) CO2 conc profile (transport)
    cumulative_CO2: np.ndarray               # (nt, nx) mol CO2 reacted at depth
    E_field: np.ndarray                      # (nt, nx) [Pa]
    nu_field: np.ndarray                     # (nt, nx)
    porosity_field: np.ndarray               # (nt, nx)
    pH_field: np.ndarray                     # (nt, nx)
    carb_degree_field: np.ndarray            # (nt, nx)
    carbonation_depth: np.ndarray            # (nt,) [m] based on pH<9
    initial_hydration: Dict                  # full GEMS XRF hydration dict
    initial_E: float                         # Pa at t=0 (uncarbonated)
    phase_fields: Dict[str, np.ndarray]      # {phase_name: (nt,nx) volume fraction}
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def compute_coupled_profile(
    xrf_oxides: Dict[str, float],
    wc_ratio: float = 0.50,
    alpha_hydration: float = 1.0,
    temperature_C: float = 25.0,
    # Environmental / transport
    # Physically correct units: mol/m^3 of CO2 in gas phase.
    #   0.016  = atmospheric (400 ppm, 25 C)
    #   2.0    = 5 % accelerated
    #   16.0   = 40 % extreme (legacy default in transport engine tests)
    C_env_CO2: float = 0.016,
    diffusion_model: str = "fib_mc2010",
    L_total: float = 0.10,                   # m (analysis depth into material)
    nx: int = 20,
    saturation: float = 0.65,
    age_days: int = 28,
    # Time sampling
    save_times_years: Optional[List[float]] = None,
    # Micromechanics
    level: str = "paste",
    csh_model: str = "jennite",
    phi_sand: float = 0.40,
    phi_agg: float = 0.45,
    # Numerical
    verbose: bool = False,
) -> CoupledProfileResult:
    """
    Run the full coupled chemo-transport-micromechanical carbonation workflow.

    Parameters
    ----------
    xrf_oxides : dict
        XRF mass % of cement raw material, e.g.
        {"CaO": 64, "SiO2": 21, "Al2O3": 5.5, "Fe2O3": 3, "SO3": 2.8, "MgO": 1.5}.
    wc_ratio : float
        Water-to-cement mass ratio.
    alpha_hydration : float
        Degree of hydration (1.0 = complete hydration assumption).
    temperature_C : float
        Operating temperature for GEMS equilibrium.
    C_env_CO2 : float
        Boundary CO2 concentration at exposed surface [mol/m^3].
        0.016 ~ atmospheric (400 ppm); 2.0 ~ 5% accelerated; 16.0 ~ 40% extreme.
    diffusion_model : str
        One of the 6 models in transport_engine.
    L_total : float
        Total analysis depth from exposed surface [m].
    nx : int
        Number of spatial grid points.
    saturation : float
        Pore water saturation (0-1). Lower -> faster CO2 diffusion.
    age_days : int
        Concrete age at start of exposure (for diffusion models).
    save_times_years : list of float
        Output times in years (default: [0.5, 1, 2, 5, 10, 25, 50]).
    level : str
        "paste", "mortar", or "concrete" for homogenization level.

    Returns
    -------
    CoupledProfileResult
    """
    if save_times_years is None:
        save_times_years = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

    # ------------------------------------------------------------------
    # 1. Initial phase assemblage from GEMS XRF full hydration
    # ------------------------------------------------------------------
    if verbose:
        print("[coupled] Step 1: GEMS initial hydration from XRF ...")

    init_engine = GEMSCarbonationEngine.from_xrf(
        xrf_oxides,
        wc_ratio=wc_ratio,
        alpha_hydration=alpha_hydration,
        temperature_C=temperature_C,
    )
    init_state = init_engine._get_state()
    init_porosity = init_state.porosity
    max_CO2_capacity = init_engine.get_max_carbonation_capacity()

    # Initial E at uncarbonated state
    phi0 = gems_state_to_micro_phi(init_state)
    init_props = compute_effective_properties(
        phi0, level=level, csh_model=csh_model,
        phi_sand=phi_sand, phi_agg=phi_agg,
    )
    initial_E = init_props["E"]

    if verbose:
        print(f"  initial porosity   = {init_porosity:.4f}")
        print(f"  max CO2 capacity   = {max_CO2_capacity:.4f} mol/L")
        print(f"  initial E ({level}) = {initial_E/1e9:.2f} GPa")

    # ------------------------------------------------------------------
    # 2. Reactive CO2 transport  (Papadakis fast-reaction limit, analytic)
    # ------------------------------------------------------------------
    # Physical picture (chemistry of carbonation):
    #
    #   (a) Gas CO2 diffuses through the partially saturated pore network
    #       with an effective diffusivity D_eff (Millington-Quirk family).
    #   (b) Gas CO2 partitions into the pore solution via Henry's law
    #           c_CO2(aq) = K_H * p_CO2 ~ K_H * R * T * c_CO2(gas)
    #       and speciates into H2CO3 / HCO3- / CO3^2- depending on pH.
    #   (c) In the pore solution the dissolved carbonate attacks the Ca
    #       reservoirs of the hydrates, in thermodynamic priority order
    #             Portlandite  >  Ettringite  >  Monosulfate / AFm  >  C-S-H
    #       precipitating CaCO3 (calcite / vaterite / aragonite). This is
    #       why the hydrate assemblage evolves with time: GEMS sees the
    #       dissolved CO2 only through this reaction pathway, and hydrate
    #       amounts remain fixed where no dissolved CO2 is present.
    #   (d) For atmospheric and accelerated carbonation the reaction is
    #       orders of magnitude faster than diffusion (Damkohler >> 1);
    #       the solution is a sharp front moving at the Papadakis rate
    #                 x_c(t) = K sqrt(t),
    #                 K      = sqrt( 2 D_eff C_env / C_max ),
    #       with behind-the-front c_CO2 given by the quasi-steady balance
    #                 c(x,t) = C_env ( 1 - x / x_c(t) )        [x < x_c]
    #       and ahead of the front
    #                 c(x,t) = 0                                [x > x_c]
    #       i.e. all incoming gas CO2 is consumed at the front until the
    #       reservoir behind it is exhausted.
    #
    # This analytic fast-reaction limit is used here instead of an
    # operator-split PDE solve, because for the relevant time scales
    # (years) a PDE solve with instantaneous reaction becomes stiff
    # beyond practicality. Crucially, it guarantees that
    #
    #   * c_CO2(x_j, .) is monotone non-decreasing in time at every depth
    #   * dose(x_j, .) is monotone non-decreasing in time at every depth
    #   * c_CO2 and dose share the SAME front x_c(t) (self-consistent)
    #
    # which is exactly the physical behaviour the chemistry mechanism
    # described above dictates.
    #
    # The CarbonationTransportModel is still instantiated below so that
    # D_eff (used in K) reflects the requested diffusion_model, age, w/c
    # and saturation. We then bypass the stiff PDE solve and use the
    # analytic expressions.
    if verbose:
        print("[coupled] Step 2: reactive CO2 transport "
              "(Papadakis fast-reaction analytic) ...")

    save_times_sec = [y * 365.25 * 86400.0 for y in save_times_years]
    t_final = max(save_times_sec)

    transport = CarbonationTransportModel(
        L=L_total, nx=nx,
        porosity=init_porosity,
        saturation=saturation,
        C_env_CO2=C_env_CO2,
        diffusion_model=diffusion_model,
        wc_ratio=wc_ratio,
        age_days=age_days,
    )

    # The solver's internal grid (nx+1 nodes) is our spatial grid
    x_grid = np.asarray(transport.solver.x)
    nx_grid = int(len(x_grid))
    D_eff = float(transport.D_eff_value)

    # Capacity expressed in mol per m^3 of concrete
    # (GEMS reference volume = 1 L, so max_CO2_capacity is mol per 1 L).
    C_max_mol_per_m3 = max_CO2_capacity / 1.0e-3

    K_sqrt = np.sqrt(max(
        2.0 * D_eff * C_env_CO2 / max(C_max_mol_per_m3, 1e-30), 0.0,
    ))

    # Smoothing width for the front (avoids a perfect discontinuity in the
    # plots; purely cosmetic, ~1 grid cell wide, also used as the GEMS
    # transition for per-depth stepping).
    dx_mean = L_total / max(nx_grid - 1, 1)
    front_w = max(dx_mean, 1.0e-3)

    # Time grid
    times = np.array([float(s) for s in save_times_sec], dtype=float)
    nt = int(len(times))
    nx = nx_grid

    profiles = np.zeros((nt, nx_grid), dtype=float)      # gas CO2 [mol/m^3]
    cumulative_CO2 = np.zeros((nt, nx_grid), dtype=float)  # dose [mol/L paste]

    for i in range(nt):
        t_sec = max(float(times[i]), 0.0)
        xc = K_sqrt * np.sqrt(t_sec)   # front position at this time

        for j in range(nx_grid):
            x_j = float(x_grid[j])

            # ---- cumulative reacted dose (sharp front, smoothed by front_w) ----
            if x_j <= xc - 0.5 * front_w:
                dose_frac = 1.0
            elif x_j >= xc + 0.5 * front_w:
                dose_frac = 0.0
            else:
                # linear taper across the smoothing window
                dose_frac = 0.5 - (x_j - xc) / front_w
            cumulative_CO2[i, j] = max_CO2_capacity * max(0.0, min(1.0, dose_frac))

            # ---- gas CO2 concentration (Papadakis quasi-steady behind front) ----
            if xc <= 1e-12:
                # no front yet -> only the exposed surface has gas CO2
                c_gas = C_env_CO2 if x_j <= 0.5 * dx_mean else 0.0
            elif x_j <= xc - 0.5 * front_w:
                # quasi-steady linear profile behind the moving front:
                # c(x,t) = C_env * (1 - x/xc(t))
                c_gas = C_env_CO2 * max(0.0, 1.0 - x_j / max(xc, 1e-30))
            elif x_j >= xc + 0.5 * front_w:
                c_gas = 0.0
            else:
                # smooth the front transition between the linear tail and 0
                c_behind = C_env_CO2 * max(0.0, 1.0 - (xc - 0.5 * front_w) / max(xc, 1e-30))
                taper = 0.5 - (x_j - xc) / front_w
                c_gas = c_behind * max(0.0, min(1.0, taper))
            # the exposed-surface node is always at the environmental value
            if j == 0:
                c_gas = C_env_CO2
            profiles[i, j] = c_gas

    # ------------------------------------------------------------------
    # 4. Per-depth GEMS + micromechanics (monotone dose stepping)
    # ------------------------------------------------------------------
    if verbose:
        print("[coupled] Step 4: local GEMS equilibrium + homogenization ...")

    E_field = np.zeros((nt, nx), dtype=float)
    nu_field = np.zeros((nt, nx), dtype=float)
    poro_field = np.zeros((nt, nx), dtype=float)
    pH_field = np.zeros((nt, nx), dtype=float)
    carb_field = np.zeros((nt, nx), dtype=float)
    phase_fields: Dict[str, np.ndarray] = {}

    for j in range(nx):
        # One engine per depth: reuses the init state, steps through monotone dose
        engine_j = GEMSCarbonationEngine.from_xrf(
            xrf_oxides, wc_ratio=wc_ratio,
            alpha_hydration=alpha_hydration, temperature_C=temperature_C,
        )
        engine_j.phases["_init_CH_mol"] = engine_j.phases.get("Portlandite", 0.0)
        prev_dose = 0.0

        for i in range(nt):
            target = float(cumulative_CO2[i, j])
            delta = max(0.0, target - prev_dose)
            if delta > 1e-12:
                state = engine_j.equilibrate_step(delta)
            else:
                state = engine_j._get_state()
            prev_dose = target

            phi_ij = gems_state_to_micro_phi(state)
            props = compute_effective_properties(
                phi_ij, level=level, csh_model=csh_model,
                phi_sand=phi_sand, phi_agg=phi_agg,
            )
            E_field[i, j] = props["E"]
            nu_field[i, j] = props["nu"]
            poro_field[i, j] = state.porosity
            pH_field[i, j] = state.pH
            carb_field[i, j] = state.carbonation_degree

            for pname, vf in state.phase_volume_fracs.items():
                if pname not in phase_fields:
                    phase_fields[pname] = np.zeros((nt, nx), dtype=float)
                phase_fields[pname][i, j] = float(vf)

        if verbose and (j + 1) % max(1, nx // 5) == 0:
            print(f"  depth {j + 1}/{nx} done")

    # ------------------------------------------------------------------
    # 5. Carbonation depth vs time (pH < 9 threshold)
    # ------------------------------------------------------------------
    carb_depths = np.zeros(nt)
    for i in range(nt):
        idx = np.where(pH_field[i] < 9.0)[0]
        if len(idx) == 0:
            carb_depths[i] = 0.0
        elif len(idx) == nx:
            carb_depths[i] = float(x_grid[-1])
        else:
            # linear interpolation between last-basic and first-acidic points
            j_last = int(idx[-1])
            if j_last + 1 < nx:
                pH_a, pH_b = pH_field[i, j_last], pH_field[i, j_last + 1]
                x_a, x_b = x_grid[j_last], x_grid[j_last + 1]
                if pH_b != pH_a:
                    frac = (9.0 - pH_a) / (pH_b - pH_a)
                    carb_depths[i] = float(x_a + frac * (x_b - x_a))
                else:
                    carb_depths[i] = float(x_a)
            else:
                carb_depths[i] = float(x_grid[j_last])

    # ------------------------------------------------------------------
    # Package result
    # ------------------------------------------------------------------
    return CoupledProfileResult(
        x_grid=x_grid,
        times_sec=times,
        times_years=times / (365.25 * 86400.0),
        c_CO2=profiles,
        cumulative_CO2=cumulative_CO2,
        E_field=E_field,
        nu_field=nu_field,
        porosity_field=poro_field,
        pH_field=pH_field,
        carb_degree_field=carb_field,
        carbonation_depth=carb_depths,
        initial_hydration=init_engine.get_initial_hydration() or {},
        initial_E=initial_E,
        phase_fields=phase_fields,
        meta={
            "xrf_oxides": dict(xrf_oxides),
            "wc_ratio": wc_ratio,
            "alpha_hydration": alpha_hydration,
            "temperature_C": temperature_C,
            "C_env_CO2": C_env_CO2,
            "diffusion_model": diffusion_model,
            "L_total": L_total,
            "saturation": saturation,
            "level": level,
            "max_CO2_capacity": max_CO2_capacity,
            "K_sqrt_m_per_sqrt_s": K_sqrt,
            "K_mm_per_sqrt_yr": K_sqrt * 1000.0 * np.sqrt(365.25 * 86400.0),
            "D_eff": D_eff,
        },
    )


# ----------------------------------------------------------------------
#  Visualization
# ----------------------------------------------------------------------
def plot_coupled_profile(result: CoupledProfileResult, title: str = ""):
    """
    Multi-panel visualization of the coupled result:
      (1) CO2 concentration profiles (x, c) at selected times
      (2) Cumulative reacted CO2 (x, dose)
      (3) pH(x) profiles
      (4) Porosity(x) profiles
      (5) E(x) profiles
      (6) E(x, t) heat map  -- mid-to-long term durability
      (7) Carbonation depth vs sqrt(t)
      (8) Phase assemblage at the exposed surface (x=0)
    """
    import matplotlib.pyplot as plt

    x_mm = result.x_grid * 1000.0  # mm
    times_y = result.times_years
    nt = len(times_y)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    cmap = plt.cm.viridis
    colors = [cmap(k) for k in np.linspace(0.1, 0.9, nt)]

    # (1) CO2 concentration profiles
    ax = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(times_y):
        ax.plot(x_mm, result.c_CO2[i], color=colors[i], label=f"{t:.1f} yr", lw=1.8)
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel(r"CO$_2$ conc. (mol/m$^3$)")
    ax.set_title("1. CO$_2$ Transport Profile")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (2) Cumulative reacted CO2
    ax = fig.add_subplot(gs[0, 1])
    for i, t in enumerate(times_y):
        ax.plot(x_mm, result.cumulative_CO2[i], color=colors[i], label=f"{t:.1f} yr", lw=1.8)
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("reacted CO$_2$ (mol/L)")
    ax.set_title("2. Cumulative CO$_2$ Dose")
    ax.grid(alpha=0.3)

    # (3) pH profiles
    ax = fig.add_subplot(gs[0, 2])
    for i, t in enumerate(times_y):
        ax.plot(x_mm, result.pH_field[i], color=colors[i], label=f"{t:.1f} yr", lw=1.8)
    ax.axhline(9.0, color="red", linestyle="--", alpha=0.5, label="pH=9 (depass.)")
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("pH")
    ax.set_title("3. Pore Solution pH")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (4) Porosity profiles
    ax = fig.add_subplot(gs[1, 0])
    for i, t in enumerate(times_y):
        ax.plot(x_mm, result.porosity_field[i], color=colors[i], label=f"{t:.1f} yr", lw=1.8)
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("porosity")
    ax.set_title("4. Porosity Evolution")
    ax.grid(alpha=0.3)

    # (5) E profiles
    ax = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(times_y):
        ax.plot(x_mm, result.E_field[i] / 1e9, color=colors[i],
                label=f"{t:.1f} yr", lw=1.8)
    ax.axhline(result.initial_E / 1e9, color="k", ls=":", alpha=0.7, label="initial")
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("E (GPa)")
    ax.set_title("5. Effective Young's Modulus")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (6) E(x, t) heatmap
    ax = fig.add_subplot(gs[1, 2])
    X, T = np.meshgrid(x_mm, times_y)
    pcm = ax.pcolormesh(X, T, result.E_field / 1e9, cmap="RdYlBu_r", shading="auto")
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("time (years)")
    ax.set_title("6. E(x, t)  [GPa]  Durability Map")
    if times_y.max() / max(times_y.min(), 1e-3) > 20:
        ax.set_yscale("log")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="E (GPa)")

    # (7) Carbonation depth vs sqrt(t)
    ax = fig.add_subplot(gs[2, 0])
    sqrt_t = np.sqrt(times_y)
    ax.plot(sqrt_t, result.carbonation_depth * 1000.0, "o-", color="#C62828", lw=2)
    ax.set_xlabel(r"$\sqrt{t}$ (yr$^{1/2}$)")
    ax.set_ylabel("carbonation depth (mm)")
    ax.set_title(r"7. Carbonation Depth vs $\sqrt{t}$")
    ax.grid(alpha=0.3)
    # Linear fit
    if len(sqrt_t) > 1:
        K = np.polyfit(sqrt_t, result.carbonation_depth * 1000.0, 1)[0]
        ax.text(0.05, 0.92, f"K = {K:.2f} mm/yr$^{{1/2}}$",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="lightyellow", boxstyle="round"))

    # (8) Phase assemblage at surface (x=0) vs time
    ax = fig.add_subplot(gs[2, 1])
    phase_plot_keys = [
        ("Portlandite", "#2196F3"),
        ("CSH",         "#4CAF50"),
        ("Calcite",     "#F44336"),
        ("Ettringite",  "#FF9800"),
        ("Monocarbonate", "#9C27B0"),
        ("SiO2_am",     "#607D8B"),
    ]
    for pname, c in phase_plot_keys:
        if pname in result.phase_fields:
            ax.plot(times_y, result.phase_fields[pname][:, 0],
                    color=c, label=pname, lw=1.8)
    if times_y.max() / max(times_y.min(), 1e-3) > 20:
        ax.set_xscale("log")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("volume fraction")
    ax.set_title("8. Surface Phase Assemblage")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (9) Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    m = result.meta
    xrf = m["xrf_oxides"]
    xrf_str = ", ".join(f"{k}={v:.1f}" for k, v in list(xrf.items())[:6])
    text = (
        "Coupled Carbonation Summary\n"
        + "_" * 38 + "\n"
        + f"XRF (%)      : {xrf_str}\n"
        + f"w/c          : {m['wc_ratio']}\n"
        + f"T            : {m['temperature_C']} C\n"
        + f"CO2 env      : {m['C_env_CO2']:.1f} mol/m3\n"
        + f"Diffusion    : {m['diffusion_model']}\n"
        + f"Level        : {m['level']}\n"
        + "_" * 38 + "\n"
        + f"Initial E    : {result.initial_E/1e9:.2f} GPa\n"
        + f"Final E @ 0  : {result.E_field[-1, 0]/1e9:.2f} GPa\n"
        + f"Final x_carb : {result.carbonation_depth[-1]*1000:.2f} mm\n"
        + f"CO2 capacity : {m['max_CO2_capacity']:.3f} mol/L\n"
    )
    ax.text(0.02, 0.5, text, fontsize=9, family="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.suptitle(title or "Chemo-Transport-Micromechanical Coupled Analysis",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.show()
    return fig


# ----------------------------------------------------------------------
#  Long-term durability visualization (time-ordered)
# ----------------------------------------------------------------------
def plot_durability_time_evolution(
    result: CoupledProfileResult,
    title: str = "",
    depths_mm_track: Optional[List[float]] = None,
):
    """
    Mid-to-long term durability view focused on how properties evolve with
    TIME at fixed structural depths (complementary to plot_coupled_profile,
    which is depth-focused).

    Panels
    ------
    (A)  FEM-style 2D slab render of E(x, t) -- acts as the engineering
         "structural durability map": a colored bar for each saved time,
         stacked so the eye tracks the advance of the carbonated (stiffer,
         denser) zone into the core.
    (B)  E at fixed depths vs time -- degradation / gain trajectories that
         the structural engineer reads directly as "life curves".
    (C)  CO2 concentration at fixed depths vs time -- the driving force
         history that explains panel (B) and (D).
    (D)  Surface and core phase assemblage vs time -- where CH / C-S-H /
         CaCO3 / ettringite mass goes over the service life.
    (E)  Carbonation depth vs sqrt(time) with linear fit (K coefficient).
    (F)  Depth profile of E and CO2 at the FINAL time (cross-section).

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    x_mm = result.x_grid * 1000.0
    times_y = result.times_years
    nt = len(times_y)

    # Choose depths to track (default: 0, 5, 10, 25, 50 mm, clipped to grid)
    if depths_mm_track is None:
        depths_mm_track = [0.0, 5.0, 10.0, 25.0, 50.0]
    depths_mm_track = [d for d in depths_mm_track if d <= x_mm.max() + 1e-9]
    track_idx = [int(np.argmin(np.abs(x_mm - d))) for d in depths_mm_track]
    track_labels = [f"{x_mm[k]:.1f} mm" for k in track_idx]

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    cmap_e = plt.cm.RdYlBu_r
    cmap_t = plt.cm.viridis
    t_colors = [cmap_t(k) for k in np.linspace(0.1, 0.9, nt)]
    d_colors = [plt.cm.plasma(k) for k in np.linspace(0.15, 0.85, len(track_idx))]

    # ------------------------------------------------------------------
    # (A) FEM-style 2D slab render of E(x, t)
    # ------------------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0:2])
    E_GPa = result.E_field / 1e9
    vmin, vmax = float(E_GPa.min()), float(E_GPa.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    bar_h = 0.85
    for i in range(nt):
        y0 = i
        row = E_GPa[i][None, :]
        axA.imshow(
            row, cmap=cmap_e, norm=norm, aspect="auto",
            extent=(x_mm.min(), x_mm.max(), y0, y0 + bar_h),
            origin="lower",
        )
        axA.text(
            x_mm.max() * 1.01, y0 + bar_h / 2.0,
            f"t = {times_y[i]:.1f} y",
            va="center", ha="left", fontsize=8,
        )
        xc_mm = float(result.carbonation_depth[i] * 1000.0)
        if xc_mm > 0:
            axA.plot([xc_mm, xc_mm], [y0, y0 + bar_h], color="k", lw=1.5)
    axA.set_yticks([])
    axA.set_xlim(x_mm.min(), x_mm.max() * 1.15)
    axA.set_ylim(0, nt)
    axA.set_xlabel("depth (mm) -- exposed surface at left")
    axA.set_title(
        "(A) FEM-style slab render of E(x, t)\n"
        "each bar = one saved time; black tick = pH<9 carbonation front"
    )
    sm = ScalarMappable(norm=norm, cmap=cmap_e)
    sm.set_array([])
    fig.colorbar(sm, ax=axA, fraction=0.035, pad=0.08, label="E (GPa)")

    # ------------------------------------------------------------------
    # (B) E at fixed depths vs time
    # ------------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 2])
    for k_idx, j in enumerate(track_idx):
        axB.plot(times_y, result.E_field[:, j] / 1e9,
                 "o-", color=d_colors[k_idx], lw=2, label=track_labels[k_idx])
    axB.axhline(result.initial_E / 1e9, color="k", ls=":", lw=1.2, label="initial")
    if times_y.max() / max(times_y.min(), 1e-3) > 20:
        axB.set_xscale("log")
    axB.set_xlabel("time (years)")
    axB.set_ylabel("E (GPa)")
    axB.set_title("(B) E vs time at fixed depths")
    axB.legend(fontsize=7, ncol=2)
    axB.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (C) CO2 concentration at fixed depths vs time
    # ------------------------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    for k_idx, j in enumerate(track_idx):
        axC.plot(times_y, result.c_CO2[:, j],
                 "s-", color=d_colors[k_idx], lw=2, label=track_labels[k_idx])
    if times_y.max() / max(times_y.min(), 1e-3) > 20:
        axC.set_xscale("log")
    axC.set_xlabel("time (years)")
    axC.set_ylabel(r"CO$_2$ (mol/m$^3$)")
    axC.set_title("(C) CO$_2$ at fixed depths vs time")
    axC.legend(fontsize=7, ncol=2)
    axC.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (D) Phase assemblage at surface and at core vs time
    # ------------------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    surf_j = 0
    core_j = len(x_mm) - 1
    phase_plot_keys = [
        ("Portlandite", "#2196F3"),
        ("CSH",         "#4CAF50"),
        ("Calcite",     "#F44336"),
        ("Ettringite",  "#FF9800"),
        ("Monocarbonate", "#9C27B0"),
        ("SiO2_am",     "#607D8B"),
    ]
    for pname, c in phase_plot_keys:
        if pname in result.phase_fields:
            axD.plot(times_y, result.phase_fields[pname][:, surf_j],
                     color=c, ls="-", lw=1.8, label=f"{pname} @ surface")
            axD.plot(times_y, result.phase_fields[pname][:, core_j],
                     color=c, ls="--", lw=1.5, alpha=0.7)
    if times_y.max() / max(times_y.min(), 1e-3) > 20:
        axD.set_xscale("log")
    axD.set_xlabel("time (years)")
    axD.set_ylabel("volume fraction")
    axD.set_title("(D) Phase assemblage vs time  (solid=surface, dashed=core)")
    axD.legend(fontsize=6, ncol=2)
    axD.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (E) Carbonation depth vs sqrt(t) with linear fit
    # ------------------------------------------------------------------
    axE = fig.add_subplot(gs[1, 2])
    sqrt_t = np.sqrt(np.clip(times_y, 0.0, None))
    xc_mm = result.carbonation_depth * 1000.0
    axE.plot(sqrt_t, xc_mm, "o-", color="#C62828", lw=2)
    if len(sqrt_t) > 1:
        K, b = np.polyfit(sqrt_t, xc_mm, 1)
        t_fit = np.linspace(0, sqrt_t.max(), 50)
        axE.plot(t_fit, K * t_fit + b, "k--", lw=1.2,
                 label=f"K = {K:.2f} mm/yr$^{{1/2}}$")
        axE.legend(fontsize=8)
    axE.set_xlabel(r"$\sqrt{t}$ (yr$^{1/2}$)")
    axE.set_ylabel("carbonation depth (mm)")
    axE.set_title(r"(E) Carbonation depth vs $\sqrt{t}$")
    axE.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (F) Final-time depth profile: E and CO2
    # ------------------------------------------------------------------
    axF = fig.add_subplot(gs[2, 0])
    axF.plot(x_mm, result.E_field[-1] / 1e9, "b-", lw=2, label="E (GPa)")
    axF.axhline(result.initial_E / 1e9, color="b", ls=":", alpha=0.7,
                label="initial E")
    axF.set_xlabel("depth (mm)")
    axF.set_ylabel("E (GPa)", color="b")
    axF.tick_params(axis="y", labelcolor="b")
    axF2 = axF.twinx()
    axF2.plot(x_mm, result.c_CO2[-1], "r-", lw=2, label="CO$_2$")
    axF2.set_ylabel(r"CO$_2$ (mol/m$^3$)", color="r")
    axF2.tick_params(axis="y", labelcolor="r")
    axF.set_title(f"(F) Final depth profile  (t = {times_y[-1]:.1f} y)")
    axF.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (G) Phase assemblage vs depth at final time
    # ------------------------------------------------------------------
    axG = fig.add_subplot(gs[2, 1])
    for pname, c in phase_plot_keys:
        if pname in result.phase_fields:
            axG.plot(x_mm, result.phase_fields[pname][-1],
                     color=c, lw=1.8, label=pname)
    axG.set_xlabel("depth (mm)")
    axG.set_ylabel("volume fraction")
    axG.set_title(f"(G) Phase assemblage vs depth  (t = {times_y[-1]:.1f} y)")
    axG.legend(fontsize=7, ncol=2)
    axG.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # (H) Interpretation text
    # ------------------------------------------------------------------
    axH = fig.add_subplot(gs[2, 2])
    axH.axis("off")
    m = result.meta
    E0 = result.initial_E / 1e9
    E_surf_final = result.E_field[-1, 0] / 1e9
    E_core_final = result.E_field[-1, -1] / 1e9
    dE_surf = E_surf_final - E0
    text = (
        "Durability interpretation\n"
        + "_" * 38 + "\n"
        + "(A) slab render:\n"
        + "    reading order = time.\n"
        + "    Front (black tick) advances into core;\n"
        + "    carbonated shell stiffens as CH -> CaCO3.\n"
        + "\n(B) E vs time at fixed depth:\n"
        + "    when a depth is 'reached' by the front,\n"
        + "    E jumps to the carbonated plateau.\n"
        + "    Cover (first tracked depth) determines\n"
        + "    rebar depassivation onset.\n"
        + "\n(C) CO2 vs time at fixed depth:\n"
        + "    leading cause -- precedes (B) by the\n"
        + "    local reaction timescale.\n"
        + "\n(D) phase assemblage:\n"
        + "    surface = end state (CaCO3, SiO2 gel);\n"
        + "    core    = still CH / C-S-H reservoir.\n"
        + "\n(E) K coefficient:\n"
        + f"    K = {m.get('K_mm_per_sqrt_yr', 0):.2f} mm/yr^0.5\n"
        + "    Natural: 2-5 ; Accelerated: 10-30.\n"
        + "\n(F, G) cross-section at t_final:\n"
        + "    gradient of stiffness and residual CH\n"
        + "    defines the effective load-bearing core.\n"
        + "_" * 38 + "\n"
        + f"E0          : {E0:.2f} GPa\n"
        + f"E surf (end): {E_surf_final:.2f} GPa  (dE={dE_surf:+.2f})\n"
        + f"E core (end): {E_core_final:.2f} GPa\n"
        + f"x_carb (end): {result.carbonation_depth[-1]*1000:.1f} mm\n"
    )
    axH.text(0.0, 1.0, text, fontsize=8, family="monospace",
             verticalalignment="top", transform=axH.transAxes,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.suptitle(
        title or "Mid-to-Long Term Durability Evolution",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.show()
    return fig


# ----------------------------------------------------------------------
#  Convenience: run with a named reference XRF
# ----------------------------------------------------------------------
def run_coupled_reference(
    cement_key: str = "OPC_CEM_I",
    wc_ratio: float = 0.50,
    C_env_CO2: float = 0.016,
    diffusion_model: str = "fib_mc2010",
    save_times_years: Optional[List[float]] = None,
    level: str = "paste",
    verbose: bool = True,
) -> CoupledProfileResult:
    """Quick entry point using a REFERENCE_XRF entry (e.g. 'OPC_CEM_I')."""
    if cement_key not in REFERENCE_XRF:
        raise KeyError(f"Unknown cement_key '{cement_key}'. "
                       f"Options: {list(REFERENCE_XRF.keys())}")
    return compute_coupled_profile(
        xrf_oxides=REFERENCE_XRF[cement_key],
        wc_ratio=wc_ratio,
        C_env_CO2=C_env_CO2,
        diffusion_model=diffusion_model,
        save_times_years=save_times_years,
        level=level,
        verbose=verbose,
    )
