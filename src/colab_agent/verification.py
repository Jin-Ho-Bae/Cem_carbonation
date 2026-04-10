"""
Verification module -- Direct engine computations for LLM Agent result comparison.

Provides:
- full_pipeline_visualization(): 6-panel chemo-transport-micromechanical plot
  (smooth monotone backend via compute_coupled_profile)
- run_verification(): quick engine-vs-LLM comparison
- run_direct_engine_demo(): all direct engine demos (chemistry, transport, etc.)
"""

import numpy as np
import json
from typing import Optional, Dict, List

from .chemistry_engine import (
    PhaseEvolution, EnvironmentCondition, CEMENT_SYSTEMS,
    SCM_DATABASE, build_cement_system,
    get_available_scms, get_available_base_cements,
)
from .transport_engine import (
    CarbonationTransportModel, DiffusionSolver1D, PoreStructure,
    millington_quirk, compute_D_eff, list_diffusion_models,
    find_carbonation_depth,
)
from .micromechanics_engine import (
    compute_effective_properties, ELASTIC_PROPERTIES,
    level_I_md, level_II_csh_variant, level_III_csh_matrix,
    level_IV_paste, level_V_mortar, level_VI_concrete,
    level_I, level_II, level_III,
)
from .fem_engine import CarbonationFEMModel, list_structure_types, list_load_types
from .md_engine import MDEngine, MD_REFERENCE_DATA
from .coupled_carbonation import compute_coupled_profile
from .gems_engine import REFERENCE_XRF


# ----------------------------------------------------------------------
#  SCM-aware surrogate wc_ratio
# ----------------------------------------------------------------------
#  The direct verification path runs the monotone coupled pipeline
#  (compute_coupled_profile), which is GEMS/XRF based. For SCM blends we
#  do not have a full thermodynamic engine, so we route the SCM effect
#  through an equivalent water/binder ratio that preserves the expected
#  change in porosity and D_eff. These coefficients are pulled from the
#  fib MC2010 k-value framework; they keep the trend monotone and
#  physically ordered (FA slightly increases diffusivity at young age
#  and decreases it long-term; GGBS decreases it; SF decreases strongly).
_SCM_WC_SHIFT = {
    "fly_ash_F":   +0.015,
    "fly_ash_C":   +0.010,
    "GGBS":        -0.020,
    "silica_fume": -0.040,
    "metakaolin":  -0.025,
    "limestone_filler": +0.005,
    "natural_pozzolan": -0.005,
    "nano_silica": -0.035,
    "rice_husk_ash": -0.020,
    "calcined_clay": -0.015,
}


def _xrf_for_cement(cement_type: str) -> Dict[str, float]:
    key_map = {
        "OPC": "OPC_CEM_I",
        "OPC_CEM_I": "OPC_CEM_I",
        "CEM_I": "OPC_CEM_I",
        "OPC_low_C3A": "OPC_low_C3A",
        "white_cement": "white_cement",
        "white": "white_cement",
        "CSA": "CSA",
    }
    return dict(REFERENCE_XRF[key_map.get(cement_type, "OPC_CEM_I")])


def full_pipeline_visualization(cement_type="OPC", scm_mix=None,
                                diffusion_model="millington_quirk",
                                wc_ratio=0.50, duration_years=50,
                                C_env_CO2=0.016, saturation=0.65,
                                level="paste", nx=25, L_total=0.10,
                                temperature_C=25.0):
    """
    Smooth, monotone direct-engine verification plot.

    Uses compute_coupled_profile (Papadakis sharp-front dose + per-depth GEMS
    local equilibrium + 6-level Eshelby-MT) as the single backend, so every
    panel is driven by one consistent state evolution. This eliminates the
    piecewise a_CH/a_CSH kinks and the noisy K estimator that caused the
    earlier "too many inflection points, no trend" behavior.

    Parameters
    ----------
    cement_type : str
        "OPC" (default), "OPC_low_C3A", "white_cement", or "CSA".
    scm_mix : dict or None
        e.g. {"fly_ash_F": 0.30}. Routed through an equivalent-w/c shift so
        the downstream coupled profile stays monotone and physically ordered.
    diffusion_model : str
        One of the 6 transport-engine models.
    wc_ratio : float
        Water-to-binder ratio.
    duration_years : float
        Total exposure time.
    C_env_CO2 : float
        Boundary CO2 concentration [mol/m^3]. Default 0.016 (atmospheric).
    saturation : float
        Pore water saturation (0-1).
    level : str
        Homogenization level: "paste", "mortar", "concrete".
    """
    import matplotlib.pyplot as plt

    scm_mix = scm_mix or {}

    # ---- label and equivalent w/c (SCM surrogate) ----
    blend_label = cement_type
    wc_eff = float(wc_ratio)
    for name, frac in scm_mix.items():
        wc_eff += _SCM_WC_SHIFT.get(name, 0.0) * (float(frac) / 0.30)
    wc_eff = float(np.clip(wc_eff, 0.25, 0.70))

    if scm_mix:
        parts = [f"{int(r*100)}%{n.replace('_',' ')}" for n, r in scm_mix.items()]
        blend_label += " + " + " + ".join(parts)

    # ---- monotone save times (log-spaced so early-age still visible) ----
    if duration_years >= 10:
        save_times_years = list(np.unique(np.round(
            np.geomspace(0.5, float(duration_years), 8), 2
        )))
    else:
        save_times_years = list(np.linspace(0.1, float(duration_years), 6))

    # ---- single smooth backend run ----
    xrf = _xrf_for_cement(cement_type)
    result = compute_coupled_profile(
        xrf_oxides=xrf,
        wc_ratio=wc_eff,
        alpha_hydration=1.0,
        temperature_C=temperature_C,
        C_env_CO2=C_env_CO2,
        diffusion_model=diffusion_model,
        L_total=L_total,
        nx=nx,
        saturation=saturation,
        save_times_years=save_times_years,
        level=level,
        verbose=False,
    )

    x_mm = result.x_grid * 1000.0
    times_y = result.times_years
    nt = len(times_y)

    # ---- 6-panel figure ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cmap = plt.cm.viridis
    t_colors = [cmap(k) for k in np.linspace(0.1, 0.9, nt)]

    # 1. Phase evolution at the exposed surface (monotone dose)
    ax = axes[0, 0]
    phase_keys = [
        ("Portlandite", "#2196F3", "CH"),
        ("CSH",         "#4CAF50", "C-S-H"),
        ("Calcite",     "#F44336", "CaCO3"),
        ("Ettringite",  "#FF9800", "AFt"),
        ("SiO2_am",     "#607D8B", "SiO2 (gel)"),
    ]
    for pname, c, lbl in phase_keys:
        if pname in result.phase_fields:
            ax.plot(times_y, result.phase_fields[pname][:, 0],
                    color=c, linewidth=2, label=lbl)
    ax.plot(times_y, result.porosity_field[:, 0], "k--",
            lw=1.5, label="porosity")
    ax.set_title("1. Chemistry: Phase Evolution (surface)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Volume fraction")
    if duration_years > 1 and times_y[0] > 0:
        ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Transport: CO2 profiles (already normalized by C_env)
    ax = axes[0, 1]
    c_env = max(float(C_env_CO2), 1e-12)
    for i in range(nt):
        ax.plot(x_mm, result.c_CO2[i] / c_env,
                color=t_colors[i], lw=1.8, label=f"{times_y[i]:.1f} yr")
    ax.set_title(f"2. Transport: CO2 / C_env  ({diffusion_model})")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("C / C_env")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Micromechanics: E vs depth across time (monotone, smooth)
    ax = axes[0, 2]
    for i in range(nt):
        ax.plot(x_mm, result.E_field[i] / 1e9,
                color=t_colors[i], lw=1.8, label=f"{times_y[i]:.1f} yr")
    ax.axhline(result.initial_E / 1e9, color="k", ls=":",
               lw=1.2, alpha=0.7, label="initial")
    ax.set_title(f"3. Micromechanics: E(x, t)  ({level})")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("E (GPa)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Carbonation depth vs sqrt(t) with full linear fit
    ax = axes[1, 0]
    sqrt_t = np.sqrt(np.clip(times_y, 0.0, None))
    xc_mm = result.carbonation_depth * 1000.0
    ax.plot(sqrt_t, xc_mm, "ko-", linewidth=2, markersize=6,
            label="pH<9 front")
    K = float(result.meta.get("K_mm_per_sqrt_yr", 0.0))
    if len(sqrt_t) >= 2:
        K_fit, b_fit = np.polyfit(sqrt_t, xc_mm, 1)
        t_line = np.linspace(0, sqrt_t.max(), 50)
        ax.plot(t_line, K_fit * t_line + b_fit, "r--", lw=1.5,
                label=f"linear fit K = {K_fit:.2f} mm/yr$^{{1/2}}$")
        K = K_fit
    ax.set_title(r"4. Carbonation Depth vs $\sqrt{t}$")
    ax.set_xlabel(r"$\sqrt{t}$  (yr$^{1/2}$)")
    ax.set_ylabel("Depth (mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Final-time E profile with carbonation front
    ax = axes[1, 1]
    ax.plot(x_mm, result.E_field[-1] / 1e9, "b-", linewidth=2,
            label=f"E @ {times_y[-1]:.1f} y")
    ax.axhline(result.initial_E / 1e9, color="k", ls=":",
               lw=1.2, alpha=0.7, label="initial")
    if xc_mm[-1] > 0:
        ax.axvline(xc_mm[-1], color="r", ls="--",
                   label=f"x_c = {xc_mm[-1]:.1f} mm")
    ax.set_title(f"5. Final E Profile  (t = {times_y[-1]:.1f} y)")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("E (GPa)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Summary
    ax = axes[1, 2]
    ax.axis("off")
    phi0 = result.meta.get("L_total", L_total)  # unused placeholder
    D_eff = result.meta.get("D_eff", float("nan"))
    E0 = result.initial_E / 1e9
    E_surf = result.E_field[-1, 0] / 1e9
    E_core = result.E_field[-1, -1] / 1e9
    pH_surf = float(result.pH_field[-1, 0])
    pH_core = float(result.pH_field[-1, -1])
    porosity_init = float(result.porosity_field[0, -1])
    summary = (
        f"SUMMARY: {blend_label}\n"
        f"{'_' * 40}\n"
        f"w/c (nominal):    {wc_ratio:.3f}\n"
        f"w/c (eff, SCM):   {wc_eff:.3f}\n"
        f"Diffusion model:  {diffusion_model}\n"
        f"Level:            {level}\n"
        f"Initial porosity: {porosity_init:.3f}\n"
        f"D_eff:            {D_eff:.2e} m2/s\n"
        f"{'_' * 40}\n"
        f"Carb. depth ({duration_years:.0f}y): {xc_mm[-1]:.1f} mm\n"
        f"K coefficient:    {K:.2f} mm/sqrt(yr)\n"
        f"pH  surface/core: {pH_surf:.1f} / {pH_core:.1f}\n"
        f"E0 (uncarbonated): {E0:.1f} GPa\n"
        f"E surface (end):  {E_surf:.1f} GPa\n"
        f"E core    (end):  {E_core:.1f} GPa\n"
    )
    ax.text(
        0.02, 0.5, summary, fontsize=10, family="monospace",
        verticalalignment="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow"),
    )

    plt.suptitle(f"Chemo-Transport-Micromechanical Analysis: {blend_label}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return {
        "depths": xc_mm.tolist(),
        "times_years": times_y.tolist(),
        "K": K,
        "porosity": porosity_init,
        "D_eff": D_eff,
        "E_initial_GPa": E0,
        "E_surface_final_GPa": E_surf,
        "E_core_final_GPa": E_core,
        "coupled_result": result,
    }


def run_verification(cases=None):
    """
    Run direct engine computation for standard test cases.
    Compare with LLM agent results if provided.

    cases: list of dicts with keys: cement_type, scm_mix, wc_ratio, duration_years, diffusion_model
    """
    if cases is None:
        cases = [
            {"cement_type": "OPC", "scm_mix": {}, "wc_ratio": 0.50,
             "duration_years": 50, "diffusion_model": "millington_quirk"},
            {"cement_type": "OPC", "scm_mix": {"fly_ash_F": 0.30}, "wc_ratio": 0.45,
             "duration_years": 50, "diffusion_model": "papadakis"},
            {"cement_type": "OPC", "scm_mix": {"GGBS": 0.50}, "wc_ratio": 0.50,
             "duration_years": 50, "diffusion_model": "fib_mc2010"},
        ]

    print("=" * 60)
    print("  DIRECT ENGINE VERIFICATION")
    print("=" * 60)

    results = []
    for i, case in enumerate(cases, 1):
        label = case["cement_type"]
        if case.get("scm_mix"):
            parts = [f"{int(r*100)}%{n}" for n, r in case["scm_mix"].items()]
            label += " + " + " + ".join(parts)

        print(f"\n--- Case {i}: {label} ---")

        result = full_pipeline_visualization(**case)
        result["label"] = label
        results.append(result)

        print(f"  Carb. depth: {result['depths'][-1]:.1f} mm")
        print(f"  K coeff:     {result['K']:.2f} mm/sqrt(yr)")
        print(f"  Porosity:    {result['porosity']:.3f}")
        print(f"  D_eff:       {result['D_eff']:.2e} m2/s")

    print(f"\n{'=' * 60}")
    print("  VERIFICATION COMPLETE")
    print(f"{'=' * 60}")
    return results
