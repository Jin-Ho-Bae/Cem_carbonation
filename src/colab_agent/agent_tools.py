"""
LangChain Agent Tools -- Wraps computation engines as callable tools.

Each tool is a @tool-decorated function that LangChain agents can invoke.
Updated for:
- All SCMs (fly ash, metakaolin, silica fume, GGBS, limestone, nano-silica, etc.)
- Multiple diffusion models (user-selectable)
- 6-level micromechanics (nano to macro)
- Civil structure types (beam, column, beam-column) with diverse load conditions
- MD nanoscale fallback
"""

import json
import numpy as np
from typing import Optional

from langchain_core.tools import tool

from .chemistry_engine import (
    PhaseEvolution, EnvironmentCondition, CEMENT_SYSTEMS,
    MOLAR_VOLUMES, SCM_DATABASE, BASE_CEMENTS,
    build_cement_system, get_available_scms, get_available_base_cements,
)
from .micromechanics_engine import (
    compute_effective_properties, level_I, level_II, level_III,
    level_I_md, level_II_csh_variant, level_III_csh_matrix,
    level_IV_paste, level_V_mortar, level_VI_concrete,
    crack_evolution, ELASTIC_PROPERTIES,
)
from .transport_engine import (
    CarbonationTransportModel, DiffusionSolver1D, PoreStructure,
    millington_quirk, compute_D_eff, list_diffusion_models,
    find_carbonation_depth, sqrt_t_fit,
)
from .md_engine import MDEngine
from .gems_engine import (
    GEMSCarbonationEngine, CEMDATA18_PHASES, CSHModel,
    compute_gems_carbonation, list_cemdata18_phases, get_cemdata18_phase,
    compute_initial_hydration_from_xrf, REFERENCE_XRF, bogue_calculation,
)
from .coupled_carbonation import compute_coupled_profile, run_coupled_reference

# Self-correction system integration
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.self_correction import get_self_correction

_sc = get_self_correction()


# ══════════════════════ Chemistry Tools ══════════════════════

@tool
def compute_phase_assemblage(
    base_cement: str = "OPC",
    scm_mix: str = "{}",
    wc_ratio: float = 0.5,
    exposure_type: str = "natural_carbonation",
    CO2_ppm: float = 400,
    temperature_C: float = 20.0,
    relative_humidity: float = 0.65,
    duration_days: float = 365.0,
    sulfate_mol_L: float = 0.0,
) -> str:
    """
    Compute phase assemblage evolution under carbonation or sulfate attack.

    base_cement: "OPC", "CSA", or "white_cement"
    scm_mix: JSON string of SCM replacement ratios, e.g. '{"fly_ash_F": 0.25, "silica_fume": 0.05}'.
             Available SCMs: fly_ash_F, fly_ash_C, metakaolin, silica_fume, GGBS,
             limestone_filler, natural_pozzolan, nano_silica, rice_husk_ash, calcined_clay.
    exposure_type: natural_carbonation, accelerated_carbonation, sulfate_attack, weathering

    Returns phase volume fractions, pH, and carbonation degree.
    """
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix

    env = EnvironmentCondition(
        CO2_ppm=CO2_ppm,
        temperature_K=temperature_C + 273.15,
        relative_humidity=relative_humidity,
        sulfate_concentration=sulfate_mol_L,
        exposure_type=exposure_type,
    )

    pe = PhaseEvolution(cement_type=base_cement, env=env, scm_mix=scm if scm else None)

    CO2_mol_m3 = CO2_ppm * 1e-6 * 101325 / (8.314 * env.temperature_K)
    t_seconds = duration_days * 86400

    sol = pe.evolve(
        t_span=(0, t_seconds),
        C_CO2_func=CO2_mol_m3,
        S_func=relative_humidity,
        T_func=env.temperature_K,
    )

    alpha_CH_final = float(sol.y[0, -1])
    alpha_CSH_final = float(sol.y[1, -1])
    phi_final = pe.get_volume_fractions(alpha_CH_final, alpha_CSH_final)
    pH_final = pe.pore_solution_pH(alpha_CH_final, alpha_CSH_final)
    carb_deg = pe.carbonation_degree(alpha_CH_final, alpha_CSH_final)

    result = {
        "blend": pe.get_blend_description(),
        "base_cement": base_cement,
        "scm_mix": scm,
        "exposure": exposure_type,
        "duration_days": duration_days,
        "alpha_CH": round(alpha_CH_final, 4),
        "alpha_CSH": round(alpha_CSH_final, 4),
        "carbonation_degree": round(carb_deg, 4),
        "pH": round(pH_final, 2),
        "volume_fractions": {k: round(v, 4) for k, v in phi_final.items()},
        "initial_phases": {k: round(v, 4) for k, v in pe.phi0.items()},
    }
    return json.dumps(result, indent=2)


@tool
def get_available_cement_systems() -> str:
    """
    List all available base cements and SCMs (supplementary cementitious materials).
    Use this to help the user choose materials for their cement blend.
    """
    result = {
        "base_cements": get_available_base_cements(),
        "available_SCMs": {name: data["description"]
                          for name, data in SCM_DATABASE.items()},
        "pre_built_systems": [k for k, v in CEMENT_SYSTEMS.items() if v is not None],
        "usage_example": {
            "base_cement": "OPC",
            "scm_mix": {"fly_ash_F": 0.25, "silica_fume": 0.05},
            "description": "OPC + 25% Class F fly ash + 5% silica fume",
        },
    }
    return json.dumps(result, indent=2)


# ══════════════════════ Transport Tools ══════════════════════

@tool
def compute_pore_structure(
    wc_ratio: float = 0.5,
    alpha_hydration: float = 0.85,
    scm_mix: str = "{}",
) -> str:
    """
    Compute pore structure from w/c ratio and raw material composition.
    Uses Powers model with SCM corrections.

    Returns total porosity, capillary porosity, gel porosity,
    critical pore diameter, tortuosity.
    """
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix
    ps = PoreStructure(wc_ratio, alpha_hydration, scm)

    result = {
        "wc_ratio": wc_ratio,
        "alpha_hydration": alpha_hydration,
        "total_porosity": round(ps.total_porosity(), 4),
        "capillary_porosity": round(ps.capillary_porosity(), 4),
        "gel_porosity": round(ps.gel_porosity(), 4),
        "critical_pore_diameter_nm": round(ps.critical_pore_diameter_nm(), 1),
        "tortuosity": round(ps.tortuosity(), 3),
    }
    return json.dumps(result, indent=2)


@tool
def compute_carbonation_depth(
    base_cement: str = "OPC",
    scm_mix: str = "{}",
    wc_ratio: float = 0.5,
    CO2_ppm: float = 400,
    temperature_C: float = 20.0,
    relative_humidity: float = 0.65,
    duration_days: float = 365.0,
    specimen_length_mm: float = 50.0,
    n_grid: int = 50,
    diffusion_model: str = "millington_quirk",
) -> str:
    """
    Compute CO2 carbonation depth vs time using modified Fick's 2nd law.

    diffusion_model: select the effective diffusivity model:
      - "standard"         : D_eff = D0 * phi * (1-S), simplest
      - "millington_quirk" : D_eff = D0 * phi^1.74 * (1-S)^3.2 (default)
      - "papadakis"        : Papadakis et al. (1991) empirical
      - "ceb_fip"          : CEB-FIP Model Code approach
      - "fib_mc2010"       : fib Model Code 2010 / DuraCrete with aging
      - "eshelby_mt"       : Mori-Tanaka effective D from inclusion theory

    Returns carbonation depth at specified duration, intermediate times, and K coefficient.
    """
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix

    L = specimen_length_mm / 1000
    ps = PoreStructure(wc_ratio, 0.85, scm)
    porosity = ps.total_porosity()
    D_CO2 = 1.6e-5
    T_K = temperature_C + 273.15
    CO2_mol = CO2_ppm * 1e-6 * 101325 / (8.314 * T_K)
    t_final = duration_days * 86400

    D_eff = compute_D_eff(
        diffusion_model, D_CO2, porosity, relative_humidity,
        wc_ratio=wc_ratio, cement_type=base_cement,
        temperature_K=T_K, age_days=duration_days,
    )

    model = CarbonationTransportModel(
        L=L, nx=n_grid, D_CO2_air=D_CO2,
        porosity=porosity, saturation=relative_humidity,
        C_env_CO2=CO2_mol,
        diffusion_model=diffusion_model,
        wc_ratio=wc_ratio, cement_type=base_cement,
    )

    save_days = [7, 14, 28, 56, 91, 182, 365, 730, 1825, 3650]
    save_times = [d * 86400 for d in save_days if d <= duration_days]
    if duration_days not in save_days:
        save_times.append(t_final)
    save_times = sorted(set(save_times))

    # Crank-Nicolson is unconditionally stable: use large dt for speed
    dt = t_final / 200  # ~200 steps, sufficient accuracy for CN
    results = model.solve(t_final, dt, save_times=save_times)

    depths = []
    for t, profile in zip(results["t"], results["profiles"]):
        d = find_carbonation_depth(profile, results["x"], threshold=0.5)
        depths.append({"time_days": round(t / 86400, 1), "depth_mm": round(d * 1000, 3)})

    times_arr = np.array([d["time_days"] for d in depths])
    depths_arr = np.array([d["depth_mm"] for d in depths])
    K = sqrt_t_fit(depths_arr, times_arr * 365.25) if len(times_arr) > 1 else 0

    return json.dumps({
        "diffusion_model": diffusion_model,
        "base_cement": base_cement,
        "wc_ratio": wc_ratio,
        "CO2_ppm": CO2_ppm,
        "D_eff_m2s": f"{D_eff:.3e}",
        "porosity": round(porosity, 3),
        "K_mm_per_sqrt_year": round(K, 3),
        "depth_profile": depths,
    }, indent=2)


@tool
def list_available_diffusion_models() -> str:
    """List all available modified Fick's 2nd law diffusion models for transport analysis."""
    models = list_diffusion_models()
    return json.dumps(models, indent=2)


# ══════════════════════ Micromechanics Tools ══════════════════════

@tool
def compute_mechanical_properties(
    phi_CSH: float = 0.40,
    phi_CH: float = 0.20,
    phi_CaCO3: float = 0.0,
    phi_porosity: float = 0.25,
    phi_SiO2_gel: float = 0.0,
    level: str = "paste",
    phi_sand: float = 0.3,
    phi_aggregate: float = 0.45,
    phi_crack: float = 0.005,
    csh_model: str = "jennite",
    agg_type: str = "limestone_agg",
    fibers: str = "[]",
) -> str:
    """
    Compute effective mechanical properties via 6-level Eshelby-Mori-Tanaka homogenization.

    6 levels: MD crystal -> CSH variants -> CSH matrix -> Paste -> Mortar -> Concrete

    level: "paste" (Level IV), "mortar" (Level V), "concrete" (Level VI)
    csh_model: "jennite" or "tobermorite_14A" (Level I MD crystal)
    agg_type: "limestone_agg", "granite_agg", "basalt_agg", "sandstone_agg"
    fibers: JSON array, e.g. '[{"type":"steel_fiber","phi":0.02,"aspect_ratio":60}]'

    Returns E, nu, kappa, mu at the requested scale.
    """
    phi = {
        "CSH": phi_CSH, "CH": phi_CH, "CaCO3": phi_CaCO3,
        "porosity": phi_porosity, "SiO2_gel": phi_SiO2_gel,
    }
    fiber_list = json.loads(fibers) if isinstance(fibers, str) else fibers

    props = compute_effective_properties(
        phi, level=level, phi_sand=phi_sand, phi_agg=phi_aggregate,
        phi_crack=phi_crack, csh_model=csh_model, agg_type=agg_type,
        fibers=fiber_list if fiber_list else None,
    )

    result = {
        "level": level,
        "homogenization_scheme": "6-level Eshelby-Mori-Tanaka",
        "csh_model": csh_model,
        "E_GPa": round(props["E"] / 1e9, 3),
        "nu": round(props["nu"], 4),
        "kappa_GPa": round(props["kappa"] / 1e9, 3),
        "mu_GPa": round(props["mu"] / 1e9, 3),
        "input_phases": {k: round(v, 4) for k, v in phi.items()},
    }
    return json.dumps(result, indent=2)


@tool
def compute_properties_at_carbonation_depths(
    base_cement: str = "OPC",
    scm_mix: str = "{}",
    n_points: int = 10,
    level: str = "paste",
    phi_sand: float = 0.3,
    phi_aggregate: float = 0.45,
    csh_model: str = "jennite",
) -> str:
    """
    Compute mechanical properties at different carbonation degrees (0 to 1).
    Useful for generating property profiles for FEM analysis.
    """
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix
    pe = PhaseEvolution(cement_type=base_cement, scm_mix=scm if scm else None)
    results = []

    for i in range(n_points + 1):
        alpha = i / n_points
        alpha_CH = min(alpha * 1.2, 1.0)
        alpha_CSH = max(0, (alpha - 0.4) / 0.6) if alpha > 0.4 else 0.0

        phi = pe.get_volume_fractions(alpha_CH, alpha_CSH)
        props = compute_effective_properties(
            phi, level=level, phi_sand=phi_sand, phi_agg=phi_aggregate,
            csh_model=csh_model,
        )

        results.append({
            "carbonation_degree": round(alpha, 2),
            "alpha_CH": round(alpha_CH, 3),
            "alpha_CSH": round(alpha_CSH, 3),
            "E_GPa": round(props["E"] / 1e9, 3),
            "nu": round(props["nu"], 4),
        })

    return json.dumps(results, indent=2)


# ══════════════════════ FEM Tools ══════════════════════

@tool
def run_fem_analysis(
    structure_type: str = "beam",
    length_m: float = 0.5,
    height_m: float = 0.1,
    support: str = "cantilever",
    carbonation_depth_mm: float = 10.0,
    E_carbonated_GPa: float = 35.0,
    nu_carbonated: float = 0.22,
    E_neat_GPa: float = 25.0,
    nu_neat: float = 0.20,
    load_type: str = "tension",
    load_MPa: float = 1.0,
    mesh_density: int = 30,
) -> str:
    """
    Run FEniCS FEM analysis with carbonation-partitioned material properties.

    structure_type: "beam", "column", "beam_column", "slab", "wall"
    support: "cantilever", "simply_supported", "fixed_fixed", "pinned_roller"
    load_type: "tension", "compression", "shear", "bending", "pressure", "combined"
    load_MPa: magnitude in MPa

    The mesh is partitioned by carbonation depth: carbonated zones near surfaces
    get different E/nu from the neat cement core.
    Requires FEniCS (auto-installed in Colab).
    """
    from .fem_engine import CarbonationFEMModel

    model = CarbonationFEMModel(
        geometry={
            "type": structure_type, "length": length_m,
            "height": height_m, "support": support,
        },
        carbonation_depth=carbonation_depth_mm / 1000,
        E_carbonated=E_carbonated_GPa * 1e9,
        nu_carbonated=nu_carbonated,
        E_neat=E_neat_GPa * 1e9,
        nu_neat=nu_neat,
    )

    load = {"type": load_type, "magnitude": load_MPa * 1e6}
    results = model.solve(load=load, mesh_density=mesh_density)

    results.pop("dolfin_objects", None)
    results.pop("mesh_coords", None)
    results.pop("displacement", None)

    return json.dumps({
        "structure": structure_type,
        "support": support,
        "dimensions_m": f"{length_m} x {height_m}",
        "carbonation_depth_mm": carbonation_depth_mm,
        "E_carbonated_GPa": E_carbonated_GPa,
        "E_neat_GPa": E_neat_GPa,
        "load_type": load_type,
        "load_MPa": load_MPa,
        "max_displacement_mm": round(results["max_displacement"] * 1000, 6),
        "von_mises_max_MPa": round(results["von_mises_max"] / 1e6, 3),
        "von_mises_mean_MPa": round(results["von_mises_mean"] / 1e6, 3),
        "n_carbonated_elements": results["n_carbonated_cells"],
        "n_neat_elements": results["n_neat_cells"],
    }, indent=2)


@tool
def list_fem_options() -> str:
    """List available structure types, load types, and support conditions for FEM analysis."""
    from .fem_engine import list_structure_types, list_load_types
    return json.dumps({
        "structure_types": list_structure_types(),
        "load_types": list_load_types(),
        "supports": ["cantilever", "simply_supported", "fixed_fixed", "pinned_roller"],
    }, indent=2)


# ══════════════════════ MD Tools ══════════════════════

@tool
def get_md_properties(
    phase: str = "portlandite",
    property_type: str = "elastic",
    temperature_C: float = 25.0,
) -> str:
    """
    Get material properties from MD simulation or reference database.
    Fallback for nanoscale properties when experimental data is unavailable.

    Supported phases: portlandite, calcite, tobermorite_14A, tobermorite_11A,
    jennite, ettringite, vaterite, aragonite, monosulfate, gypsum, quartz,
    clinker_C3S, clinker_C2S
    Property types: elastic, diffusion
    """
    md = MDEngine()
    if property_type == "elastic":
        result = md.get_elastic_properties(phase, temperature_C + 273.15)
    elif property_type == "diffusion":
        result = md.get_diffusion_coefficient(
            species="CO2", medium=phase, temperature=temperature_C + 273.15,
        )
    else:
        return json.dumps({"error": f"Unknown property type: {property_type}"})

    return json.dumps({
        "phase": phase, "property": result.property_name,
        "value": result.value, "unit": result.unit,
        "method": result.method, "confidence": result.confidence,
        "temperature_C": temperature_C,
    }, indent=2)


@tool
def generate_lammps_script(
    calculation_type: str = "elastic",
    structure: str = "portlandite",
    potential: str = "reaxff",
    temperature_C: float = 25.0,
) -> str:
    """
    Generate LAMMPS input script for MD simulation.
    Use when properties are not in the reference database.
    """
    from .md_engine import generate_elastic_constants_script, generate_diffusion_script
    T = temperature_C + 273.15
    struct_file = f"{structure}.data"
    pot_file = f"ffield.{potential}"
    if calculation_type == "elastic":
        script = generate_elastic_constants_script(struct_file, pot_file, potential, T)
    else:
        script = generate_diffusion_script(struct_file, pot_file, "CO2", T)
    return json.dumps({
        "calculation_type": calculation_type, "structure": structure,
        "script": script,
        "instructions": "Save script as 'in.lammps', provide structure file and potential, then run: lmp -in in.lammps",
    }, indent=2)


# ══════════════════════ Integrated Pipeline ══════════════════════

@tool
def run_full_chemo_transport_mechanical(
    base_cement: str = "OPC",
    scm_mix: str = "{}",
    wc_ratio: float = 0.5,
    exposure_type: str = "natural_carbonation",
    CO2_ppm: float = 400,
    temperature_C: float = 20.0,
    relative_humidity: float = 0.65,
    duration_days: float = 365.0,
    level: str = "concrete",
    phi_sand: float = 0.3,
    phi_aggregate: float = 0.45,
    diffusion_model: str = "millington_quirk",
    csh_model: str = "jennite",
) -> str:
    """
    Run complete chemo-transport-micromechanical analysis pipeline:
    1. Chemistry: phase assemblage evolution (with SCMs)
    2. Transport: carbonation depth via selected Fick's law model
    3. Micromechanics: 6-level effective properties (carbonated vs neat)
    4. Summary with FEM-ready property partitioning
    """
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix

    # Self-correction: read past failure guides
    lessons = _sc.get_lessons_learned("chemo_transport_mechanical")
    if lessons:
        pass  # Lessons available for diagnosis if errors occur

    try:
        return _run_pipeline_inner(
            base_cement, scm, wc_ratio, exposure_type, CO2_ppm,
            temperature_C, relative_humidity, duration_days, level,
            phi_sand, phi_aggregate, diffusion_model, csh_model,
        )
    except Exception as e:
        _sc.log_failure(
            analysis_type="chemo_transport_mechanical",
            error=str(e),
            context={
                "base_cement": base_cement, "scm_mix": str(scm),
                "wc_ratio": wc_ratio, "diffusion_model": diffusion_model,
                "level": level, "duration_days": duration_days,
            },
            module="agent_tools",
            function="run_full_chemo_transport_mechanical",
        )
        raise


def _run_pipeline_inner(base_cement, scm, wc_ratio, exposure_type, CO2_ppm,
                        temperature_C, relative_humidity, duration_days, level,
                        phi_sand, phi_aggregate, diffusion_model, csh_model):
    """Inner pipeline logic separated for self-correction wrapping."""
    # Step 1: Chemistry
    env = EnvironmentCondition(
        CO2_ppm=CO2_ppm, temperature_K=temperature_C + 273.15,
        relative_humidity=relative_humidity, exposure_type=exposure_type,
    )
    pe = PhaseEvolution(cement_type=base_cement, env=env,
                        scm_mix=scm if scm else None)
    CO2_mol = CO2_ppm * 1e-6 * 101325 / (8.314 * env.temperature_K)
    t_sec = duration_days * 86400

    sol = pe.evolve((0, t_sec), CO2_mol, relative_humidity, env.temperature_K)
    a_CH = float(sol.y[0, -1])
    a_CSH = float(sol.y[1, -1])
    phi_carb = pe.get_volume_fractions(a_CH, a_CSH)
    phi_neat = pe.get_volume_fractions(0, 0)

    # Step 2: Transport
    ps = PoreStructure(wc_ratio, 0.85, scm)
    porosity = ps.total_porosity()
    D_eff = compute_D_eff(diffusion_model, 1.6e-5, porosity, relative_humidity,
                          wc_ratio=wc_ratio, cement_type=base_cement)
    K_approx = np.sqrt(2 * D_eff * CO2_mol * t_sec /
                        max(phi_neat.get("CH", 0.2) / MOLAR_VOLUMES["CH"], 1e-10))
    x_carb_mm = K_approx * 1000

    # Step 3: Micromechanics
    props_carb = compute_effective_properties(
        phi_carb, level, phi_sand, phi_aggregate, csh_model=csh_model)
    props_neat = compute_effective_properties(
        phi_neat, level, phi_sand, phi_aggregate, csh_model=csh_model)

    result = {
        "blend": pe.get_blend_description(),
        "exposure": exposure_type,
        "duration_days": duration_days,
        "diffusion_model": diffusion_model,
        "chemistry": {
            "carbonation_degree": round(pe.carbonation_degree(a_CH, a_CSH), 4),
            "pH": round(pe.pore_solution_pH(a_CH, a_CSH), 2),
            "alpha_CH": round(a_CH, 4),
            "alpha_CSH": round(a_CSH, 4),
        },
        "transport": {
            "carbonation_depth_mm": round(x_carb_mm, 3),
            "D_eff_m2s": f"{D_eff:.3e}",
            "porosity": round(porosity, 3),
        },
        "micromechanics_carbonated": {
            "E_GPa": round(props_carb["E"] / 1e9, 3),
            "nu": round(props_carb["nu"], 4),
        },
        "micromechanics_neat": {
            "E_GPa": round(props_neat["E"] / 1e9, 3),
            "nu": round(props_neat["nu"], 4),
        },
        "fem_partition": {
            "carbonated_zone_depth_mm": round(x_carb_mm, 3),
            "E_carbonated_GPa": round(props_carb["E"] / 1e9, 3),
            "E_neat_GPa": round(props_neat["E"] / 1e9, 3),
            "property_ratio": round(props_carb["E"] / props_neat["E"], 3) if props_neat["E"] > 0 else 0,
        },
    }
    return json.dumps(result, indent=2)


# ══════════════════════ Thermodynamics Tools (GEMS backend) ══════════════════════
#
# GEMS (cemdata18 Gibbs energy minimization) is NOT exposed as a standalone
# program to the LLM agents. Instead, it serves as the numerical BACKEND that
# these thermodynamics-expert tools call internally. The thermodynamics agent
# sees domain-focused capabilities (initial hydration from XRF, phase evolution
# vs CO2 dose, local equilibrium at a depth, etc.) — never a raw GEMS API.
# ------------------------------------------------------------------------------

@tool
def compute_initial_phase_assemblage(
    xrf_oxides: str = "{}",
    cement_reference: str = "",
    wc_ratio: float = 0.50,
    alpha_hydration: float = 1.0,
    temperature_C: float = 25.0,
) -> str:
    """
    [Thermodynamics] Compute the initial (uncarbonated) phase assemblage of a
    hydrated cement from raw material composition, via internal GEMS (cemdata18)
    Gibbs energy minimization assuming the specified degree of hydration.

    This is the PRIMARY entry point when only the XRF of the cement and the
    water/cement ratio are known. Uses Bogue clinker calculation + stoichiometric
    hydration + Powers porosity model.

    Args
    ----
    xrf_oxides : JSON string of oxide mass %, e.g.
        '{"CaO":63.5,"SiO2":20,"Al2O3":5,"Fe2O3":3,"MgO":1.5,"SO3":3}'
    cement_reference : if xrf_oxides is empty, use a named reference composition.
        Options: "OPC_CEM_I", "OPC_low_C3A", "white_cement", "CSA".
    wc_ratio : water/cement mass ratio
    alpha_hydration : degree of hydration (1.0 = full)
    temperature_C : hydration temperature

    Returns: Bogue clinker, volume fractions of hydrate phases (CH, C-S-H,
    ettringite, monosulfate, AFm, hydrotalcite, ...), porosity, Ca/Si, and
    chemically bound water.
    """
    if isinstance(xrf_oxides, str):
        xrf = json.loads(xrf_oxides) if xrf_oxides.strip() else {}
    else:
        xrf = xrf_oxides
    if not xrf:
        if cement_reference not in REFERENCE_XRF:
            return json.dumps({
                "error": "No xrf_oxides provided and cement_reference not recognized.",
                "available_references": list(REFERENCE_XRF.keys()),
            })
        xrf = REFERENCE_XRF[cement_reference]

    result = compute_initial_hydration_from_xrf(
        xrf, wc_ratio=wc_ratio,
        alpha_hydration=alpha_hydration, temperature_C=temperature_C,
    )
    # Trim to JSON-serializable summary
    return json.dumps({
        "xrf_input": xrf,
        "wc_ratio": wc_ratio,
        "alpha_hydration": alpha_hydration,
        "temperature_C": temperature_C,
        "bogue_clinker_pct": result["clinker_bogue"],
        "phases_volume_frac": {k: round(v, 4) for k, v in result["phases_volume_frac"].items()},
        "porosity_total": round(result["porosity"], 4),
        "capillary_porosity": round(result["capillary_porosity"], 4),
        "gel_porosity": round(result["gel_porosity"], 4),
        "CSH_CaSi": result["CSH_CaSi"],
        "bound_water_g_per_kg": result["bound_water_g_per_kg"],
        "note": "Backend: GEMS (cemdata18 subset, CSHQ solid-solution model)",
    }, indent=2)


@tool
def compute_phase_evolution_vs_CO2(
    xrf_oxides: str = "{}",
    cement_reference: str = "OPC_CEM_I",
    base_cement: str = "OPC",
    scm_mix: str = "{}",
    wc_ratio: float = 0.50,
    temperature_C: float = 25.0,
    CO2_total_mol: float = 1.0,
    n_steps: int = 100,
) -> str:
    """
    [Thermodynamics] Compute the evolution of phase assemblage as CO2 is
    progressively added to the hydrated system. Returns assemblage, pH, Ca/Si,
    porosity, and carbonation degree as a function of cumulative CO2 dose.

    Source of the initial state:
      - if xrf_oxides provided   -> GEMS.from_xrf()  (Gibbs minimization)
      - elif cement_reference     -> REFERENCE_XRF[cement_reference]
      - else                      -> chemistry_engine BASE_CEMENTS + SCMs

    Args
    ----
    xrf_oxides : JSON string of XRF mass % (optional)
    cement_reference : "OPC_CEM_I", "OPC_low_C3A", "white_cement", "CSA"
    base_cement : fallback when no XRF/reference: "OPC", "CSA", "white_cement"
    scm_mix : JSON string, e.g. '{"fly_ash_F":0.30}' (only when using base_cement path)
    CO2_total_mol : total mol CO2 to add per L of paste
    n_steps : number of equilibrium steps

    Returns: initial + intermediate (10 snapshots) + final phase assemblage,
    pH, Ca/Si, porosity, carbonation degree at each CO2 dose point.
    """
    if isinstance(xrf_oxides, str):
        xrf = json.loads(xrf_oxides) if xrf_oxides.strip() else {}
    else:
        xrf = xrf_oxides
    scm = json.loads(scm_mix) if isinstance(scm_mix, str) else scm_mix

    if xrf:
        engine = GEMSCarbonationEngine.from_xrf(
            xrf, wc_ratio=wc_ratio, temperature_C=temperature_C,
        )
        origin = "XRF"
    elif cement_reference in REFERENCE_XRF:
        engine = GEMSCarbonationEngine.from_xrf(
            REFERENCE_XRF[cement_reference],
            wc_ratio=wc_ratio, temperature_C=temperature_C,
        )
        origin = f"reference/{cement_reference}"
    else:
        engine = GEMSCarbonationEngine(
            cement_type=base_cement, scm_mix=scm or None,
            wc_ratio=wc_ratio, temperature_C=temperature_C,
        )
        origin = f"{base_cement}"
        if scm:
            origin += "+" + "+".join(f"{int(v*100)}%{k}" for k, v in scm.items())

    engine.phases["_init_CH_mol"] = engine.phases.get("Portlandite", 0.0)
    states = engine.run_carbonation(CO2_total_mol, n_steps)

    # Sample 10 evenly spaced snapshots + final
    idxs = list(np.linspace(0, len(states) - 1, 11).astype(int))
    evolution = []
    for idx in idxs:
        s = states[idx]
        evolution.append({
            "CO2_mol": round(s.CO2_added_total, 4),
            "pH": round(s.pH, 2),
            "CaSi_CSH": round(s.CaSi_CSH, 3),
            "porosity": round(s.porosity, 4),
            "carbonation_degree": round(s.carbonation_degree, 4),
            "top_phases": {k: round(v, 4)
                            for k, v in sorted(s.phase_volume_fracs.items(),
                                               key=lambda kv: -kv[1])[:6]},
        })

    # Key milestones
    ch_depletion = next((round(s.CO2_added_total, 4)
                          for s in states
                          if s.phase_amounts.get("Portlandite", 1) < 1e-6), None)
    pH9 = next((round(s.CO2_added_total, 4) for s in states if s.pH < 9.0), None)

    return json.dumps({
        "source": origin,
        "wc_ratio": wc_ratio,
        "temperature_C": temperature_C,
        "CO2_total_mol": CO2_total_mol,
        "n_steps": n_steps,
        "milestones": {
            "CO2_at_CH_depletion": ch_depletion,
            "CO2_at_pH_9": pH9,
            "max_carbonation_capacity": round(engine.get_max_carbonation_capacity(), 4),
        },
        "final": engine.summary(),
        "evolution": evolution,
        "note": "Backend: GEMS (cemdata18 subset). Each step = Gibbs reequilibration.",
    }, indent=2)


@tool
def compute_local_phase_assemblage_at_dose(
    CO2_dose_mol: float = 0.5,
    xrf_oxides: str = "{}",
    cement_reference: str = "OPC_CEM_I",
    wc_ratio: float = 0.50,
    temperature_C: float = 25.0,
) -> str:
    """
    [Thermodynamics] Compute the LOCAL phase assemblage at a given cumulative
    CO2 dose (mol per L of paste). This is the per-point equilibrium used by
    the transport-coupled workflow: given CO3^2- supplied by diffusion at a
    specific depth and time, return the phase assemblage, pH, porosity, and
    Ca/Si at that point.

    Internally runs one GEMS equilibrium from the initial hydrated state.

    Use inside a loop over (depth, time) from the transport solver, or directly
    to answer "what does the system look like after absorbing X mol of CO2?".

    Args
    ----
    CO2_dose_mol : total cumulative CO2 delivered [mol per L of paste]
    xrf_oxides   : JSON XRF (optional, takes precedence over cement_reference)
    cement_reference : reference cement when xrf_oxides not given
    wc_ratio, temperature_C : material / environment
    """
    if isinstance(xrf_oxides, str):
        xrf = json.loads(xrf_oxides) if xrf_oxides.strip() else {}
    else:
        xrf = xrf_oxides
    if not xrf:
        if cement_reference not in REFERENCE_XRF:
            return json.dumps({"error": "Need xrf_oxides or valid cement_reference."})
        xrf = REFERENCE_XRF[cement_reference]

    engine = GEMSCarbonationEngine.from_xrf(
        xrf, wc_ratio=wc_ratio, temperature_C=temperature_C,
    )
    engine.phases["_init_CH_mol"] = engine.phases.get("Portlandite", 0.0)
    # Add dose in 20 small steps to maintain reasonable thermodynamic path
    n_step = 20
    step = CO2_dose_mol / n_step
    for _ in range(n_step):
        engine.equilibrate_step(step)
    state = engine._get_state()

    return json.dumps({
        "CO2_dose_mol": CO2_dose_mol,
        "source": "XRF" if isinstance(xrf_oxides, str) and xrf_oxides.strip() else cement_reference,
        "pH": round(state.pH, 2),
        "porosity": round(state.porosity, 4),
        "CaSi_CSH": round(state.CaSi_CSH, 3),
        "carbonation_degree": round(state.carbonation_degree, 4),
        "phase_volume_fracs": {k: round(v, 4)
                                for k, v in state.phase_volume_fracs.items()
                                if v > 1e-4},
        "aqueous_mol_L": {k: f"{v:.3e}" for k, v in state.aqueous.items()},
        "note": "Backend: GEMS local equilibrium (cemdata18).",
    }, indent=2)


@tool
def list_thermodynamic_phase_database() -> str:
    """
    [Thermodynamics] List the phases in the internal GEMS cemdata18
    (carbonation subset): name, formula, log Ksp at 25 C, molar volume,
    molar mass, category. Reference database used by all thermodynamics tools.
    """
    phases = []
    for name, pd in CEMDATA18_PHASES.items():
        phases.append({
            "name": name,
            "formula": pd.formula,
            "log_Ksp_25C": pd.log_Ksp_25C,
            "molar_volume_m3_mol": f"{pd.molar_volume:.2e}",
            "molar_mass_g_mol": pd.molar_mass,
            "category": pd.category,
        })
    return json.dumps({
        "database": "cemdata18 (carbonation-focused subset)",
        "n_phases": len(phases), "phases": phases,
        "references_available": list(REFERENCE_XRF.keys()),
    }, indent=2)


@tool
def compute_csh_decalcification_profile(
    initial_CaSi: float = 1.67,
    CO2_steps: int = 50,
    CO2_per_step_mol: float = 0.01,
) -> str:
    """
    [Thermodynamics] Track C-S-H decalcification (CSHQ, Kulik 2011) as CO2
    progressively removes Ca from C-S-H, converting it to CaCO3 + SiO2(am).

    Returns Ca/Si evolution and log activity of Ca/Si at each step.
    Useful for diagnosing how far along the decalcification curve a given
    carbonation state is.
    """
    csh = CSHModel(initial_CaSi=initial_CaSi, amount_mol=1.0)
    evolution = []
    for i in range(CO2_steps):
        csh.decalcify(CO2_per_step_mol)
        log_a_Ca, log_a_Si = csh.get_solubility()
        evolution.append({
            "step": i + 1,
            "CO2_added": round((i + 1) * CO2_per_step_mol, 4),
            "CaSi": round(csh.CaSi, 4),
            "amount_mol": round(csh.amount, 4),
            "log_a_Ca": round(log_a_Ca, 3),
            "log_a_Si": round(log_a_Si, 3),
            "carbonation_degree": round(csh.carbonation_degree, 4),
        })
        if csh.amount <= 0:
            break

    return json.dumps({
        "model": "CSHQ (Kulik 2011, cemdata18)",
        "initial_CaSi": initial_CaSi,
        "final_CaSi": round(csh.CaSi, 4),
        "final_carbonation_degree": round(csh.carbonation_degree, 4),
        "fully_dissolved": csh.amount <= 0,
        "n_steps": len(evolution),
        "evolution_sampled": evolution[::max(1, len(evolution) // 10)],
    }, indent=2)


# ══════════════════════ Coupled Engineering Tool ══════════════════════

@tool
def compute_coupled_chemo_transport_mechanical_profile(
    xrf_oxides: str = "{}",
    cement_reference: str = "OPC_CEM_I",
    wc_ratio: float = 0.50,
    alpha_hydration: float = 1.0,
    temperature_C: float = 25.0,
    C_env_CO2_mol_m3: float = 0.016,
    diffusion_model: str = "fib_mc2010",
    L_total_mm: float = 100.0,
    nx: int = 20,
    saturation: float = 0.65,
    save_times_years: str = "[0.5,1,5,10,25,50]",
    level: str = "paste",
) -> str:
    """
    [Engineering] Run the FULL coupled chemo-transport-micromechanical pipeline:

       XRF -> GEMS full hydration (initial assemblage, porosity)
            -> 1D CO2 diffusion (transport engine)
            -> per-(depth,time) GEMS equilibrium (local phase assemblage)
            -> 6-level Eshelby-Mori-Tanaka homogenization
            -> E(x,t), porosity(x,t), pH(x,t), carbonation depth x_c(t)

    Use this for "mid- to long-term durability" questions where the user wants
    BOTH the chemical state and the effective stiffness as a function of
    depth and exposure time in one frame.

    Args
    ----
    xrf_oxides : JSON XRF oxide mass %. Empty -> use cement_reference.
    cement_reference : "OPC_CEM_I", "OPC_low_C3A", "white_cement", "CSA"
    C_env_CO2_mol_m3 : 0.016 atmospheric, 2.0 accelerated 5%, 16.0 extreme 40%
    diffusion_model : standard | millington_quirk | papadakis | ceb_fip |
                      fib_mc2010 | eshelby_mt
    L_total_mm : analysis depth into material (mm)
    save_times_years : JSON list of years at which to output snapshots
    level : "paste" | "mortar" | "concrete"

    Returns a compact JSON summary: carbonation depth vs time, E(surface),
    E(mid-depth), E at each save time, initial vs final assemblage, K coef.
    """
    if isinstance(xrf_oxides, str):
        xrf = json.loads(xrf_oxides) if xrf_oxides.strip() else {}
    else:
        xrf = xrf_oxides
    if not xrf:
        if cement_reference not in REFERENCE_XRF:
            return json.dumps({
                "error": "Need xrf_oxides or valid cement_reference.",
                "available": list(REFERENCE_XRF.keys())})
        xrf = REFERENCE_XRF[cement_reference]

    times_years = json.loads(save_times_years) if isinstance(save_times_years, str) \
        else list(save_times_years)

    # Self-correction logging
    try:
        r = compute_coupled_profile(
            xrf_oxides=xrf,
            wc_ratio=wc_ratio,
            alpha_hydration=alpha_hydration,
            temperature_C=temperature_C,
            C_env_CO2=C_env_CO2_mol_m3,
            diffusion_model=diffusion_model,
            L_total=L_total_mm / 1000.0,
            nx=nx,
            saturation=saturation,
            save_times_years=times_years,
            level=level,
        )
    except Exception as e:
        _sc.log_failure(
            analysis_type="coupled_chemo_transport_mechanical",
            error=str(e),
            context={"xrf": xrf, "wc_ratio": wc_ratio,
                     "C_env": C_env_CO2_mol_m3, "diffusion_model": diffusion_model},
            module="agent_tools",
            function="compute_coupled_chemo_transport_mechanical_profile",
        )
        raise

    nt = len(r.times_years)
    mid = r.E_field.shape[1] // 2
    time_series = []
    for i in range(nt):
        time_series.append({
            "time_years": round(float(r.times_years[i]), 3),
            "carbonation_depth_mm": round(float(r.carbonation_depth[i]) * 1000, 3),
            "E_surface_GPa": round(float(r.E_field[i, 0]) / 1e9, 3),
            "E_middepth_GPa": round(float(r.E_field[i, mid]) / 1e9, 3),
            "E_core_GPa": round(float(r.E_field[i, -1]) / 1e9, 3),
            "pH_surface": round(float(r.pH_field[i, 0]), 2),
            "pH_core": round(float(r.pH_field[i, -1]), 2),
            "porosity_surface": round(float(r.porosity_field[i, 0]), 4),
        })

    return json.dumps({
        "source": "XRF" if xrf_oxides and xrf_oxides != "{}" else cement_reference,
        "xrf_input": xrf,
        "wc_ratio": wc_ratio,
        "C_env_CO2_mol_m3": C_env_CO2_mol_m3,
        "diffusion_model": diffusion_model,
        "level": level,
        "initial_E_GPa": round(r.initial_E / 1e9, 3),
        "K_mm_per_sqrt_yr": round(r.meta["K_mm_per_sqrt_yr"], 3),
        "D_eff_m2_s": f"{r.meta['D_eff']:.3e}",
        "max_CO2_capacity_mol_L": round(r.meta["max_CO2_capacity"], 3),
        "time_series": time_series,
        "note": ("Backend pipeline: GEMS(cemdata18) -> transport_engine -> "
                 "GEMS local equilibrium -> 6-level Eshelby-Mori-Tanaka."),
    }, indent=2)


def get_chemistry_tools():
    """Chemistry domain tools: phase assemblage, cement systems."""
    return [compute_phase_assemblage, get_available_cement_systems]


def get_transport_tools():
    """Transport domain tools: pore structure, carbonation depth, diffusion models."""
    return [compute_pore_structure, compute_carbonation_depth, list_available_diffusion_models]


def get_micromechanics_tools():
    """Micromechanics domain tools: 6-level homogenization, property profiles."""
    return [compute_mechanical_properties, compute_properties_at_carbonation_depths]


def get_fem_tools():
    """FEM/Structural domain tools: FEniCS analysis, structure/load options."""
    return [run_fem_analysis, list_fem_options]


def get_md_tools():
    """MD domain tools: nanoscale properties, LAMMPS scripts."""
    return [get_md_properties, generate_lammps_script]


def get_engineering_tools():
    """
    Engineering assessment tools: integrated pipelines with durability evaluation.
    Includes both the legacy chemistry-based pipeline and the new GEMS-coupled
    chemo-transport-micromechanical driver.
    """
    return [
        run_full_chemo_transport_mechanical,
        compute_coupled_chemo_transport_mechanical_profile,
        compute_properties_at_carbonation_depths,
    ]


def get_thermodynamics_tools():
    """
    Thermodynamics domain tools -- all backed by GEMS (cemdata18) INTERNALLY.
    The LLM agent sees thermodynamics-level capabilities, not raw GEMS calls.
      - compute_initial_phase_assemblage           (XRF -> hydrated state)
      - compute_phase_evolution_vs_CO2             (assemblage vs mol CO2)
      - compute_local_phase_assemblage_at_dose     (local equilibrium at x,t)
      - compute_csh_decalcification_profile        (CSHQ Ca/Si tracking)
      - list_thermodynamic_phase_database          (cemdata18 info)
    """
    return [
        compute_initial_phase_assemblage,
        compute_phase_evolution_vs_CO2,
        compute_local_phase_assemblage_at_dose,
        compute_csh_decalcification_profile,
        list_thermodynamic_phase_database,
    ]


def get_all_tools():
    """Return list of all available LangChain tools (RBAC registry)."""
    return [
        # Chemistry (kinetics-based)
        compute_phase_assemblage,
        get_available_cement_systems,
        # Thermodynamics (GEMS-backed domain tools)
        compute_initial_phase_assemblage,
        compute_phase_evolution_vs_CO2,
        compute_local_phase_assemblage_at_dose,
        compute_csh_decalcification_profile,
        list_thermodynamic_phase_database,
        # Transport
        compute_pore_structure,
        compute_carbonation_depth,
        list_available_diffusion_models,
        # Micromechanics
        compute_mechanical_properties,
        compute_properties_at_carbonation_depths,
        # FEM
        run_fem_analysis,
        list_fem_options,
        # MD
        get_md_properties,
        generate_lammps_script,
        # Engineering (integrated pipelines)
        run_full_chemo_transport_mechanical,
        compute_coupled_chemo_transport_mechanical_profile,
    ]
