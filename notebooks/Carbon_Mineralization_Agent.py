# %% [markdown]
# # Carbon Mineralization in Construction Materials — LLM Agent System
#
# **Chemo-Transport-Micromechanical Modeling via Multi-Agent AI**
#
# This notebook implements an LLM-based multi-agent system for comprehensive
# analysis of carbon mineralization in construction materials.
#
# ## Architecture
# ```
# User Query
#     ↓
# Orchestrator Agent (GPT-4o + LangChain)
#     ├── Chemistry Agent → Phase assemblage, reaction kinetics
#     ├── Transport Agent → Fick's 2nd law, carbonation depth
#     ├── Micromechanics Agent → Eshelby-Mori-Tanaka, effective properties
#     ├── FEM Agent → FEniCS structural analysis (partitioned mesh)
#     └── MD Agent → LAMMPS (fallback for missing data)
# ```

# %% [markdown]
# ## 1. Environment Setup

# %%
# ── Install dependencies ──
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["langchain", "langchain-openai", "langchain-core", "openai", "scipy", "numpy", "matplotlib"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# ── Install FEniCS for Colab ──
try:
    import dolfin
    print("FEniCS already installed")
except ImportError:
    print("Installing FEniCS (this takes ~2 minutes)...")
    import os
    os.system('wget "https://fem-on-colab.github.io/releases/fenics-install-release-real.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')
    import dolfin
    print("FEniCS installed successfully")

print("All dependencies ready.")

# %%
# ── Clone/setup project code ──
import os, sys

# If running in Colab, mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = "/content/drive/MyDrive/1 Cement carbonation"
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from colab_agent.chemistry_engine import PhaseEvolution, EnvironmentCondition, CEMENT_SYSTEMS
from colab_agent.micromechanics_engine import compute_effective_properties, level_I, level_II, level_III
from colab_agent.transport_engine import CarbonationTransportModel, DiffusionSolver1D, millington_quirk, find_carbonation_depth
from colab_agent.fem_engine import CarbonationFEMModel
from colab_agent.md_engine import MDEngine
from colab_agent.agent_tools import get_all_tools
from colab_agent.orchestrator import CarbonMineralizationOrchestrator, create_agent

import numpy as np
import matplotlib.pyplot as plt

print(f"Project root: {PROJECT_ROOT}")
print(f"Available cement systems: {list(CEMENT_SYSTEMS.keys())}")

# %% [markdown]
# ## 2. Set OpenAI API Key

# %%
import os
# Option 1: Set directly (replace with your key)
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Option 2: From Colab secrets
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("API key loaded from Colab secrets")
except:
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter OpenAI API Key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    print("API key set")

# %% [markdown]
# ## 3. Direct Engine Usage (No LLM)
#
# Use the computation engines directly for deterministic analysis.

# %% [markdown]
# ### 3.1 Chemistry: Phase Assemblage Evolution

# %%
# Compare OPC vs CSA under natural carbonation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, cement_type in enumerate(["OPC", "CSA"]):
    pe = PhaseEvolution(cement_type=cement_type)
    sol = pe.evolve(
        t_span=(0, 10*365*86400),  # 10 years
        C_CO2_func=16.0,           # natural CO₂
        S_func=0.65,
        T_func=293.15,
    )

    t_years = sol.t / (365 * 86400)

    # Phase evolution
    ax = axes[idx, 0]
    for j in range(len(t_years)):
        phi = pe.get_volume_fractions(sol.y[0,j], sol.y[1,j])
    # Plot final key phases at each time
    phases_over_time = {"CH": [], "CSH": [], "CaCO3": [], "porosity": []}
    for j in range(len(t_years)):
        phi = pe.get_volume_fractions(sol.y[0,j], sol.y[1,j])
        for k in phases_over_time:
            phases_over_time[k].append(phi.get(k, 0))

    for name, vals in phases_over_time.items():
        ax.plot(t_years, vals, label=name, linewidth=2)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Volume fraction")
    ax.set_title(f"{cement_type} — Phase Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # pH evolution
    ax2 = axes[idx, 1]
    pHs = [pe.pore_solution_pH(sol.y[0,j], sol.y[1,j]) for j in range(len(t_years))]
    ax2.plot(t_years, pHs, 'r-', linewidth=2)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("pH")
    ax2.set_title(f"{cement_type} — pH Evolution")
    ax2.set_ylim(7, 14)
    ax2.axhline(y=9.0, color='gray', linestyle='--', label='Phenolphthalein')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "_workspace", "phase_evolution.png"), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3.2 Transport: Carbonation Depth

# %%
fig, ax = plt.subplots(figsize=(10, 6))

for wc in [0.4, 0.5, 0.6]:
    porosity = 0.12 + 0.26 * wc
    D_eff = millington_quirk(1.6e-5, porosity, 0.65)

    model = CarbonationTransportModel(
        L=0.05, nx=200, porosity=porosity, saturation=0.65, C_env_CO2=16.0,
    )

    save_times = [d * 86400 for d in [7, 28, 91, 182, 365, 730, 1825, 3650]]
    dt = min(3600, 0.4 * (0.05/200)**2 / D_eff)
    results = model.solve(3650*86400, dt, save_times=save_times)

    depths = [find_carbonation_depth(p, results["x"]) * 1000
              for p in results["profiles"]]
    times = [t / (365*86400) for t in results["t"]]

    ax.plot(times, depths, 'o-', label=f"w/c = {wc}", linewidth=2, markersize=4)

ax.set_xlabel("Time (years)")
ax.set_ylabel("Carbonation depth (mm)")
ax.set_title("Carbonation Depth vs Time (Natural CO₂)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Micromechanics: Effective Properties vs Carbonation

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
levels = ["paste", "mortar", "concrete"]

for ax, level in zip(axes, levels):
    pe = PhaseEvolution("OPC")
    alphas = np.linspace(0, 1, 50)
    Es, nus = [], []

    for a in alphas:
        a_CH = min(a * 1.2, 1.0)
        a_CSH = max(0, (a - 0.4) / 0.6) if a > 0.4 else 0.0
        phi = pe.get_volume_fractions(a_CH, a_CSH)
        props = compute_effective_properties(phi, level=level)
        Es.append(props["E"] / 1e9)
        nus.append(props["nu"])

    ax.plot(alphas, Es, 'b-', linewidth=2, label="E (GPa)")
    ax2 = ax.twinx()
    ax2.plot(alphas, nus, 'r--', linewidth=2, label="ν")
    ax.set_xlabel("Carbonation degree")
    ax.set_ylabel("E (GPa)", color='b')
    ax2.set_ylabel("Poisson's ratio ν", color='r')
    ax.set_title(f"Level: {level.capitalize()}")
    ax.grid(True, alpha=0.3)

plt.suptitle("Effective Properties vs Carbonation Degree (OPC)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 FEM: Structural Analysis with Partitioned Properties

# %%
# Compute carbonated vs neat properties
pe = PhaseEvolution("OPC")
phi_carb = pe.get_volume_fractions(1.0, 0.5)  # heavily carbonated
phi_neat = pe.get_volume_fractions(0.0, 0.0)  # neat cement

props_carb = compute_effective_properties(phi_carb, "concrete")
props_neat = compute_effective_properties(phi_neat, "concrete")

print(f"Carbonated: E = {props_carb['E']/1e9:.1f} GPa, ν = {props_carb['nu']:.3f}")
print(f"Neat:       E = {props_neat['E']/1e9:.1f} GPa, ν = {props_neat['nu']:.3f}")
print(f"Ratio:      E_carb/E_neat = {props_carb['E']/props_neat['E']:.3f}")

# FEM analysis
fem = CarbonationFEMModel(
    geometry={"type": "beam", "length": 0.5, "height": 0.1},
    carbonation_depth=0.015,  # 15 mm
    E_carbonated=props_carb["E"],
    nu_carbonated=props_carb["nu"],
    E_neat=props_neat["E"],
    nu_neat=props_neat["nu"],
)

results = fem.solve(
    load={"type": "uniaxial", "magnitude": 5e6},
    mesh_density=30,
)

print(f"\nFEM Results:")
print(f"  Max displacement: {results['max_displacement']*1000:.4f} mm")
print(f"  Von Mises max: {results['von_mises_max']/1e6:.2f} MPa")
print(f"  Von Mises mean: {results['von_mises_mean']/1e6:.2f} MPa")
print(f"  Carbonated elements: {results['n_carbonated_cells']}")
print(f"  Neat elements: {results['n_neat_cells']}")

# Plot von Mises
import dolfin as df
plt.figure(figsize=(12, 4))
p = df.plot(results["dolfin_objects"]["von_mises"])
plt.colorbar(p, label="Von Mises Stress (Pa)")
plt.title("Von Mises Stress — Carbonated Beam (15mm depth)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.5 MD: Fallback Properties

# %%
md = MDEngine()

phases_to_check = ["portlandite", "calcite", "tobermorite_14A", "ettringite", "jennite"]

print("Molecular Dynamics Reference Data:")
print(f"{'Phase':<20} {'E (GPa)':<12} {'ν':<10} {'Confidence':<12} {'Method':<15}")
print("-" * 70)

for phase in phases_to_check:
    E_result = md.get_elastic_properties(phase)
    nu_data = md.get_elastic_properties(phase)
    E_val = E_result.value / 1e9 if E_result.value > 0 else "N/A"
    print(f"{phase:<20} {str(E_val):<12} {'—':<10} {E_result.confidence:<12} {E_result.method:<15}")

# %% [markdown]
# ## 4. LLM Agent System

# %%
orchestrator = CarbonMineralizationOrchestrator(model_name="gpt-4o")

# %% [markdown]
# ### 4.1 Single Query Analysis

# %%
result = orchestrator.analyze(
    "Analyze natural carbonation of OPC concrete (w/c=0.5) for 50 years. "
    "Compute phase assemblage, carbonation depth, and effective mechanical "
    "properties at paste and concrete levels."
)

# %% [markdown]
# ### 4.2 Material Comparison

# %%
result = orchestrator.compare_materials(
    materials=["OPC", "CSA", "OPC_FA", "OPC_GGBS"],
    condition="accelerated carbonation at 5% CO₂",
    duration_days=90,
)

# %% [markdown]
# ### 4.3 Sensitivity Analysis

# %%
result = orchestrator.sensitivity_analysis(
    base_material="OPC",
    parameter="relative_humidity",
    values=[0.3, 0.5, 0.65, 0.8, 0.95],
)

# %% [markdown]
# ### 4.4 Complex Multi-Condition Query

# %%
result = orchestrator.analyze(
    "Compare the structural performance of an OPC concrete beam "
    "(0.5m × 0.1m, w/c=0.5) under:\n"
    "1. Natural carbonation (400 ppm CO₂) for 50 years\n"
    "2. Accelerated carbonation (50000 ppm CO₂) for 90 days\n"
    "3. Sulfate attack (0.5 mol/L Na₂SO₄) for 2 years\n\n"
    "For each case, compute the full chemo-transport-micromechanical "
    "pipeline and run FEM analysis. If any material data is missing, "
    "use molecular dynamics to estimate it."
)

# %% [markdown]
# ## 5. Vulnerability Testing

# %%
from colab_agent.vulnerability_test import run_all_vulnerability_tests

# Run without agent queries first (fast)
results = run_all_vulnerability_tests(orchestrator=None)

# %%
# Run with agent queries (slower, requires API calls)
# results_full = run_all_vulnerability_tests(orchestrator=orchestrator)

# %% [markdown]
# ## 6. Full Pipeline Visualization

# %%
def full_pipeline_visualization(cement_type="OPC", duration_years=50):
    """Complete chemo-transport-micromechanical pipeline with visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Chemistry
    pe = PhaseEvolution(cement_type)
    t_final = duration_years * 365 * 86400
    sol = pe.evolve((0, t_final), 16.0, 0.65, 293.15)
    t_years = sol.t / (365 * 86400)

    ax = axes[0, 0]
    phases_t = {"CH": [], "CSH": [], "CaCO3": [], "porosity": []}
    for j in range(len(t_years)):
        phi = pe.get_volume_fractions(sol.y[0,j], sol.y[1,j])
        for k in phases_t:
            phases_t[k].append(phi.get(k, 0))
    for name, vals in phases_t.items():
        ax.plot(t_years, vals, label=name, linewidth=2)
    ax.set_title("1. Chemistry: Phase Evolution")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Volume fraction")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Transport
    porosity = 0.25
    D_eff = millington_quirk(1.6e-5, porosity, 0.65)
    model = CarbonationTransportModel(L=0.05, nx=200, porosity=porosity, saturation=0.65, C_env_CO2=16.0)
    save_times = [y*365*86400 for y in [1, 5, 10, 20, 50] if y <= duration_years]
    dt = min(3600, 0.3 * (0.05/200)**2 / D_eff)
    res = model.solve(t_final, dt, save_times=save_times)

    ax = axes[0, 1]
    for i, (t, prof) in enumerate(zip(res["t"], res["profiles"])):
        ax.plot(res["x"]*1000, prof/16.0, label=f"{t/(365*86400):.0f}yr", linewidth=1.5)
    ax.set_title("2. Transport: CO₂ Profiles")
    ax.set_xlabel("Depth (mm)"); ax.set_ylabel("C/C₀")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Micromechanics vs carbonation
    ax = axes[0, 2]
    alphas = np.linspace(0, 1, 30)
    for lvl, color in [("paste", "b"), ("mortar", "g"), ("concrete", "r")]:
        Es = []
        for a in alphas:
            a_CH = min(a*1.2, 1.0)
            a_CSH = max(0, (a-0.4)/0.6) if a>0.4 else 0
            phi = pe.get_volume_fractions(a_CH, a_CSH)
            props = compute_effective_properties(phi, level=lvl)
            Es.append(props["E"]/1e9)
        ax.plot(alphas, Es, color=color, linewidth=2, label=lvl)
    ax.set_title("3. Micromechanics: E vs Carbonation")
    ax.set_xlabel("Carbonation degree"); ax.set_ylabel("E (GPa)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Carbonation depth
    ax = axes[1, 0]
    depths = [find_carbonation_depth(p, res["x"])*1000 for p in res["profiles"]]
    times_yr = [t/(365*86400) for t in res["t"]]
    ax.plot(times_yr, depths, 'ko-', linewidth=2, markersize=6)
    # sqrt-t fit
    if len(times_yr) > 1:
        t_fit = np.linspace(0.1, max(times_yr), 100)
        K = depths[-1] / np.sqrt(times_yr[-1]) if times_yr[-1] > 0 else 0
        ax.plot(t_fit, K*np.sqrt(t_fit), 'r--', label=f'K={K:.1f} mm/√yr', linewidth=1.5)
    ax.set_title("4. Carbonation Depth vs Time")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Depth (mm)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Property profiles at final time
    ax = axes[1, 1]
    x_mm = res["x"] * 1000
    final_prof = res["profiles"][-1] if res["profiles"] else np.zeros_like(res["x"])
    c_norm = final_prof / max(np.max(final_prof), 1e-10)

    E_profile = []
    for cn in c_norm:
        a_local = cn
        a_CH_l = min(a_local*1.2, 1.0)
        a_CSH_l = max(0, (a_local-0.4)/0.6) if a_local>0.4 else 0
        phi = pe.get_volume_fractions(a_CH_l, a_CSH_l)
        props = compute_effective_properties(phi, "concrete")
        E_profile.append(props["E"]/1e9)

    ax.plot(x_mm, E_profile, 'b-', linewidth=2)
    ax.set_title(f"5. E Profile at {duration_years}yr")
    ax.set_xlabel("Depth (mm)"); ax.set_ylabel("E (GPa)")
    ax.axvline(x=depths[-1] if depths else 0, color='r', linestyle='--', label='Carb. front')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"SUMMARY: {cement_type} — {duration_years} years\n"
        f"{'─'*35}\n"
        f"Carbonation depth: {depths[-1] if depths else 0:.1f} mm\n"
        f"K coefficient: {K:.2f} mm/√yr\n"
        f"pH (surface): {pe.pore_solution_pH(sol.y[0,-1], sol.y[1,-1]):.1f}\n"
        f"E (carbonated): {E_profile[0]:.1f} GPa\n"
        f"E (neat core): {E_profile[-1]:.1f} GPa\n"
        f"E ratio: {E_profile[0]/E_profile[-1]:.3f}\n"
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle(f"Chemo-Transport-Micromechanical Analysis: {cement_type}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

full_pipeline_visualization("OPC", duration_years=50)

# %%
full_pipeline_visualization("CSA", duration_years=50)

# %%
full_pipeline_visualization("OPC_GGBS", duration_years=50)
