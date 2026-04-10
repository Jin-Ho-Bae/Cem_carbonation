# Carbon Mineralization in Construction Materials

LLM Multi-Agent System for chemo-transport-micromechanical modeling of cement carbonation. Runs on Google Colab with GPT-4o + Claude backend.

## Architecture

```
User Query
    |
Manager (GPT-4o)                    -- interprets query, dispatches commands
    |
Orchestrator (Claude)               -- routes experts, coordinates coder, feedback loops
    |
+-- 7 Domain Experts (GPT-4o)      -- parallel execution, RBAC-enforced tool access
|     Chemistry      : phase assemblage, 10 SCM blends, pH evolution
|     Thermodynamics : GEMS equilibrium, cemdata18, C-S-H decalcification
|     Transport      : Fick's 2nd law, 6 diffusion models, carbonation depth
|     Micromechanics : 6-level Eshelby-Mori-Tanaka homogenization
|     Structural     : FEniCS FEM, 5 structures x 6 load conditions
|     MD             : LAMMPS nanoscale properties, 13-phase fallback DB
|     Engineering    : durability assessment, service life prediction
|
+-- Coder (Claude)                  -- writes computation code, self-learning from history
+-- Reviewer (Claude)               -- vulnerability checks, PASS-only save to workspace
```

**LLM role assignment:**
- **GPT-4o** -- Manager + 6 Domain Experts (tool calling)
- **Claude** -- Orchestrator + Coder + Reviewer (reasoning, coding, verification)

## Project Structure

```
.
├── notebooks/
│   └── Carbonation.ipynb                  # Main Colab notebook (entry point)
│
├── src/
│   ├── colab_agent/                       # LLM multi-agent system
│   │   ├── orchestrator.py                # Manager/Orchestrator/Expert/Coder/Reviewer
│   │   ├── agent_tools.py                 # 15 LangChain @tool wrappers + RBAC getters
│   │   ├── gems_engine.py                 # GEMS thermodynamic engine (cemdata18, ~20 phases)
│   │   ├── chemistry_engine.py            # Phase evolution, reaction kinetics, pH
│   │   ├── transport_engine.py            # Pore structure, diffusion, FDM solver
│   │   ├── micromechanics_engine.py       # 6-level Eshelby-MT homogenization
│   │   ├── fem_engine.py                  # FEniCS structural analysis
│   │   ├── md_engine.py                   # LAMMPS MD interface + reference DB
│   │   ├── verification.py                # Direct engine computation + 6-panel plots
│   │   └── vulnerability_test.py          # 8 categories, 115 tests, N-round repeat
│   │
│   └── models/                            # MATLAB -> Python reimplementation (standalone)
│       ├── tensor_ops.py                  # 4th-order tensor ops (C_ASM, Tinv, Lame)
│       ├── eshelby.py                     # Eshelby tensor (sphere, n-layered, oblate)
│       ├── homogenization.py              # Mori-Tanaka (standard, n-layered, multi-phase)
│       ├── hashin_bounds.py               # Hashin-Shtrikman bounds
│       ├── damage.py                      # Crack evolution, Mazars damage model
│       ├── multiscale.py                  # Level I-III multiscale homogenization
│       ├── effective_diffusivity.py       # Eshelby-based effective diffusion coefficient
│       ├── chemo_transport.py             # FDM 1D diffusion solver
│       ├── chemo_mechanical.py            # N-layered ettringite + coupled solver
│       ├── self_correction.py             # Failure logging + experience inheritance
│       └── tests/test_all.py              # 17 verification tests (vs MATLAB results)
│
├── data/validation/                       # MATLAB validation data
│   ├── micromechanics/                    # E_star_N.xlsx, N_layered_inclusion.xlsx
│   ├── chemo_transport/                   # M0-M2 phase/stress/ion profiles, sulfate data
│   └── chemo_mechanical/                  # E/kappa/mu, PhaseValidation (LA/LB x WA-WC)
│
├── _workspace/                            # Agent runtime artifacts
│   ├── logs/agent_errors.json             # Self-correction failure/success log
│   └── coder/                             # Coder output + history
│
└── CLAUDE.md                              # Project rules and conventions
```

## Computation Engines

### GEMS Thermodynamic Engine (`gems_engine.py`)

Lightweight Gibbs Energy Minimization solver inspired by [GEM-Selektor](https://github.com/gemshub) with the [cemdata18](https://www.empa.ch/web/s308/cemdata) thermodynamic database. The original GEMS handles 129 phases / 291 species -- this engine retains only ~20 phases relevant to concrete carbonation.

- **Database:** cemdata18 subset -- Portlandite, C-S-H (CSHQ), Calcite/Vaterite/Aragonite, Ettringite, Monosulfate, Monocarbonate, Hemicarbonate, OH-AFm, C3AH6, Hydrotalcite, SiO2(am), Gypsum, Stratlingite, Brucite, Thaumasite
- **C-S-H model:** CSHQ (Kulik, 2011) -- variable Ca/Si from 1.67 (jennite) to 0.83 (tobermorite), with solubility curve interpolation
- **Carbonation sequence** (thermodynamic priority):
  1. Portlandite dissolves first (most soluble Ca source)
  2. Ettringite decomposes -> gypsum + Al(OH)3
  3. Monosulfate -> hemicarbonate -> monocarbonate -> decomposition
  4. C-S-H decalcifies (Ca/Si decreases progressively)
  5. CaCO3 precipitates (calcite dominant)
  6. pH drops: 13.0 -> 12.5 -> ~10 -> 8.3
- **Temperature:** van't Hoff correction (0-60 C)
- **Outputs:** phase volumes, pH, Ca/Si of C-S-H, carbonation degree, porosity evolution

```python
from colab_agent.gems_engine import compute_gems_carbonation, plot_gems_carbonation

result = compute_gems_carbonation("OPC", scm_mix={"fly_ash_F": 0.30},
                                   wc_ratio=0.45, CO2_total_mol=0.5)
plot_gems_carbonation(result)  # 6-panel plot
```

### Chemistry Engine (`chemistry_engine.py`)

ODE-based phase evolution for carbonation and sulfate attack (kinetics approach, complementary to GEMS equilibrium).

- **Base cements:** OPC, CSA, white cement
- **10 SCMs:** fly ash (F/C), metakaolin, silica fume, GGBS, limestone filler, natural pozzolan, nano-silica, rice husk ash, calcined clay
- **Outputs:** volume fractions (CH, CSH, CaCO3, ettringite, porosity), pH, carbonation degree
- `PhaseEvolution(cement_type, scm_mix={"fly_ash_F": 0.30})` -- arbitrary blends via `build_cement_system()`

### Transport Engine (`transport_engine.py`)

CO2 diffusion modeling with Powers pore structure.

- **Pore structure:** `PoreStructure(wc, alpha_hyd, scm_mix)` -- capillary + gel porosity (Powers model)
- **6 diffusion models** (user-selectable):
  - `standard` -- basic tortuosity
  - `millington_quirk` -- gas-phase tortuosity
  - `papadakis` -- empirical cement-specific
  - `ceb_fip` -- CEB-FIP Model Code
  - `fib_mc2010` -- fib Model Code 2010
  - `eshelby_mt` -- Eshelby-Mori-Tanaka effective diffusivity
- **FDM solver:** Crank-Nicolson (unconditionally stable, nx=50, dt=t_final/200)
- **Outputs:** carbonation depth, K coefficient (mm/sqrt(yr)), CO2 concentration profiles

### Micromechanics Engine (`micromechanics_engine.py`)

6-level Eshelby-Mori-Tanaka homogenization (Haile et al. 2019):

| Level | Scale | Description |
|-------|-------|-------------|
| I | Nano | MD crystal properties (jennite, tobermorite) |
| II | Nano-micro | CSH variants: LD (phi_gel=0.37), HD (0.24), UHD (0.13) |
| III | Micro | CSH matrix: HD matrix + LD/UHD inclusions |
| IV | Micro-meso | Cement paste: CSH + CH + CaCO3 + capillary pores + clinker |
| V | Meso | Mortar: paste + sand + air voids + microcracks |
| VI | Macro | Concrete: mortar + coarse aggregates + ITZ + fibers |

- **Outputs:** effective E (GPa), nu, K (bulk), G (shear) at each scale
- Compares carbonated vs neat properties

### FEM Engine (`fem_engine.py`)

FEniCS-based structural analysis with carbonation-partitioned material properties.

- **Structures:** beam, column, beam_column, slab, wall
- **Loads:** tension, compression, shear, bending, pressure, combined
- **Supports:** cantilever, simply_supported, fixed_fixed, pinned_roller
- Mesh partitioned by carbonation depth: carbonated and neat zones get different E, nu

### MD Engine (`md_engine.py`)

LAMMPS molecular dynamics interface with pre-computed fallback database.

- **13 phases:** jennite, tobermorite (11A/14A), portlandite, calcite, ettringite, monosulfate, C3S, C2S, C3A, C4AF, gypsum, quartz
- **Outputs:** elastic constants (E, nu, K, G), diffusion coefficients
- LAMMPS script generation for custom simulations

## LangChain Tools (15 tools)

| # | Tool | Domain | Description |
|---|------|--------|-------------|
| 1 | `compute_phase_assemblage` | Chemistry | SCM blend phase evolution (kinetics) |
| 2 | `get_available_cement_systems` | Chemistry | List base cements + SCMs |
| 3 | `compute_gems_equilibrium` | Thermodynamics | GEMS carbonation equilibrium (cemdata18) |
| 4 | `get_gems_phase_database` | Thermodynamics | List ~20 cemdata18 phases |
| 5 | `compute_csh_decalcification` | Thermodynamics | CSHQ Ca/Si evolution model |
| 6 | `compute_pore_structure` | Transport | Powers pore structure calculation |
| 7 | `compute_carbonation_depth` | Transport | Diffusion model-selectable depth |
| 8 | `list_available_diffusion_models` | Transport | 6 model descriptions |
| 9 | `compute_mechanical_properties` | Micromechanics | 6-level effective properties |
| 10 | `compute_properties_at_carbonation_depths` | Micromechanics | Property profile vs depth |
| 11 | `run_fem_analysis` | Structural | FEniCS structure/load/support analysis |
| 12 | `list_fem_options` | Structural | Available structures/loads/supports |
| 13 | `get_md_properties` | MD | Nanoscale elastic/diffusion data |
| 14 | `generate_lammps_script` | MD | LAMMPS input script generation |
| 15 | `run_full_chemo_transport_mechanical` | Engineering | Full pipeline integration |

**RBAC:** Each expert agent accesses ONLY its domain tools via `get_tools_for_role()`.

## Key Features

### Resilient API Calls
- `_resilient_invoke()` with 3 retries and exponential backoff (5s, 10s, 20s)
- Automatic GPT-4o fallback when Claude API returns 529 (overloaded)
- Ultimate fallback: raw expert results returned without synthesis

### Self-Correction
- `SelfCorrectionLog` persists failure/success to `_workspace/logs/agent_errors.json`
- All agents receive past failure lessons injected into their prompts
- Coder maintains separate history in `_workspace/coder/logs/history.json`

### Vulnerability Testing
8 categories, 115 tests, N-round repeat execution:

1. **Hard-coded edge cases** (16) -- tensor inverse, conservation, boundary
2. **Extreme conditions** (18) -- temperature, CO2, porosity, duration
3. **Conservation laws** (27) -- volume sum <= 1, no negative values, pH monotone
4. **SCM blend robustness** (16) -- all 10 SCMs, multi-SCM combos
5. **Transport model consistency** (11) -- all 6 diffusion models
6. **6-level micromechanics** (15) -- Level I-VI chain, fiber inclusions
7. **FEM structural inputs** (8) -- beam/column, bending/compression
8. **Tough queries** (14) -- contradictory, ambiguous, out-of-range

## Quick Start (Google Colab)

### 1. Install dependencies

```python
!pip install -q langchain langchain-openai langchain-anthropic langchain-core langgraph numpy scipy matplotlib
```

### 2. Mount Drive and import

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, "/content/drive/MyDrive/1 Cement carbonation/src")

from colab_agent.orchestrator import CarbonMineralizationSystem
```

### 3. Set API keys

```python
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

### 4. Run analysis

```python
system = CarbonMineralizationSystem(workspace="/content/drive/MyDrive/1 Cement carbonation/_workspace")

# Single query
result = system.analyze(
    "Analyze natural carbonation of OPC + 30% fly ash F concrete (w/c=0.45) for 50 years. "
    "Use the fib MC2010 diffusion model."
)

# Material comparison
result = system.compare_materials(
    materials=["OPC", "OPC + 30% fly_ash_F", "OPC + 50% GGBS"],
    condition="accelerated carbonation at 5% CO2, w/c=0.50",
    duration_days=90,
)

# Sensitivity analysis
result = system.sensitivity_analysis(
    base_material="OPC",
    parameter="relative_humidity",
    values=[0.3, 0.5, 0.65, 0.8, 0.95],
)
```

### 5. Verification (direct engine, no LLM)

```python
from colab_agent.verification import run_verification, full_pipeline_visualization

# 3 standard test cases with 6-panel plots
run_verification()

# Custom case
full_pipeline_visualization("OPC", scm_mix={"metakaolin": 0.10}, diffusion_model="eshelby_mt")
```

## Dependencies

| Package | Role |
|---------|------|
| `langchain`, `langchain-core` | Agent framework |
| `langchain-openai` | GPT-4o integration |
| `langchain-anthropic` | Claude integration |
| `langgraph` | ReAct agent execution |
| `numpy`, `scipy` | Numerical computation |
| `matplotlib` | Visualization |
| `FEniCS` (optional) | FEM structural analysis |
| `LAMMPS` (optional) | Molecular dynamics |

## References

- **cemdata18:** Lothenbach, B., Kulik, D.A., Matschei, T., Balonis, M., et al. (2019). Cemdata18: A chemical thermodynamic database for hydrated Portland cements and alkali-activated materials. *Cement and Concrete Research*, 115, 472-506.
- **CSHQ model:** Kulik, D.A. (2011). Improving the structural consistency of C-S-H solid solution thermodynamic models. *Cement and Concrete Research*, 41(5), 477-495.
- **GEM-IPM:** Kulik, D.A., Wagner, T., Dmytrieva, S.V., et al. (2013). GEM-Selektor geochemical modeling package. *Computational Geosciences*, 17, 1-24.
- **GEM-Selektor:** https://github.com/gemshub
- Haile, B.F., et al. (2019). Multi-level homogenization for the prediction of the mechanical properties of ultra-high-performance concrete. *Construction and Building Materials*.
- Papadakis, V.G. (1991). Fundamental modeling and experimental investigation of concrete carbonation. *ACI Materials Journal*.
- fib Model Code 2010. *International Federation for Structural Concrete*.
- Powers, T.C. & Brownyard, T.L. (1947). Studies of the physical properties of hardened Portland cement paste. *ACI Journal Proceedings*.
