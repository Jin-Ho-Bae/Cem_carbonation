"""
Chemistry Engine for Carbon Mineralization in Construction Materials.

Handles phase assemblage evolution, reaction kinetics, and thermodynamic
calculations for arbitrary cement+SCM blends under various conditions.

Supports: OPC, CSA, and all major SCMs (fly ash, metakaolin, silica fume,
GGBS, limestone filler, natural pozzolans, nano-silica).
Users specify a base cement + any combination of SCMs with replacement ratios.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from scipy.integrate import solve_ivp
from scipy.constants import R as GAS_CONST


# ──────────────────────── Base Cement Database ────────────────────────

BASE_CEMENTS = {
    "OPC": {
        "phases": {
            "C3S": 0.55, "C2S": 0.20, "C3A": 0.08, "C4AF": 0.10,
            "gypsum": 0.05, "other": 0.02,
        },
        "hydration_products": {
            "CH": 0.20, "CSH": 0.40, "ettringite": 0.05,
            "monosulfate": 0.03, "porosity": 0.25, "other": 0.07,
        },
        "CSH_CaSi": 1.7,
    },
    "CSA": {
        "phases": {
            "yeelimite": 0.50, "belite": 0.25, "anhydrite": 0.15,
            "C4AF": 0.05, "other": 0.05,
        },
        "hydration_products": {
            "ettringite": 0.35, "AH3": 0.15, "stratlingite": 0.10,
            "CSH": 0.10, "CH": 0.02, "porosity": 0.20, "other": 0.08,
        },
        "CSH_CaSi": 1.0,
    },
    "white_cement": {
        "phases": {
            "C3S": 0.65, "C2S": 0.20, "C3A": 0.10, "C4AF": 0.01,
            "gypsum": 0.03, "other": 0.01,
        },
        "hydration_products": {
            "CH": 0.24, "CSH": 0.42, "ettringite": 0.04,
            "monosulfate": 0.02, "porosity": 0.22, "other": 0.06,
        },
        "CSH_CaSi": 1.7,
    },
}

# ──────────────────────── SCM Database ────────────────────────

SCM_DATABASE = {
    "fly_ash_F": {
        "description": "Class F fly ash (low calcium, <10% CaO)",
        "composition": {"SiO2": 0.52, "Al2O3": 0.25, "Fe2O3": 0.10, "CaO": 0.05, "other": 0.08},
        "reactivity": 0.4,  # fraction reactive at 28d
        "pozzolanic_factor": 0.80,  # CH consumption per unit SCM reacted
        "CSH_CaSi_modifier": -0.3,  # lowers Ca/Si in CSH
        "porosity_modifier": 0.03,  # increases initial porosity slightly
        "extra_CSH_factor": 0.6,   # extra CSH from pozzolanic reaction
        "hydration_products_modifier": {
            "CH": -0.60,  # consumes 60% of available CH at full reaction
            "CSH": 0.15,  # produces additional CSH
        },
    },
    "fly_ash_C": {
        "description": "Class C fly ash (high calcium, >20% CaO)",
        "composition": {"SiO2": 0.35, "Al2O3": 0.20, "Fe2O3": 0.06, "CaO": 0.25, "other": 0.14},
        "reactivity": 0.5,
        "pozzolanic_factor": 0.50,
        "CSH_CaSi_modifier": -0.15,
        "porosity_modifier": 0.01,
        "extra_CSH_factor": 0.5,
        "hydration_products_modifier": {
            "CH": -0.35,
            "CSH": 0.12,
            "ettringite": 0.02,
        },
    },
    "metakaolin": {
        "description": "Calcined kaolin (Al2Si2O7), highly reactive pozzolan",
        "composition": {"SiO2": 0.53, "Al2O3": 0.43, "Fe2O3": 0.01, "CaO": 0.00, "other": 0.03},
        "reactivity": 0.7,
        "pozzolanic_factor": 1.0,
        "CSH_CaSi_modifier": -0.4,
        "porosity_modifier": -0.03,  # reduces porosity (finer)
        "extra_CSH_factor": 0.7,
        "hydration_products_modifier": {
            "CH": -0.80,
            "CSH": 0.20,
            "stratlingite": 0.05,
        },
    },
    "silica_fume": {
        "description": "Amorphous SiO2, ultrafine (<1 um), highly reactive",
        "composition": {"SiO2": 0.92, "Al2O3": 0.01, "Fe2O3": 0.01, "CaO": 0.01, "other": 0.05},
        "reactivity": 0.85,
        "pozzolanic_factor": 1.2,
        "CSH_CaSi_modifier": -0.5,
        "porosity_modifier": -0.05,
        "extra_CSH_factor": 0.8,
        "hydration_products_modifier": {
            "CH": -0.90,
            "CSH": 0.25,
        },
    },
    "GGBS": {
        "description": "Ground granulated blast-furnace slag",
        "composition": {"SiO2": 0.35, "Al2O3": 0.12, "Fe2O3": 0.01, "CaO": 0.40, "MgO": 0.08, "other": 0.04},
        "reactivity": 0.55,
        "pozzolanic_factor": 0.60,
        "CSH_CaSi_modifier": -0.25,
        "porosity_modifier": 0.02,
        "extra_CSH_factor": 0.65,
        "hydration_products_modifier": {
            "CH": -0.50,
            "CSH": 0.18,
            "hydrotalcite": 0.05,
        },
    },
    "limestone_filler": {
        "description": "Ground limestone (CaCO3), mostly inert filler",
        "composition": {"CaCO3": 0.95, "SiO2": 0.02, "other": 0.03},
        "reactivity": 0.05,
        "pozzolanic_factor": 0.0,
        "CSH_CaSi_modifier": 0.0,
        "porosity_modifier": -0.02,
        "extra_CSH_factor": 0.05,
        "hydration_products_modifier": {
            "CH": 0.0,
            "monocarboaluminate": 0.03,
        },
    },
    "natural_pozzolan": {
        "description": "Volcanic ash / pumice / trass",
        "composition": {"SiO2": 0.55, "Al2O3": 0.18, "Fe2O3": 0.08, "CaO": 0.07, "other": 0.12},
        "reactivity": 0.3,
        "pozzolanic_factor": 0.55,
        "CSH_CaSi_modifier": -0.2,
        "porosity_modifier": 0.04,
        "extra_CSH_factor": 0.45,
        "hydration_products_modifier": {
            "CH": -0.40,
            "CSH": 0.10,
        },
    },
    "nano_silica": {
        "description": "Nano-SiO2 (5-50 nm), extremely high reactivity",
        "composition": {"SiO2": 0.99, "other": 0.01},
        "reactivity": 0.95,
        "pozzolanic_factor": 1.5,
        "CSH_CaSi_modifier": -0.6,
        "porosity_modifier": -0.06,
        "extra_CSH_factor": 0.9,
        "hydration_products_modifier": {
            "CH": -0.95,
            "CSH": 0.30,
        },
    },
    "rice_husk_ash": {
        "description": "Amorphous SiO2 from rice husk combustion",
        "composition": {"SiO2": 0.87, "Al2O3": 0.01, "Fe2O3": 0.01, "CaO": 0.01, "K2O": 0.03, "other": 0.07},
        "reactivity": 0.70,
        "pozzolanic_factor": 1.0,
        "CSH_CaSi_modifier": -0.45,
        "porosity_modifier": -0.02,
        "extra_CSH_factor": 0.75,
        "hydration_products_modifier": {
            "CH": -0.80,
            "CSH": 0.22,
        },
    },
    "calcined_clay": {
        "description": "LC3-type calcined clay (kaolinite-rich)",
        "composition": {"SiO2": 0.50, "Al2O3": 0.40, "Fe2O3": 0.04, "CaO": 0.01, "other": 0.05},
        "reactivity": 0.60,
        "pozzolanic_factor": 0.90,
        "CSH_CaSi_modifier": -0.35,
        "porosity_modifier": -0.02,
        "extra_CSH_factor": 0.65,
        "hydration_products_modifier": {
            "CH": -0.70,
            "CSH": 0.18,
            "stratlingite": 0.04,
        },
    },
}

# Legacy compatibility: pre-built cement systems
CEMENT_SYSTEMS = {
    "OPC": BASE_CEMENTS["OPC"],
    "CSA": BASE_CEMENTS["CSA"],
    "OPC_FA": None,  # Built dynamically
    "OPC_GGBS": None,
    "OPC_SF": None,
    "OPC_MK": None,
}


def build_cement_system(base_cement: str, scm_mix: Dict[str, float]) -> dict:
    """
    Build a cement system from base cement + SCM replacement ratios.

    Args:
        base_cement: "OPC", "CSA", "white_cement"
        scm_mix: {"fly_ash_F": 0.25, "silica_fume": 0.05}
                 values are mass replacement ratios (sum <= 0.90)

    Returns:
        Complete cement system dict with phases, hydration_products, CSH_CaSi.
    """
    if base_cement not in BASE_CEMENTS:
        raise ValueError(f"Unknown base cement: {base_cement}. Available: {list(BASE_CEMENTS)}")

    total_replacement = sum(scm_mix.values())
    if total_replacement > 0.90:
        raise ValueError(f"Total SCM replacement {total_replacement:.0%} exceeds 90%")

    for scm_name in scm_mix:
        if scm_name not in SCM_DATABASE:
            raise ValueError(f"Unknown SCM: {scm_name}. Available: {list(SCM_DATABASE)}")

    base = BASE_CEMENTS[base_cement]
    cement_fraction = 1.0 - total_replacement

    # Scale clinker phases
    phases = {}
    for p, v in base["phases"].items():
        phases[p] = v * cement_fraction
    for scm_name, ratio in scm_mix.items():
        phases[scm_name] = ratio

    # Compute hydration products with SCM effects
    hp = {}
    for p, v in base["hydration_products"].items():
        hp[p] = v * cement_fraction

    csh_casi = base["CSH_CaSi"]
    porosity_mod = 0.0

    for scm_name, ratio in scm_mix.items():
        scm = SCM_DATABASE[scm_name]
        reactivity = scm["reactivity"]
        effective_ratio = ratio * reactivity

        for product, modifier in scm.get("hydration_products_modifier", {}).items():
            if modifier < 0:
                # Consumes existing product
                if product in hp:
                    hp[product] = max(0, hp[product] + modifier * effective_ratio)
            else:
                # Produces new product
                hp[product] = hp.get(product, 0) + modifier * effective_ratio

        # Extra CSH from pozzolanic reaction
        hp["CSH"] = hp.get("CSH", 0) + scm["extra_CSH_factor"] * effective_ratio

        csh_casi += scm["CSH_CaSi_modifier"] * ratio
        porosity_mod += scm["porosity_modifier"] * ratio

    hp["porosity"] = max(0.05, hp.get("porosity", 0.25) + porosity_mod)

    # Normalize so total <= 1
    total = sum(hp.values())
    if total > 1.0:
        scale = 1.0 / total
        hp = {k: v * scale for k, v in hp.items()}

    return {
        "phases": phases,
        "hydration_products": hp,
        "CSH_CaSi": max(0.6, min(2.0, csh_casi)),
        "base_cement": base_cement,
        "scm_mix": scm_mix,
    }


def _init_legacy_systems():
    """Initialize legacy pre-built systems."""
    CEMENT_SYSTEMS["OPC_FA"] = build_cement_system("OPC", {"fly_ash_F": 0.25})
    CEMENT_SYSTEMS["OPC_GGBS"] = build_cement_system("OPC", {"GGBS": 0.45})
    CEMENT_SYSTEMS["OPC_SF"] = build_cement_system("OPC", {"silica_fume": 0.08})
    CEMENT_SYSTEMS["OPC_MK"] = build_cement_system("OPC", {"metakaolin": 0.15})

_init_legacy_systems()


# ──────────────────────── Constants ────────────────────────

MOLAR_VOLUMES = {
    "CH": 33.1e-6,
    "CSH": 78.0e-6,
    "CaCO3_calcite": 36.9e-6,
    "CaCO3_vaterite": 38.7e-6,
    "CaCO3_aragonite": 34.2e-6,
    "SiO2_gel": 29.0e-6,
    "ettringite": 707.0e-6,
    "gypsum": 74.7e-6,
    "monosulfate": 309.0e-6,
    "thaumasite": 490.0e-6,
    "AH3": 32.0e-6,
    "monocarboaluminate": 262.0e-6,
    "stratlingite": 216.0e-6,
    "hydrotalcite": 220.0e-6,
}

REACTION_ENTHALPIES = {
    "CH_carbonation": -113.0e3,
    "CSH_carbonation": -100.0e3,
    "ettringite_carbonation": -120.0e3,
    "CH_sulfate": -17.0e3,
}


@dataclass
class ReactionParams:
    k_ref: float
    Ea: float
    T_ref: float = 298.15
    C_ref: float = 16.0
    a_sat: float = 2.2
    b_sat: float = 0.5
    alpha_threshold: float = 0.0


@dataclass
class EnvironmentCondition:
    CO2_ppm: float = 400.0
    temperature_K: float = 293.15
    relative_humidity: float = 0.65
    sulfate_concentration: float = 0.0
    exposure_type: str = "natural_carbonation"


@dataclass
class PhaseState:
    alpha_CH: float = 0.0
    alpha_CSH: float = 0.0
    alpha_ettringite: float = 0.0
    phi: dict = field(default_factory=dict)
    pH: float = 13.0
    porosity: float = 0.25
    carbonation_degree: float = 0.0


# ──────────────────────── Kinetics ────────────────────────

def arrhenius(k_ref: float, Ea: float, T: float, T_ref: float = 298.15) -> float:
    return k_ref * np.exp(-Ea / GAS_CONST * (1.0/T - 1.0/T_ref))


def ch_carbonation_rate(alpha_CH, C_CO2, S, T, params: ReactionParams):
    if alpha_CH >= 1.0 or C_CO2 <= 0:
        return 0.0
    k = arrhenius(params.k_ref, params.Ea, T, params.T_ref)
    sat = max(0, (1 - S))**params.a_sat * max(0, S)**params.b_sat
    return k * (1 - alpha_CH) * (C_CO2 / params.C_ref) * sat


def csh_carbonation_rate(alpha_CSH, C_CO2, alpha_CH, T, params: ReactionParams):
    if alpha_CSH >= 1.0 or C_CO2 <= 0:
        return 0.0
    if alpha_CH < params.alpha_threshold:
        return 0.0
    k = arrhenius(params.k_ref, params.Ea, T, params.T_ref)
    return k * (1 - alpha_CSH)**(2.0/3.0) * (C_CO2 / params.C_ref)


def ettringite_sulfate_rate(alpha_ett, C_sulfate, T, params: ReactionParams):
    if alpha_ett >= 1.0 or C_sulfate <= 0:
        return 0.0
    k = arrhenius(params.k_ref, params.Ea, T, params.T_ref)
    return k * (1 - alpha_ett) * C_sulfate


# ──────────────────────── Phase Evolution ────────────────────────

class PhaseEvolution:
    """
    Computes time evolution of phase assemblage under carbonation or sulfate attack.
    Supports arbitrary cement+SCM blends.
    """

    DEFAULT_KINETICS = {
        "carbonation": {
            "CH": ReactionParams(k_ref=1e-4, Ea=30e3, a_sat=2.2, b_sat=0.5),
            "CSH": ReactionParams(k_ref=1e-5, Ea=45e3, alpha_threshold=0.85),
            "ettringite": ReactionParams(k_ref=5e-6, Ea=35e3),
        },
        "sulfate_attack": {
            "ettringite_formation": ReactionParams(k_ref=5e-5, Ea=40e3),
            "CH_consumption": ReactionParams(k_ref=2e-5, Ea=25e3),
        },
        "weathering": {
            "CH": ReactionParams(k_ref=2e-5, Ea=30e3, a_sat=1.5, b_sat=0.8),
            "CSH": ReactionParams(k_ref=5e-6, Ea=45e3, alpha_threshold=0.90),
        },
    }

    def __init__(self, cement_type: str = "OPC",
                 env: Optional[EnvironmentCondition] = None,
                 kinetics: Optional[dict] = None,
                 scm_mix: Optional[Dict[str, float]] = None):
        """
        Args:
            cement_type: Base cement or pre-built system name
            scm_mix: Optional SCM replacement ratios. If provided, builds
                     custom system from base cement + SCMs.
                     e.g., {"fly_ash_F": 0.20, "silica_fume": 0.05}
        """
        if scm_mix is not None:
            # Custom blend
            self.cement = build_cement_system(cement_type, scm_mix)
        elif cement_type in CEMENT_SYSTEMS and CEMENT_SYSTEMS[cement_type] is not None:
            self.cement = CEMENT_SYSTEMS[cement_type]
        elif cement_type in BASE_CEMENTS:
            self.cement = BASE_CEMENTS[cement_type]
        else:
            raise ValueError(
                f"Unknown cement: {cement_type}. "
                f"Base cements: {list(BASE_CEMENTS)}. "
                f"Pre-built systems: {[k for k,v in CEMENT_SYSTEMS.items() if v]}. "
                f"Or provide scm_mix for custom blend."
            )

        self.cement_type = cement_type
        self.env = env or EnvironmentCondition()
        self.phi0 = dict(self.cement["hydration_products"])
        self.CaSi = self.cement["CSH_CaSi"]
        self.scm_mix = scm_mix or self.cement.get("scm_mix", {})

        mode = self.env.exposure_type.split("_")[0]
        if mode not in ("natural", "accelerated", "sulfate", "weathering"):
            mode = "carbonation"
        if mode in ("natural", "accelerated"):
            mode = "carbonation"
        self.kinetics = kinetics or self.DEFAULT_KINETICS.get(mode, self.DEFAULT_KINETICS["carbonation"])

    def ode_system(self, t, y, C_CO2_func, S_func, T_func):
        alpha_CH, alpha_CSH, alpha_ett = y
        C = C_CO2_func(t) if callable(C_CO2_func) else C_CO2_func
        S = S_func(t) if callable(S_func) else S_func
        T = T_func(t) if callable(T_func) else T_func

        if self.env.exposure_type in ("sulfate_attack",):
            d_CH = 0.0
            d_CSH = 0.0
            params_ett = self.kinetics.get("ettringite_formation",
                                           ReactionParams(k_ref=5e-5, Ea=40e3))
            d_ett = ettringite_sulfate_rate(alpha_ett, self.env.sulfate_concentration, T, params_ett)
        else:
            params_CH = self.kinetics.get("CH", ReactionParams(k_ref=1e-4, Ea=30e3))
            params_CSH = self.kinetics.get("CSH", ReactionParams(k_ref=1e-5, Ea=45e3, alpha_threshold=0.85))
            params_ett = self.kinetics.get("ettringite", ReactionParams(k_ref=5e-6, Ea=35e3))
            d_CH = ch_carbonation_rate(alpha_CH, C, S, T, params_CH)
            d_CSH = csh_carbonation_rate(alpha_CSH, C, alpha_CH, T, params_CSH)
            d_ett = ch_carbonation_rate(alpha_ett, C, S, T, params_ett)

        return [d_CH, d_CSH, d_ett]

    def evolve(self, t_span, C_CO2_func, S_func=0.5, T_func=293.15,
               n_eval=200):
        t_eval = np.linspace(t_span[0], t_span[1], n_eval)
        sol = solve_ivp(
            lambda t, y: self.ode_system(t, y, C_CO2_func, S_func, T_func),
            t_span, [0.0, 0.0, 0.0],
            t_eval=t_eval, method="BDF", rtol=1e-8, atol=1e-10,
            max_step=t_span[1] / 100,
        )
        return sol

    def get_volume_fractions(self, alpha_CH, alpha_CSH, alpha_ett=0.0):
        phi = {}
        phi["CH"] = self.phi0.get("CH", 0) * (1 - alpha_CH)
        phi["CSH"] = self.phi0.get("CSH", 0) * (1 - alpha_CSH)
        phi["ettringite"] = self.phi0.get("ettringite", 0) * (1 - alpha_ett)

        Vm_CH = MOLAR_VOLUMES["CH"]
        Vm_CC = MOLAR_VOLUMES["CaCO3_calcite"]
        Vm_CSH = MOLAR_VOLUMES["CSH"]
        Vm_SiO2 = MOLAR_VOLUMES["SiO2_gel"]

        phi["CaCO3"] = (self.phi0.get("CH", 0) * alpha_CH * Vm_CC / Vm_CH +
                        self.phi0.get("CSH", 0) * alpha_CSH * self.CaSi * Vm_CC / Vm_CSH)
        phi["SiO2_gel"] = self.phi0.get("CSH", 0) * alpha_CSH * Vm_SiO2 / Vm_CSH

        for k in ("monosulfate", "AH3", "stratlingite", "hydrotalcite",
                  "monocarboaluminate", "other"):
            if k in self.phi0:
                phi[k] = self.phi0[k]

        total_solid = sum(phi.values())
        phi["porosity"] = max(0, 1.0 - total_solid)

        return phi

    def get_source_terms(self, alpha_CH, alpha_CSH, C_CO2, S, T):
        params_CH = self.kinetics.get("CH", ReactionParams(k_ref=1e-4, Ea=30e3))
        params_CSH = self.kinetics.get("CSH", ReactionParams(k_ref=1e-5, Ea=45e3, alpha_threshold=0.85))

        r_CH = ch_carbonation_rate(alpha_CH, C_CO2, S, T, params_CH)
        r_CSH = csh_carbonation_rate(alpha_CSH, C_CO2, alpha_CH, T, params_CSH)

        n_CH0 = self.phi0.get("CH", 0) / MOLAR_VOLUMES["CH"]
        n_CSH0 = self.phi0.get("CSH", 0) / MOLAR_VOLUMES["CSH"]

        R_CO2 = n_CH0 * r_CH + self.CaSi * n_CSH0 * r_CSH
        R_H2O = n_CH0 * r_CH
        Q_rxn = (n_CH0 * r_CH * abs(REACTION_ENTHALPIES["CH_carbonation"]) +
                 n_CSH0 * r_CSH * abs(REACTION_ENTHALPIES["CSH_carbonation"]))

        return {"R_CO2": R_CO2, "R_H2O": R_H2O, "Q_rxn": Q_rxn}

    def pore_solution_pH(self, alpha_CH, alpha_CSH):
        if alpha_CH < 0.9:
            return 13.0 - 0.5 * alpha_CH
        elif alpha_CSH < 0.5:
            return 12.5 - 4.0 * alpha_CSH
        else:
            return max(8.3, 10.5 - 4.4 * (alpha_CSH - 0.5))

    def carbonation_degree(self, alpha_CH, alpha_CSH):
        w_CH = self.phi0.get("CH", 0)
        w_CSH = self.phi0.get("CSH", 0)
        total = w_CH + w_CSH
        if total == 0:
            return 0.0
        return (w_CH * alpha_CH + w_CSH * alpha_CSH) / total

    def get_blend_description(self) -> str:
        """Human-readable description of the cement blend."""
        desc = self.cement_type
        if self.scm_mix:
            parts = [f"{name} {ratio*100:.0f}%" for name, ratio in self.scm_mix.items()]
            desc += " + " + " + ".join(parts)
        return desc


def get_available_scms() -> Dict[str, str]:
    """Return available SCMs with descriptions."""
    return {name: data["description"] for name, data in SCM_DATABASE.items()}


def get_available_base_cements() -> List[str]:
    """Return available base cements."""
    return list(BASE_CEMENTS.keys())
