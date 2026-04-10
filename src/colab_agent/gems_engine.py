"""
Lightweight GEMS Thermodynamic Engine for Concrete Carbonation.

Inspired by GEM-Selektor (https://github.com/gemshub) and the cemdata18
thermodynamic database (Lothenbach et al., 2019).

The original GEMS system handles 129 phases / 291 species / 24 elements --
far too heavy for focused carbonation studies. This engine retains only the
~20 phases relevant to concrete carbonation and implements Gibbs energy
minimization via sequential equilibrium (solubility products + mass balance).

Key references:
  - cemdata18: Lothenbach, Kulik, Matschei, Balonis (Cement & Concrete Research, 2019)
  - GEM-IPM: Kulik et al. (Computational Geosciences, 2013)
  - CSHQ model: Kulik (Cement & Concrete Research, 2011)

Capabilities:
  - Stepwise CO2 ingress equilibrium (Portlandite -> C-S-H -> AFm/AFt dissolution)
  - C-S-H decalcification with variable Ca/Si (CSHQ-like model)
  - AFm/AFt phase transformations under carbonation
  - pH evolution from phase equilibria
  - Volume fraction tracking for transport/micromechanics coupling
  - Temperature dependence via van't Hoff (0-60 C)

──────────────────────────────────────────────────────────────────────────
  CO2 gas / aqueous / reaction mechanism (IMPORTANT)
──────────────────────────────────────────────────────────────────────────
  GEMS full-hydration output (`from_xrf`) gives the initial saturated
  (un-carbonated) phase assemblage. The hydrates (Portlandite, C-S-H,
  Ettringite, AFm, ...) stay exactly as computed UNLESS dissolved CO2
  reaches them. In other words:

    hydrates do NOT react with gaseous CO2 in the pore space.
    Only dissolved CO2 (carbonic acid / bicarbonate / carbonate in the
    pore solution) participates in the reaction chain that transforms
    portlandite and C-S-H into carbonates.

  The full 3-step chain is:

    (1) Gas-phase transport
        CO2(g) diffuses through the air-filled fraction of the capillary
        pore network (Millington-Quirk-type D_eff(phi, Sr)). This is
        handled OUTSIDE GEMS, by `transport_engine.py`.

    (2) Gas -> aqueous partitioning (Henry's law)
        At each pore wall the gas partial pressure equilibrates with the
        pore solution:
            c_CO2,aq = K_H(T) * p_CO2
        with K_H(25 C) ~ 3.3e-4 mol / (m^3 * Pa). Inside the solution
        CO2 speciates as:
            CO2(aq) + H2O  <->  H2CO3
            H2CO3         <->  H+ + HCO3-     (pKa1 ~ 6.35)
            HCO3-         <->  H+ + CO3^2-    (pKa2 ~ 10.33)
        At cement pH (>=12.5 before carbonation) virtually all dissolved
        inorganic carbon is present as CO3^2-.

    (3) Aqueous reaction with hydrates (what `equilibrate_step` does)
        Dissolved CO3^2- is consumed by Ca^2+ released from hydrate
        dissolution:
            Ca(OH)2        + CO3^2-  ->  CaCO3 + 2 OH-        (Portlandite)
            3CaO.Al2O3.3CaSO4.32H2O + 3CO3^2- -> 3CaCO3 + ... (Ettringite)
            C-S-H(Ca/Si=x) + CO3^2-  ->  C-S-H(Ca/Si=x-dx) + CaCO3
        Priority is set by thermodynamic undersaturation:
            Portlandite > Ettringite > Monosulfate/AFm > C-S-H
        `equilibrate_step(CO2_mol)` takes the *dissolved* CO2 that has
        reached this control volume (not the gas field) and applies the
        four dissolution routines in priority order.

  Because the aqueous reactions are fast (Damkoehler number >> 1 for
  typical transport time scales), the pore solution is assumed to be
  in local equilibrium with the current hydrate assemblage. This is the
  Papadakis fast-reaction limit: the moving sharp front x_c(t) = K*sqrt(t)
  with K = sqrt(2 D_eff C_env / C_max) — implemented in
  `coupled_carbonation.py`. `equilibrate_step` is then called per depth
  with the cumulative *dose* of dissolved CO2 that has crossed that cell.

  Consequences for users:
    * Never pass gaseous CO2 concentration directly to `equilibrate_step`.
      Pass the cumulative dissolved-CO2 dose (mol per system volume).
    * If no dissolved CO2 has reached a depth, its phase assemblage is
      identical to the GEMS hydration output (hydrates unchanged).
    * Time-evolution of the phase assemblage at a fixed depth comes
      entirely from the monotone cumulative dose — never decreasing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from scipy.constants import R as R_GAS


# ══════════════════════════════════════════════════════════════
#  CEMDATA18 THERMODYNAMIC DATABASE (carbonation-relevant subset)
# ══════════════════════════════════════════════════════════════

@dataclass
class PhaseData:
    """Thermodynamic data for a single phase from cemdata18."""
    name: str
    formula: str
    log_Ksp_25C: float           # log10(Ksp) at 25 C
    delta_H_rxn: float           # Enthalpy of dissolution reaction [J/mol]
    molar_volume: float          # [m3/mol]
    molar_mass: float            # [g/mol]
    density: float               # [kg/m3]
    category: str                # clinker, hydrate, carbonate, sulfate, other
    n_Ca: float = 0.0            # moles of Ca per formula unit
    n_Si: float = 0.0            # moles of Si per formula unit
    n_Al: float = 0.0            # moles of Al per formula unit
    n_C: float = 0.0             # moles of CO3 per formula unit (inorganic carbon)
    n_S_bar: float = 0.0         # moles of SO4 per formula unit
    n_H2O: float = 0.0           # moles of H2O per formula unit


# cemdata18 values -- dissolution reactions written as:
# Phase -> aqueous species (log Ksp at 25 C, 1 bar)
# delta_H from van't Hoff for temperature correction
CEMDATA18_PHASES: Dict[str, PhaseData] = {
    # ── Portlandite ──
    "Portlandite": PhaseData(
        name="Portlandite", formula="Ca(OH)2",
        log_Ksp_25C=-5.20, delta_H_rxn=-17600.0,
        molar_volume=33.1e-6, molar_mass=74.09, density=2240.0,
        category="hydrate", n_Ca=1, n_H2O=1,
    ),

    # ── C-S-H (CSHQ model end-members) ──
    # Tobermorite-like (Ca/Si ~ 0.83)
    "CSHQ_TobH": PhaseData(
        name="CSHQ_TobH", formula="(CaO)0.83(SiO2)(H2O)1.33",
        log_Ksp_25C=-8.0, delta_H_rxn=-25000.0,
        molar_volume=59.0e-6, molar_mass=130.0, density=2200.0,
        category="hydrate", n_Ca=0.83, n_Si=1, n_H2O=1.33,
    ),
    # Jennite-like (Ca/Si ~ 1.67)
    "CSHQ_JenH": PhaseData(
        name="CSHQ_JenH", formula="(CaO)1.67(SiO2)(H2O)2.1",
        log_Ksp_25C=-13.2, delta_H_rxn=-40000.0,
        molar_volume=78.0e-6, molar_mass=197.0, density=2530.0,
        category="hydrate", n_Ca=1.67, n_Si=1, n_H2O=2.1,
    ),

    # ── Carbonates ──
    "Calcite": PhaseData(
        name="Calcite", formula="CaCO3",
        log_Ksp_25C=-8.48, delta_H_rxn=-10400.0,
        molar_volume=36.9e-6, molar_mass=100.09, density=2710.0,
        category="carbonate", n_Ca=1, n_C=1,
    ),
    "Vaterite": PhaseData(
        name="Vaterite", formula="CaCO3",
        log_Ksp_25C=-7.91, delta_H_rxn=-9800.0,
        molar_volume=38.7e-6, molar_mass=100.09, density=2650.0,
        category="carbonate", n_Ca=1, n_C=1,
    ),
    "Aragonite": PhaseData(
        name="Aragonite", formula="CaCO3",
        log_Ksp_25C=-8.34, delta_H_rxn=-10200.0,
        molar_volume=34.2e-6, molar_mass=100.09, density=2930.0,
        category="carbonate", n_Ca=1, n_C=1,
    ),

    # ── AFt phases ──
    "Ettringite": PhaseData(
        name="Ettringite", formula="Ca6Al2(SO4)3(OH)12*26H2O",
        log_Ksp_25C=-44.9, delta_H_rxn=-200000.0,
        molar_volume=707.0e-6, molar_mass=1255.1, density=1775.0,
        category="hydrate", n_Ca=6, n_Al=2, n_S_bar=3, n_H2O=32,
    ),
    "Tricarboaluminate": PhaseData(
        name="Tricarboaluminate", formula="Ca6Al2(CO3)3(OH)12*26H2O",
        log_Ksp_25C=-46.0, delta_H_rxn=-210000.0,
        molar_volume=650.0e-6, molar_mass=1146.0, density=1763.0,
        category="hydrate", n_Ca=6, n_Al=2, n_C=3, n_H2O=32,
    ),

    # ── AFm phases ──
    "Monosulfate": PhaseData(
        name="Monosulfate", formula="Ca4Al2(SO4)(OH)12*6H2O",
        log_Ksp_25C=-29.3, delta_H_rxn=-120000.0,
        molar_volume=309.0e-6, molar_mass=622.5, density=2015.0,
        category="hydrate", n_Ca=4, n_Al=2, n_S_bar=1, n_H2O=12,
    ),
    "Monocarbonate": PhaseData(
        name="Monocarbonate", formula="Ca4Al2(CO3)(OH)12*5H2O",
        log_Ksp_25C=-31.5, delta_H_rxn=-130000.0,
        molar_volume=262.0e-6, molar_mass=568.5, density=2170.0,
        category="hydrate", n_Ca=4, n_Al=2, n_C=1, n_H2O=11,
    ),
    "Hemicarbonate": PhaseData(
        name="Hemicarbonate", formula="Ca4Al2(CO3)0.5(OH)13*5.5H2O",
        log_Ksp_25C=-29.1, delta_H_rxn=-125000.0,
        molar_volume=285.0e-6, molar_mass=564.5, density=1980.0,
        category="hydrate", n_Ca=4, n_Al=2, n_C=0.5, n_H2O=11,
    ),
    "OH_AFm": PhaseData(
        name="OH_AFm", formula="Ca4Al2(OH)14*6H2O",
        log_Ksp_25C=-25.4, delta_H_rxn=-100000.0,
        molar_volume=274.0e-6, molar_mass=560.5, density=2045.0,
        category="hydrate", n_Ca=4, n_Al=2, n_H2O=12,
    ),

    # ── Hydrogarnet ──
    "C3AH6": PhaseData(
        name="C3AH6", formula="Ca3Al2(OH)12",
        log_Ksp_25C=-20.5, delta_H_rxn=-80000.0,
        molar_volume=150.0e-6, molar_mass=378.3, density=2520.0,
        category="hydrate", n_Ca=3, n_Al=2, n_H2O=6,
    ),

    # ── Hydrotalcite ──
    "Hydrotalcite_OH": PhaseData(
        name="Hydrotalcite_OH", formula="Mg4Al2(OH)14*3H2O",
        log_Ksp_25C=-56.0, delta_H_rxn=-250000.0,
        molar_volume=220.0e-6, molar_mass=443.5, density=2016.0,
        category="hydrate", n_Al=2, n_H2O=10,
    ),

    # ── Silica gel (carbonation product of C-S-H) ──
    "SiO2_am": PhaseData(
        name="SiO2_am", formula="SiO2*nH2O",
        log_Ksp_25C=-2.71, delta_H_rxn=-3200.0,
        molar_volume=29.0e-6, molar_mass=60.08, density=2070.0,
        category="other", n_Si=1,
    ),

    # ── Gypsum ──
    "Gypsum": PhaseData(
        name="Gypsum", formula="CaSO4*2H2O",
        log_Ksp_25C=-4.58, delta_H_rxn=-700.0,
        molar_volume=74.7e-6, molar_mass=172.2, density=2305.0,
        category="sulfate", n_Ca=1, n_S_bar=1, n_H2O=2,
    ),

    # ── Strätlingite ──
    "Stratlingite": PhaseData(
        name="Stratlingite", formula="Ca2Al2SiO2(OH)10*3H2O",
        log_Ksp_25C=-19.7, delta_H_rxn=-85000.0,
        molar_volume=216.0e-6, molar_mass=418.3, density=1935.0,
        category="hydrate", n_Ca=2, n_Al=2, n_Si=1, n_H2O=7,
    ),

    # ── Brucite (for GGBS / MgO-rich blends) ──
    "Brucite": PhaseData(
        name="Brucite", formula="Mg(OH)2",
        log_Ksp_25C=-11.16, delta_H_rxn=-26000.0,
        molar_volume=24.6e-6, molar_mass=58.32, density=2370.0,
        category="hydrate", n_H2O=1,
    ),

    # ── Thaumasite (low-T carbonation-sulfate) ──
    "Thaumasite": PhaseData(
        name="Thaumasite", formula="Ca3Si(SO4)(CO3)(OH)6*12H2O",
        log_Ksp_25C=-25.0, delta_H_rxn=-140000.0,
        molar_volume=490.0e-6, molar_mass=622.6, density=1870.0,
        category="hydrate", n_Ca=3, n_Si=1, n_S_bar=1, n_C=1, n_H2O=15,
    ),
}


# ══════════════════════════════════════════════════════════════
#  C-S-H DECALCIFICATION MODEL (CSHQ-inspired)
# ══════════════════════════════════════════════════════════════

class CSHModel:
    """
    Variable Ca/Si C-S-H model inspired by CSHQ (Kulik, 2011).

    Tracks C-S-H decalcification during carbonation:
      Ca/Si 1.7 (OPC) -> 1.2 -> 0.83 -> amorphous SiO2

    The solubility curve is parameterized as a function of Ca/Si.
    """

    # Ca/Si vs log10(Ca2+ activity) -- from CSHQ model / cemdata18
    CASI_SOLUBILITY = np.array([
        # (Ca/Si, log_a_Ca, log_a_Si)
        (0.83, -5.2, -2.8),
        (1.00, -4.0, -3.5),
        (1.20, -3.2, -4.2),
        (1.40, -2.6, -4.8),
        (1.67, -2.2, -5.3),
        (1.80, -2.0, -5.5),
    ])

    def __init__(self, initial_CaSi: float = 1.67, amount_mol: float = 1.0):
        self.CaSi = initial_CaSi
        self.amount = amount_mol  # mol of C-S-H (per SiO2 basis)
        self.initial_CaSi = initial_CaSi
        self.initial_amount = amount_mol

    def decalcify(self, Ca_removed_mol: float):
        """Remove Ca from C-S-H, lowering Ca/Si. Returns actual Ca removed."""
        if self.amount <= 0 or self.CaSi <= 0:
            return 0.0
        max_Ca = self.CaSi * self.amount
        actual = min(Ca_removed_mol, max_Ca)
        if actual > 0:
            self.CaSi = max(0.0, (max_Ca - actual) / self.amount)
        # If Ca/Si drops to ~0, C-S-H dissolves to silica gel
        if self.CaSi < 0.05:
            dissolved_Si = self.amount
            self.amount = 0.0
            self.CaSi = 0.0
            return actual, dissolved_Si
        return actual, 0.0

    def get_solubility(self) -> Tuple[float, float]:
        """Interpolate log(a_Ca), log(a_Si) from CSHQ table at current Ca/Si."""
        casi = np.clip(self.CaSi, 0.83, 1.80)
        log_a_Ca = np.interp(casi, self.CASI_SOLUBILITY[:, 0], self.CASI_SOLUBILITY[:, 1])
        log_a_Si = np.interp(casi, self.CASI_SOLUBILITY[:, 0], self.CASI_SOLUBILITY[:, 2])
        return float(log_a_Ca), float(log_a_Si)

    @property
    def volume(self) -> float:
        """Volume in m3 from interpolated molar volume."""
        Vm_tob = CEMDATA18_PHASES["CSHQ_TobH"].molar_volume
        Vm_jen = CEMDATA18_PHASES["CSHQ_JenH"].molar_volume
        t = np.clip((self.CaSi - 0.83) / (1.67 - 0.83), 0, 1)
        Vm = Vm_tob + t * (Vm_jen - Vm_tob)
        return self.amount * Vm

    @property
    def carbonation_degree(self) -> float:
        """Fraction of Ca removed from initial C-S-H."""
        if self.initial_CaSi * self.initial_amount == 0:
            return 0.0
        Ca_now = self.CaSi * self.amount
        Ca_init = self.initial_CaSi * self.initial_amount
        return max(0, min(1, 1 - Ca_now / Ca_init))


# ══════════════════════════════════════════════════════════════
#  XRF -> INITIAL PHASE ASSEMBLAGE (GEMS full hydration)
# ══════════════════════════════════════════════════════════════

# Oxide molar masses [g/mol]
OXIDE_MW = {
    "CaO": 56.08, "SiO2": 60.08, "Al2O3": 101.96, "Fe2O3": 159.69,
    "MgO": 40.30, "SO3": 80.06, "Na2O": 61.98, "K2O": 94.20,
    "TiO2": 79.87, "P2O5": 141.94, "MnO": 70.94, "LOI": 44.01,
}

# Clinker phase molar masses [g/mol]
CLINKER_MW = {
    "C3S": 228.32,  # 3CaO.SiO2
    "C2S": 172.24,  # 2CaO.SiO2
    "C3A": 270.20,  # 3CaO.Al2O3
    "C4AF": 485.96, # 4CaO.Al2O3.Fe2O3
}


def bogue_calculation(xrf_oxides: Dict[str, float]) -> Dict[str, float]:
    """
    Bogue calculation (ASTM C150): oxide % -> clinker phase %.

    Args:
        xrf_oxides: Mass % of oxides (e.g. {"CaO": 63.0, "SiO2": 20.0, ...})

    Returns:
        Mass % of C3S, C2S, C3A, C4AF
    """
    CaO = xrf_oxides.get("CaO", 0)
    SiO2 = xrf_oxides.get("SiO2", 0)
    Al2O3 = xrf_oxides.get("Al2O3", 0)
    Fe2O3 = xrf_oxides.get("Fe2O3", 0)
    SO3 = xrf_oxides.get("SO3", 0)

    # Free lime correction (simplified: assume minimal)
    CaO_free = xrf_oxides.get("CaO_free", 0)
    CaO_bound = CaO - CaO_free

    # Bogue equations (ASTM C150)
    C4AF = 3.0432 * Fe2O3
    C3A = 2.6504 * Al2O3 - 1.6920 * Fe2O3
    C3A = max(0, C3A)
    C3S = (4.0710 * CaO_bound - 7.6024 * SiO2 -
           6.7187 * Al2O3 - 1.4297 * Fe2O3 - 2.8522 * SO3)
    C3S = max(0, C3S)
    C2S = 2.8675 * SiO2 - 0.7544 * C3S
    C2S = max(0, C2S)

    return {
        "C3S": round(C3S, 2),
        "C2S": round(C2S, 2),
        "C3A": round(C3A, 2),
        "C4AF": round(C4AF, 2),
        "gypsum": round(SO3 * 2.15, 2),  # SO3 -> CaSO4.2H2O equivalent
    }


def compute_initial_hydration_from_xrf(
    xrf_oxides: Dict[str, float],
    wc_ratio: float,
    alpha_hydration: float = 1.0,
    temperature_C: float = 25.0,
) -> Dict:
    """
    Compute initial phase assemblage from XRF oxide data via GEMS-style
    full hydration (assumes Gibbs energy minimization at complete hydration).

    Stoichiometric hydration reactions (at 25 C, from cemdata18):
      C3S + 5.3 H   -> C1.7SH4 + 1.3 CH
      C2S + 4.3 H   -> C1.7SH4 + 0.3 CH
      C3A + 3CSH2 + 26H -> C6AS3H32 (ettringite, if enough gypsum)
      2C3A + C6AS3H32 + 4H -> 3 C4ASH12 (monosulfate, if excess C3A)
      C3A + CH + 12 H -> C4AH13 (if no sulfate)
      C4AF -> similar to C3A with Fe substitution -> hydrogarnet + Fe-AFm

    Args:
        xrf_oxides: XRF oxide composition (mass %)
        wc_ratio: water/cement ratio by mass
        alpha_hydration: degree of hydration (0-1), default 1 (full)
        temperature_C: hydration temperature

    Returns:
        Dict with:
          - "clinker_bogue": Bogue clinker phases
          - "phases_mol_per_kg": mol of each hydrate per kg cement
          - "phases_volume_frac": volume fractions (including water/air)
          - "porosity": total porosity (capillary + gel)
          - "CSH_CaSi": Ca/Si of C-S-H
          - "water_bound_g_per_kg": chemically bound water
          - "xrf": input XRF
          - "wc_ratio", "alpha"
    """
    clinker = bogue_calculation(xrf_oxides)

    # Per kg cement
    n_C3S  = clinker["C3S"]  / 100 * 1000 / CLINKER_MW["C3S"]   # mol/kg
    n_C2S  = clinker["C2S"]  / 100 * 1000 / CLINKER_MW["C2S"]
    n_C3A  = clinker["C3A"]  / 100 * 1000 / CLINKER_MW["C3A"]
    n_C4AF = clinker["C4AF"] / 100 * 1000 / CLINKER_MW["C4AF"]

    # Apply degree of hydration
    n_C3S  *= alpha_hydration
    n_C2S  *= alpha_hydration
    n_C3A  *= alpha_hydration
    n_C4AF *= alpha_hydration

    # Gypsum amount (mol/kg cement)
    SO3 = xrf_oxides.get("SO3", 0)
    n_gypsum = (SO3 / 100) * 1000 / OXIDE_MW["SO3"]  # mol SO4 / kg

    # ── C-S-H and CH from silicate hydration ──
    # C3S + 5.3 H2O -> C1.7SH4 + 1.3 CH
    # C2S + 4.3 H2O -> C1.7SH4 + 0.3 CH
    n_CSH = n_C3S + n_C2S  # mol C1.7SH4 per kg cement
    n_CH  = 1.3 * n_C3S + 0.3 * n_C2S

    # ── Aluminate hydration: distribute C3A between ettringite and monosulfate ──
    # Priority: all gypsum forms ettringite first, excess C3A -> monosulfate
    # Ettringite: 1 C3A + 3 CaSO4 + 26 H -> 1 C6AS3H32
    # Monosulfate: 2 C3A + 1 C6AS3H32 + 4 H -> 3 C4ASH12
    # We model this via SO3/Al2O3 ratio in moles

    n_C3A_total = n_C3A + 0.5 * n_C4AF  # C4AF contributes ~half to aluminate pool

    # Phase 1: Ettringite formation (limited by gypsum)
    n_ettringite_max = n_gypsum / 3  # each ettringite uses 3 sulfate
    n_ettringite = min(n_ettringite_max, n_C3A_total)
    n_C3A_after_ett = n_C3A_total - n_ettringite
    n_gypsum_used = 3 * n_ettringite

    # Phase 2: Remaining C3A converts ettringite to monosulfate (if SO3 limited)
    # Simplified: if excess C3A, form monosulfate and monocarbonate
    n_monosulfate = 0.0
    if n_C3A_after_ett > 0 and n_ettringite > 0:
        # 2 C3A + 1 Ett + 4H -> 3 Ms
        n_convert = min(n_C3A_after_ett / 2, n_ettringite)
        n_ettringite -= n_convert
        n_monosulfate = 3 * n_convert
        n_C3A_after_ett -= 2 * n_convert

    # Phase 3: Remaining C3A -> C4AH13 (OH-AFm) or hydrogarnet (C3AH6)
    # In well-hydrated OPC, we get mostly C4AH13 at early ages, converting to C3AH6
    n_OH_AFm = max(0, n_C3A_after_ett) * 0.5
    n_C3AH6 = max(0, n_C3A_after_ett) * 0.5

    # ── Fe: form hydrogarnet or Fe-AFm ──
    # Simplified: all Fe ends up in hydrogarnet + monosulfate (already counted)
    # Add Fe contribution to hydrogarnet
    n_hydrogarnet = n_C4AF * 0.5 + n_C3AH6

    # ── Mg: form hydrotalcite or brucite ──
    MgO = xrf_oxides.get("MgO", 0)
    n_MgO = (MgO / 100) * 1000 / OXIDE_MW["MgO"]
    n_hydrotalcite = n_MgO / 4  # Mg4Al2(OH)14.3H2O
    n_brucite = max(0, n_MgO - 4 * n_hydrotalcite)

    # ── Water consumption (chemically bound water) ──
    # Per reaction: CH: 1 H, CSH: 4 H (in C1.7SH4), ettringite: 32 H, monosulfate: 12 H
    w_CH = n_CH * 1
    w_CSH = n_CSH * 4.0  # C1.7SH4 has 4 H2O
    w_ett = n_ettringite * 32
    w_Ms = n_monosulfate * 12
    w_AFm = n_OH_AFm * 12
    w_HG = n_hydrogarnet * 6
    w_HT = n_hydrotalcite * 10
    n_H2O_bound = w_CH + w_CSH + w_ett + w_Ms + w_AFm + w_HG + w_HT  # mol/kg cement
    bound_water_g = n_H2O_bound * 18.015  # g/kg cement

    # ── Volume calculation (per kg cement) ──
    # Convert mol -> m3 using molar volumes from cemdata18
    def Vm(name):
        return CEMDATA18_PHASES[name].molar_volume if name in CEMDATA18_PHASES else 0

    V_CH          = n_CH * Vm("Portlandite")
    V_CSH         = n_CSH * Vm("CSHQ_JenH")  # Ca/Si = 1.67 OPC
    V_ett         = n_ettringite * Vm("Ettringite")
    V_Ms          = n_monosulfate * Vm("Monosulfate")
    V_AFm         = n_OH_AFm * Vm("OH_AFm")
    V_HG          = n_hydrogarnet * Vm("C3AH6")
    V_HT          = n_hydrotalcite * Vm("Hydrotalcite_OH")
    V_brucite     = n_brucite * Vm("Brucite")

    V_hydrates = V_CH + V_CSH + V_ett + V_Ms + V_AFm + V_HG + V_HT + V_brucite

    # ── Powers model for total volume ──
    # Initial mix: V_cement + V_water
    rho_cement = 3150  # kg/m3 (typical OPC)
    V_cement_initial = 1.0 / rho_cement  # m3/kg
    V_water_initial = wc_ratio / 1000.0  # m3/kg (water density 1000 kg/m3)
    V_total = V_cement_initial + V_water_initial

    # Unreacted cement
    V_cement_unreacted = V_cement_initial * (1 - alpha_hydration)

    # Capillary water remaining
    V_water_consumed = bound_water_g / 1000 / 1000  # g -> kg -> m3 (rho_H2O=1000)
    V_capillary = max(0, V_water_initial - V_water_consumed)

    # Gel porosity (Powers: ~28% of C-S-H volume)
    V_gel_pore = 0.28 * V_CSH

    # Total solid volume
    V_solid = V_hydrates + V_cement_unreacted

    # Chemical shrinkage (Le Chatelier): hydrates are smaller than cement + water
    V_shrinkage = max(0, V_total - V_solid - V_capillary - V_gel_pore)

    # Volume fractions (normalized to V_total)
    vf = {
        "CH":             V_CH / V_total,
        "CSH":            V_CSH / V_total,
        "ettringite":     V_ett / V_total,
        "monosulfate":    V_Ms / V_total,
        "OH_AFm":         V_AFm / V_total,
        "C3AH6":          V_HG / V_total,
        "hydrotalcite":   V_HT / V_total,
        "brucite":        V_brucite / V_total,
        "unreacted_cement": V_cement_unreacted / V_total,
        "capillary_pore": V_capillary / V_total,
        "gel_pore":       V_gel_pore / V_total,
        "chem_shrinkage": V_shrinkage / V_total,
    }
    vf = {k: v for k, v in vf.items() if v > 1e-8}
    porosity_total = vf.get("capillary_pore", 0) + vf.get("gel_pore", 0)

    # Mol amounts (per m3 of paste)
    phases_mol_m3 = {
        "Portlandite": n_CH / V_total,
        "CSH":         n_CSH / V_total,
        "Ettringite":  n_ettringite / V_total,
        "Monosulfate": n_monosulfate / V_total,
        "OH_AFm":      n_OH_AFm / V_total,
        "C3AH6":       n_hydrogarnet / V_total,
        "Hydrotalcite_OH": n_hydrotalcite / V_total,
        "Brucite":     n_brucite / V_total,
    }
    phases_mol_m3 = {k: v for k, v in phases_mol_m3.items() if v > 1e-8}

    return {
        "xrf": xrf_oxides,
        "wc_ratio": wc_ratio,
        "alpha_hydration": alpha_hydration,
        "temperature_C": temperature_C,
        "clinker_bogue": clinker,
        "phases_mol_per_m3": phases_mol_m3,
        "phases_volume_frac": vf,
        "porosity": porosity_total,
        "capillary_porosity": vf.get("capillary_pore", 0),
        "gel_porosity": vf.get("gel_pore", 0),
        "CSH_CaSi": 1.67,  # OPC default, can be modified by SCMs
        "bound_water_g_per_kg": round(bound_water_g, 2),
        "V_total_m3_per_kg": V_total,
    }


# Reference XRF compositions for common cements
REFERENCE_XRF = {
    "OPC_CEM_I": {
        "CaO": 63.5, "SiO2": 20.0, "Al2O3": 5.0, "Fe2O3": 3.0,
        "MgO": 1.5, "SO3": 3.0, "Na2O": 0.2, "K2O": 0.8, "LOI": 3.0,
    },
    "OPC_low_C3A": {
        "CaO": 64.0, "SiO2": 21.0, "Al2O3": 3.5, "Fe2O3": 5.0,
        "MgO": 1.5, "SO3": 2.5, "LOI": 2.5,
    },
    "white_cement": {
        "CaO": 68.0, "SiO2": 23.0, "Al2O3": 4.5, "Fe2O3": 0.3,
        "MgO": 0.8, "SO3": 2.2, "LOI": 1.2,
    },
    "CSA": {  # Calcium sulfoaluminate
        "CaO": 44.0, "SiO2": 8.0, "Al2O3": 32.0, "Fe2O3": 2.0,
        "MgO": 1.5, "SO3": 11.0, "LOI": 1.5,
    },
}


# ══════════════════════════════════════════════════════════════
#  GEMS-LITE EQUILIBRIUM SOLVER
# ══════════════════════════════════════════════════════════════

@dataclass
class GEMSState:
    """State of the thermodynamic system at a given equilibrium step."""
    phase_amounts: Dict[str, float]        # mol of each phase
    phase_volumes: Dict[str, float]        # m3 of each phase
    phase_volume_fracs: Dict[str, float]   # volume fractions
    aqueous: Dict[str, float]              # mol of dissolved species
    pH: float
    Ca_aq: float          # mol/L Ca2+ in solution
    CO3_aq: float         # mol/L CO3^2- in solution
    total_volume: float   # m3
    porosity: float
    CaSi_CSH: float       # current Ca/Si of C-S-H
    carbonation_degree: float
    CO2_added_total: float  # cumulative mol CO2 added


class GEMSCarbonationEngine:
    """
    Lightweight Gibbs Energy Minimization solver for concrete carbonation.

    Implements sequential local equilibrium: at each step, a dose of
    DISSOLVED CO2 (already partitioned from gas via Henry's law, already
    speciated to CO3^2- at high pore-solution pH) is added to the system,
    and phases are re-equilibrated based on their relative stability
    (solubility products from cemdata18).

    Carbonation sequence (from thermodynamics):
      1. Portlandite dissolves first (most soluble Ca source)
      2. Ettringite/Monosulfate decompose -> gypsum + Al(OH)3 + carbonate
      3. C-S-H decalcifies (Ca/Si decreases)
      4. CaCO3 precipitates (calcite >> vaterite >> aragonite)
      5. pH drops: 13.0 -> 12.5 -> ~10 -> 8.3

    Gas/aqueous separation of concerns:
      * Gas-phase CO2 transport through the air-filled pore network is
        handled by `transport_engine.py` (Millington-Quirk family).
      * Henry's law partitioning c_aq = K_H(T) * p_CO2 converts the gas
        field to a dissolved field at each depth.
      * THIS class only consumes the dissolved carbonate dose. Hydrates
        produced by `from_xrf` stay unchanged until such a dose arrives.

    Fast-reaction (Damkoehler >> 1) assumption:
      Transport time scales (years) >> local aqueous reaction time
      scales (seconds). Therefore the hydrates and the pore solution at
      every depth are in local chemical equilibrium at every instant,
      and the carbonation front is a moving sharp surface
      x_c(t) = K * sqrt(t), as in the Papadakis analytical solution.

    Temperature dependence via van't Hoff equation.
    """

    # Carbonation reaction priority (lower = reacts first)
    REACTION_PRIORITY = [
        "Portlandite",      # dissolves first
        "Ettringite",       # decomposes -> gypsum + Al(OH)3
        "Monosulfate",      # decomposes -> monocarbonate
        "Hemicarbonate",    # converts to monocarbonate
        "OH_AFm",           # carbonated
        "CSH",              # decalcifies last (slowest)
    ]

    def __init__(self,
                 cement_type: str = "OPC",
                 scm_mix: Optional[Dict[str, float]] = None,
                 wc_ratio: float = 0.50,
                 temperature_C: float = 20.0,
                 system_volume_L: float = 1.0):
        """
        Args:
            cement_type: "OPC", "CSA", "white_cement"
            scm_mix: e.g. {"fly_ash_F": 0.30, "GGBS": 0.20}
            wc_ratio: water/cement ratio
            temperature_C: Temperature in Celsius
            system_volume_L: Reference volume in liters
        """
        self.cement_type = cement_type
        self.scm_mix = scm_mix or {}
        self.wc_ratio = wc_ratio
        self.T_C = temperature_C
        self.T_K = temperature_C + 273.15
        self.system_volume = system_volume_L * 1e-3  # m3

        # Import phase data from chemistry_engine for blend construction
        from .chemistry_engine import build_cement_system, BASE_CEMENTS
        if scm_mix:
            blend = build_cement_system(cement_type, scm_mix)
        else:
            blend = BASE_CEMENTS[cement_type]

        hp = blend["hydration_products"]
        self.CaSi_initial = blend.get("CSH_CaSi", 1.67)

        # Convert volume fractions to moles (normalize to system_volume)
        self._init_phases(hp)
        self._co2_added_total = 0.0
        self._history: List[GEMSState] = []

    @classmethod
    def from_xrf(cls,
                 xrf_oxides: Dict[str, float],
                 wc_ratio: float = 0.50,
                 alpha_hydration: float = 1.0,
                 temperature_C: float = 25.0,
                 system_volume_L: float = 1.0) -> "GEMSCarbonationEngine":
        """
        Create engine from XRF oxide data via GEMS full hydration.

        The thermodynamics expert uses this as its PRIMARY entry point: given
        raw material composition (XRF) and w/c, compute the initial phase
        assemblage assuming Gibbs energy minimization at complete hydration.

        Args:
            xrf_oxides: XRF mass % {"CaO": ..., "SiO2": ..., ...}
            wc_ratio: water/cement mass ratio
            alpha_hydration: degree of hydration (1 = full)
            temperature_C: hydration temperature
            system_volume_L: reference volume for GEMS calculation

        Returns:
            GEMSCarbonationEngine initialized at hydrated (un-carbonated) state.
        """
        engine = cls.__new__(cls)  # bypass __init__
        engine.cement_type = "XRF-custom"
        engine.scm_mix = {}
        engine.wc_ratio = wc_ratio
        engine.T_C = temperature_C
        engine.T_K = temperature_C + 273.15
        engine.system_volume = system_volume_L * 1e-3
        engine._co2_added_total = 0.0
        engine._history = []

        # Full hydration from XRF
        hydration = compute_initial_hydration_from_xrf(
            xrf_oxides, wc_ratio, alpha_hydration, temperature_C,
        )
        engine._initial_hydration = hydration
        engine.CaSi_initial = hydration["CSH_CaSi"]

        # Transfer volume fractions into the engine using same API
        hp = hydration["phases_volume_frac"].copy()
        # Map XRF output names -> chemistry_engine names used by _init_phases
        hp_mapped = {
            "CH":           hp.get("CH", 0),
            "CSH":          hp.get("CSH", 0),
            "ettringite":   hp.get("ettringite", 0),
            "monosulfate":  hp.get("monosulfate", 0),
            "AH3":          hp.get("C3AH6", 0),  # hydrogarnet
            "hydrotalcite": hp.get("hydrotalcite", 0),
            "porosity":     hp.get("capillary_pore", 0) + hp.get("gel_pore", 0),
        }
        engine._init_phases(hp_mapped)
        return engine

    def get_initial_hydration(self) -> Optional[Dict]:
        """Return the XRF-derived initial hydration result (if from_xrf used)."""
        return getattr(self, "_initial_hydration", None)

    def get_max_carbonation_capacity(self) -> float:
        """
        Max mol CO2 that can be absorbed before full carbonation (stoichiometric).

        Each mol Ca in CH and CSH can bind 1 mol CO2.
        """
        n_Ca_CH = self.phases.get("Portlandite", 0)
        n_Ca_CSH = self.csh.CaSi * self.csh.amount
        n_Ca_ett = self.phases.get("Ettringite", 0) * 6
        n_Ca_Ms = self.phases.get("Monosulfate", 0) * 4
        return n_Ca_CH + n_Ca_CSH + n_Ca_ett + n_Ca_Ms

    def _init_phases(self, hp: Dict[str, float]):
        """Initialize phase amounts from hydration product volume fractions."""
        self.phases: Dict[str, float] = {}  # mol

        # Map chemistry_engine names -> GEMS phase names
        NAME_MAP = {
            "CH": "Portlandite",
            "CSH": None,  # handled separately via CSHModel
            "ettringite": "Ettringite",
            "monosulfate": "Monosulfate",
            "monocarboaluminate": "Monocarbonate",
            "stratlingite": "Stratlingite",
            "hydrotalcite": "Hydrotalcite_OH",
            "AH3": "C3AH6",
        }

        porosity_frac = hp.get("porosity", 0.25)
        solid_volume = self.system_volume * (1 - porosity_frac)

        for name, vol_frac in hp.items():
            if name in ("porosity", "other"):
                continue
            gems_name = NAME_MAP.get(name)
            if gems_name and gems_name in CEMDATA18_PHASES:
                pd = CEMDATA18_PHASES[gems_name]
                phase_vol = vol_frac * self.system_volume
                self.phases[gems_name] = phase_vol / pd.molar_volume if pd.molar_volume > 0 else 0
            elif name == "CSH":
                pass  # handled below

        # C-S-H model
        csh_vol = hp.get("CSH", 0.40) * self.system_volume
        Vm_csh = CEMDATA18_PHASES["CSHQ_JenH"].molar_volume
        csh_mol = csh_vol / Vm_csh if Vm_csh > 0 else 0
        self.csh = CSHModel(initial_CaSi=self.CaSi_initial, amount_mol=csh_mol)

        # Carbonates start at 0
        self.phases["Calcite"] = 0.0
        self.phases["Vaterite"] = 0.0
        self.phases["SiO2_am"] = 0.0
        self.phases["Gypsum"] = 0.0

        # Pore solution
        self.pore_water_mol = (porosity_frac * self.system_volume * 1e6 *
                               0.65 / 18.015)  # approximate moles of H2O at S=0.65
        self.Ca_aq = 0.020  # mol/L initial (saturated wrt CH)
        self.CO3_aq = 0.0   # mol/L
        self.pH = 13.0

    def log_Ksp_at_T(self, phase_name: str) -> float:
        """Temperature-corrected log Ksp via van't Hoff equation."""
        pd = CEMDATA18_PHASES[phase_name]
        if abs(self.T_K - 298.15) < 0.1:
            return pd.log_Ksp_25C
        # van't Hoff: ln(K2/K1) = -dH/R * (1/T2 - 1/T1)
        ln_K_ratio = -pd.delta_H_rxn / R_GAS * (1/self.T_K - 1/298.15)
        return pd.log_Ksp_25C + ln_K_ratio / np.log(10)

    def _compute_pH(self) -> float:
        """Compute pH from phase equilibria.

        Simplified model:
          - CH present: pH ~ 12.4-13.0 (controlled by Ca(OH)2 solubility)
          - CH gone, CSH Ca/Si > 1.0: pH ~ 10.5-12.5
          - CSH Ca/Si < 1.0: pH ~ 9.0-10.5
          - Only CaCO3: pH ~ 8.3
        """
        ch = self.phases.get("Portlandite", 0)
        if ch > 1e-6:
            # CH buffered: pH depends on amount remaining
            return 12.45 + 0.1 * np.log10(max(ch, 1e-10))

        casi = self.csh.CaSi
        if self.csh.amount > 1e-6 and casi > 1.0:
            return 10.5 + 2.0 * (casi - 1.0) / 0.67
        elif self.csh.amount > 1e-6 and casi > 0.5:
            return 9.0 + 3.0 * (casi - 0.5) / 0.5
        else:
            return 8.3  # calcite-water equilibrium

    def _dissolve_portlandite(self, CO2_mol: float) -> float:
        """CH + CO2 -> CaCO3 + H2O. Returns CO2 consumed."""
        ch = self.phases.get("Portlandite", 0)
        if ch <= 0 or CO2_mol <= 0:
            return 0.0
        consumed = min(ch, CO2_mol)  # 1:1 stoichiometry
        self.phases["Portlandite"] -= consumed
        self.phases["Calcite"] += consumed
        return consumed

    def _decompose_ettringite(self, CO2_mol: float) -> float:
        """Ettringite + 3CO2 -> 3CaCO3 + 3Gypsum + 2Al(OH)3 + 26H2O.
        Simplified: each mol ettringite consumes 3 mol CO2."""
        ett = self.phases.get("Ettringite", 0)
        if ett <= 0 or CO2_mol <= 0:
            return 0.0
        # Each mol ettringite can consume 3 mol CO2 (3 Ca -> CaCO3)
        max_co2 = ett * 3
        consumed = min(max_co2, CO2_mol)
        ett_dissolved = consumed / 3
        self.phases["Ettringite"] -= ett_dissolved
        self.phases["Calcite"] += consumed  # 3 CaCO3 per ettringite
        self.phases["Gypsum"] = self.phases.get("Gypsum", 0) + ett_dissolved * 3
        return consumed

    def _decompose_monosulfate(self, CO2_mol: float) -> float:
        """Ms + CO2 -> Monocarbonate (or Calcite + Gypsum at higher CO2).
        At low CO2: Ms -> Mc (monocarbonate). At high CO2: Mc decomposes too."""
        ms = self.phases.get("Monosulfate", 0)
        mc = self.phases.get("Monocarbonate", 0)
        hc = self.phases.get("Hemicarbonate", 0)

        consumed = 0.0

        # Step 1: Hemicarbonate -> Monocarbonate (consumes 0.5 CO2 per mol)
        if hc > 0 and CO2_mol > 0:
            c = min(hc, CO2_mol / 0.5)
            self.phases["Hemicarbonate"] -= c
            self.phases["Monocarbonate"] = mc + c
            mc += c
            used = c * 0.5
            CO2_mol -= used
            consumed += used

        # Step 2: Monosulfate -> Monocarbonate (consumes 1 CO2 per mol)
        if ms > 0 and CO2_mol > 0:
            c = min(ms, CO2_mol)
            self.phases["Monosulfate"] -= c
            self.phases["Monocarbonate"] = self.phases.get("Monocarbonate", 0) + c
            CO2_mol -= c
            consumed += c

        # Step 3: at high CO2, Monocarbonate decomposes -> CaCO3 + Al(OH)3
        if CO2_mol > 0:
            mc_now = self.phases.get("Monocarbonate", 0)
            if mc_now > 0:
                c = min(mc_now, CO2_mol / 3)  # 3 Ca per Mc -> 3 CaCO3
                self.phases["Monocarbonate"] -= c
                self.phases["Calcite"] += c * 4  # 4 Ca in monocarbonate
                used = c * 3
                CO2_mol -= used
                consumed += used

        return consumed

    def _decalcify_csh(self, CO2_mol: float) -> float:
        """C-S-H decalcification: Ca removed from C-S-H -> CaCO3."""
        if CO2_mol <= 0 or self.csh.amount <= 0:
            return 0.0
        Ca_removed, Si_released = self.csh.decalcify(CO2_mol)
        self.phases["Calcite"] += Ca_removed
        if Si_released > 0:
            self.phases["SiO2_am"] = self.phases.get("SiO2_am", 0) + Si_released
        return Ca_removed

    def equilibrate_step(self, CO2_mol: float) -> GEMSState:
        """
        Add dissolved CO2 (mol) to the pore solution and re-equilibrate.

        IMPORTANT — physical meaning of `CO2_mol`:
            This is NOT the gaseous CO2 in the pore air. It is the
            cumulative DISSOLVED carbonate dose (as CO3^2- equivalent)
            that has reached this control volume via the sequence
                CO2(g) ---diffusion---> CO2(g) at wall
                     ---Henry---> CO2(aq) ---speciation---> CO3^2-
            The transport solver (or Papadakis fast-reaction limit)
            provides this monotone-in-time dose field; GEMS only handles
            the aqueous reaction step.

        Effect on the phase assemblage:
            Hydrates remain exactly as produced by `from_xrf` UNLESS
            dissolved CO2 is delivered through this routine. When called,
            dissolved carbonate is consumed in thermodynamic priority
            order (most undersaturated first):

                1. Portlandite  (CH + CO3^2- -> CaCO3 + 2 OH-)
                2. Ettringite   (decomposes, releasing Ca and sulfate)
                3. Monosulfate / AFm phases
                4. C-S-H decalcification (Ca/Si drops towards SiO2(am))

            Each depleted hydrate releases Ca^2+ which immediately
            precipitates as CaCO3 (calcite). Excess dissolved CO2 (after
            all Ca sinks are saturated) stays in solution and drops the
            pH towards 8.3.
        """
        remaining = CO2_mol

        # Priority-ordered reactions
        if remaining > 0:
            remaining -= self._dissolve_portlandite(remaining)
        if remaining > 0:
            remaining -= self._decompose_ettringite(remaining)
        if remaining > 0:
            remaining -= self._decompose_monosulfate(remaining)
        if remaining > 0:
            remaining -= self._decalcify_csh(remaining)

        # Excess CO2 stays in solution
        self.CO3_aq += remaining * 1000 / max(self.system_volume * 1e3, 1e-10)

        self._co2_added_total += CO2_mol
        self.pH = self._compute_pH()

        # Ca in solution
        if self.phases.get("Portlandite", 0) > 1e-6:
            log_Ksp = self.log_Ksp_at_T("Portlandite")
            self.Ca_aq = 10**(log_Ksp / 2)  # simplified: Ksp ~ [Ca][OH]^2
        else:
            log_a_Ca, _ = self.csh.get_solubility()
            self.Ca_aq = 10**log_a_Ca

        state = self._get_state()
        self._history.append(state)
        return state

    def _get_state(self) -> GEMSState:
        """Compute current state snapshot."""
        phase_volumes = {}
        for name, mol in self.phases.items():
            if name in CEMDATA18_PHASES and mol > 0:
                phase_volumes[name] = mol * CEMDATA18_PHASES[name].molar_volume
            else:
                phase_volumes[name] = 0.0

        # C-S-H volume
        phase_volumes["CSH"] = self.csh.volume
        phase_amounts = dict(self.phases)
        phase_amounts["CSH"] = self.csh.amount

        total_solid = sum(v for v in phase_volumes.values() if v > 0)
        total_vol = max(self.system_volume, total_solid * 1.01)
        porosity = max(0, 1 - total_solid / total_vol)

        phase_vf = {k: v / total_vol for k, v in phase_volumes.items() if v > 0}

        # Carbonation degree: fraction of initial Ca in CH+CSH now in carbonates
        Ca_init = (self.phases.get("_init_CH_mol", 0) * 1 +
                   self.csh.initial_CaSi * self.csh.initial_amount)
        if Ca_init == 0:
            Ca_init = 1.0  # prevent division by zero
        Ca_carb = (self.phases.get("Calcite", 0) +
                   self.phases.get("Vaterite", 0) +
                   self.phases.get("Aragonite", 0))
        carb_deg = min(1.0, Ca_carb / Ca_init)

        return GEMSState(
            phase_amounts={k: v for k, v in phase_amounts.items() if v > 1e-12},
            phase_volumes={k: v for k, v in phase_volumes.items() if v > 1e-12},
            phase_volume_fracs=phase_vf,
            aqueous={"Ca2+": self.Ca_aq, "CO3_2-": self.CO3_aq},
            pH=self.pH,
            Ca_aq=self.Ca_aq,
            CO3_aq=self.CO3_aq,
            total_volume=total_vol,
            porosity=porosity,
            CaSi_CSH=self.csh.CaSi,
            carbonation_degree=carb_deg,
            CO2_added_total=self._co2_added_total,
        )

    def run_carbonation(self, total_CO2_mol: float, n_steps: int = 100) -> List[GEMSState]:
        """
        Run stepwise carbonation by adding CO2 incrementally.

        Args:
            total_CO2_mol: Total CO2 to add (in moles per system_volume)
            n_steps: Number of equilibrium steps

        Returns:
            List of GEMSState at each step
        """
        # Store initial CH amount for carbonation degree calculation
        self.phases["_init_CH_mol"] = self.phases.get("Portlandite", 0)

        co2_per_step = total_CO2_mol / n_steps
        results = []
        for _ in range(n_steps):
            state = self.equilibrate_step(co2_per_step)
            results.append(state)
        return results

    def get_phase_evolution_arrays(self, results: Optional[List[GEMSState]] = None
                                   ) -> Dict[str, np.ndarray]:
        """Convert results to arrays for plotting. Keys are phase names, values are arrays."""
        if results is None:
            results = self._history
        if not results:
            return {}

        co2 = np.array([r.CO2_added_total for r in results])
        phases = set()
        for r in results:
            phases.update(r.phase_volume_fracs.keys())

        out = {"CO2_added": co2, "pH": np.array([r.pH for r in results]),
               "carbonation_degree": np.array([r.carbonation_degree for r in results]),
               "porosity": np.array([r.porosity for r in results]),
               "CaSi_CSH": np.array([r.CaSi_CSH for r in results])}
        for p in sorted(phases):
            out[p] = np.array([r.phase_volume_fracs.get(p, 0) for r in results])
        return out

    @property
    def history(self) -> List[GEMSState]:
        return self._history

    def summary(self) -> Dict:
        """Return summary of current state."""
        state = self._get_state()
        return {
            "cement_type": self.cement_type,
            "scm_mix": self.scm_mix,
            "temperature_C": self.T_C,
            "wc_ratio": self.wc_ratio,
            "pH": round(state.pH, 2),
            "carbonation_degree": round(state.carbonation_degree, 4),
            "CaSi_CSH": round(state.CaSi_CSH, 3),
            "porosity": round(state.porosity, 4),
            "Portlandite_mol": round(self.phases.get("Portlandite", 0), 6),
            "Calcite_mol": round(self.phases.get("Calcite", 0), 6),
            "CSH_mol": round(self.csh.amount, 6),
            "Ettringite_mol": round(self.phases.get("Ettringite", 0), 6),
            "phase_volumes": {k: f"{v:.3e}" for k, v in state.phase_volumes.items()},
            "phase_volume_fracs": {k: round(v, 4) for k, v in state.phase_volume_fracs.items()},
        }


# ══════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_cemdata18_phase(name: str) -> Optional[PhaseData]:
    """Look up a phase in the cemdata18 database."""
    return CEMDATA18_PHASES.get(name)


def list_cemdata18_phases() -> Dict[str, str]:
    """Return dict of phase name -> formula."""
    return {name: pd.formula for name, pd in CEMDATA18_PHASES.items()}


def compute_gems_carbonation(
    cement_type: str = "OPC",
    scm_mix: Optional[Dict[str, float]] = None,
    wc_ratio: float = 0.50,
    temperature_C: float = 20.0,
    CO2_total_mol: float = 0.5,
    n_steps: int = 100,
) -> Dict:
    """
    Run a complete GEMS carbonation analysis.

    Args:
        cement_type: Base cement
        scm_mix: SCM blend ratios
        wc_ratio: w/c ratio
        temperature_C: Temperature
        CO2_total_mol: Total CO2 added (mol per L of paste)
        n_steps: Equilibration steps

    Returns:
        Dict with phase evolution arrays and summary
    """
    engine = GEMSCarbonationEngine(
        cement_type=cement_type, scm_mix=scm_mix,
        wc_ratio=wc_ratio, temperature_C=temperature_C,
    )
    results = engine.run_carbonation(CO2_total_mol, n_steps)
    arrays = engine.get_phase_evolution_arrays(results)
    return {
        "summary": engine.summary(),
        "arrays": arrays,
        "states": results,
        "engine": engine,
    }


def plot_gems_carbonation(results: Dict, title: str = ""):
    """6-panel GEMS carbonation plot."""
    import matplotlib.pyplot as plt

    arrays = results["arrays"]
    co2 = arrays["CO2_added"]
    summary = results["summary"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Phase volume fractions
    ax = axes[0, 0]
    phase_keys = ["Portlandite", "CSH", "Calcite", "Ettringite",
                  "Monosulfate", "Monocarbonate", "SiO2_am", "Gypsum"]
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800",
              "#9C27B0", "#795548", "#607D8B", "#FFEB3B"]
    for pk, color in zip(phase_keys, colors):
        if pk in arrays and np.max(arrays[pk]) > 1e-5:
            ax.plot(co2, arrays[pk], label=pk, color=color, linewidth=2)
    ax.set_xlabel("CO$_2$ added (mol)")
    ax.set_ylabel("Volume fraction")
    ax.set_title("1. Phase Assemblage (cemdata18)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. pH evolution
    ax = axes[0, 1]
    ax.plot(co2, arrays["pH"], "b-", linewidth=2)
    ax.axhline(y=9.0, color="r", linestyle="--", alpha=0.5, label="pH 9 (depassivation)")
    ax.set_xlabel("CO$_2$ added (mol)")
    ax.set_ylabel("pH")
    ax.set_title("2. pH Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Ca/Si of C-S-H
    ax = axes[0, 2]
    ax.plot(co2, arrays["CaSi_CSH"], "g-", linewidth=2)
    ax.axhline(y=0.83, color="gray", linestyle=":", label="Tobermorite limit")
    ax.axhline(y=1.67, color="gray", linestyle="--", label="Jennite (OPC)")
    ax.set_xlabel("CO$_2$ added (mol)")
    ax.set_ylabel("Ca/Si ratio")
    ax.set_title("3. C-S-H Decalcification (CSHQ)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Carbonation degree
    ax = axes[1, 0]
    ax.plot(co2, arrays["carbonation_degree"], "r-", linewidth=2)
    ax.set_xlabel("CO$_2$ added (mol)")
    ax.set_ylabel("Carbonation degree")
    ax.set_title("4. Carbonation Degree")
    ax.grid(True, alpha=0.3)

    # 5. Porosity evolution
    ax = axes[1, 1]
    ax.plot(co2, arrays["porosity"], "k-", linewidth=2)
    ax.set_xlabel("CO$_2$ added (mol)")
    ax.set_ylabel("Porosity")
    ax.set_title("5. Porosity Change")
    ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    blend = summary["cement_type"]
    if summary["scm_mix"]:
        parts = [f"{int(r*100)}%{n}" for n, r in summary["scm_mix"].items()]
        blend += " + " + " + ".join(parts)
    text = (
        f"GEMS Carbonation Summary\n"
        f"{'_' * 40}\n"
        f"Blend:           {blend}\n"
        f"w/c ratio:       {summary['wc_ratio']}\n"
        f"Temperature:     {summary['temperature_C']} C\n"
        f"{'_' * 40}\n"
        f"Final pH:        {summary['pH']}\n"
        f"Carb. degree:    {summary['carbonation_degree']:.4f}\n"
        f"Ca/Si (CSH):     {summary['CaSi_CSH']:.3f}\n"
        f"Porosity:        {summary['porosity']:.4f}\n"
        f"{'_' * 40}\n"
        f"CH remaining:    {summary['Portlandite_mol']:.4f} mol\n"
        f"Calcite formed:  {summary['Calcite_mol']:.4f} mol\n"
        f"CSH remaining:   {summary['CSH_mol']:.4f} mol\n"
        f"Database:        cemdata18\n"
    )
    ax.text(0.05, 0.5, text, fontsize=10, family="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    suptitle = title or f"GEMS Thermodynamic Carbonation: {blend}"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig
