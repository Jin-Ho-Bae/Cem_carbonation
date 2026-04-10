"""
Micromechanics Engine -- Multi-Level Eshelby-Mori-Tanaka Homogenization.

6-level homogenization for cementitious materials (Haile et al. 2019):

  Level I:   MD nanoscale -- jennite/tobermorite crystal properties
  Level II:  CSH variants -- LD-CSH, HD-CSH, UHD-CSH (crystal + gel porosity)
  Level III: CSH matrix   -- combine LD/HD/UHD into general CSH matrix
  Level IV:  Cement paste  -- CSH matrix + CH + capillary pores + unreacted clinker
  Level V:   Mortar        -- paste + sand + air voids + microcracks
  Level VI:  Concrete      -- mortar + coarse aggregates + ITZ + fibers

All levels use elastic Mori-Tanaka homogenization.
"""

import numpy as np
from typing import Optional, Dict, List


# ================================================================
#  PHASE ELASTIC PROPERTIES DATABASE
# ================================================================

ELASTIC_PROPERTIES = {
    # Level I: Nanoscale crystals (from MD)
    "jennite":         {"E": 71.39e9, "nu": 0.38, "scale": "nano"},
    "tobermorite_14A": {"E": 91.62e9, "nu": 0.28, "scale": "nano"},
    "tobermorite_11A": {"E": 85.0e9,  "nu": 0.30, "scale": "nano"},
    "portlandite":     {"E": 55.74e9, "nu": 0.40, "scale": "nano"},
    "calcite":         {"E": 91.28e9, "nu": 0.248, "scale": "nano"},
    "vaterite":        {"E": 70.0e9,  "nu": 0.26, "scale": "nano"},
    "aragonite":       {"E": 80.0e9,  "nu": 0.25, "scale": "nano"},
    "gypsum":          {"E": 45.7e9,  "nu": 0.33, "scale": "nano"},

    # Level II: CSH variants (from Level II homogenization)
    "CSH_LD":          {"E": 21.7e9,  "nu": 0.24, "scale": "micro"},
    "CSH_HD":          {"E": 29.4e9,  "nu": 0.24, "scale": "micro"},
    "CSH_UHD":         {"E": 36.0e9,  "nu": 0.22, "scale": "micro"},

    # Hydration products
    "ettringite":      {"E": 22.4e9,  "nu": 0.20, "scale": "micro"},
    "monosulfate":     {"E": 42.3e9,  "nu": 0.324, "scale": "micro"},
    "AH3":             {"E": 25.0e9,  "nu": 0.25, "scale": "micro"},
    "stratlingite":    {"E": 30.0e9,  "nu": 0.25, "scale": "micro"},
    "hydrotalcite":    {"E": 55.0e9,  "nu": 0.28, "scale": "micro"},
    "monocarboaluminate": {"E": 40.0e9, "nu": 0.30, "scale": "micro"},
    "SiO2_gel":        {"E": 20.0e9,  "nu": 0.22, "scale": "micro"},
    "thaumasite":      {"E": 38.2e9,  "nu": 0.33, "scale": "micro"},

    # Unreacted phases (clinker)
    "clinker":         {"E": 145.0e9, "nu": 0.30, "scale": "micro"},
    "quartz_powder":   {"E": 73.0e9,  "nu": 0.17, "scale": "micro"},

    # Aggregates
    "sand":            {"E": 73.0e9,  "nu": 0.17, "scale": "meso"},
    "silica_sand":     {"E": 73.0e9,  "nu": 0.17, "scale": "meso"},
    "aggregate":       {"E": 16.5e9,  "nu": 0.20, "scale": "macro"},
    "limestone_agg":   {"E": 65.6e9,  "nu": 0.27, "scale": "macro"},
    "granite_agg":     {"E": 50.0e9,  "nu": 0.20, "scale": "macro"},
    "basalt_agg":      {"E": 70.0e9,  "nu": 0.25, "scale": "macro"},
    "sandstone_agg":   {"E": 30.0e9,  "nu": 0.15, "scale": "macro"},

    # Fibers
    "steel_fiber":     {"E": 200.0e9, "nu": 0.30, "scale": "macro"},
    "carbon_fiber":    {"E": 240.0e9, "nu": 0.20, "scale": "macro"},
    "glass_fiber":     {"E": 72.0e9,  "nu": 0.22, "scale": "macro"},
    "PVA_fiber":       {"E": 40.0e9,  "nu": 0.35, "scale": "macro"},
    "PP_fiber":        {"E": 3.5e9,   "nu": 0.40, "scale": "macro"},

    # Voids
    "porosity":        {"E": 1e-3,    "nu": 0.0, "scale": "all"},
    "gel_pore":        {"E": 1e-3,    "nu": 0.0, "scale": "nano"},
    "capillary_pore":  {"E": 1e-3,    "nu": 0.0, "scale": "micro"},
    "air_void":        {"E": 1e-3,    "nu": 0.0, "scale": "meso"},
    "crack":           {"E": 1e-6,    "nu": 0.0, "scale": "all"},

    # ITZ
    "ITZ":             {"E": 12.0e9,  "nu": 0.25, "scale": "meso"},
}


# ================================================================
#  TENSOR UTILITIES
# ================================================================

def lame_from_E_nu(E, nu):
    if abs(1 - 2*nu) < 1e-15:
        nu = 0.4999
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def E_nu_from_lame(lam, mu):
    denom = lam + mu
    if denom < 1e-30:
        return 0.0, 0.0
    E = mu * (3 * lam + 2 * mu) / denom
    nu = lam / (2 * denom)
    return E, nu


def c_asm(lam, mu):
    """6x6 isotropic stiffness in Voigt notation."""
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] = lam + 2 * mu
    for i in range(3, 6):
        C[i, i] = mu
    return C


def tinv(Q):
    """Invert 4th-rank isotropic tensor in Voigt form."""
    m = Q[0, 1]
    w = (Q[0, 0] - Q[0, 1]) / 2.0
    if abs(w) < 1e-30:
        return np.zeros_like(Q)

    a = -m / (2.0 * w * (3.0 * m + 2.0 * w))
    b = 1.0 / (2.0 * w)

    Qinv = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            Qinv[i, j] = a
        Qinv[i, i] = a + b
    for i in range(3, 6):
        Qinv[i, i] = 1.0 / (4.0 * w) if abs(w) > 1e-30 else 0.0
    return Qinv


# ================================================================
#  ESHELBY TENSORS
# ================================================================

def eshelby_sphere(nu, bc="N"):
    S = np.zeros((6, 6))
    diag = (7 - 5*nu) / (15*(1 - nu))
    off = (5*nu - 1) / (15*(1 - nu))
    shear = (4 - 5*nu) / (15*(1 - nu))
    for i in range(3):
        S[i, i] = diag
        for j in range(3):
            if i != j:
                S[i, j] = off
    for i in range(3, 6):
        S[i, i] = shear
    return S


def eshelby_crack(nu):
    S = np.zeros((6, 6))
    S[0, 0] = 1.0
    S[0, 1] = nu / (1 - nu)
    S[0, 2] = nu / (1 - nu)
    S[3, 3] = 0.5
    S[4, 4] = 0.5
    return S


def eshelby_oblate(nu, aspect_ratio):
    if aspect_ratio > 0.99:
        return eshelby_sphere(nu)
    if aspect_ratio < 0.01:
        return eshelby_crack(nu)
    r = aspect_ratio
    g = r / (1 - r**2)**1.5 * (np.arccos(r) - r * np.sqrt(1 - r**2))
    S = np.zeros((6, 6))
    S11 = (3*r**2)/(8*(1-nu)*(1-r**2)) + g*(1-2*nu-9/(4*(1-r**2)))/(4*(1-nu))
    S22 = S11
    S33 = 1/(2*(1-nu)) * (1-2*nu + (3*r**2-1)/(1-r**2)) - g/(2*(1-nu)) * (1-2*nu + (3*r**2)/(1-r**2))
    S12 = 1/(4*(1-nu)) * (r**2/(2*(1-r**2)) - g*(1-2*nu+3/(4*(1-r**2))))
    S13 = -r**2/(2*(1-nu)*(1-r**2)) + g/(4*(1-nu))*(3*r**2/(1-r**2) - (1-2*nu))
    S31 = -1/(2*(1-nu)) * (1-2*nu+1/(1-r**2)) + g/(2*(1-nu))*(1-2*nu+3/(2*(1-r**2)))
    S[0,0] = S11; S[1,1] = S22; S[2,2] = S33
    S[0,1] = S12; S[1,0] = S12
    S[0,2] = S13; S[2,0] = S31
    S[1,2] = S13; S[2,1] = S31
    S[3,3] = (S11 - S12) / 2
    S[4,4] = 1/(4*(1-nu)) * (1 - (2*r**2+1)/(1-r**2) - g*(1 - (3*(2*r**2+1))/(4*(1-r**2))))
    S[5,5] = S[4,4]
    return S


def eshelby_prolate(nu, aspect_ratio):
    """Eshelby tensor for prolate spheroid (fibers), a1 > a2 = a3."""
    if aspect_ratio < 1.01:
        return eshelby_sphere(nu)
    a = aspect_ratio
    g = a / (a**2 - 1)**1.5 * (a * np.sqrt(a**2 - 1) - np.arccosh(a))
    S = np.zeros((6, 6))
    S11 = 1/(2*(1-nu)) * (1-2*nu + (3*a**2-1)/(a**2-1)) - g/(2*(1-nu)) * (1-2*nu + (3*a**2)/(a**2-1))
    S22 = (3*a**2)/(16*(1-nu)*(a**2-1)) + g*(1-2*nu-9/(4*(a**2-1)))/(4*(1-nu))
    S12 = -1/(2*(1-nu)) * (1-2*nu+1/(a**2-1)) + g/(2*(1-nu))*(1-2*nu+3/(2*(a**2-1)))
    S21 = -a**2/(2*(1-nu)*(a**2-1)) + g/(4*(1-nu))*(3*a**2/(a**2-1) - (1-2*nu))
    S23 = 1/(4*(1-nu)) * (a**2/(2*(a**2-1)) - g*(1-2*nu+3/(4*(a**2-1))))
    S[0,0] = S11; S[1,1] = S22; S[2,2] = S22
    S[0,1] = S12; S[0,2] = S12
    S[1,0] = S21; S[2,0] = S21
    S[1,2] = S23; S[2,1] = S23
    S[3,3] = (S22 - S23)/2
    S[4,4] = 1/(4*(1-nu)) * (1 - (2*a**2+1)/(a**2-1)*0.5 + g/2*(1 + 3*(2*a**2+1)/(4*(a**2-1))))
    S[5,5] = S[4,4]
    return S


def get_eshelby_tensor(nu, shape="sphere", aspect_ratio=1.0):
    if shape == "sphere":
        return eshelby_sphere(nu)
    elif shape == "crack":
        return eshelby_crack(nu)
    elif shape == "oblate":
        return eshelby_oblate(nu, aspect_ratio)
    elif shape == "prolate":
        return eshelby_prolate(nu, aspect_ratio)
    else:
        return eshelby_sphere(nu)


# ================================================================
#  MORI-TANAKA HOMOGENIZATION
# ================================================================

def _safe_inv(M):
    """Invert 6x6 matrix; use tinv for isotropic, np.linalg.inv otherwise."""
    try:
        if np.linalg.cond(M) < 1e12:
            return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        pass
    # Fallback to isotropic tinv
    result = tinv(M)
    if np.max(np.abs(result)) > 1e-30:
        return result
    return np.zeros_like(M)


def mori_tanaka(C_matrix, phases):
    """
    N-phase Mori-Tanaka effective stiffness.

    Standard MT scheme:
      T_r = [I + S_r : C_m^-1 : (C_r - C_m)]^-1
      C_eff = (f_m * C_m + sum f_r * C_r : T_r) : (f_m * I + sum f_r * T_r)^-1

    phases: [{"C": 6x6, "phi": float, "S": 6x6}, ...]
    """
    I6 = np.eye(6)
    phi_m = 1.0 - sum(p["phi"] for p in phases)
    phi_m = max(phi_m, 0.01)

    # Compute strain concentration tensors T_r
    try:
        C_m_inv = np.linalg.inv(C_matrix) if np.linalg.det(C_matrix) > 1e-30 else np.zeros((6, 6))
    except np.linalg.LinAlgError:
        C_m_inv = np.zeros((6, 6))

    sum_fC_T = phi_m * C_matrix.copy()
    sum_f_T = phi_m * I6.copy()

    for p in phases:
        if p["phi"] < 1e-15:
            continue
        C_r = p["C"]
        S_r = p["S"]
        C_diff = C_r - C_matrix

        # T_r = [I + S : C_m^-1 : (C_r - C_m)]^-1
        M = I6 + S_r @ C_m_inv @ C_diff
        try:
            T_r = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            T_r = I6

        sum_fC_T += p["phi"] * C_r @ T_r
        sum_f_T += p["phi"] * T_r

    try:
        C_eff = sum_fC_T @ np.linalg.inv(sum_f_T)
    except np.linalg.LinAlgError:
        C_eff = C_matrix

    return C_eff


def extract_moduli(C):
    C11 = C[0, 0]
    C12 = C[0, 1]
    C44 = C[3, 3]
    mu = C44
    kappa = (C11 + 2*C12) / 3
    E = 9*kappa*mu / (3*kappa + mu) if (3*kappa + mu) > 0 else 0
    nu = (3*kappa - 2*mu) / (2*(3*kappa + mu)) if (3*kappa + mu) > 0 else 0
    return {"E": E, "nu": nu, "kappa": kappa, "mu": mu}


def _make_phase(name, phi, nu_matrix, shape="sphere", aspect_ratio=1.0):
    """Helper to build a phase dict for mori_tanaka()."""
    if phi < 1e-15:
        return None
    props = ELASTIC_PROPERTIES.get(name, {"E": 1e-3, "nu": 0.0})
    lam_i, mu_i = lame_from_E_nu(props["E"], props["nu"])
    C_i = c_asm(lam_i, mu_i)
    S_i = get_eshelby_tensor(nu_matrix, shape, aspect_ratio)
    return {"C": C_i, "phi": phi, "S": S_i}


# ================================================================
#  6-LEVEL HOMOGENIZATION SCHEME
# ================================================================

def level_I_md(csh_model: str = "jennite") -> dict:
    """
    Level I: Nanoscale MD properties.
    Returns elastic properties of the CSH crystal building block.
    Uses Voigt-Reuss-Hill approximation for polycrystalline average.
    """
    props = ELASTIC_PROPERTIES.get(csh_model, ELASTIC_PROPERTIES["jennite"])
    return {"E": props["E"], "nu": props["nu"], "phase": csh_model}


def level_II_csh_variant(csh_crystal_E: float, csh_crystal_nu: float,
                          phi_gel_pore: float) -> dict:
    """
    Level II: CSH variant (LD, HD, or UHD-CSH).
    Matrix: CSH crystal. Inclusion: gel porosity (spherical).

    Typical gel porosities:
      LD-CSH: ~0.37, HD-CSH: ~0.24, UHD-CSH: ~0.13
    """
    lam_m, mu_m = lame_from_E_nu(csh_crystal_E, csh_crystal_nu)
    C_matrix = c_asm(lam_m, mu_m)

    phases = []
    if phi_gel_pore > 1e-15:
        p = _make_phase("gel_pore", phi_gel_pore, csh_crystal_nu, "sphere")
        if p:
            phases.append(p)

    if not phases:
        return {"E": csh_crystal_E, "nu": csh_crystal_nu}

    C_eff = mori_tanaka(C_matrix, phases)
    return extract_moduli(C_eff)


def level_III_csh_matrix(phi_LD: float = 0.30, phi_HD: float = 0.40,
                          phi_UHD: float = 0.30,
                          csh_model: str = "jennite",
                          heat_treated: bool = False) -> dict:
    """
    Level III: General CSH matrix from LD + HD + UHD-CSH.

    For heat-treated: UHD-CSH is matrix, LD/HD are inclusions.
    For non-heat-treated: HD-CSH is matrix, LD/UHD are inclusions.
    """
    md = level_I_md(csh_model)

    # Level II: compute each CSH variant
    ld = level_II_csh_variant(md["E"], md["nu"], 0.37)  # LD-CSH
    hd = level_II_csh_variant(md["E"], md["nu"], 0.24)  # HD-CSH
    uhd = level_II_csh_variant(md["E"], md["nu"], 0.13)  # UHD-CSH

    total = phi_LD + phi_HD + phi_UHD
    if total < 1e-10:
        return hd

    if heat_treated:
        # UHD is matrix
        lam_m, mu_m = lame_from_E_nu(uhd["E"], uhd["nu"])
        C_matrix = c_asm(lam_m, mu_m)
        nu_m = uhd["nu"]
        incl_phi_LD = phi_LD / total
        incl_phi_HD = phi_HD / total
    else:
        # HD is matrix
        lam_m, mu_m = lame_from_E_nu(hd["E"], hd["nu"])
        C_matrix = c_asm(lam_m, mu_m)
        nu_m = hd["nu"]
        incl_phi_LD = phi_LD / total
        incl_phi_HD = phi_UHD / total  # UHD as inclusion

    phases = []
    if not heat_treated:
        p = _make_phase_from_moduli(ld["E"], ld["nu"], incl_phi_LD, nu_m)
        if p:
            phases.append(p)
        p = _make_phase_from_moduli(uhd["E"], uhd["nu"], incl_phi_HD, nu_m)
        if p:
            phases.append(p)
    else:
        p = _make_phase_from_moduli(ld["E"], ld["nu"], incl_phi_LD, nu_m)
        if p:
            phases.append(p)
        p = _make_phase_from_moduli(hd["E"], hd["nu"], incl_phi_HD, nu_m)
        if p:
            phases.append(p)

    if not phases:
        return extract_moduli(C_matrix)

    C_eff = mori_tanaka(C_matrix, phases)
    return extract_moduli(C_eff)


def _make_phase_from_moduli(E, nu, phi, nu_matrix, shape="sphere", ar=1.0):
    if phi < 1e-15:
        return None
    lam, mu = lame_from_E_nu(E, nu)
    C = c_asm(lam, mu)
    S = get_eshelby_tensor(nu_matrix, shape, ar)
    return {"C": C, "phi": phi, "S": S}


def level_IV_paste(csh_matrix: dict,
                   phi_CH: float = 0.15,
                   phi_CaCO3: float = 0.0,
                   phi_capillary: float = 0.10,
                   phi_clinker: float = 0.08,
                   phi_quartz: float = 0.0,
                   phi_SiO2_gel: float = 0.0,
                   additional_phases: Optional[Dict[str, float]] = None) -> dict:
    """
    Level IV: Cement paste.
    Matrix: CSH matrix (from Level III).
    Inclusions: CH, CaCO3, capillary porosity, unreacted clinker, quartz, etc.
    All inclusions are spherical.
    """
    E_m = csh_matrix.get("E", 25e9)
    nu_m = csh_matrix.get("nu", 0.24)
    lam_m, mu_m = lame_from_E_nu(E_m, nu_m)
    C_matrix = c_asm(lam_m, mu_m)

    phase_defs = [
        ("portlandite", phi_CH),
        ("calcite", phi_CaCO3),
        ("capillary_pore", phi_capillary),
        ("clinker", phi_clinker),
        ("quartz_powder", phi_quartz),
        ("SiO2_gel", phi_SiO2_gel),
    ]
    if additional_phases:
        for name, phi in additional_phases.items():
            if name in ELASTIC_PROPERTIES:
                phase_defs.append((name, phi))

    phases = []
    for name, phi in phase_defs:
        p = _make_phase(name, phi, nu_m, "sphere")
        if p:
            phases.append(p)

    if not phases:
        return extract_moduli(C_matrix)

    C_eff = mori_tanaka(C_matrix, phases)
    return extract_moduli(C_eff)


def level_V_mortar(paste: dict,
                   phi_sand: float = 0.40,
                   phi_air: float = 0.03,
                   phi_crack: float = 0.005,
                   sand_type: str = "silica_sand") -> dict:
    """
    Level V: Mortar = Paste matrix + Sand + Air voids + Microcracks.
    """
    E_m = paste.get("E", 20e9)
    nu_m = paste.get("nu", 0.22)
    lam_m, mu_m = lame_from_E_nu(E_m, nu_m)
    C_matrix = c_asm(lam_m, mu_m)

    phases = []

    p = _make_phase(sand_type, phi_sand, nu_m, "sphere")
    if p:
        phases.append(p)

    p = _make_phase("air_void", phi_air, nu_m, "sphere")
    if p:
        phases.append(p)

    if phi_crack > 1e-10:
        C_crack = c_asm(0, 0)
        S_crack = eshelby_crack(nu_m)
        phases.append({"C": C_crack, "phi": phi_crack, "S": S_crack})

    if not phases:
        return paste.copy()

    C_eff = mori_tanaka(C_matrix, phases)
    return extract_moduli(C_eff)


def level_VI_concrete(mortar: dict,
                      phi_agg: float = 0.40,
                      agg_type: str = "limestone_agg",
                      phi_ITZ: float = 0.0,
                      fibers: Optional[List[Dict]] = None) -> dict:
    """
    Level VI: Concrete = Mortar + Coarse aggregates + ITZ + Fibers.

    fibers: [{"type": "steel_fiber", "phi": 0.02, "aspect_ratio": 60}, ...]
    """
    E_m = mortar.get("E", 30e9)
    nu_m = mortar.get("nu", 0.20)
    lam_m, mu_m = lame_from_E_nu(E_m, nu_m)
    C_matrix = c_asm(lam_m, mu_m)

    phases = []

    # Coarse aggregates (spherical)
    p = _make_phase(agg_type, phi_agg, nu_m, "sphere")
    if p:
        phases.append(p)

    # ITZ (spherical shell approximation as soft inclusion)
    if phi_ITZ > 1e-10:
        p = _make_phase("ITZ", phi_ITZ, nu_m, "sphere")
        if p:
            phases.append(p)

    # Fibers (prolate spheroids)
    if fibers:
        for fiber in fibers:
            f_type = fiber.get("type", "steel_fiber")
            f_phi = fiber.get("phi", 0.0)
            f_ar = fiber.get("aspect_ratio", 60)
            if f_phi > 1e-15 and f_type in ELASTIC_PROPERTIES:
                p = _make_phase(f_type, f_phi, nu_m, "prolate", f_ar)
                if p:
                    phases.append(p)

    if not phases:
        return mortar.copy()

    C_eff = mori_tanaka(C_matrix, phases)
    return extract_moduli(C_eff)


# ================================================================
#  LEGACY-COMPATIBLE LEVEL FUNCTIONS
# ================================================================

def level_I(phi_CSH=0.0, phi_CH=0.0, phi_CaCO3=0.0,
            phi_porosity=0.0, phi_SiO2=0.0,
            additional_phases=None) -> dict:
    """Legacy Level I (cement paste) using 6-level scheme internally."""
    csh_matrix = level_III_csh_matrix()
    total_input = phi_CSH + phi_CH + phi_CaCO3 + phi_porosity + phi_SiO2
    if total_input < 1e-10:
        total_input = 1.0

    return level_IV_paste(
        csh_matrix,
        phi_CH=phi_CH,
        phi_CaCO3=phi_CaCO3,
        phi_capillary=phi_porosity,
        phi_clinker=0.0,
        phi_SiO2_gel=phi_SiO2,
        additional_phases=additional_phases,
    )


def level_II(paste_E, paste_nu, phi_sand=0.3, phi_crack=0.005,
             sand_type="sand") -> dict:
    """Legacy Level II (mortar)."""
    paste = {"E": paste_E, "nu": paste_nu}
    return level_V_mortar(paste, phi_sand=phi_sand, phi_crack=phi_crack,
                          sand_type="silica_sand" if sand_type == "sand" else sand_type)


def level_III(mortar_E, mortar_nu, phi_agg=0.45,
              agg_type="aggregate") -> dict:
    """Legacy Level III (concrete)."""
    mortar = {"E": mortar_E, "nu": mortar_nu}
    return level_VI_concrete(mortar, phi_agg=phi_agg, agg_type=agg_type)


# ================================================================
#  DAMAGE MODEL
# ================================================================

def crack_evolution(phi_crack_ini, strain, eps_threshold,
                    c1=0.25, c2=2.0):
    if strain <= eps_threshold or eps_threshold <= 0:
        return phi_crack_ini
    ratio = 1.0 - eps_threshold / strain
    return phi_crack_ini + c1 * ratio**c2


def damage_permeability_factor(damage, xi=4.0):
    return 10.0**(xi * np.clip(damage, 0, 1))


# ================================================================
#  FULL MULTISCALE (user-friendly wrapper)
# ================================================================

def compute_effective_properties(phi: dict,
                                 level: str = "paste",
                                 phi_sand: float = 0.3,
                                 phi_agg: float = 0.45,
                                 phi_crack: float = 0.005,
                                 csh_model: str = "jennite",
                                 heat_treated: bool = False,
                                 agg_type: str = "limestone_agg",
                                 fibers: Optional[List[Dict]] = None) -> dict:
    """
    Full 6-level homogenization from phase volume fractions.

    Args:
        phi: dict of phase volume fractions from chemistry engine
        level: "paste", "mortar", or "concrete"
        csh_model: "jennite" or "tobermorite_14A" for Level I MD
        heat_treated: True for heat-treated UHPC
        agg_type: aggregate type for Level VI
        fibers: fiber list for Level VI
    """
    # Levels I-III: CSH matrix
    csh_matrix = level_III_csh_matrix(csh_model=csh_model, heat_treated=heat_treated)

    # Level IV: Cement paste
    extra = {k: v for k, v in phi.items()
             if k not in ("CSH", "CH", "CaCO3", "porosity", "SiO2_gel")
             and k in ELASTIC_PROPERTIES and v > 1e-10}

    paste = level_IV_paste(
        csh_matrix,
        phi_CH=phi.get("CH", 0),
        phi_CaCO3=phi.get("CaCO3", 0),
        phi_capillary=phi.get("porosity", 0.25),
        phi_SiO2_gel=phi.get("SiO2_gel", 0),
        additional_phases=extra if extra else None,
    )

    if level == "paste":
        return paste

    # Level V: Mortar
    mortar = level_V_mortar(paste, phi_sand=phi_sand, phi_crack=phi_crack)

    if level == "mortar":
        return mortar

    # Level VI: Concrete
    concrete = level_VI_concrete(mortar, phi_agg=phi_agg,
                                  agg_type=agg_type, fibers=fibers)
    return concrete
