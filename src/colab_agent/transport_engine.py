"""
Transport Engine -- Modified Fick's 2nd Law Diffusion Solver.

Provides:
- Pore structure model based on w/c ratio and raw materials (Powers model, MIP-based)
- Multiple modified Fick's 2nd law models (user-selectable):
    1. standard         -- D_eff = const, pure Fick's 2nd law
    2. millington_quirk  -- D_eff = D0 * phi^a * (1-S)^b
    3. papadakis        -- Papadakis et al. (1991) empirical model
    4. ceb_fip          -- CEB-FIP Model Code approach
    5. fib_mc2010       -- fib Model Code 2010 (DuraCrete)
    6. eshelby_mt       -- Mori-Tanaka effective D from inclusion theory
- 1D FDM solver (explicit Euler + Crank-Nicolson)
- Reactive source/sink terms

Based on MATLAB: Chemo_transport_validation_wc04.m, wc06.m
"""

import numpy as np
from typing import Optional, Callable, Union, Dict
import warnings


# ================================================================
#  1. PORE STRUCTURE MODEL
# ================================================================

class PoreStructure:
    """
    Compute pore structure from w/c ratio and raw material composition.
    Based on Powers model with SCM corrections.
    """

    def __init__(self, wc_ratio: float, alpha_hyd: float = 0.85,
                 scm_mix: Optional[Dict[str, float]] = None):
        """
        Args:
            wc_ratio: water-to-cement (or binder) ratio
            alpha_hyd: degree of hydration (0 to 1)
            scm_mix: SCM replacements, e.g. {"fly_ash_F": 0.25}
        """
        self.wc = wc_ratio
        self.alpha = alpha_hyd
        self.scm_mix = scm_mix or {}

    def total_porosity(self) -> float:
        """Powers model: phi_cap = (w/c - 0.36*alpha) / (w/c + 0.32)"""
        phi_cap = max(0, (self.wc - 0.36 * self.alpha) / (self.wc + 0.32))
        phi_gel = 0.28 * (1 - phi_cap) * 0.26 * self.alpha
        phi_total = phi_cap + phi_gel

        # SCM corrections
        for scm_name, ratio in self.scm_mix.items():
            from .chemistry_engine import SCM_DATABASE
            if scm_name in SCM_DATABASE:
                phi_total += SCM_DATABASE[scm_name].get("porosity_modifier", 0) * ratio

        return np.clip(phi_total, 0.02, 0.60)

    def capillary_porosity(self) -> float:
        return max(0, (self.wc - 0.36 * self.alpha) / (self.wc + 0.32))

    def gel_porosity(self) -> float:
        return 0.28 * 0.26 * self.alpha

    def critical_pore_diameter_nm(self) -> float:
        """Estimate critical pore diameter from MIP (empirical)."""
        phi_cap = self.capillary_porosity()
        if phi_cap < 0.05:
            return 10.0  # nm, dense paste
        return 10.0 + 200.0 * phi_cap  # rough correlation

    def tortuosity(self) -> float:
        """Tortuosity from porosity (Boudreau 1996)."""
        phi = self.total_porosity()
        return 1.0 / (1.0 - np.log(phi**2))

    def pore_size_distribution(self, n_bins: int = 50) -> Dict:
        """Simple log-normal pore size distribution."""
        phi_gel = self.gel_porosity()
        phi_cap = self.capillary_porosity()
        d_crit = self.critical_pore_diameter_nm()

        diameters = np.logspace(0, 5, n_bins)  # 1 nm to 100 um

        gel_peak = 5.0  # nm
        cap_peak = d_crit

        gel_dist = phi_gel * np.exp(-0.5 * ((np.log10(diameters) - np.log10(gel_peak)) / 0.4)**2)
        cap_dist = phi_cap * np.exp(-0.5 * ((np.log10(diameters) - np.log10(cap_peak)) / 0.6)**2)

        total_dist = gel_dist + cap_dist
        total_dist /= np.trapz(total_dist, np.log10(diameters)) + 1e-30
        total_dist *= (phi_gel + phi_cap)

        return {
            "diameters_nm": diameters,
            "dv_dlogd": total_dist,
            "phi_gel": phi_gel,
            "phi_cap": phi_cap,
            "phi_total": self.total_porosity(),
            "d_crit_nm": d_crit,
        }


# ================================================================
#  2. EFFECTIVE DIFFUSIVITY MODELS (user-selectable)
# ================================================================

DIFFUSION_MODELS = {}

def register_model(name):
    def decorator(func):
        DIFFUSION_MODELS[name] = func
        return func
    return decorator


@register_model("standard")
def D_eff_standard(D0: float, porosity: float, saturation: float,
                   **kwargs) -> float:
    """Constant D_eff = D0 * porosity * (1-S)."""
    return D0 * porosity * max(0, 1 - saturation)


@register_model("millington_quirk")
def D_eff_millington_quirk(D0: float, porosity: float, saturation: float,
                            a: float = 1.74, b: float = 3.20,
                            **kwargs) -> float:
    """Millington-Quirk: D_eff = D0 * phi^a * (1-S)^b"""
    return D0 * porosity**a * max(0, 1 - saturation)**b


@register_model("papadakis")
def D_eff_papadakis(D0: float, porosity: float, saturation: float,
                     wc_ratio: float = 0.5, **kwargs) -> float:
    """
    Papadakis et al. (1991):
    D_eff = 6.1e-6 * (phi_cap)^3 * (1-S)^2.2  [m^2/s for CO2]
    phi_cap from w/c ratio.
    """
    phi_cap = max(0, (wc_ratio - 0.36 * 0.85) / (wc_ratio + 0.32))
    return 6.1e-6 * phi_cap**3 * max(0, 1 - saturation)**2.2


@register_model("ceb_fip")
def D_eff_ceb_fip(D0: float, porosity: float, saturation: float,
                   temperature_K: float = 293.15, **kwargs) -> float:
    """
    CEB-FIP Model Code approach:
    D_eff = D_ref * f1(RH) * f2(T) * f3(t)
    D_ref from porosity, f1 accounts for RH effect.
    """
    RH = saturation  # using saturation as proxy for RH
    f1 = (1 - RH)**2.5
    f2 = np.exp(5000 * (1/293.15 - 1/temperature_K))  # Arrhenius T factor
    D_ref = D0 * porosity**1.8
    return D_ref * f1 * f2


@register_model("fib_mc2010")
def D_eff_fib_mc2010(D0: float, porosity: float, saturation: float,
                      wc_ratio: float = 0.5, cement_type: str = "OPC",
                      age_days: float = 28, **kwargs) -> float:
    """
    fib Model Code 2010 / DuraCrete:
    D(t) = D_28 * k_e * k_c * (t_ref/t)^n
    k_e = environmental factor (RH)
    k_c = curing factor
    n = aging exponent (depends on cement type)
    """
    # Aging exponents by cement type
    aging_n = {
        "OPC": 0.30, "CSA": 0.25,
        "OPC_FA": 0.60, "OPC_GGBS": 0.55,
        "OPC_SF": 0.45, "OPC_MK": 0.50,
    }
    n = aging_n.get(cement_type, 0.30)

    # Base diffusion at 28 days
    D_28 = D0 * porosity**1.8

    # Environmental factor (RH effect)
    RH = saturation
    k_e = (1 - RH)**2.5 if RH < 1.0 else 1e-10

    # Aging
    t_ref = 28.0
    age_factor = (t_ref / max(age_days, 1))**n

    return D_28 * k_e * age_factor


@register_model("eshelby_mt")
def D_eff_eshelby_mt(D0: float, porosity: float, saturation: float,
                      inclusions: Optional[list] = None, **kwargs) -> float:
    """
    Mori-Tanaka effective diffusivity from Eshelby inclusion theory.
    inclusions: [(D_i, phi_i, aspect_ratio), ...]
    """
    D_gas = D0 * max(0, 1 - saturation)
    D_matrix = D_gas * porosity

    if inclusions is None:
        # Default: pores as spherical inclusions in solid matrix
        D_solid = 1e-15  # effectively zero diffusion in solid
        # MT: D_eff = D_matrix * (1 + 2*phi*(D_solid - D_matrix)/(D_solid + 2*D_matrix)) / ...
        # Simplified for gas in porous medium
        return D_matrix

    D_eff = D_matrix
    for D_i, phi_i, ar in inclusions:
        if phi_i < 1e-15:
            continue
        S_iso = 1.0 / 3.0
        contrast = D_i / D_matrix - 1 if D_matrix > 0 else 0
        denom = 1 + S_iso * contrast
        A = contrast / denom if abs(denom) > 1e-30 else 0
        denom2 = 1 + (1 - phi_i) * S_iso * A
        D_eff += D_matrix * phi_i * A / denom2 if abs(denom2) > 1e-30 else 0

    return max(D_eff, 1e-20)


# Convenience alias
def millington_quirk(D0, porosity, saturation, a=1.74, b=3.20):
    return D_eff_millington_quirk(D0, porosity, saturation, a=a, b=b)


def compute_D_eff(model_name: str, D0: float = 1.6e-5,
                  porosity: float = 0.25, saturation: float = 0.65,
                  **kwargs) -> float:
    """
    Compute effective diffusivity using the selected model.

    Args:
        model_name: one of "standard", "millington_quirk", "papadakis",
                    "ceb_fip", "fib_mc2010", "eshelby_mt"
    """
    if model_name not in DIFFUSION_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(DIFFUSION_MODELS)}")
    return DIFFUSION_MODELS[model_name](D0, porosity, saturation, **kwargs)


def list_diffusion_models() -> Dict[str, str]:
    """Return available diffusion models with docstrings."""
    return {name: (func.__doc__ or "").strip().split('\n')[0]
            for name, func in DIFFUSION_MODELS.items()}


# Legacy aliases
def effective_D_mori_tanaka(D_matrix, inclusions):
    return D_eff_eshelby_mt(D_matrix, 1.0, 0.0, inclusions)

def effective_D_porosity(D_CO2_air, porosity, saturation, damage=0.0, xi=4.0):
    D_base = millington_quirk(D_CO2_air, porosity, saturation)
    return D_base * 10.0**(xi * np.clip(damage, 0, 1))

def eshelby_diffusion_Q(aspect_ratio):
    r = aspect_ratio
    if r >= 1.0:
        return 1.0/3.0
    if r < 1e-6:
        return 0.0
    return 0.5 * (1 + 1/(r**2-1) * (1 - r/np.sqrt(1-r**2) * np.arctan(np.sqrt(1-r**2)/r)))


# ================================================================
#  3. 1D DIFFUSION SOLVER
# ================================================================

class DiffusionSolver1D:
    """
    Solve dc/dt = d/dx(D(x,t)*dc/dx) - R(x,t)

    Supports:
    - Explicit Euler (CFL-limited)
    - Crank-Nicolson (implicit, unconditionally stable)
    - Spatially varying D_eff
    - Reactive source/sink term R(x,t)
    """

    def __init__(self, L: float, nx: int,
                 D_eff: Union[float, np.ndarray, Callable] = 1e-8,
                 source_func: Optional[Callable] = None):
        self.L = L
        self.nx = nx
        self.dx = L / nx
        self.x = np.linspace(0, L, nx + 1)
        self._D_eff = D_eff
        self.source_func = source_func
        self.c = np.zeros(nx + 1)
        self.bc_left = ("dirichlet", 0.0)
        self.bc_right = ("neumann", 0.0)

    def set_initial(self, c0):
        if isinstance(c0, (int, float)):
            self.c = np.full(self.nx + 1, float(c0))
        else:
            self.c = np.array(c0, dtype=float)

    def set_bc(self, left=("dirichlet", 0.0), right=("neumann", 0.0)):
        self.bc_left = left
        self.bc_right = right

    def get_D_eff(self, t=0):
        if callable(self._D_eff):
            return np.array([self._D_eff(xi, t) for xi in self.x])
        elif isinstance(self._D_eff, np.ndarray):
            return self._D_eff
        else:
            return np.full(self.nx + 1, float(self._D_eff))

    def update_D_eff(self, D_new):
        self._D_eff = D_new

    def _apply_bc(self, c):
        if self.bc_left[0] == "dirichlet":
            c[0] = self.bc_left[1]
        elif self.bc_left[0] == "neumann":
            c[0] = c[1] - self.bc_left[1] * self.dx
        if self.bc_right[0] == "dirichlet":
            c[-1] = self.bc_right[1]
        elif self.bc_right[0] == "neumann":
            c[-1] = c[-2] + self.bc_right[1] * self.dx
        return c

    def step_explicit(self, dt, t=0):
        D = self.get_D_eff(t)
        c = self.c.copy()
        cfl = np.max(D) * dt / self.dx**2
        if cfl > 0.5:
            warnings.warn(f"CFL={cfl:.3f}>0.5; reduce dt or use implicit scheme")
        for i in range(1, self.nx):
            D_plus = 0.5 * (D[i] + D[i+1])
            D_minus = 0.5 * (D[i] + D[i-1])
            diffusion = (D_plus*(c[i+1]-c[i]) - D_minus*(c[i]-c[i-1])) / self.dx**2
            source = self.source_func(self.x[i], t, c[i]) if self.source_func else 0.0
            self.c[i] = c[i] + dt * (diffusion - source)
        self.c = np.maximum(self.c, 0)
        self.c = self._apply_bc(self.c)

    def step_crank_nicolson(self, dt, t=0):
        D = self.get_D_eff(t)
        theta = 0.5
        n = self.nx + 1

        a = np.zeros(n)
        b = np.ones(n)
        cc = np.zeros(n)
        d = self.c.copy()

        for i in range(1, n-1):
            D_p = 0.5*(D[i]+D[i+1])
            D_m = 0.5*(D[i]+D[i-1])
            coeff = dt / self.dx**2
            a[i] = -theta * coeff * D_m
            b[i] = 1 + theta * coeff * (D_p + D_m)
            cc[i] = -theta * coeff * D_p
            d[i] = (self.c[i]
                    + (1-theta)*coeff*D_m*self.c[i-1]
                    - (1-theta)*coeff*(D_p+D_m)*self.c[i]
                    + (1-theta)*coeff*D_p*self.c[i+1])
            if self.source_func:
                d[i] -= dt * self.source_func(self.x[i], t, self.c[i])

        if self.bc_left[0] == "dirichlet":
            b[0] = 1; cc[0] = 0; d[0] = self.bc_left[1]
        else:
            b[0] = 1; cc[0] = -1; d[0] = self.bc_left[1] * self.dx
        if self.bc_right[0] == "dirichlet":
            b[-1] = 1; a[-1] = 0; d[-1] = self.bc_right[1]
        else:
            b[-1] = 1; a[-1] = -1; d[-1] = self.bc_right[1] * self.dx

        # Thomas algorithm
        for i in range(1, n):
            if abs(b[i-1]) < 1e-30:
                continue
            w = a[i] / b[i-1]
            b[i] -= w * cc[i-1]
            d[i] -= w * d[i-1]

        self.c[-1] = d[-1] / b[-1] if abs(b[-1]) > 1e-30 else 0
        for i in range(n-2, -1, -1):
            self.c[i] = (d[i] - cc[i]*self.c[i+1]) / b[i] if abs(b[i]) > 1e-30 else 0
        self.c = np.maximum(self.c, 0)

    def solve(self, t_final, dt, scheme="crank_nicolson", save_times=None,
              verbose=False):
        t = 0.0
        results = {"t": [], "profiles": [], "x": self.x.copy()}
        step_func = self.step_crank_nicolson if scheme == "crank_nicolson" else self.step_explicit
        if save_times is None:
            save_times = np.linspace(0, t_final, 20).tolist()
        save_times = sorted(save_times)
        save_idx = 0

        # Adaptive dt for Crank-Nicolson: unconditionally stable, use large dt
        if scheme == "crank_nicolson":
            # CN stability allows large Fourier numbers
            # Use at least 200 steps for accuracy, but cap by diffusion timescale
            dt_min_steps = t_final / 200
            dt = max(dt, min(dt_min_steps, t_final / 50))

        total_steps = int(t_final / dt) + 1
        log_interval = max(total_steps // 10, 1)
        step_count = 0

        while t < t_final:
            actual_dt = min(dt, t_final - t)
            # Snap to next save_time if close
            if save_idx < len(save_times) and t + actual_dt > save_times[save_idx]:
                actual_dt = save_times[save_idx] - t
            step_func(actual_dt, t)
            t += actual_dt
            step_count += 1

            if save_idx < len(save_times) and t >= save_times[save_idx] - 1e-6:
                results["t"].append(t)
                results["profiles"].append(self.c.copy())
                save_idx += 1

            if verbose and step_count % log_interval == 0:
                pct = min(100, t / t_final * 100)
                print(f"  Diffusion solver: {pct:.0f}% (t={t:.1e}s, step {step_count})")

        return results


# ================================================================
#  4. CARBONATION FRONT
# ================================================================

def find_carbonation_depth(c_profile, x_grid, threshold=0.5):
    c_max = np.max(c_profile)
    if c_max < 1e-15:
        return 0.0
    c_norm = c_profile / c_max
    idx = np.where(c_norm < threshold)[0]
    if len(idx) == 0:
        return x_grid[-1]
    return x_grid[idx[0]]


def sqrt_t_fit(depths, times):
    sqrt_t = np.sqrt(times[times > 0])
    x = depths[times > 0]
    if len(sqrt_t) < 2:
        return 0.0
    K = np.polyfit(sqrt_t, x, 1)[0]
    return K


# ================================================================
#  5. COUPLED TRANSPORT MODEL
# ================================================================

class CarbonationTransportModel:
    """
    High-level CO2 carbonation transport model with selectable diffusion model.
    """

    def __init__(self, L=0.05, nx=50, D_CO2_air=1.6e-5,
                 porosity=0.25, saturation=0.5, C_env_CO2=16.0,
                 diffusion_model="millington_quirk",
                 wc_ratio=0.5, cement_type="OPC",
                 age_days=28, scm_mix=None):
        self.L = L
        self.nx = nx
        self.porosity = porosity
        self.saturation = saturation
        self.C_env = C_env_CO2
        self.model_name = diffusion_model

        D_eff = compute_D_eff(
            diffusion_model, D_CO2_air, porosity, saturation,
            wc_ratio=wc_ratio, cement_type=cement_type,
            age_days=age_days,
        )
        self.D_eff_value = D_eff

        self.solver = DiffusionSolver1D(L, nx, D_eff)
        self.solver.set_initial(0.0)
        self.solver.set_bc(left=("dirichlet", C_env_CO2),
                           right=("neumann", 0.0))

    def solve(self, t_final, dt=3600.0, source_func=None, save_times=None):
        if source_func:
            self.solver.source_func = source_func
        return self.solver.solve(t_final, dt, save_times=save_times)
