
import os
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
import numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings


PLOT_DARK  = "#0d1117"
PLOT_PANEL = "#161b22"
PLOT_TEXT  = "#e6edf3"
PLOT_GRID  = "#30363d"

try:
    from scipy.stats import qmc as _qmc
    _HAS_QMC = True
except ImportError:
    _HAS_QMC = False
    _qmc = None


_LN2 = np.log(2.0)

DECAY_CONSTANTS = {
    "Rn222": _LN2 / (3.824 * 86400.0),
    "Po218": _LN2 / 185.88,
    "Pb214": _LN2 / 1613.0,
    "Bi214": _LN2 / 1194.0,
    "Po214": _LN2 / 1.643e-4,
    "Rn220": _LN2 / 55.6,
    "Po216": _LN2 / 0.145,
    "Pb212": _LN2 / (10.64 * 3600.0),
}


GEOLITHOTYPES = {


    "tuff_welded": {
        "vesicle_fraction": 0.05,
        "phi_m": 0.30, "phi_f": 0.05,
        "k0": 1.5e-14, "k_frac": 3e-12,
        "D0": 1.2e-6,
        "Ra226_Bqkg": 120.0, "rho_bulk": 1800.0,
        "soil_alpha": 1.5, "soil_n": 1.6,
        "Ksat0": 1e-6,
        "A_crack_m2": 0.10,
        "grain_size_mm": 0.05,
        "eps_max": 0.38, "Sw_peak_eman": 0.28,
        "label": "Tufo saldato",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.38, "Acrack_m2": 0.10,
    },
    "tuff_unwelded": {
        "vesicle_fraction": 0.10,
        "phi_m": 0.48, "phi_f": 0.08,
        "k0": 8e-14, "k_frac": 2e-11,
        "D0": 2.5e-6,
        "Ra226_Bqkg": 90.0, "rho_bulk": 1350.0,
        "soil_alpha": 2.0, "soil_n": 1.5,
        "Ksat0": 5e-6,
        "A_crack_m2": 0.50,
        "grain_size_mm": 0.08,
        "eps_max": 0.42, "Sw_peak_eman": 0.22,
        "label": "Tufo non saldato",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.42, "Acrack_m2": 0.50,
    },
    "pumice": {
        "vesicle_fraction": 0.40,
        "phi_m": 0.60, "phi_f": 0.12,
        "k0": 3e-13, "k_frac": 1e-10,
        "D0": 4.0e-6,
        "Ra226_Bqkg": 55.0, "rho_bulk": 900.0,
        "soil_alpha": 3.5, "soil_n": 1.45,
        "Ksat0": 2e-5,
        "A_crack_m2": 1.00,
        "grain_size_mm": 2.0,
        "eps_max": 0.28, "Sw_peak_eman": 0.15,
        "label": "Pomice",
        "seismic_site": "Etna",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.28, "Acrack_m2": 1.00,
    },
    "scoria_basaltic": {
        "vesicle_fraction": 0.25,
        "phi_m": 0.45, "phi_f": 0.10,
        "k0": 1e-13, "k_frac": 5e-11,
        "D0": 3.0e-6,
        "Ra226_Bqkg": 40.0, "rho_bulk": 1200.0,
        "soil_alpha": 2.5, "soil_n": 1.5,
        "Ksat0": 1e-5,
        "A_crack_m2": 0.80,
        "grain_size_mm": 5.0,
        "eps_max": 0.22, "Sw_peak_eman": 0.18,
        "label": "Scoria basaltica",
        "seismic_site": "Etna",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.22, "Acrack_m2": 0.80,
    },
    "ignimbrite": {
        "phi_m": 0.35, "phi_f": 0.06,
        "k0": 4e-14, "k_frac": 8e-12,
        "D0": 1.8e-6,
        "Ra226_Bqkg": 100.0, "rho_bulk": 1600.0,
        "soil_alpha": 1.8, "soil_n": 1.55,
        "Ksat0": 2e-6,
        "A_crack_m2": 0.30,
        "grain_size_mm": 0.1,
        "eps_max": 0.36, "Sw_peak_eman": 0.25,
        "vesicle_fraction": 0.08,
        "label": "Ignimbrite",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.36, "Acrack_m2": 0.30,
    },
    "basalt_dense": {
        "phi_m": 0.08, "phi_f": 0.02,
        "k0": 5e-17, "k_frac": 1e-13,
        "D0": 3.0e-7,
        "Ra226_Bqkg": 25.0, "rho_bulk": 2700.0,
        "soil_alpha": 0.8, "soil_n": 1.8,
        "Ksat0": 5e-8,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.001,
        "eps_max": 0.12, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.01,
        "label": "Basalto denso",
        "seismic_site": "Stromboli",
        "regime": "fracture_dominant",
        "f_src_to_fracture": 0.2,
        "emanation_model": "crystalline_microfracture",
        "epsmax": 0.12, "Acrack_m2": 0.05,
    },
    "sandy_soil": {
        "phi_m": 0.38, "phi_f": 0.04,
        "k0": 2e-12, "k_frac": 5e-12,
        "D0": 5.0e-6,
        "Ra226_Bqkg": 60.0, "rho_bulk": 1600.0,
        "soil_alpha": 4.0, "soil_n": 2.0,
        "Ksat0": 5e-5,
        "A_crack_m2": 0.15,
        "grain_size_mm": 0.5,
        "eps_max": 0.22, "Sw_peak_eman": 0.15,
        "vesicle_fraction": 0.0,
        "label": "Suolo sabbioso (riferimento)",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "epsmax": 0.22, "Acrack_m2": 0.15,
    },


    "granite": {
        "phi_m": 0.02, "phi_f": 0.005,
        "k0": 1e-17, "k_frac": 5e-13,
        "D0": 6.0e-08,
        "Ra226_Bqkg": 80.0, "rho_bulk": 2650.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.03,
        "grain_size_mm": 5.0,
        "eps_max": 0.10, "Sw_peak_eman": 0.35,
        "vesicle_fraction": 0.0,
        "label": "Granito",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "f_src_to_fracture": 0.2,
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.08,
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 5e-13,
        "fracture_density_factor": 1.0,
        "texture": "fractured_rock",
        "epsmax": 0.10, "Acrack_m2": 0.03,
    },
    "gneiss": {
        "phi_m": 0.03, "phi_f": 0.007,
        "k0": 5e-17, "k_frac": 3e-13,
        "D0": 1.0e-07,
        "Ra226_Bqkg": 60.0, "rho_bulk": 2700.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 3e-9,
        "A_crack_m2": 0.04,
        "grain_size_mm": 3.0,
        "eps_max": 0.12, "Sw_peak_eman": 0.32,
        "vesicle_fraction": 0.0,
        "label": "Gneiss",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "f_src_to_fracture": 0.19,
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.09,
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 3e-13,
        "fracture_density_factor": 1.1,
        "texture": "fractured_rock",
        "epsmax": 0.12, "Acrack_m2": 0.04,
    },
    "schist": {
        "phi_m": 0.05, "phi_f": 0.015,
        "k0": 2e-16, "k_frac": 8e-13,
        "D0": 2.0e-07,
        "Ra226_Bqkg": 55.0, "rho_bulk": 2750.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.06,
        "grain_size_mm": 0.5,
        "eps_max": 0.15, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.0,
        "label": "Scisto/Filladi",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "f_src_to_fracture": 0.23,
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.12,
        "D_matrix_crystalline": 5e-12,
        "k_frac_eff": 8e-13,
        "fracture_density_factor": 1.3,
        "texture": "fractured_rock",
        "epsmax": 0.15, "Acrack_m2": 0.06,
    },
    "limestone_compact": {
        "phi_m": 0.08, "phi_f": 0.02,
        "k0": 1e-15, "k_frac": 1e-12,
        "D0": 3.8e-07,
        "Ra226_Bqkg": 15.0, "rho_bulk": 2500.0,
        "soil_alpha": 1.2, "soil_n": 1.50,
        "Ksat0": 1e-7,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.1,
        "eps_max": 0.08, "Sw_peak_eman": 0.28,
        "vesicle_fraction": 0.0,
        "label": "Calcare compatto",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.08,
        "texture": "karst_matrix",
        "epsmax": 0.08, "Acrack_m2": 0.05,
    },
    "limestone_karst": {
        "phi_m": 0.15, "phi_f": 0.05,
        "k0": 5e-14, "k_frac": 1e-10,
        "D0": 8.8e-07,
        "Ra226_Bqkg": 20.0, "rho_bulk": 2300.0,
        "soil_alpha": 1.2, "soil_n": 1.50,
        "Ksat0": 2e-6,
        "A_crack_m2": 0.15,
        "grain_size_mm": 0.1,
        "eps_max": 0.25, "Sw_peak_eman": 0.40,
        "vesicle_fraction": 0.0,
        "label": "Calcare carsico",
        "seismic_site": "N/A",
        "regime": "karst_conduit",
        "emanation_model": "karst_terra_rossa",
        "eps_karst": 0.22,
        "f_karst_conduit": 0.08,
        "D_conduit": 5e-6,
        "k_karst_conduit": 1e-10,
        "f_karst_active": 0.05,
        "texture": "karst_matrix",
        "epsmax": 0.25, "Acrack_m2": 0.15,
    },
    "dolomite": {
        "phi_m": 0.08, "phi_f": 0.015,
        "k0": 5e-15, "k_frac": 5e-13,
        "D0": 3.8e-07,
        "Ra226_Bqkg": 12.0, "rho_bulk": 2600.0,
        "soil_alpha": 1.0, "soil_n": 1.55,
        "Ksat0": 5e-8,
        "A_crack_m2": 0.04,
        "grain_size_mm": 0.2,
        "eps_max": 0.07, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.0,
        "label": "Dolomia",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.07,
        "texture": "karst_matrix",
        "epsmax": 0.07, "Acrack_m2": 0.04,
    },
    "sandstone": {
        "phi_m": 0.20, "phi_f": 0.03,
        "k0": 5e-13, "k_frac": 2e-12,
        "D0": 1.3e-06,
        "Ra226_Bqkg": 50.0, "rho_bulk": 2100.0,
        "soil_alpha": 7.5, "soil_n": 1.89,
        "Ksat0": 1e-5,
        "A_crack_m2": 0.10,
        "grain_size_mm": 0.30,
        "eps_max": 0.30, "Sw_peak_eman": 0.18,
        "vesicle_fraction": 0.0,
        "label": "Arenaria",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "texture": "sandy_loam",
        "epsmax": 0.30, "Acrack_m2": 0.10,
    },
    "clay": {
        "phi_m": 0.45, "phi_f": 0.02,
        "k0": 5e-16, "k_frac": 1e-15,
        "D0": 3.8e-06,
        "Ra226_Bqkg": 55.0, "rho_bulk": 1400.0,
        "soil_alpha": 0.8, "soil_n": 1.09,
        "Ksat0": 5e-7,
        "A_crack_m2": 0.02,
        "grain_size_mm": 0.002,
        "eps_max": 0.45, "Sw_peak_eman": 0.12,
        "vesicle_fraction": 0.0,
        "label": "Argilla",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay",
        "epsmax": 0.45, "Acrack_m2": 0.02,
    },
    "silt": {
        "phi_m": 0.40, "phi_f": 0.02,
        "k0": 1e-14, "k_frac": 4e-14,
        "D0": 3.2e-06,
        "Ra226_Bqkg": 45.0, "rho_bulk": 1500.0,
        "soil_alpha": 2.0, "soil_n": 1.41,
        "Ksat0": 1e-6,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.02,
        "eps_max": 0.40, "Sw_peak_eman": 0.14,
        "vesicle_fraction": 0.0,
        "label": "Limo/Silt",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "silt_loam",
        "epsmax": 0.40, "Acrack_m2": 0.05,
    },
    "alluvium_mixed": {
        "phi_m": 0.35, "phi_f": 0.04,
        "k0": 5e-13, "k_frac": 2e-12,
        "D0": 2.7e-06,
        "Ra226_Bqkg": 45.0, "rho_bulk": 1700.0,
        "soil_alpha": 3.6, "soil_n": 1.56,
        "Ksat0": 3e-6,
        "A_crack_m2": 0.12,
        "grain_size_mm": 0.5,
        "eps_max": 0.28, "Sw_peak_eman": 0.16,
        "vesicle_fraction": 0.0,
        "label": "Alluvione mista",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "texture": "loam",
        "epsmax": 0.28, "Acrack_m2": 0.12,
    },
    "glacial_till": {
        "phi_m": 0.30, "phi_f": 0.03,
        "k0": 1e-14, "k_frac": 5e-14,
        "D0": 2.2e-06,
        "Ra226_Bqkg": 35.0, "rho_bulk": 2000.0,
        "soil_alpha": 1.9, "soil_n": 1.31,
        "Ksat0": 7e-7,
        "A_crack_m2": 0.06,
        "grain_size_mm": 0.5,
        "eps_max": 0.24, "Sw_peak_eman": 0.18,
        "vesicle_fraction": 0.0,
        "label": "Till glaciale",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay_loam",
        "epsmax": 0.24, "Acrack_m2": 0.06,
    },
    "marl": {
        "phi_m": 0.25, "phi_f": 0.025,
        "k0": 5e-15, "k_frac": 2e-14,
        "D0": 1.7e-06,
        "Ra226_Bqkg": 30.0, "rho_bulk": 2100.0,
        "soil_alpha": 1.9, "soil_n": 1.31,
        "Ksat0": 8e-7,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.01,
        "eps_max": 0.22, "Sw_peak_eman": 0.16,
        "vesicle_fraction": 0.0,
        "label": "Marna",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay_loam",
        "epsmax": 0.22, "Acrack_m2": 0.05,
    },
    "uraniferous_granite": {
        "phi_m": 0.02, "phi_f": 0.006,
        "k0": 2e-17, "k_frac": 8e-13,
        "D0": 6.0e-08,
        "Ra226_Bqkg": 300.0, "rho_bulk": 2680.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.04,
        "grain_size_mm": 4.0,
        "eps_max": 0.12, "Sw_peak_eman": 0.35,
        "vesicle_fraction": 0.0,
        "label": "Granitoide uranifero",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "f_src_to_fracture": 0.23,
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.10,
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 8e-13,
        "fracture_density_factor": 1.2,
        "texture": "fractured_rock",
        "epsmax": 0.12, "Acrack_m2": 0.04,
    },
}

VOLCANIC_LITHOTYPES = {
    k: v for k, v in GEOLITHOTYPES.items()
    if k in ("tuff_welded", "tuff_unwelded", "pumice", "scoria_basaltic",
             "ignimbrite", "basalt_dense", "sandy_soil")
}


VG_TEXTURE_PARAMS = {
    "sand":           (14.5,   2.68,  0.045,   0.430,  8.25e-5),
    "loamy_sand":     (12.4,   2.28,  0.057,   0.410,  4.05e-5),
    "sandy_loam":     ( 7.5,   1.89,  0.065,   0.410,  1.23e-5),
    "loam":           ( 3.6,   1.56,  0.078,   0.430,  2.89e-6),
    "silt_loam":      ( 2.0,   1.41,  0.067,   0.450,  1.25e-6),
    "clay_loam":      ( 1.9,   1.31,  0.095,   0.410,  7.22e-7),
    "clay":           ( 0.8,   1.09,  0.068,   0.380,  5.56e-7),
    "fractured_rock": ( 0.5,   1.80,  0.010,   0.050,  5.00e-8),
    "karst_matrix":   ( 1.2,   1.50,  0.020,   0.150,  1.00e-6),
}


REGIMES = {
    "porous_diffusive": {
        "phi_m_range": (0.05, 0.65),
        "transport": "millington_quirk",
        "description": "Argille, limi, marne — diffusione dominante, k bassa",
    },
    "porous_advective": {
        "phi_m_range": (0.05, 0.65),
        "transport": "millington_quirk + darcy",
        "description": "Sabbie, alluvioni — advection + diffusion, k moderata-alta",
    },
    "fracture_dominant": {
        "phi_m_range": (0.01, 0.15),
        "transport": "cubic_law_network",
        "description": "Graniti, gneiss, scisti, basalto denso — trasporto in fratture",
    },
    "karst_conduit": {
        "phi_m_range": (0.05, 0.30),
        "transport": "conduit_diffuse_dual",
        "description": "Calcari carsici — doppia permeabilità matrice + condotti",
    },
    "geothermal_active": {
        "phi_m_range": (0.10, 0.65),
        "transport": "buoyant_advection",
        "description": "Vulcanici attivi — gradiente termico + sovrapressione magmatica",
    },
}


SEISMIC_SITES = {
    "Etna": {
        "M_ref": 3.5, "gamma_seis": 1.8,
        "A_seis": 0.015, "tau_seis": 8 * 3600.0,
        "f_seismic": 1.5 / 86400.0,
    },
    "Campi Flegrei": {
        "M_ref": 2.5, "gamma_seis": 2.2,
        "A_seis": 0.025, "tau_seis": 12 * 3600.0,
        "f_seismic": 3.0 / 86400.0,
    },
    "Stromboli": {
        "M_ref": 1.5, "gamma_seis": 1.5,
        "A_seis": 0.010, "tau_seis": 4 * 3600.0,
        "f_seismic": 8.0 / 86400.0,
    },
    "N/A": {
        "M_ref": 0.0, "gamma_seis": 0.0,
        "A_seis": 0.0, "tau_seis": 1.0,
        "f_seismic": 0.0,
    },
}


_DIFF_FACE_HARMONIC = True

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def _pos(x, eps=1e-30):
    return np.maximum(np.asarray(x, dtype=float), eps)

def _smooth_step(x, x0, s=0.03):
    return 0.5 * (1.0 + np.tanh((np.asarray(x) - x0) / max(s, 1e-12)))

def _face_gradient(P, dz, P_top=None, P_bot=None):
    P = np.asarray(P, dtype=float)
    n = len(P)
    g = np.empty(n + 1)
    if n == 1:
        g[:] = 0.0
        return g
    g[1:n] = (P[1:] - P[:-1]) / dz
    g[0]   = (P[0] - float(P_top)) / (0.5*dz) if P_top is not None else g[1]
    g[-1]  = (float(P_bot) - P[-1]) / (0.5*dz) if P_bot is not None else g[-2]
    return g

def _cn_diffusion_step(C, D, dz, h, C_top=0.0, C_bot=None, harmonic=True, beta=None):
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    n = len(C)
    if n == 1:
        return C.copy()
    Dl = D[:-1]; Dr = D[1:]
    if harmonic:
        D_int = 2.0 * Dl * Dr / np.maximum(Dl + Dr, 1e-300)
    else:
        D_int = 0.5 * (Dl + Dr)
    r = h / (dz * dz)
    Dface = np.empty(n + 1)
    Dface[1:n] = D_int
    Dface[0]   = D[0]
    Dface[n]   = D[-1]
    aW = Dface[:-1]
    aE = Dface[1:]
    lo = np.zeros(n); di = np.zeros(n); up = np.zeros(n)
    rhs = np.zeros(n)
    if beta is None:
        half_c = np.full(n, 0.5 * r)
    else:
        beta_c = np.maximum(np.broadcast_to(np.asarray(beta, dtype=float),
                                            (n,)).astype(float), 1e-30)
        half_c = 0.5 * r / beta_c
    for i in range(n):
        cW = aW[i]; cE = aE[i]; half = half_c[i]
        if i == 0:
            di[i]  = 1.0 + half * (2.0*cW + cE)
            up[i]  = -half * cE
            b_expl = 1.0 - half * (2.0*cW + cE)
            rhs[i] = b_expl * C[i] + half*cE*C[i+1] + 2.0*(2.0*half*cW)*C_top
        elif i == n-1:
            if C_bot is None:
                di[i] = 1.0 + half * (cW)
                lo[i] = -half * cW
                b_expl = 1.0 - half * (cW)
                rhs[i] = b_expl * C[i] + half*cW*C[i-1]
            else:
                di[i]  = 1.0 + half * (cW + 2.0*cE)
                lo[i]  = -half * cW
                b_expl = 1.0 - half * (cW + 2.0*cE)
                rhs[i] = b_expl * C[i] + half*cW*C[i-1] + 2.0*(2.0*half*cE)*C_bot
        else:
            di[i] = 1.0 + half * (cW + cE)
            lo[i] = -half * cW
            up[i] = -half * cE
            b_expl = 1.0 - half * (cW + cE)
            rhs[i] = b_expl * C[i] + half*cW*C[i-1] + half*cE*C[i+1]
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = up[0]/di[0] if abs(di[0])>1e-300 else 0.0
    dp[0] = rhs[0]/di[0] if abs(di[0])>1e-300 else C[0]
    for i in range(1, n):
        m = di[i] - lo[i]*cp[i-1]
        if abs(m) < 1e-300:
            cp[i]=0.0; dp[i]=C[i]; continue
        cp[i] = up[i]/m
        dp[i] = (rhs[i] - lo[i]*dp[i-1])/m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def _upwind(C, q, C_top=0.0, C_bot_inlet=None):
    C = np.asarray(C, dtype=float)
    C_ext = np.empty(len(C) + 2)
    C_ext[1:-1] = C
    C_ext[0]    = C_top
    if C_bot_inlet is not None and q[-1] < 0.0:
        C_ext[-1] = C_bot_inlet
    else:
        C_ext[-1] = C[-1]
    q_face = np.asarray(q, dtype=float)
    F = np.where(q_face >= 0, C_ext[:-1], C_ext[1:]) * q_face
    return F

def _div_diff_flux(C, D, dz, C_top=0.0, C_bot=None):
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    n = len(C)
    C_ext = np.empty(n + 2)
    C_ext[1:-1] = C
    C_ext[0]    = 2.0 * C_top - C[0]
    C_ext[-1]   = C[-1] if C_bot is None else 2.0 * C_bot - C[-1]
    D_ext = np.empty(n + 2)
    D_ext[1:-1] = D
    D_ext[0]    = D[0]
    D_ext[-1]   = D[-1]
    if _DIFF_FACE_HARMONIC:
        Dl = D_ext[:-1]; Dr = D_ext[1:]
        D_face = 2.0 * Dl * Dr / np.maximum(Dl + Dr, 1e-300)
    else:
        D_face = 0.5 * (D_ext[:-1] + D_ext[1:])
    F = D_face * (C_ext[1:] - C_ext[:-1]) / dz
    return (F[1:] - F[:-1]) / dz


class VolcanicRadonModelV2:

    def __init__(self, p: dict):
        self.p = p.copy()
        self.nz   = p.get("nz", 60)
        self.H    = p["H"]
        self.dz   = self.H / self.nz
        self.z    = (np.arange(self.nz) + 0.5) * self.dz

        self.lam_rn  = DECAY_CONSTANTS["Rn222"]
        self.lam_po218 = DECAY_CONSTANTS["Po218"]
        self.lam_pb214 = DECAY_CONSTANTS["Pb214"]
        self.lam_bi214 = DECAY_CONSTANTS["Bi214"]
        self.lam_po214 = DECAY_CONSTANTS["Po214"]
        self.lam_rn220 = DECAY_CONSTANTS["Rn220"]

        self._eps_mem_init = None
        self._source_cache_rn222 = None
        self._source_cache_rn220 = None
        self.cracks = []
        self.horizontal_network = []
        self._k_seismic = np.ones(self.nz)
        self._t_seismic_last = -1e9

        self._build_dynamic_fracture_network(
            n_cracks=p.get("n_cracks", 18),
            seed=p.get("crack_seed", 42)
        )
        self._build_horizontal_fracture_layer(
            n_frac=p.get("n_frac_horiz", 20),
            seed=p.get("horiz_seed", 7)
        )

        site = str(p.get("seismic_site", "Campi Flegrei"))
        _site_norm = site.replace("_", " ").strip()
        self.seis = SEISMIC_SITES.get(site,
                    SEISMIC_SITES.get(_site_norm, SEISMIC_SITES["N/A"]))

        self._vg_alpha_d = p["soil_alpha"]
        self._vg_n_d     = p["soil_n"]
        self._vg_alpha_w = p.get("soil_alpha_wet", p["soil_alpha"] * 2.0)
        self._vg_n_w     = p.get("soil_n_wet",     p["soil_n"]     * 0.92)

        self.nstate = 10 * self.nz + 7


    def _Se_drainage(self, theta):
        theta = np.asarray(theta, dtype=float)
        phi   = self.p["phi_m"]
        Se    = np.clip(theta / max(phi, 1e-12), 1e-8, 0.9999)
        return Se

    def _h_from_theta_drainage(self, theta):
        Se    = self._Se_drainage(theta)
        alpha = self._vg_alpha_d
        n     = self._vg_n_d
        m     = 1.0 - 1.0 / n
        h     = -((Se**(-1.0/m) - 1.0)**(1.0/n)) / alpha
        return h

    def _kr_vg_drainage(self, Se):
        Se    = np.clip(Se, 1e-8, 0.9999)
        n     = self._vg_n_d
        m     = 1.0 - 1.0 / n
        kr    = np.sqrt(Se) * (1.0 - (1.0 - Se**(1.0/m))**m)**2
        return kr

    def _kr_gas_vg(self, Se):
        Se = np.clip(Se, 1e-8, 0.9999)
        n  = self._vg_n_d
        m  = 1.0 - 1.0 / n
        return np.sqrt(1.0 - Se) * (1.0 - Se ** (1.0 / m)) ** (2.0 * m)

    def _K_unsat(self, theta, z):
        Se   = self._Se_drainage(theta)
        Ksat = self.p["Ksat0"] * (1.0 + self.p.get("Ksat_grad", 0.2) * z / self.H)
        return np.clip(Ksat * self._kr_vg_drainage(Se), 1e-12, 1e-4)

    def richards_rhs(self, theta, t):
        theta_raw = np.asarray(theta, dtype=float)
        theta = np.clip(theta_raw, 1e-8, 0.95 * self.p["phi_m"])
        h     = self._h_from_theta_drainage(theta)
        K     = self._K_unsat(theta, self.z)
        q     = np.empty(self.nz + 1)
        rain  = self.p.get("rain0", 2e-7) * max(np.sin(2*np.pi*t / self.p.get("T_rain", 864000)), 0.0)
        evap  = self.p.get("evap0", 1e-7) * max(np.sin(2*np.pi*t / self.p.get("T_evap", 86400) + self.p.get("evap_phase", np.pi/2)), 0.0)
        q_pot     = rain - evap
        h_pond    = self.p.get("richards_h_pond", 0.0)
        K_surf    = float(np.clip(self.p["Ksat0"], 1e-12, 1e-4))
        K_face    = 0.5 * (K_surf + K[0])
        q_inf_max = -K_face * ((h[0] - h_pond) / (0.5 * self.dz) - 1.0)
        _phi_m_r  = self.p["phi_m"]
        _x_sat    = float(np.clip((theta_raw[0] - 0.95*_phi_m_r) / (0.05*_phi_m_r), 0.0, 1.0))
        q_inf_max_eff = (1.0 - _x_sat) * q_inf_max + _x_sat * K[0]
        q[0]      = min(q_pot, q_inf_max_eff)
        dhdz_int = (h[1:] - h[:-1]) / self.dz
        Kf_int   = 0.5 * (K[1:] + K[:-1])
        q[1:self.nz] = -Kf_int * (dhdz_int - 1.0)
        if theta_raw[0] >= _phi_m_r:
            q[0] = min(q[0], q[1])
        self._runoff_rate = max(q_pot - q[0], 0.0)
        bc_mode = self.p.get("richards_bottom_bc", "water_table")
        if bc_mode == "free_drainage":
            q[-1] = K[-1]
        else:
            h_bot   = self.p.get("richards_h_bottom", h[-1])
            dz_half = 0.5 * self.dz
            dhdz_bot = (h_bot - h[-1]) / dz_half
            q[-1] = -K[-1] * (dhdz_bot - 1.0)
        return -(q[1:] - q[:-1]) / self.dz

    def T_field(self, t):
        z      = self.z
        T0     = self.p.get("T0", 288.0)
        grad   = self.p.get("T_grad", 0.025)
        dT_s   = self.p.get("T_surf_amp", 8.0)
        z_damp = self.p.get("T_damp_depth", 0.12)
        omega  = 2*np.pi / 86400.0
        T      = T0 + grad * z + dT_s * np.exp(-z / z_damp) * np.sin(omega * t - z / z_damp)
        return np.clip(T, 260.0, 370.0)

    def T_outdoor(self, t):
        T_out0 = self.p.get("T_out0", 283.15)
        A_day  = self.p.get("T_out_amp", 5.0)
        A_year = self.p.get("T_out_year_amp", 0.0)
        phi_y  = self.p.get("T_out_year_phase", -np.pi/2)
        T_year = self.p.get("T_year", 365.25 * 86400.0)
        T_day  = self.p.get("T_day", 86400.0)
        return (T_out0
                + A_year * np.sin(2*np.pi * t / max(T_year, 1.0) + phi_y)
                + A_day  * np.sin(2*np.pi * t / max(T_day, 1.0)))

    def T_indoor(self, t):
        T_in0  = self.p.get("T_in0", 293.15)
        A_day  = self.p.get("T_in_amp", 2.0)
        A_year = self.p.get("T_in_year_amp", 0.0)
        phi_y  = self.p.get("T_in_year_phase", -np.pi/2)
        T_year = self.p.get("T_year", 365.25 * 86400.0)
        T_day  = self.p.get("T_day", 86400.0)
        return (T_in0
                + A_year * np.sin(2*np.pi * t / max(T_year, 1.0) + phi_y)
                + A_day  * np.sin(2*np.pi * t / max(T_day, 1.0)))

    def stack_pressure(self, t, Tin=None, Tout=None, Patm=None):
        g    = self.p.get("g", 9.81)
        H_s  = self.p.get("H_stack", 2.5)
        Mg   = self.p.get("M_g", 0.02897)
        Rg   = self.p.get("R_g", 8.314)
        Patm = self.p.get("P_atm0", 101325.0) if Patm is None else float(Patm)
        Tin  = self.T_indoor(t)  if Tin  is None else Tin
        Tout = self.T_outdoor(t) if Tout is None else Tout
        rho_in  = Patm * Mg / (Rg * np.maximum(Tin,  1.0))
        rho_out = Patm * Mg / (Rg * np.maximum(Tout, 1.0))
        return -0.5 * g * H_s * (rho_out - rho_in)

    def stack_effect_advective_factor(self, t):
        dP = np.abs(self.stack_pressure(t))
        return dP

    def _k_gas_standard(self, theta, z, seismic_mult=None):
        theta    = np.asarray(theta, dtype=float)
        phi      = self.p["phi_m"]
        Sw       = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg       = 1.0 - Sw
        Sg_crit  = self.p.get("Sg_crit", 0.08)
        Sg_sharp = self.p.get("Sg_sharp", 0.02)
        nk       = self.p.get("nk", 3.5)
        k0       = self.p["k0"] * np.exp(self.p.get("kz_grad", 0.0) * z / self.H)
        open_mask= 0.5*(1.0 + np.tanh((Sg - Sg_crit) / max(Sg_sharp, 1e-12)))
        rel      = np.clip((Sg - Sg_crit) / max(1.0 - Sg_crit, 1e-12), 0.0, 1.0)
        k        = k0 * open_mask * rel**nk
        k        = np.clip(k, self.p.get("k_min", 1e-18), self.p.get("k_max", 1e-10))
        if seismic_mult is not None:
            k = k * seismic_mult
        return k

    def _k_gas(self, theta, z, seismic_mult=None):
        regime = self.p.get("regime", "porous_diffusive")

        if regime == "fracture_dominant":
            theta    = np.asarray(theta, dtype=float)
            phi      = self.p["phi_m"]
            Sw       = np.clip(theta / max(phi, 1e-12), 0.0, 1.0)
            k_matrix = self.p["k0"] * np.exp(self.p.get("kz_grad", 0.0) * z / self.H)
            k_frac_eff = self.p.get("k_frac_eff", self.p.get("k_frac", 1e-13))
            f_frac     = self.p.get("fracture_density_factor", 1.0)
            frac_open  = np.clip(1.0 - Sw**2, 0.01, 1.0)
            k_eff      = k_matrix + k_frac_eff * f_frac * frac_open
            k_eff      = np.clip(k_eff, self.p.get("k_min", 1e-18),
                                         self.p.get("k_max", 1e-10))
            if seismic_mult is not None:
                k_eff = k_eff * seismic_mult
            return k_eff

        elif regime == "karst_conduit":
            k_base    = self._k_gas_standard(theta, z, seismic_mult)
            k_conduit = self.p.get("k_karst_conduit", 1e-10)
            f_active  = self.p.get("f_karst_active", 0.05)
            k_eff     = k_base + k_conduit * f_active
            return np.clip(k_eff, self.p.get("k_min", 1e-18),
                                   self.p.get("k_max", 1e-9))

        else:
            return self._k_gas_standard(theta, z, seismic_mult)

    def _barometric_transient_profile(self, theta, t, seismic_mult=None):
        amp   = self.p.get("P_atm_amp", 150.0)
        T_a   = self.p.get("T_atm", 86400.0)
        omega = 2.0 * np.pi / max(T_a, 1.0)
        mu    = self.p.get("mu", 1.82e-5)
        Mg    = self.p.get("M_g", 0.02897); Rg = self.p.get("R_g", 8.314)
        k     = self._k_gas(theta, self.z, seismic_mult)
        phi_g = np.maximum(self.p["phi_m"] - np.asarray(theta, dtype=float), 1e-6)
        P0    = self.p.get("P_atm0", 101325.0)
        D_P   = np.maximum(k * P0 / np.maximum(phi_g * mu, 1e-30), 1e-12)
        D_P_bar = float(np.mean(D_P))
        delta   = float(np.sqrt(2.0 * D_P_bar / max(omega, 1e-30)))
        delta   = max(delta, 1e-3)
        zeta    = self.z / delta
        return amp * np.exp(-zeta) * np.sin(omega * t - zeta)

    def gas_pressure_rhs(self, theta, Pg, t, seismic_mult=None):
        theta   = np.asarray(theta, dtype=float)
        Pg      = np.asarray(Pg,    dtype=float)
        nz      = self.nz
        mu      = self.p.get("mu", 1.82e-5)
        k       = self._k_gas(theta, self.z, seismic_mult)
        dz      = self.dz
        Patm    = self._Patm(t)
        if self.p.get("barometric_pumping_on", True):
            Pbot = self.p.get("P_atm0", 101325.0) + self.p.get("dP_bottom", 5.0)
        else:
            Pbot = Patm + self.p.get("dP_bottom", 5.0)

        if nz == 1:
            Pg_ss = np.array([0.5 * (Patm + Pbot)])
            phi_g1 = max(self.p["phi_m"] - float(np.mean(np.asarray(theta))), 1e-12)
            D_P = k[0] * 0.5 * (Patm + Pbot) / (mu * phi_g1)
            tau = max(dz**2 / (2.0 * D_P), 1.0)
            return (Pg_ss - Pg) / tau

        k_face = np.empty(nz + 1)
        k_face[0]  = k[0]
        k_face[-1] = k[-1]
        for i in range(1, nz):
            k_face[i] = 2.0 * k[i-1] * k[i] / max(k[i-1] + k[i], 1e-40)
        T_lam = k_face / (mu * dz)
        T_lam[0]  *= 2.0
        T_lam[-1] *= 2.0

        a = np.zeros(nz); b = np.zeros(nz); c = np.zeros(nz); rhs = np.zeros(nz)
        for i in range(nz):
            tL = T_lam[i]; tR = T_lam[i+1]
            b[i] = -(tL + tR)
            if i > 0:     a[i] = tL
            if i < nz-1:  c[i] = tR
        rhs[0]  = -T_lam[0] * Patm
        rhs[-1] = -T_lam[-1] * Pbot

        c_ = np.zeros(nz); d_ = np.zeros(nz)
        c_[0] = c[0] / b[0] if abs(b[0]) > 1e-40 else 0.0
        d_[0] = rhs[0] / b[0] if abs(b[0]) > 1e-40 else Patm
        for i in range(1, nz):
            denom = b[i] - a[i] * c_[i-1]
            if abs(denom) < 1e-40:
                c_[i] = 0.0; d_[i] = Pg[i]
            else:
                c_[i] = c[i] / denom
                d_[i] = (rhs[i] - a[i] * d_[i-1]) / denom
        Pg_ss = np.zeros(nz)
        Pg_ss[-1] = d_[-1]
        for i in range(nz-2, -1, -1):
            Pg_ss[i] = d_[i] - c_[i] * Pg_ss[i+1]

        P_lo = min(Patm, Pbot) - 50.0
        P_hi = max(Patm, Pbot) + 50.0
        Pg_ss = np.clip(Pg_ss, P_lo, P_hi)

        phi_g = np.maximum(self.p["phi_m"] - theta, 1e-12)
        D_P = k * Pg_ss / (mu * phi_g)
        D_flux = k * Pg_ss / mu
        phi_g_div = np.maximum(phi_g, 0.05 * self.p["phi_m"])
        return _div_diff_flux(Pg, D_flux, dz, C_top=Patm, C_bot=Pbot) / phi_g_div


    def _Patm(self, t):
        P0  = self.p.get("P_atm0", 101325.0)
        amp = self.p.get("P_atm_amp", 150.0)
        T_a = self.p.get("T_atm", 86400.0)
        return P0 + amp * np.sin(2*np.pi*t / T_a)

    def _dPatm_dt(self, t):
        amp = self.p.get("P_atm_amp", 150.0)
        T_a = self.p.get("T_atm", 86400.0)
        return amp * (2*np.pi/T_a) * np.cos(2*np.pi*t / T_a)


    def seismic_permeability_multiplier(self, t):
        gr_on = self.p.get("seismic_gr_magnitudes_on", True)
        if not hasattr(self, "_seis_events"):
            rng  = np.random.default_rng(self.p.get("seis_seed", 99))
            rate = self.seis["f_seismic"]
            T_sim= self.p.get("t_total_seis", 7*86400.0)
            n_ev = max(int(rng.poisson(rate * T_sim)), 1)
            self._seis_events = sorted(rng.uniform(0, T_sim, n_ev))
            if gr_on:
                b     = self.p.get("GR_b_value", 1.0)
                M_min = self.p.get("M_min_seis", 1.0)
                M_max = self.p.get("M_max_seis", 5.0)
                beta  = b * np.log(10.0)
                u     = rng.uniform(0.0, 1.0, n_ev)
                denom = 1.0 - np.exp(-beta * (M_max - M_min))
                self._seis_mags = M_min - np.log(1.0 - u * denom) / beta
                self._seis_mags = np.clip(self._seis_mags, M_min, M_max)
            else:
                self._seis_mags = rng.exponential(self.seis["M_ref"], n_ev)

        A      = self.seis["A_seis"]
        gamma  = self.seis["gamma_seis"]
        tau    = self.seis["tau_seis"]
        M_ref  = self.seis.get("M_ref", 2.5)
        M_shift = M_ref if gr_on else 0.0
        mult   = 1.0
        for t_ev, M in zip(self._seis_events, self._seis_mags):
            if t >= t_ev:
                dt   = t - t_ev
                mult += A * np.exp(gamma * (M - M_shift)) * np.exp(-dt / max(tau, 1e-12))
        return np.clip(mult, 1.0, self.p.get("k_seis_max", 50.0)) * np.ones(self.nz)

    def seismic_permeability_multiplier_recovery(self, t, t0=None):
        if not self.p.get("seismic_recovery_on", False):
            return np.ones(self.nz)
        t0 = float(self.p.get("seis_t0_s", 0.0)) if t0 is None else float(t0)
        if "k_seis_peak" in self.p:
            k_peak = float(self.p["k_seis_peak"])
        else:
            M_w   = float(self.p.get("M_w_unrest", 0.0))
            A     = float(self.seis.get("A_seis", 0.30))
            g     = float(self.seis.get("gamma_seis", 0.90))
            M_ref = float(self.seis.get("M_ref", 2.5))
            k_peak = 1.0 + A * math.exp(g * (M_w - M_ref))
        tau = float(self.p.get(
            "tau_seis_recovery",
            _tau_seis_recovery_seconds(site=self.p.get("seismic_site"))))
        k_floor = float(self.p.get("k_seis_floor", 1.0))
        k_t = seismic_recovery_factor(t, t0, k_peak, tau, k_floor=k_floor)
        k_t = float(np.clip(k_t, 1.0, self.p.get("k_seis_max", 50.0)))
        return k_t * np.ones(self.nz)


    def _Se_wetting(self, theta):
        phi = self.p["phi_m"]
        Se  = np.clip(theta / max(phi, 1e-12), 1e-8, 0.9999)
        return Se

    def hydraulic_hysteresis_indicator(self, theta, dtheta_dt):
        on_drainage = (np.mean(dtheta_dt) <= 0.0)
        return 1.0 if on_drainage else 0.0

    def _fracture_activation_metric(self, theta, Pg):
        theta  = np.asarray(theta, dtype=float)
        Pg     = np.asarray(Pg,    dtype=float)
        phi    = self.p["phi_m"]
        Sw     = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        chi    = np.zeros(self.nz)
        Pg_ref = float(np.mean(Pg))
        for crack in self.cracks:
            a     = crack.get("aperture", crack.get("aperture_0", 1e-4))
            a0    = crack.get("aperture_0", a)
            conn  = self._crack_connectivity_scalar(Sw, crack)
            chi  += conn * np.clip(a / max(a0, 1e-12), 0.1, 10.0)
        return np.nan_to_num(chi / max(len(self.cracks), 1), nan=0.0)

    def radon_accessibility_rhs(self, Ar, theta, dtheta_dt, Pg):
        Ar         = np.asarray(Ar,         dtype=float)
        theta      = np.asarray(theta,      dtype=float)
        dtheta_dt  = np.asarray(dtheta_dt,  dtype=float)
        Pg         = np.asarray(Pg,         dtype=float)
        phi        = self.p["phi_m"]
        Sw         = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg         = np.clip(1.0 - Sw, 1e-12, 1.0)
        drying     = np.maximum(-dtheta_dt, 0.0)
        wetting    = np.maximum( dtheta_dt, 0.0)
        beta_d     = self.p.get("A_beta_dry",    1.0e9)
        dry_crit   = self.p.get("A_dtheta_crit", 1e-8)
        dry_boost  = self.p.get("A_dry_boost",   1.0)
        dry_gate   = 1.0 + dry_boost / (1.0 + np.exp(-beta_d * (drying - dry_crit)))
        ng         = self.p.get("A_ng",  2.0)
        nw         = self.p.get("A_nw",  2.0)
        chi_f      = self._fracture_activation_metric(theta, Pg)
        beta_f     = self.p.get("A_beta_frac", 1.5)
        gradPg     = np.abs(np.gradient(Pg, self.dz)) if len(Pg) > 1 else np.zeros(len(Pg))
        beta_pg    = self.p.get("A_beta_gradPg", 0.0)
        pg_ref     = self.p.get("A_gradPg_ref",  100.0)
        press_gate = 1.0 + beta_pg * gradPg / max(pg_ref, 1e-12)
        k_open     = self.p.get("A_k_open",   1.0/3600.0)
        k_close    = self.p.get("A_k_close",  0.35/3600.0)
        beta_rewet = self.p.get("A_beta_rewet", 1e6)
        phi_open   = (Sg**ng) * dry_gate * (1.0 + beta_f * chi_f) * press_gate
        phi_close  = (Sw**nw) * (1.0 + beta_rewet * wetting)
        dAr_dt     = k_open * phi_open * (1.0 - Ar) - k_close * phi_close * Ar
        return np.clip(dAr_dt, -1e3, 1e3)

    def emanation_hysteresis_rhs(self, eps_mem, theta, dtheta_dt, T):
        eps_inst   = self._emanation_factor_inst(theta, T)
        tau_dry    = self.p.get("tau_hyst_dry",  2 * 3600.0)
        tau_wet    = self.p.get("tau_hyst_wet", 48 * 3600.0)
        if self.p.get("hysteresis_per_cell", True):
            dtheta_dt = np.asarray(dtheta_dt, dtype=float)
            tau_h = np.where(dtheta_dt <= 0.0, tau_dry, tau_wet)
            return (eps_inst - eps_mem) / np.maximum(tau_h, 1e-12)
        else:
            drying = (np.mean(dtheta_dt) <= 0.0)
            tau_h  = tau_dry if drying else tau_wet
            return (eps_inst - eps_mem) / max(tau_h, 1e-12)

    def _emanation_vesicular(self, theta, T):
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_max = self.p.get("eps_max",       0.30)
        Sw_pk   = self.p.get("Sw_peak_eman",  0.22)
        sigma_s = self.p.get("Sw_sigma_eman", 0.14)
        eps_dry = self.p.get("eps_dry",        0.06)
        rise    = 1.0 - np.exp(-Sw / max(Sw_pk, 1e-6))
        eps_sw  = eps_dry + (eps_max - eps_dry) * rise
        Sw_blk  = self.p.get("Sw_block_eman",  0.90)
        blk_sh  = self.p.get("Sw_block_sharp",  0.03)
        block   = 1.0 / (1.0 + np.exp((Sw - Sw_blk) / max(blk_sh, 1e-6)))
        T_ref   = self.p.get("T_ref", 293.15)
        beta_T  = self.p.get("beta_T_eman", 0.003)
        temp_f  = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        g_mm    = self.p.get("grain_size_mm",  0.2)
        g_ref   = self.p.get("grain_ref_mm",  0.25)
        g_exp   = self.p.get("grain_exp_eman", 0.20)
        grain_f = np.clip((g_ref / max(g_mm, 1e-6))**g_exp, 0.7, 1.6)
        return np.clip(eps_sw * block * temp_f * grain_f, 0.01, 0.80)

    def _emanation_granular(self, theta, T):
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        grain_mm = self.p.get("grain_size_mm", 0.25)
        eps_dry  = self.p.get("eps_dry",  0.06)
        eps_sat  = 0.50 * (0.1 / max(grain_mm, 0.005))**0.30
        eps_sat  = float(np.clip(eps_sat, 0.15, 0.52))
        Sw_sat   = float(np.clip(0.04 + 0.12 * grain_mm, 0.01, 0.20))
        f_moist  = 1.0 - np.exp(-Sw / max(Sw_sat, 1e-6))
        eps_sw   = eps_dry + (eps_sat - eps_dry) * f_moist
        block    = 1.0 / (1.0 + np.exp((Sw - 0.85) / 0.03))
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_sw * block * temp_f, 0.01, 0.60)

    def _emanation_crystalline(self, theta, T):
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_base = self.p.get("eps_crystalline", 0.08)
        eps_sw   = eps_base * (1.0 + 0.05 * Sw)
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_sw * temp_f, 0.01, 0.25)

    def _emanation_karst(self, theta, T):
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_base = self.p.get("eps_karst", 0.22)
        f_wash   = 1.0 + 0.30 * np.clip(Sw - 0.50, 0.0, 1.0)
        block    = 1.0 / (1.0 + np.exp((Sw - 0.88) / 0.03))
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_base * f_wash * block * temp_f, 0.05, 0.45)


    def _moisture_emanation_factor(self, Sw):
        Sw      = np.clip(np.asarray(Sw, dtype=float), 0.0, 1.0)
        Sw_pk   = self.p.get("Sw_peak_moist",   0.17)
        sigma   = self.p.get("Sw_sigma_moist",  0.12)
        f_dry   = self.p.get("f_emanation_dry", 0.45)
        Sw_blk  = self.p.get("Sw_block_moist",  0.88)
        blk_sh  = self.p.get("Sw_block_sharp_moist", 0.04)
        bump    = np.exp(-((Sw - Sw_pk) / max(sigma, 1e-6))**2)
        rise    = f_dry + (1.0 - f_dry) * bump
        block   = 1.0 / (1.0 + np.exp((Sw - Sw_blk) / max(blk_sh, 1e-6)))
        norm_pk = (f_dry + (1.0 - f_dry)) / (1.0 + np.exp((Sw_pk - Sw_blk) / max(blk_sh, 1e-6)))
        f = rise * block / max(norm_pk, 1e-9)
        return np.clip(f, 0.05, 1.05)

    def _moisture_diffusivity_choke(self, Sw):
        Sw      = np.clip(np.asarray(Sw, dtype=float), 0.0, 0.9999)
        Sw_perc = self.p.get("Sw_perc_gas", 0.95)
        p_exp   = self.p.get("diff_choke_exp", 3.0)
        chk_min = self.p.get("diff_choke_min", 1e-4)
        x       = np.clip(1.0 - Sw / max(Sw_perc, 1e-6), 0.0, 1.0)
        return np.clip(x**p_exp, chk_min, 1.0)

    def _emanation_factor_inst(self, theta, T):
        eman_model = self.p.get("emanation_model", "volcanic_vesicular")
        if eman_model == "grain_size_moisture":
            eps = self._emanation_granular(theta, T)
        elif eman_model == "crystalline_microfracture":
            eps = self._emanation_crystalline(theta, T)
        elif eman_model == "karst_terra_rossa":
            eps = self._emanation_karst(theta, T)
        else:
            eps = self._emanation_vesicular(theta, T)

        if self.p.get("moisture_emanation_on", False):
            phi = self.p["phi_m"]
            Sw  = np.clip(np.asarray(theta, dtype=float) / max(phi, 1e-12), 1e-12, 0.9999)
            eps = eps * self._moisture_emanation_factor(Sw)
            eps = np.clip(eps, 0.005, 0.80)
        return eps

    def source_rn222(self, theta, T, Ar, eps_mem):
        theta    = np.asarray(theta,   dtype=float)
        T        = np.asarray(T,       dtype=float)
        Ar       = np.clip(np.asarray(Ar,     dtype=float), 0.0, 1.0)
        eps_mem  = np.clip(np.asarray(eps_mem, dtype=float), 0.01, 0.80)
        phi      = self.p["phi_m"]
        theta    = np.clip(theta, 1e-12, 0.9999*phi)
        Sw       = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        phi_g    = np.maximum(phi - theta, 1e-12)
        phi_w    = np.maximum(theta,        1e-12)
        H        = self._henry_rn(T)
        A_part   = np.maximum(phi_g + H * phi_w, 1e-12)
        rho_b    = self.p.get("rho_bulk",   1600.0)
        Ra226    = self.p.get("Ra226_Bqkg",   35.0)
        S_bulk   = rho_b * Ra226 * eps_mem * Ar
        S_gas    = self.lam_rn * S_bulk / A_part
        return np.maximum(S_gas, 0.0)

    def source_rn220(self, theta, T, Ar, eps_mem):
        theta    = np.asarray(theta,  dtype=float)
        T        = np.asarray(T,      dtype=float)
        Ar       = np.clip(np.asarray(Ar,    dtype=float), 0.0, 1.0)
        eps_mem  = np.clip(np.asarray(eps_mem,dtype=float), 0.01, 0.80)
        phi      = self.p["phi_m"]
        theta    = np.clip(theta, 1e-12, 0.9999*phi)
        phi_g    = np.maximum(phi - theta, 1e-12)
        phi_w    = np.maximum(theta,        1e-12)
        H        = self._henry_rn(T)
        A_part   = np.maximum(phi_g + H * phi_w, 1e-12)
        rho_b    = self.p.get("rho_bulk",   1600.0)
        Th232    = self.p.get("Th232_Bqkg",   25.0)
        eps_220  = eps_mem * self.p.get("eps_ratio_220_222", 1.0)
        S_gas    = self.lam_rn220 * rho_b * Th232 * eps_220 * Ar / A_part
        return np.maximum(S_gas, 0.0)

    def _henry_rn(self, T):
        H0  = self.p.get("H_rn0",  0.23)
        dHR = self.p.get("dH_rn", 2800.0)
        T0  = 293.15
        return H0 * np.exp(dHR * (1.0 / np.maximum(np.asarray(T, dtype=float), 1.0) - 1.0/T0))

    def _beta_partition(self, theta, T, domain="m"):
        theta = np.asarray(theta, dtype=float)
        phi   = self.p["phi_m"] if domain == "m" else self.p["phi_f"]
        phi_g = np.maximum(phi - theta, 1e-12)
        L     = self._henry_rn(T)
        return np.maximum(phi_g + L * np.maximum(theta, 0.0), 1e-12)

    def _D_eff_millington_quirk(self, theta, domain="m"):
        theta  = np.asarray(theta, dtype=float)
        phi    = self.p["phi_m"] if domain == "m" else self.p["phi_f"]
        Sw     = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg     = 1.0 - Sw
        if domain == "f":
            D0 = self.p.get("D0_f", 1.1e-5)
        else:
            D0 = self.p.get("D0", 1.1e-5)
        nD     = self.p.get("nD",  2.5)
        betaD  = self.p.get("betaD", 2.0)
        Dmin   = self.p.get("Dmin", 1e-12)
        Dmax   = self.p.get("Dmax", 1.1e-5)
        tau_mq = (Sg**nD) * np.exp(-betaD * Sw)
        _bulk_f = phi if domain == "f" else 1.0
        D_mq   = Dmin + _bulk_f * D0 * tau_mq
        if self.p.get("moisture_diffusivity_on", False):
            D_mq = Dmin + (D_mq - Dmin) * self._moisture_diffusivity_choke(Sw)
        return np.clip(D_mq, Dmin, Dmax)

    def _D_eff(self, theta, domain="m"):
        regime = self.p.get("regime", "porous_diffusive")

        if regime == "fracture_dominant":
            if domain == "f":
                return self._D_eff_millington_quirk(theta, "f")
            D_matrix = self.p.get("D_matrix_crystalline", 1e-11)
            return np.full(self.nz, D_matrix)

        elif regime == "karst_conduit":
            D_matrix   = self._D_eff_millington_quirk(theta, domain)
            f_conduit  = self.p.get("f_karst_conduit", 0.08)
            D_conduit  = self.p.get("D_conduit", 5e-6)
            _bf = self.p["phi_f"] if domain == "f" else 1.0
            return D_matrix * (1.0 - f_conduit) + _bf * D_conduit * f_conduit

        else:
            return self._D_eff_millington_quirk(theta, domain)

    def gas_water_exchange(self, Cg, Cw, theta, T, domain="m"):
        phi  = self.p["phi_m"] if domain == "m" else self.p["phi_f"]
        Sw   = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg   = 1.0 - Sw
        H    = self._henry_rn(T)
        Cw_eq= H * Cg
        a_int= self.p.get("a_int0", 1200.0) * (Sw**2.1) * (Sg**1.7)
        kla  = self.p.get("k_la0",  3.2e-5) * a_int * (1.0 + 0.8*np.abs(Sw - 0.5))
        return kla * (Cw_eq - Cw)


    def memory_kernel(self, s):
        nu  = self.p.get("mem_nu",  0.70)
        tau = self.p.get("mem_tau", 6*3600.0)
        a   = self.p.get("mem_alpha", 1.0/3600.0)
        s   = np.maximum(np.asarray(s, dtype=float), 0.0)
        return (a**nu) * (s**(nu-1.0)) * np.exp(-s / max(tau, 1e-12)) / gamma(nu)

    def exchange_memory_state_rhs(self, M, Cm, Cf):
        tau = self.p.get("mem_tau", 6*3600.0)
        return (Cm - Cf - M) / max(tau, 1e-12)

    @staticmethod
    def _mf_Dref(Dm, like):
        Dm_a = np.atleast_1d(np.asarray(Dm, dtype=float))
        if Dm_a.size == np.size(like):
            return Dm_a.reshape(np.shape(np.atleast_1d(like)))
        return float(np.mean(Dm_a))

    def matrix_fracture_flux(self, Cm, Cf, M, theta_m, theta_f, Pg_m, Pg_f):
        phi_m     = self.p["phi_m"]
        phi_f     = self.p["phi_f"]
        Swm       = np.clip(theta_m / max(phi_m, 1e-12), 1e-12, 0.9999)
        Swf       = np.clip(theta_f / max(phi_f, 1e-12), 1e-12, 0.9999)
        sat_f     = np.clip((1.0 - 0.5*(Swm+Swf))**self.p.get("n_mf_sat", 2.0), 0.0, 1.0)
        press_f   = 1.0 + self.p.get("beta_mf_p", 1e-6) * np.abs(Pg_m - Pg_f)
        Dm        = self._D_eff(theta_m, "m")
        Df        = self._D_eff(theta_f, "f")
        Dref      = self._mf_Dref(Dm, sat_f)
        alpha0    = self.p.get("a_mf0", 1.0) * sat_f * press_f * Dref / max(self.p.get("d_mf", 0.50), 1e-12)**2
        eta_inst  = self.p.get("eta_inst", 1.0)
        eta_mem   = self.p.get("eta_mem",  0.5)
        return alpha0 * (eta_inst*(Cm-Cf) + eta_mem*M)

    def _mf_alpha(self, theta_m, theta_f, Pg_m, Pg_f):
        phi_m   = self.p["phi_m"]
        phi_f   = self.p["phi_f"]
        Swm     = np.clip(np.asarray(theta_m) / max(phi_m, 1e-12), 1e-12, 0.9999)
        Swf     = np.clip(np.asarray(theta_f) / max(phi_f, 1e-12), 1e-12, 0.9999)
        sat_f   = np.clip((1.0 - 0.5*(Swm+Swf))**self.p.get("n_mf_sat", 2.0), 0.0, 1.0)
        press_f = 1.0 + self.p.get("beta_mf_p", 1e-6) * np.abs(np.asarray(Pg_m) - np.asarray(Pg_f))
        Dm      = self._D_eff(theta_m, "m")
        Df      = self._D_eff(theta_f, "f")
        Dref    = self._mf_Dref(Dm, sat_f)
        return (self.p.get("a_mf0", 1.0) * sat_f * press_f * Dref
                / max(self.p.get("d_mf", 0.50), 1e-12)**2)

    def _deep_equilibrium_conc(self, S, theta, Pg, T, lam, include_exchange=True):
        S_a  = np.atleast_1d(np.asarray(S, dtype=float))
        th   = np.atleast_1d(np.asarray(theta, dtype=float))
        Pg_a = np.atleast_1d(np.asarray(Pg, dtype=float))
        th_f = np.clip(self.p.get("gamma_sw_f", 0.2)
                       * (th / max(self.p["phi_m"], 1e-12)) * self.p["phi_f"],
                       1e-12, 0.9999*self.p["phi_f"])
        if include_exchange:
            alpha = np.atleast_1d(self._mf_alpha(th, th_f, Pg_a, Pg_a))
        else:
            alpha = np.zeros_like(np.atleast_1d(np.asarray(th, dtype=float)))
        beta_m = np.atleast_1d(self._beta_partition(th,   T, "m"))
        beta_f = np.atleast_1d(self._beta_partition(th_f, T, "f"))
        eta_i  = self.p.get("eta_inst", 1.0)
        f_src  = float(self.p.get("f_src_to_fracture", 0.0))
        num    = alpha * eta_i
        lam_s  = max(lam, 1e-30)
        bm_s   = np.maximum(beta_m, 1e-30)
        bf_s   = np.maximum(beta_f, 1e-30)
        S_m    = (1.0 - f_src) * S_a
        S_f    = f_src * S_a * bm_s / bf_s
        r_m    = num / bm_s
        r_f    = num / bf_s
        den    = np.maximum(lam_s * (lam_s + r_m + r_f), 1e-300)
        Cm_eq  = (S_m * (lam_s + r_f) + S_f * r_m) / den
        Cf_eq  = (S_f * (lam_s + r_m) + S_m * r_f) / den
        return Cm_eq, Cf_eq

    def adsorption_dynamic_rhs(self, Cs, Cm, theta):
        phi    = self.p["phi_m"]
        Sw     = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        f_ads  = 1.0 / (1.0 + np.exp(-10.0*(Sw - 0.25)))
        k_ads  = self.p.get("k_ads_dry", 2e-6) + (self.p.get("k_ads_wet", 5e-7) - self.p.get("k_ads_dry", 2e-6)) * f_ads
        k_des  = self.p.get("k_des_dry", 2e-7) + (self.p.get("k_des_wet", 8e-7) - self.p.get("k_des_dry", 2e-7)) * f_ads
        return k_ads * Cm - k_des * Cs - self.lam_rn * Cs


    def _build_dynamic_fracture_network(self, n_cracks=18, seed=42):
        rng = np.random.default_rng(seed)
        self.cracks = []
        for i in range(n_cracks):
            b0 = rng.lognormal(np.log(6e-4), 0.85)
            self.cracks.append({
                "id":                  i,
                "aperture_0":          float(b0),
                "aperture":            float(b0),
                "current_aperture":    float(b0),
                "length":              float(rng.uniform(0.5, 3.5)),
                "depth":               float(rng.uniform(0.02, 0.85)),
                "angle":               float(rng.uniform(0.0, np.pi)),
                "tortuosity":          float(rng.uniform(1.1, 2.6)),
                "Sw_crit":             float(rng.uniform(0.25, 0.58)),
                "memory_tau":          float(rng.uniform(2*3600, 24*3600)),
                "stress_sensitivity":  float(rng.uniform(0.8, 2.2)),
                "beta_sw":             float(rng.uniform(12.0, 24.0)),
                "conn_factor":         float(rng.uniform(0.55, 1.45)),
                "roughness":           float(rng.uniform(1.0, 2.5)),
                "Re_crit":             float(rng.uniform(1200.0, 2400.0)),
                "seismic_coupling":    float(rng.uniform(0.5, 2.0)),
                "aperture_history":    [],
            })

    def update_fracture_aperture(self, theta, Pg, t, seismic_mult=None, dt=None):
        phi      = self.p["phi_m"]
        Sw_mean  = float(np.mean(np.clip(np.asarray(theta)/max(phi,1e-12), 1e-12, 0.9999)))
        Pg_mean  = float(np.mean(np.asarray(Pg)))
        for crack in self.cracks:
            sat_f    = 1.0 / (1.0 + np.exp(18.0*(Sw_mean - crack["Sw_crit"])))
            dP       = abs(Pg_mean - self.p.get("P_atm0", 101325.0))
            press_f  = 1.0 + crack["stress_sensitivity"] * (dP / 5000.0)
            b_target = crack["aperture_0"] * sat_f * press_f
            if seismic_mult is not None:
                s_mult = float(np.mean(seismic_mult))
                b_target = b_target * (1.0 + crack["seismic_coupling"] * (s_mult - 1.0))
            tau   = max(crack["memory_tau"], 1e-12)
            relax = 1.0 if dt is None else (1.0 - np.exp(-max(float(dt), 0.0) / tau))
            crack["current_aperture"] = (1.0-relax)*crack["current_aperture"] + relax*b_target
            crack["aperture"] = float(np.clip(crack["current_aperture"], 1e-6, 5e-3))
            crack["aperture_history"].append((t, crack["aperture"]))

    def _crack_connectivity_scalar(self, Sw_scalar, crack):
        beta    = crack.get("beta_sw",    20.0)
        Swc     = crack.get("Sw_crit",    0.45)
        sat_f   = 1.0 / (1.0 + np.exp(beta*(Sw_scalar - Swc)))
        dep_f   = np.exp(-crack.get("depth", 0.0) / 0.4)
        ang_f   = np.clip(np.abs(np.cos(crack.get("angle", 0.0))), 0.1, 1.0)
        rou_f   = 1.0 / max(crack.get("roughness", 1.6), 1e-12)
        return crack.get("conn_factor", 1.0) * sat_f * dep_f * ang_f * rou_f

    def _at_depth(self, arr, z_target):
        a = np.asarray(arr, dtype=float)
        z = self.z
        zt = float(np.clip(z_target, 0.0, self.H))
        if a.size == 1:
            return float(a[0])
        return float(np.interp(zt, z, a))

    def _interface_depths(self):
        z_wall = float(self.p.get("z_wall_m", 0.10 * self.H))
        z_slab = float(self.p.get("z_slab_m", 0.90 * self.H))
        return (float(np.clip(z_wall, 0.0, self.H)),
                float(np.clip(z_slab, 0.0, self.H)))

    def single_fracture_flux(self, Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin, crack):
        phi   = self.p["phi_m"]
        Sw    = float(np.clip(np.asarray(theta0)/max(phi,1e-12), 1e-12, 0.9999))
        conn  = self._crack_connectivity_scalar(Sw, crack)
        Rg    = self.p.get("R_g", 8.314)
        Mg    = self.p.get("M_g", 0.02897)
        rho   = max(Pg0,1.0)*Mg/(Rg*Tsoil)
        mu    = self.p.get("mu", 1.8e-5)
        dP_wd = 0.5*self.p.get("rho_air",1.2)*self.p.get("C_wind",0.6)*self.p.get("U_wind",0.5)**2
        dP    = (Pg0 - Pin) + dP_wd
        a     = crack["aperture"]
        L     = crack["length"] * crack["tortuosity"]
        w     = crack["length"]
        A_c   = a * crack["length"]
        Dh    = 2.0 * a
        Cd    = self.p.get("Cd_crack", 0.61)
        if self.p.get("cubic_law_width_correct", True):
            Q_lam = (w * a**3 / (12.0*mu)) * dP / max(L, 1e-12)
            Q_ori = Cd * A_c * np.sign(dP) * np.sqrt(2.0*np.abs(dP)/max(rho,1e-12))
            u_lam = Q_lam / max(w*a, 1e-30)
            Re    = rho * np.abs(u_lam) * Dh / max(mu, 1e-12)
            _Rc   = max(float(crack.get("Re_crit", 1800.0)), 1e-12)
            _x4   = (Re / _Rc)**4
            blend = float(np.clip(_x4 / (1.0 + _x4), 0.0, 1.0))
            q     = (1.0-blend)*Q_lam + blend*Q_ori
        else:
            q_lam = (a**3 / max(L,1e-12)) * dP / (12.0*mu)
            q_ori = Cd * A_c * np.sign(dP) * np.sqrt(2.0*np.abs(dP)/max(rho,1e-12))
            Re    = rho * np.abs(q_lam) * Dh / max(mu, 1e-12)
            blend = np.clip(Re / crack.get("Re_crit", 1800.0), 0.0, 1.0)
            q     = (1.0-blend)*q_lam + blend*q_ori
        C_up  = Cf0 if q >= 0.0 else Cin
        J_adv = q * C_up
        Sg    = 1.0 - Sw
        D_e   = self.p.get("D_entry", 1e-6)
        nD_e  = self.p.get("nD_entry", 2.4)
        J_diff= -D_e * (Sg**nD_e) * A_c * (Cin-Cf0) / max(L,1e-12)
        return conn * (J_adv + J_diff)

    def crack_network_flux(self, Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin):
        total = 0.0
        for crack in self.cracks:
            total += self.single_fracture_flux(Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin, crack)
        return total

    def _build_horizontal_fracture_layer(self, n_frac=20, seed=7):
        rng = np.random.default_rng(seed)
        self.horizontal_network = []
        for _ in range(n_frac):
            self.horizontal_network.append({
                "k":       float(rng.lognormal(np.log(1e-12), 1.0)),
                "width":   float(rng.uniform(0.02, 0.25)),
                "length":  float(rng.uniform(0.5, 5.0)),
                "coupling":float(rng.uniform(0.2, 1.0)),
            })

    def foundation_2d_network_flux(self, Cf0, Cin, Pg0, Pin, theta0):
        phi    = self.p["phi_m"]
        Sw     = float(np.clip(np.mean(np.asarray(theta0))/max(phi,1e-12), 1e-12, 0.9999))
        Sw_cl  = self.p.get("Sw_close", 0.85)
        closure= np.clip(1.0 - _smooth_step(Sw, Sw_cl, 0.03), 0.0, 1.0)
        dP     = Pg0 - Pin
        mu     = self.p.get("mu", 1.8e-5)
        total  = 0.0
        for frac in self.horizontal_network:
            k_h = frac["k"] * closure
            q   = (k_h / mu) * dP / max(frac["length"], 1e-12)
            C_up = Cf0 if q >= 0.0 else Cin
            J   = q * C_up * frac["width"] * frac["coupling"]
            total += J
        return total * self.p.get("foundation_coupling", 0.35)

    def update_self_organizing_permeability(self, q_history, theta):
        phi  = self.p["phi_m"]
        Sw   = float(np.mean(np.clip(np.asarray(theta)/max(phi,1e-12), 1e-12, 0.9999)))
        q_cum= float(np.abs(np.mean(q_history[-50:])) if len(q_history) > 0 else 0.0)
        for crack in self.cracks:
            erosion  = 1.0 + 8e-4 * q_cum * (1.0-Sw)**2.5
            clogging = 1.0 - 1.2e-3 * Sw**3.5
            k_base   = crack.get("conductivity_base", crack["aperture"]**2 / 12.0)
            k_new    = np.clip(k_base * erosion * clogging, 1e-15, 5e-9)
            crack["conductivity_base"] = float(k_new)


    def _gas_density(self, P, T):
        return np.maximum(P, 1.0) * self.p.get("M_g", 0.02897) / (self.p.get("R_g", 8.314) * np.maximum(T, 1.0))

    def dPindt_rhs(self, Pin, Patm, Psoil0, Tin, Tout, t):
        V      = self.p.get("V", 170.0)
        Rg     = self.p.get("R_g", 8.314)
        Mg     = self.p.get("M_g", 0.02897)
        g      = self.p.get("g", 9.81)
        H_stk  = self.p.get("H_stack", 2.5)
        
        rho_in  = self._gas_density(float(Pin),   float(Tin))
        rho_out = self._gas_density(float(Patm),  float(Tout))
        rho_s   = self._gas_density(float(Psoil0),float(Tin))
        
        dP_stack = -0.5 * g * H_stk * (rho_out - rho_in)
        
        ach    = self._air_exchange_rate(t)
        q_vent = ach * V
        dP_vent = 0.0
        
        Pin_target = Patm + np.clip(dP_stack + dP_vent, -30.0, 5.0)
        
        Cd    = self.p.get("Cd_leak", 0.62)
        A_lk  = self.p.get("A_leak", 5e-3)
        A_tot = A_lk
        
        dP_ref = max(abs(Pin - Pin_target), 0.5)
        tau_Pin = (V * np.sqrt(rho_in * dP_ref / 2.0)
                   / max(float(Patm) * Cd * A_tot, 1e-12))
        tau_Pin = float(np.clip(tau_Pin, 10.0, 7200.0))
        
        return (Pin_target - float(Pin)) / tau_Pin

    def _air_exchange_rate(self, t):
        ach_cl = self.p.get("ACH_closed", 0.10)
        ach_op = self.p.get("ACH_open",   1.20)
        t1, t2 = self.p.get("t_open1s", 7*3600), self.p.get("t_open1e", 9*3600)
        t3, t4 = self.p.get("t_open2s",19*3600), self.p.get("t_open2e",22*3600)
        tday   = t % 86400.0
        sh     = self.p.get("ACH_sharp", 300.0)
        m1     = 0.5*(1+np.tanh((tday-t1)/sh))*0.5*(1-np.tanh((tday-t2)/sh))
        m2     = 0.5*(1+np.tanh((tday-t3)/sh))*0.5*(1-np.tanh((tday-t4)/sh))
        opn    = np.clip(m1+m2, 0.0, 1.0)
        return (ach_cl + (ach_op-ach_cl)*opn + self.p.get("ACH_mech",0.0)) / 3600.0

    def indoor_multizone_rhs(self, J_entry_bs, C_bs, C_gt, t):
        V_bs    = self.p.get("V_bs",   50.0)
        V_gt    = self.p.get("V_gt",  120.0)
        lam_bg  = self.p.get("lambda_bg", 0.2/3600.0)
        lam_v   = self._air_exchange_rate(t)
        lr      = self.lam_rn
        dep_bs  = self.p.get("dep_bs", 0.0)
        dep_gt  = self.p.get("dep_gt", 0.0)
        Q_bg    = lam_bg * V_bs
        dC_bs   = J_entry_bs/max(V_bs,1e-12) + lam_bg*(C_gt-C_bs) - (lr+lam_v+dep_bs)*C_bs
        dC_gt   = (Q_bg/max(V_gt,1e-12))*(C_bs-C_gt) - (lr+lam_v+dep_gt)*C_gt
        return dC_bs, dC_gt

    def progeny_chain_rhs(self, C_Rn, C_Po218, C_Pb214, C_Bi214, C_Po214, t):
        lv    = self._air_exchange_rate(t)
        dep   = self.p.get("dep_progeny", 1.5/3600.0)
        l1    = self.lam_po218
        l2    = self.lam_pb214
        l3    = self.lam_bi214
        l4    = self.lam_po214
        dPo218= l1*C_Rn    - (l1+lv+dep)*C_Po218
        dPb214= l2*C_Po218 - (l2+lv+dep)*C_Pb214
        dBi214= l3*C_Pb214 - (l3+lv+dep)*C_Bi214
        dPo214= l4*C_Bi214 - (l4+lv+dep)*C_Po214
        return dPo218, dPb214, dBi214, dPo214

    def equilibrium_factor(self, C_Rn, C_Po218, C_Pb214, C_Bi214, C_Po214):
        num   = 0.105*C_Po218 + 0.516*C_Pb214 + 0.379*C_Bi214
        denom = max(C_Rn, 1e-30)
        return float(np.clip(num/denom, 0.0, 1.0))


    def mass_balance_check(self, Cm, Cf, C_bs, C_gt, J_entry, S_m, lambda_v, t,
                           theta=None, Cs=None, dInv_dt=0.0, Pg=None):
        phi_m    = self.p["phi_m"]
        phi_f    = self.p["phi_f"]
        V_bs     = self.p.get("V_bs",  50.0)
        V_gt     = self.p.get("V_gt", 120.0)
        if self.p.get("massbal_legacy", False):
            soil_inv = (phi_m*np.mean(Cm) + phi_f*np.mean(Cf)) * self.H
            in_inv   = (V_bs*C_bs + V_gt*C_gt)
            prod     = np.mean(S_m) * self.H
            loss_d   = self.lam_rn * soil_inv
            loss_i   = (self.lam_rn + lambda_v) * (V_bs*C_bs + V_gt*C_gt) / (V_bs+V_gt)
            err      = (prod - loss_d - loss_i - J_entry) / max(abs(prod)+abs(loss_d)+1e-30, 1e-30)
            return float(err)
        A_fp     = self.p.get("A_footprint", self.p.get("V_bs", 50.0)/2.5)
        Cs_mean = float(np.mean(np.asarray(Cs))) if Cs is not None else 0.0
        if theta is not None:
            _T_mb   = self.T_field(t)
            _beta_m_prof = np.asarray(self._beta_partition(np.asarray(theta), _T_mb, "m"))
            _th_f_mb = np.clip(self.p.get("gamma_sw_f", 0.2)
                               * (np.asarray(theta)/max(phi_m, 1e-12)) * phi_f,
                               1e-12, 0.9999*phi_f)
            _beta_f_prof = np.asarray(self._beta_partition(_th_f_mb, _T_mb, "f"))
            _Cs_arr  = np.asarray(Cs) if Cs is not None else 0.0
            soil_inv = float(np.sum(_beta_m_prof*np.asarray(Cm)
                                    + _beta_f_prof*np.asarray(Cf)
                                    + _Cs_arr)) * self.dz * A_fp
            prod = float(np.sum(_beta_m_prof * np.asarray(S_m))) * self.dz * A_fp
        else:
            soil_inv = (phi_m*np.mean(Cm) + phi_f*np.mean(Cf) + Cs_mean) * self.H * A_fp
            prod     = np.mean(S_m) * self.H * A_fp
        loss_d   = self.lam_rn * soil_inv
        loss_i   = (self.lam_rn + lambda_v) * (V_bs*C_bs + V_gt*C_gt)
        if theta is not None:
            Dm_s = np.atleast_1d(self._D_eff(np.asarray(theta), "m"))
            theta_f_s = np.clip(self.p.get("gamma_sw_f", 0.2)
                                * (np.asarray(theta) / max(self.p["phi_m"], 1e-12))
                                * self.p["phi_f"], 1e-12, 0.9999*self.p["phi_f"])
            Df_s = np.atleast_1d(self._D_eff(theta_f_s, "f"))
            J_exh = (2.0*float(Dm_s[0])*float(np.atleast_1d(Cm)[0]) / self.dz
                     + 2.0*float(Df_s[0])*float(np.atleast_1d(Cf)[0]) / self.dz) * A_fp
        else:
            J_exh = 0.0
        J_adv = 0.0
        if theta is not None and Pg is not None:
            _mu_mb  = self.p.get("mu", 1.8e-5)
            _seis   = self.seismic_permeability_multiplier(t)
            _k_m_mb = np.asarray(self._k_gas(np.asarray(theta), self.z, _seis))
            _k_f_mb = self.p.get("k_frac", 1e-11) * np.ones(_k_m_mb.size)
            _Ptop_mb, _Pbot_mb = self._Pg_bc(t)
            _g_mb   = _face_gradient(np.asarray(Pg, dtype=float), self.dz,
                                     P_top=_Ptop_mb, P_bot=_Pbot_mb)
            _km_e   = np.concatenate(([_k_m_mb[0]], _k_m_mb, [_k_m_mb[-1]]))
            _kf_e   = np.concatenate(([_k_f_mb[0]], _k_f_mb, [_k_f_mb[-1]]))
            _km_fc  = 2.0*_km_e[:-1]*_km_e[1:]/np.maximum(_km_e[:-1]+_km_e[1:], 1e-300)
            _kf_fc  = 2.0*_kf_e[:-1]*_kf_e[1:]/np.maximum(_kf_e[:-1]+_kf_e[1:], 1e-300)
            _q_m    = -(_km_fc/_mu_mb)*_g_mb
            _q_f    = -(_kf_fc/_mu_mb)*_g_mb
            _Cm_a   = np.asarray(Cm, dtype=float)
            _Cf_a   = np.asarray(Cf, dtype=float)
            _cmb, _cfb = self._deep_equilibrium_conc(
                np.asarray(S_m)[-1], np.asarray(theta)[-1], np.asarray(Pg)[-1],
                float(np.atleast_1d(_T_mb)[-1]), self.lam_rn)
            _Cm_bot = float(np.atleast_1d(_cmb)[0])
            _Cf_bot = float(np.atleast_1d(_cfb)[0])
            if _q_m[0] < 0.0:
                J_adv += (-_q_m[0]) * float(_Cm_a[0]) * A_fp
            if _q_f[0] < 0.0:
                J_adv += (-_q_f[0]) * float(_Cf_a[0]) * A_fp
            if _q_m[-1] < 0.0:
                J_adv -= (-_q_m[-1]) * _Cm_bot * A_fp
            else:
                J_adv += _q_m[-1] * float(_Cm_a[-1]) * A_fp
            if _q_f[-1] < 0.0:
                J_adv -= (-_q_f[-1]) * _Cf_bot * A_fp
            else:
                J_adv += _q_f[-1] * float(_Cf_a[-1]) * A_fp
        err      = (prod - loss_d - loss_i - J_exh - J_adv - float(dInv_dt)) / \
                   max(abs(prod)+abs(loss_d)+abs(loss_i)+abs(J_exh)+1e-30, 1e-30)
        return float(err)

    def identifiability_fim(self, J_sensitivity):
        FIM    = J_sensitivity.T @ J_sensitivity
        evals  = np.linalg.eigvalsh(FIM)
        cond   = np.inf if np.min(np.abs(evals)) < 1e-30 else np.max(np.abs(evals))/np.min(np.abs(evals))
        rank   = np.linalg.matrix_rank(FIM)
        return {"FIM": FIM, "eigenvalues": evals, "condition_number": cond, "rank": rank}

    def buckingham_pi(self, theta, Pg, T):
        phi   = self.p["phi_m"]
        theta = np.asarray(theta, dtype=float)
        Pg    = np.asarray(Pg,    dtype=float)
        Sw    = np.clip(theta/max(phi,1e-12), 1e-12, 0.9999)
        D_eff = float(np.mean(np.maximum(self._D_eff(theta, "m"), 1e-12)))
        k     = self._k_gas(theta, self.z)
        mu    = self.p.get("mu", 1.8e-5)
        dPdz  = np.gradient(Pg, self.dz) if len(Pg) > 1 else np.zeros(len(Pg))
        u_raw = np.abs(k / mu * dPdz)
        u_eff = float(np.clip(np.mean(u_raw), 1e-15, self.p.get("vmax_gas", 5e-7)))
        H     = self.H
        phi_g_mean = float(np.mean(np.maximum(self.p["phi_m"] - np.asarray(theta), 1e-12)))
        u_pore = u_eff / max(phi_g_mean, 1e-12)
        D_pore = D_eff / max(phi_g_mean, 1e-12)
        Pe_H  = u_eff * H / max(D_eff, 1e-30)
        Da_H  = self.lam_rn * H / max(u_pore, 1e-30)
        ach   = self.p.get("ACH_closed", 0.1) / 3600.0
        Da_v  = ach * H / max(u_eff, 1e-30)
        eps_m = float(np.mean(self._emanation_factor_inst(theta, T)))
        rho_b = self.p.get("rho_bulk", 1600.0)
        Ra226 = self.p.get("Ra226_Bqkg", 35.0)
        C_ref = self.p.get("C_ref", 100.0)
        S_m   = self.lam_rn * rho_b * Ra226 * eps_m
        Pi_src= S_m * H / max(u_eff * C_ref, 1e-30)
        Pi_ven= ach / max(self.lam_rn, 1e-30)
        k_f   = self.p.get("k_frac", 1e-12)
        k_m   = float(np.mean(k))
        Pi_frac= k_f / max(k_m, 1e-30)
        Pi_iso = (self.lam_rn220 / max(self.lam_rn, 1e-30)) * 1.0
        Da_adv  = self.lam_rn * H / max(u_pore,  1e-30)
        Da_diff = self.lam_rn * H**2 / max(D_pore, 1e-30)
        return {
            "Pe_H":    Pe_H,
            "Da_H":    Da_adv,
            "DaAdv":   Da_adv,
            "DaDiff":  Da_diff,
            "Da_v":    Da_v,
            "Pi_src":  Pi_src,
            "Pi_vent": Pi_ven,
            "Pi_frac": Pi_frac,
            "Pi_iso":  Pi_iso,
            "u_eff":   u_eff,
            "D_eff":   D_eff,
        }


    def rhs(self, t, y):
        nz   = self.nz
        theta   = np.clip(y[0       :  nz],     1e-8,   0.95*self.p["phi_m"])
        Pg      = np.clip(y[  nz    :2*nz],     5e4,    2e5)
        Cm      = np.maximum(y[2*nz :3*nz],     0.0)
        Cf      = np.maximum(y[3*nz :4*nz],     0.0)
        M       =            y[4*nz :5*nz]
        Cs      = np.maximum(y[5*nz :6*nz],     0.0)
        Ar      = np.clip(   y[6*nz :7*nz],     0.0,    1.0)
        Cm220   = np.maximum(y[7*nz :8*nz],     0.0)
        Cf220   = np.maximum(y[8*nz :9*nz],     0.0)
        eps_mem = np.clip(   y[9*nz :10*nz],    0.01,   0.80)
        C_bs    = np.clip(float(y[10*nz]),   0.0, 1e7)
        C_gt    = np.clip(float(y[10*nz+1]), 0.0, 1e7)
        Pin     = np.clip(float(y[10*nz+2]), 5e4, 2e5)
        C_Po218 = max(float(y[10*nz+3]),         0.0)
        C_Pb214 = max(float(y[10*nz+4]),         0.0)
        C_Bi214 = max(float(y[10*nz+5]),         0.0)
        C_Po214 = max(float(y[10*nz+6]),         0.0)


        dtheta_dt  = self.richards_rhs(y[0:nz], t)
        T          = self.T_field(t)
        seis_mult  = self.seismic_permeability_multiplier(t)
        if self.p.get("seismic_recovery_on", False):
            seis_mult = seis_mult * self.seismic_permeability_multiplier_recovery(t)
            seis_mult = np.clip(seis_mult, 1.0, self.p.get("k_seis_max", 50.0))
        self.update_fracture_aperture(theta, Pg, t, seis_mult)
        dPg_dt     = self.gas_pressure_rhs(theta, Pg, t, seis_mult)


        dAr_dt     = self.radon_accessibility_rhs(Ar, theta, dtheta_dt, Pg)
        deps_mem   = self.emanation_hysteresis_rhs(eps_mem, theta, dtheta_dt, T)
        S_222      = self.source_rn222(theta, T, Ar, eps_mem)
        S_220      = self.source_rn220(theta, T, Ar, eps_mem)


        _Sw_f      = np.clip(self.p.get("gamma_sw_f", 0.2)
                             * (theta / max(self.p["phi_m"], 1e-12)), 1e-6, 0.9999)
        theta_f    = np.clip(_Sw_f * self.p["phi_f"], 1e-12, 0.9999*self.p["phi_f"])
        dM_dt      = self.exchange_memory_state_rhs(M, Cm, Cf)
        J_mf       = self.matrix_fracture_flux(Cm, Cf, M, theta, theta_f, Pg, Pg)
        dCs_dt     = self.adsorption_dynamic_rhs(Cs, Cm, theta)
        Sw_ads    = np.clip(theta / max(self.p["phi_m"], 1e-12), 1e-12, 0.9999)
        f_ads_gas = 1.0 / (1.0 + np.exp(-10.0*(Sw_ads - 0.25)))
        k_ads_gas = self.p.get("k_ads_dry", 2e-6) + (self.p.get("k_ads_wet", 5e-7) - self.p.get("k_ads_dry", 2e-6)) * f_ads_gas
        k_des_gas = self.p.get("k_des_dry", 2e-7) + (self.p.get("k_des_wet", 8e-7) - self.p.get("k_des_dry", 2e-7)) * f_ads_gas
        J_ads_Cm  = -(k_ads_gas*Cm - k_des_gas*Cs)

        Dm         = self._D_eff(theta,   "m")
        Df         = self._D_eff(theta_f, "f")
        k_m        = self._k_gas(theta,   self.z, seis_mult)
        k_f        = self.p.get("k_frac", 1e-11) * np.ones(nz)
        mu         = self.p.get("mu", 1.8e-5)
        if self.p.get("face_consistent_gradient", True):
            _Ptop_bc, _Pbot_bc = self._Pg_bc(t)
            gPdz_f     = _face_gradient(Pg, self.dz,
                                        P_top=_Ptop_bc, P_bot=_Pbot_bc)
            km_ext     = np.concatenate(([k_m[0]], k_m, [k_m[-1]]))
            kf_ext     = np.concatenate(([k_f[0]], k_f, [k_f[-1]]))
            km_face    = 2.0*km_ext[:-1]*km_ext[1:]/np.maximum(km_ext[:-1]+km_ext[1:], 1e-300)
            kf_face    = 2.0*kf_ext[:-1]*kf_ext[1:]/np.maximum(kf_ext[:-1]+kf_ext[1:], 1e-300)
            q_m_face   = -(km_face / mu) * gPdz_f
            q_f_face   = -(kf_face / mu) * gPdz_f
            u_m        = -(k_m / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))
            u_f        = -(k_f / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))
        else:
            dPdz       = np.gradient(Pg, self.dz) if len(Pg) > 1 else np.zeros(len(Pg))
            u_m        = -(k_m / mu) * dPdz
            u_f        = -(k_f / mu) * dPdz
            q_m_face   = np.empty(nz+1); q_m_face[0]=u_m[0]; q_m_face[-1]=u_m[-1]; q_m_face[1:-1]=0.5*(u_m[:-1]+u_m[1:])
            q_f_face   = np.empty(nz+1); q_f_face[0]=u_f[0]; q_f_face[-1]=u_f[-1]; q_f_face[1:-1]=0.5*(u_f[:-1]+u_f[1:])
        _cm_eq_b, _cf_eq_b = self._deep_equilibrium_conc(
            S_222[-1], theta[-1], Pg[-1], float(np.atleast_1d(T)[-1]), self.lam_rn)
        Cm_bot_eq  = float(np.atleast_1d(_cm_eq_b)[0])
        Cf_bot_eq  = float(np.atleast_1d(_cf_eq_b)[0])
        F_adv_m    = _upwind(Cm, q_m_face, C_top=0.0, C_bot_inlet=Cm_bot_eq)
        F_adv_f    = _upwind(Cf, q_f_face, C_top=0.0, C_bot_inlet=Cf_bot_eq)
        adv_m      = -(F_adv_m[1:] - F_adv_m[:-1]) / self.dz
        adv_f      = -(F_adv_f[1:] - F_adv_f[:-1]) / self.dz
        diff_m     = _div_diff_flux(Cm, Dm, self.dz, C_top=0.0)
        diff_f     = _div_diff_flux(Cf, Df, self.dz, C_top=0.0)
        phi_g_m    = np.maximum(self.p["phi_m"] - theta, 1e-12)
        phi_f_val  = self.p["phi_f"]
        beta_m     = self._beta_partition(theta,   T, "m")
        beta_f     = self._beta_partition(theta_f, T, "f")
        _f_src_f   = float(self.p.get("f_src_to_fracture", 0.0))
        _S222_m    = (1.0 - _f_src_f) * S_222
        _S222_f    = _f_src_f * S_222 * beta_m / beta_f
        _S220_m    = (1.0 - _f_src_f) * S_220
        _S220_f    = _f_src_f * S_220 * beta_m / beta_f
        dCm_dt     = (diff_m + adv_m)/beta_m + _S222_m - self.lam_rn*Cm - J_mf/beta_m + J_ads_Cm/beta_m
        dCf_dt     = (diff_f + adv_f)/beta_f + _S222_f - self.lam_rn*Cf + J_mf/beta_f


        _cm20_eq_b, _cf20_eq_b = self._deep_equilibrium_conc(
            S_220[-1], theta[-1], Pg[-1], float(np.atleast_1d(T)[-1]),
            self.lam_rn220, include_exchange=False)
        Cm220_bot  = float(np.atleast_1d(_cm20_eq_b)[0])
        Cf220_bot  = float(np.atleast_1d(_cf20_eq_b)[0])
        F_adv_m220 = _upwind(Cm220, q_m_face, C_top=0.0, C_bot_inlet=Cm220_bot)
        F_adv_f220 = _upwind(Cf220, q_f_face, C_top=0.0, C_bot_inlet=Cf220_bot)
        adv_m220   = -(F_adv_m220[1:] - F_adv_m220[:-1]) / self.dz
        adv_f220   = -(F_adv_f220[1:] - F_adv_f220[:-1]) / self.dz
        diff_m220  = _div_diff_flux(Cm220, Dm, self.dz, C_top=0.0)
        diff_f220  = _div_diff_flux(Cf220, Df, self.dz, C_top=0.0)
        dCm220_dt  = (diff_m220 + adv_m220)/beta_m + _S220_m - self.lam_rn220*Cm220
        dCf220_dt  = (diff_f220 + adv_f220)/beta_f + _S220_f - self.lam_rn220*Cf220


        z_wall, z_slab = self._interface_depths()
        Cf0        = self._at_depth(Cf, z_wall)
        Cf_bot     = self._at_depth(Cf, z_slab)
        Pg0        = self._at_depth(Pg, z_wall)
        Pg_bot     = self._at_depth(Pg, z_slab)
        Pg_eff     = float(np.sqrt(Pg0 * Pg_bot)) if Pg_bot > Pg0 else Pg0
        Cf_eff     = float(0.3*Cf0 + 0.7*Cf_bot)
        T0_soil    = self._at_depth(T, z_slab)
        Tin        = self.T_indoor(t)
        Tout       = self.T_outdoor(t)
        theta0     = self._at_depth(theta, z_slab)
        J_cracks   = self.crack_network_flux(Cf_eff, C_bs, Pg_eff, Pin, theta0, T0_soil, Tin)
        J_found    = self.foundation_2d_network_flux(Cf_eff, C_bs, Pg_eff, Pin, theta0)
        J_entry    = J_cracks + J_found

        Cf220_0    = self._at_depth(Cf220, z_wall)
        Cf220_bot  = self._at_depth(Cf220, z_slab)
        Cf220_eff  = 0.3*Cf220_0 + 0.7*Cf220_bot
        J_entry220 = self.crack_network_flux(Cf220_eff, 0.0, Pg_eff, Pin, theta0, T0_soil, Tin) * 0.01

        if self.p.get("entry_mass_coupling_on", True):
            A_fp     = self.p.get("A_footprint", self.p.get("V_bs", 50.0) / 2.5)
            L_infl = float(self.p.get("L_entry_influence", 0.15 * self.H))
            L_infl = max(L_infl, self.dz)
            w_top_prof = np.exp(-0.5 * ((self.z - z_wall) / L_infl) ** 2)
            w_bot_prof = np.exp(-0.5 * ((self.z - z_slab) / L_infl) ** 2)
            w_top_prof /= max(float(np.sum(w_top_prof)), 1e-30)
            w_bot_prof /= max(float(np.sum(w_bot_prof)), 1e-30)
            w_prof = 0.30 * w_top_prof + 0.70 * w_bot_prof
            vol_cell = np.maximum(A_fp * self.dz * beta_f, 1e-30)
            C_soft    = max(1e-4 * Cf_bot_eq, 1e-6)
            Cf220_bot_eq = Cf220_bot
            C_soft220 = max(1e-4 * Cf220_bot_eq, 1e-6)
            if J_entry > 0.0:
                s_avail    = Cf / (Cf + C_soft)
                sink_cf    = w_prof * s_avail * (J_entry / vol_cell)
                J_entry    = J_entry * float(np.sum(w_prof * s_avail))
            else:
                sink_cf    = w_prof * (J_entry    / vol_cell)
            if J_entry220 > 0.0:
                s_avail220 = Cf220 / (Cf220 + C_soft220)
                sink_cf220 = w_prof * s_avail220 * (J_entry220 / vol_cell)
                J_entry220 = J_entry220 * float(np.sum(w_prof * s_avail220))
            else:
                sink_cf220 = w_prof * (J_entry220 / vol_cell)
        else:
            sink_cf    = np.zeros(nz)
            sink_cf220 = np.zeros(nz)
        dCf_dt    = dCf_dt    - sink_cf
        dCf220_dt = dCf220_dt - sink_cf220


        Patm       = self._Patm(t)
        dPin_dt    = self.dPindt_rhs(Pin, Patm, Pg_bot, Tin, Tout, t)
        _Jfroz = getattr(self, "_J_entry_frozen", None)
        J_entry_indoor = _Jfroz if (_Jfroz is not None and J_entry > 0.0) else J_entry
        dC_bs, dC_gt = self.indoor_multizone_rhs(J_entry_indoor, C_bs, C_gt, t)
        dPo218, dPb214, dBi214, dPo214 = self.progeny_chain_rhs(C_bs, C_Po218, C_Pb214, C_Bi214, C_Po214, t)


        dydt = np.empty(self.nstate)
        dydt[0       :  nz] = dtheta_dt
        dydt[  nz    :2*nz] = dPg_dt
        dydt[2*nz    :3*nz] = dCm_dt
        dydt[3*nz    :4*nz] = dCf_dt
        dydt[4*nz    :5*nz] = dM_dt
        dydt[5*nz    :6*nz] = dCs_dt
        dydt[6*nz    :7*nz] = dAr_dt
        dydt[7*nz    :8*nz] = dCm220_dt
        dydt[8*nz    :9*nz] = dCf220_dt
        dydt[9*nz    :10*nz]= deps_mem
        dydt[10*nz  ]       = dC_bs
        dydt[10*nz+1]       = dC_gt
        dydt[10*nz+2]       = dPin_dt
        dydt[10*nz+3]       = dPo218
        dydt[10*nz+4]       = dPb214
        dydt[10*nz+5]       = dBi214
        dydt[10*nz+6]       = dPo214
        return np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)


    def build_y0(self):
        nz   = self.nz
        p    = self.p
        y0   = np.zeros(self.nstate)
        phi  = p["phi_m"]
        theta0_arr = np.full(nz, p.get("theta0", 0.35 * phi))
        y0[0:nz]      = theta0_arr
        Patm0 = p.get("P_atm0", 101325.0)
        dP_bot = p.get("dP_bottom", 5.0)
        Pbot0  = Patm0 + dP_bot
        y0[nz:2*nz] = np.linspace(Patm0, Pbot0, nz)
        T0_arr = p.get("T0", 288.0) + p.get("T_grad", 0.025) * self.z
        _Sw0_ic = np.clip(p.get("theta0", 0.35*phi) / max(phi, 1e-12), 0.0, 0.9999)
        _Sg0_ic = 1.0 - _Sw0_ic
        _kop_ic = p.get("A_k_open",  1.0/3600.0)
        _kcl_ic = p.get("A_k_close", 0.35/3600.0)
        Ar0_arr = np.full(nz, _kop_ic * _Sg0_ic**2
                          / max(_kop_ic*_Sg0_ic**2 + _kcl_ic*_Sw0_ic**2, 1e-12))
        eps0_arr = self._emanation_factor_inst(
            np.full(nz, p.get("theta0", 0.35*phi)), T0_arr)
        S222_0 = self.source_rn222(theta0_arr, T0_arr, Ar0_arr, eps0_arr)
        _Sw0_ads = np.clip(theta0_arr / max(phi, 1e-12), 1e-12, 0.9999)
        _f0_ads  = 1.0 / (1.0 + np.exp(-10.0*(_Sw0_ads - 0.25)))
        _kads0  = (p.get("k_ads_dry", 2e-6)
                   + (p.get("k_ads_wet", 5e-7) - p.get("k_ads_dry", 2e-6)) * _f0_ads)
        _kdes0  = (p.get("k_des_dry", 2e-7)
                   + (p.get("k_des_wet", 8e-7) - p.get("k_des_dry", 2e-7)) * _f0_ads)
        _beta0  = float(np.mean(self._beta_partition(theta0_arr, T0_arr, "m")))
        Kd_eff0 = _kads0 / np.maximum(_kdes0 + self.lam_rn, 1e-30)
        Cm_eq0 = (S222_0 / max(self.lam_rn, 1e-30)) / (1.0 + Kd_eff0 / max(_beta0, 1e-12))
        C0m    = p.get("C0_m", None)
        C0f    = p.get("C0_f", None)
        if C0m is None: C0m = Cm_eq0
        if C0f is None:
            _cm0, _cf0 = self._deep_equilibrium_conc(
                S222_0, theta0_arr, np.full(nz, p.get("P_atm0", 101325.0)),
                T0_arr, self.lam_rn)
            _ratio0 = np.asarray(_cf0) / np.maximum(np.asarray(_cm0), 1e-30)
            C0f = Cm_eq0 * _ratio0
        y0[2*nz:3*nz] = C0m
        y0[3*nz:4*nz] = C0f
        y0[4*nz:5*nz] = C0m - C0f
        y0[5*nz:6*nz] = Kd_eff0 * C0m
        Ar0   = float(Ar0_arr[0])
        y0[6*nz:7*nz] = Ar0_arr
        y0[7*nz:8*nz] = p.get("C0_m220", 0.0)
        y0[8*nz:9*nz] = p.get("C0_f220", 0.0)
        T0_arr       = p.get("T0", 288.0) + p.get("T_grad", 0.025) * self.z
        y0[9*nz:10*nz]= eps0_arr
        y0[10*nz]    = p.get("Cin0",   50.0)
        y0[10*nz+1]  = p.get("Cin_gt0",20.0)
        y0[10*nz+2]  = p.get("P_in0", Patm0 - 1.5)
        C_bs0 = p.get("Cin0", 50.0)
        lv0   = p.get("ACH_closed", 0.10) / 3600.0
        dep   = p.get("dep_progeny", 1.5/3600.0)
        l1,l2,l3,l4 = self.lam_po218, self.lam_pb214, self.lam_bi214, self.lam_po214
        loss1 = l1 + lv0 + dep
        loss2 = l2 + lv0 + dep
        loss3 = l3 + lv0 + dep
        loss4 = l4 + lv0 + dep
        CPo218_eq = l1 * C_bs0 / max(loss1, 1e-12)
        CPb214_eq = l2 * CPo218_eq / max(loss2, 1e-12)
        CBi214_eq = l3 * CPb214_eq / max(loss3, 1e-12)
        CPo214_eq = l4 * CBi214_eq / max(loss4, 1e-12)
        y0[10*nz+3]  = CPo218_eq
        y0[10*nz+4]  = CPb214_eq
        y0[10*nz+5]  = CBi214_eq
        y0[10*nz+6]  = CPo214_eq
        return y0


    def simulate(self, days=3.0, max_step=1800, rtol=1e-4, atol=1e-7, method="BDF"):
        y0  = self.build_y0()
        self._J_entry_frozen = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = solve_ivp(
                self.rhs,
                (0.0, days*86400.0),
                y0,
                method   = method,
                rtol     = rtol,
                atol     = atol,
                max_step = max_step,
                dense_output = False,
            )
        return sol


    def _seis_mult_at(self, t):
        seis_mult = self.seismic_permeability_multiplier(t)
        if self.p.get("seismic_recovery_on", False):
            seis_mult = seis_mult * self.seismic_permeability_multiplier_recovery(t)
            seis_mult = np.clip(seis_mult, 1.0, self.p.get("k_seis_max", 50.0))
        return seis_mult

    def _cfl_dt(self, theta, Pg, t=None):
        theta = np.asarray(theta, dtype=float)
        Pg    = np.asarray(Pg,    dtype=float)
        D_eff  = float(np.mean(np.maximum(self._D_eff(theta, 'm'), 1e-30)))
        k_g    = self._k_gas(theta, self.z,
                             None if t is None else self._seis_mult_at(float(t)))
        mu     = self.p.get('mu', 1.8e-5)
        gradPg = np.abs(np.gradient(Pg, self.dz)) if len(Pg) > 1 else np.zeros(1)
        Se     = self._Se_drainage(theta)
        phi_g  = np.maximum(self.p['phi_m'] * (1.0 - Se), 1e-12)
        u_g    = k_g / mu * gradPg
        u_pore = float(np.max(np.abs(u_g / phi_g)))
        if self.p.get("cfl_include_fracture", True):
            k_f = k_g * (self.p.get("k_frac", 1e-11) /
                         max(float(np.mean(k_g)), 1e-30)) if False else \
                  k_g * (self.p.get("k_frac", 1e-11) / max(self.p.get("k0", 1e-13), 1e-30))
            phi_gf = max(self.p.get("phi_f", 0.08), 1e-12)
            u_pore_f = float(np.max(np.abs(k_f / mu * gradPg / phi_gf)))
            u_pore   = max(u_pore, u_pore_f)
        dt_diff = 0.5 * self.dz**2 / max(D_eff,  1e-30)
        dt_adv  = 0.4 * self.dz    / max(u_pore, 1e-30)
        lam_Po218 = DECAY_CONSTANTS.get('Po218', 3.77e-3)
        dt_dec  = 0.3  / max(lam_Po218, 1e-30)
        h_safe  = min(dt_diff, dt_adv, dt_dec)
        return float(np.clip(h_safe, 30.0, 1800.0))

    def _rhs_no_Cm(self, t, y):
        dydt = self.rhs(t, y)
        nz = self.nz
        dydt[2*nz:3*nz] = 0.0
        dydt[3*nz:4*nz] = 0.0
        dydt[7*nz:8*nz] = 0.0
        dydt[8*nz:9*nz] = 0.0
        dydt[5*nz:6*nz] = 0.0
        dydt[nz:2*nz] = 0.0
        dydt[10*nz+2] = 0.0
        dydt[10*nz+6] = 0.0
        return dydt

    def _advance_Cm_implicit(self, Cm, Cf, Cm220, Cf220, theta, Pg, Ar, eps_mem, t, h, M=None, Cs=None):
        nz    = self.nz
        phi_m = self.p["phi_m"]
        phi_f = self.p["phi_f"]
        lam   = self.lam_rn
        lam20 = self.lam_rn220
        T     = self.T_field(t + h/2)

        S222  = self.source_rn222(theta, T, Ar, eps_mem)
        S220  = self.source_rn220(theta, T, Ar, eps_mem)
        _f_src_f  = float(self.p.get("f_src_to_fracture", 0.0))
        _th_f_s1  = np.clip(self.p.get("gamma_sw_f", 0.2)
                            * (np.asarray(theta) / max(phi_m, 1e-12)) * phi_f,
                            1e-12, 0.9999*phi_f)
        _bm_s1    = self._beta_partition(theta,   T, "m")
        _bf_s1    = self._beta_partition(_th_f_s1, T, "f")
        Cm_eq   = (1.0 - _f_src_f) * S222 / np.maximum(lam, 1e-30)
        Cm20_eq = (1.0 - _f_src_f) * S220 / np.maximum(lam20, 1e-30)
        Cf_eq_src   = (_f_src_f * S222 * _bm_s1 / np.maximum(_bf_s1, 1e-30)
                       / np.maximum(lam, 1e-30))
        Cf20_eq_src = (_f_src_f * S220 * _bm_s1 / np.maximum(_bf_s1, 1e-30)
                       / np.maximum(lam20, 1e-30))
        Cm_bc,    Cf_eq   = self._deep_equilibrium_conc(S222, theta, Pg, T, lam)
        Cm20_bc,  Cf20_eq = self._deep_equilibrium_conc(S220, theta, Pg, T, lam20,
                                                        include_exchange=False)

        e_rn   = np.exp(-lam   * h)
        e_rn20 = np.exp(-lam20 * h)
        Cm_new   = Cm_eq   + (Cm   - Cm_eq)   * e_rn
        Cf_new   = Cf_eq_src   + (Cf    - Cf_eq_src)   * e_rn
        Cm20_new = Cm20_eq     + (Cm220 - Cm20_eq)     * e_rn20
        Cf20_new = Cf20_eq_src + (Cf220 - Cf20_eq_src) * e_rn20

        Dm   = self._D_eff(theta, "m")
        Df   = self._D_eff(np.clip(np.clip(self.p.get("gamma_sw_f", 0.2)
                                           * (np.asarray(theta) / max(phi_m, 1e-12)),
                                           1e-6, 0.9999)
                                   * phi_f, 1e-12, 0.9999*phi_f), "f")
        k_m  = self._k_gas(theta, self.z, self._seis_mult_at(t))
        mu   = self.p.get("mu", 1.8e-5)
        k_f_arr = self.p.get("k_frac", 1e-11) * np.ones(nz)
        if self.p.get("face_consistent_gradient", True):
            _Ptop_tr, _Pbot_tr = self._Pg_bc(t)
            gPdz_f  = _face_gradient(Pg, self.dz, P_top=_Ptop_tr, P_bot=_Pbot_tr)
            km_ext  = np.concatenate(([k_m[0]], k_m, [k_m[-1]]))
            kf_ext  = np.concatenate(([k_f_arr[0]], k_f_arr, [k_f_arr[-1]]))
            km_face = 2.0*km_ext[:-1]*km_ext[1:]/np.maximum(km_ext[:-1]+km_ext[1:], 1e-300)
            kf_face = 2.0*kf_ext[:-1]*kf_ext[1:]/np.maximum(kf_ext[:-1]+kf_ext[1:], 1e-300)
            q_m_face = -(km_face / mu) * gPdz_f
            q_f_face = -(kf_face / mu) * gPdz_f
            u_m      = -(k_m / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))
        else:
            dPdz = np.gradient(Pg, self.dz) if nz > 1 else np.zeros(nz)
            u_m  = -(k_m / mu) * dPdz
            u_f  = -(self.p.get("k_frac", 1e-11) / mu) * dPdz
            q_m_face = np.empty(nz+1)
            q_f_face = np.empty(nz+1)
            q_m_face[0]=u_m[0]; q_m_face[-1]=u_m[-1]
            q_f_face[0]=u_f[0]; q_f_face[-1]=u_f[-1]
            if nz > 1:
                q_m_face[1:-1] = 0.5*(u_m[:-1]+u_m[1:])
                q_f_face[1:-1] = 0.5*(u_f[:-1]+u_f[1:])

        D_eff_mean = float(np.mean(np.maximum(Dm, 1e-30)))
        dt_diff    = 0.5 * self.dz**2 / max(D_eff_mean, 1e-30)
        u_pore_max = float(np.max(np.abs(u_m / np.maximum(
            self.p["phi_m"]*(1.0-self._Se_drainage(theta)), 1e-12))))
        _u_f_cell  = 0.5*(q_f_face[:-1] + q_f_face[1:])
        u_pore_f   = float(np.max(np.abs(_u_f_cell / max(self.p["phi_f"], 1e-12))))
        u_pore_max = max(u_pore_max, u_pore_f)
        dt_adv     = 0.4 * self.dz / max(u_pore_max, 1e-30)

        _T_tr      = self.T_field(t)
        _theta_f_b = np.clip(np.asarray(theta) * self.p.get("gamma_sw_f", 0.2)
                             * (self.p["phi_f"] / max(self.p["phi_m"], 1e-12)),
                             1e-12, 0.9999*self.p["phi_f"])
        phi_gm_tr  = self._beta_partition(theta, _T_tr, "m")
        phi_f_tr   = self._beta_partition(_theta_f_b, _T_tr, "f")

        use_cn = self.p.get("cn_diffusion_on", True)
        if use_cn:
            _Dpore_max = max(
                float(np.max(np.asarray(Dm) / np.maximum(self.p["phi_m"] - theta, 1e-12))),
                float(np.max(np.asarray(Df) / np.maximum(phi_f_tr, 1e-12))))
            dt_diff_mono = 0.5 * self.dz**2 / max(_Dpore_max, 1e-30)
            h_tr = np.clip(min(dt_adv, dt_diff_mono), 1.0, h)
        else:
            h_tr = np.clip(min(dt_diff, dt_adv), 1.0, h)
        n_tr       = max(1, int(np.ceil(h / h_tr)))
        h_tr       = h / n_tr
        harm       = _DIFF_FACE_HARMONIC

        _phi_m_tr  = np.maximum(phi_gm_tr, 1e-12)
        _theta_f_tr = np.clip(self.p.get("gamma_sw_f", 0.2)
                              * (np.asarray(theta) / max(self.p["phi_m"], 1e-12))
                              * self.p["phi_f"], 1e-12, 0.9999*self.p["phi_f"])
        _alpha_mf  = np.atleast_1d(self._mf_alpha(theta, _theta_f_tr, Pg, Pg))
        if _alpha_mf.size == 1 and nz > 1:
            _alpha_mf = np.full(nz, float(_alpha_mf[0]))
        _eta_i     = self.p.get("eta_inst", 1.0)
        _eta_m     = self.p.get("eta_mem",  0.5)
        _M_tr      = np.zeros(nz) if M is None else np.asarray(M, dtype=float)
        _k_ex      = _alpha_mf * _eta_i * (1.0/_phi_m_tr + 1.0/phi_f_tr)

        _Sw_ads   = np.clip(np.asarray(theta) / max(self.p["phi_m"], 1e-12), 1e-12, 0.9999)
        _f_ads    = 1.0 / (1.0 + np.exp(-10.0*(_Sw_ads - 0.25)))
        _k_ads    = self.p.get("k_ads_dry", 2e-6) + (self.p.get("k_ads_wet", 5e-7) - self.p.get("k_ads_dry", 2e-6)) * _f_ads
        _k_des    = self.p.get("k_des_dry", 2e-7) + (self.p.get("k_des_wet", 8e-7) - self.p.get("k_des_dry", 2e-7)) * _f_ads
        _Cs_tr    = np.zeros(nz) if Cs is None else np.asarray(Cs, dtype=float).copy()
        _kt_ads   = np.maximum(_k_ads + _k_des, 1e-30)
        _beta_ads = np.maximum(phi_gm_tr, 1e-12)

        def _adsorption_step(Cm_a, Cs_a, dt_ex):
            a11 = 1.0 + dt_ex * _k_ads / _beta_ads
            a12 = -dt_ex * _k_des / _beta_ads
            a21 = -dt_ex * _k_ads
            a22 = 1.0 + dt_ex * (_k_des + lam)
            det = np.maximum(a11 * a22 - a12 * a21, 1e-30)
            Cm_b = np.maximum(( a22 * Cm_a - a12 * Cs_a) / det, 0.0)
            Cs_b = np.maximum((-a21 * Cm_a + a11 * Cs_a) / det, 0.0)
            return Cm_b, Cs_b

        def _adsorption_step_unused(Cm_a, Cs_a, dt_ex):
            u0 = _k_ads * Cm_a - _k_des * Cs_a
            dQ = u0 * (1.0 - np.exp(-np.clip(_kt_ads * dt_ex, 0.0, 700.0))) / _kt_ads
            Cm_b = np.maximum(Cm_a - dQ, 0.0)
            Cs_b = np.maximum(Cs_a + dQ, 0.0) * np.exp(-lam * dt_ex)
            return Cm_b, Cs_b

        def _exchange_step(Cm_a, Cf_a, dt_ex):
            Ssum = _phi_m_tr*Cm_a + phi_f_tr*Cf_a
            u0   = Cm_a - Cf_a
            u_inf = -(_eta_m / max(_eta_i, 1e-30)) * _M_tr
            u    = u_inf + (u0 - u_inf) * np.exp(-np.clip(_k_ex*dt_ex, 0.0, 700.0))
            den  = _phi_m_tr + phi_f_tr
            return ((Ssum + phi_f_tr*u)/den, (Ssum - _phi_m_tr*u)/den)

        for _ in range(n_tr):
            F_adv_m  = _upwind(Cm_new,  q_m_face, C_top=0.0,
                               C_bot_inlet=float(np.atleast_1d(Cm_bc)[-1]))
            F_adv_f  = _upwind(Cf_new,  q_f_face, C_top=0.0, C_bot_inlet=float(Cf_eq[-1]))
            F_adv_m2 = _upwind(Cm20_new,q_m_face, C_top=0.0,
                               C_bot_inlet=float(np.atleast_1d(Cm20_bc)[-1]))
            F_adv_f2 = _upwind(Cf20_new,q_f_face, C_top=0.0, C_bot_inlet=float(Cf20_eq[-1]))
            adv_m  = -(F_adv_m[1:]  - F_adv_m[:-1])  / self.dz / phi_gm_tr
            adv_f  = -(F_adv_f[1:]  - F_adv_f[:-1])  / self.dz / phi_f_tr
            adv_m2 = -(F_adv_m2[1:] - F_adv_m2[:-1]) / self.dz / phi_gm_tr
            adv_f2 = -(F_adv_f2[1:] - F_adv_f2[:-1]) / self.dz / phi_f_tr

            if use_cn:
                Cm_new   = np.maximum(Cm_new   + h_tr*adv_m,  0.0)
                Cf_new   = np.maximum(Cf_new   + h_tr*adv_f,  0.0)
                Cm20_new = np.maximum(Cm20_new + h_tr*adv_m2, 0.0)
                Cf20_new = np.maximum(Cf20_new + h_tr*adv_f2, 0.0)
                Cm_new   = np.maximum(_cn_diffusion_step(Cm_new,  Dm, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=phi_gm_tr), 0.0)
                Cf_new   = np.maximum(_cn_diffusion_step(Cf_new,  Df, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=phi_f_tr), 0.0)
                Cm20_new = np.maximum(_cn_diffusion_step(Cm20_new,Dm, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=phi_gm_tr), 0.0)
                Cf20_new = np.maximum(_cn_diffusion_step(Cf20_new,Df, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=phi_f_tr), 0.0)
                Cm_new, Cf_new = _exchange_step(Cm_new, Cf_new, h_tr)
                Cm_new = np.maximum(Cm_new, 0.0); Cf_new = np.maximum(Cf_new, 0.0)
                Cm_new, _Cs_tr = _adsorption_step(Cm_new, _Cs_tr, h_tr)
            else:
                diff_m  = _div_diff_flux(Cm_new,  Dm, self.dz, C_top=0.0)
                diff_f  = _div_diff_flux(Cf_new,  Df, self.dz, C_top=0.0)
                diff_m2 = _div_diff_flux(Cm20_new,Dm, self.dz, C_top=0.0)
                diff_f2 = _div_diff_flux(Cf20_new,Df, self.dz, C_top=0.0)
                Cm_new   = np.maximum(Cm_new   + h_tr * (diff_m /phi_gm_tr + adv_m),  0.0)
                Cf_new   = np.maximum(Cf_new   + h_tr * (diff_f /phi_f_tr  + adv_f),  0.0)
                Cm20_new = np.maximum(Cm20_new + h_tr * (diff_m2/phi_gm_tr + adv_m2), 0.0)
                Cf20_new = np.maximum(Cf20_new + h_tr * (diff_f2/phi_f_tr  + adv_f2), 0.0)
                Cm_new, Cf_new = _exchange_step(Cm_new, Cf_new, h_tr)
                Cm_new = np.maximum(Cm_new, 0.0); Cf_new = np.maximum(Cf_new, 0.0)
                Cm_new, _Cs_tr = _adsorption_step(Cm_new, _Cs_tr, h_tr)

        Cm_new   = np.clip(Cm_new,   0.0, 5.0 * max(float(np.max(Cm_eq)),   1.0))
        Cf_new   = np.clip(Cf_new,   0.0, 5.0 * max(float(np.max(Cf_eq)),   1.0))
        Cm20_new = np.clip(Cm20_new, 0.0, 5.0 * max(float(np.max(Cm20_eq)), 1.0))
        Cf20_new = np.clip(Cf20_new, 0.0, 5.0 * max(float(np.max(Cf20_eq)), 1.0))
        return Cm_new, Cf_new, Cm20_new, Cf20_new, _Cs_tr

    def _Pg_bc(self, t):
        Patm = self._Patm(t)
        if self.p.get("barometric_pumping_on", True):
            Pbot = self.p.get("P_atm0", 101325.0) + self.p.get("dP_bottom", 5.0)
        else:
            Pbot = Patm + self.p.get("dP_bottom", 5.0)
        return float(Patm), float(Pbot)

    def _Pg_steady(self, theta, t, seismic_mult=None):
        nz  = self.nz
        mu  = self.p.get("mu", 1.82e-5)
        k   = self._k_gas(theta, self.z, seismic_mult)
        dz  = self.dz
        Patm= self._Patm(t)
        if self.p.get("barometric_pumping_on", True):
            Pbot = self.p.get("P_atm0", 101325.0) + self.p.get("dP_bottom", 5.0)
        else:
            Pbot = Patm + self.p.get("dP_bottom", 5.0)

        if nz == 1:
            return np.array([0.5*(Patm+Pbot)])

        k_face = np.empty(nz+1)
        k_face[0]  = k[0]; k_face[-1] = k[-1]
        for i in range(1, nz):
            k_face[i] = 2.0*k[i-1]*k[i]/max(k[i-1]+k[i], 1e-40)
        T_lam = k_face/(mu*dz)
        T_lam[0]  *= 2.0
        T_lam[-1] *= 2.0

        a=np.zeros(nz); b=np.zeros(nz); c=np.zeros(nz); rhs=np.zeros(nz)
        for i in range(nz):
            tL=T_lam[i]; tR=T_lam[i+1]
            b[i] = -(tL+tR)
            if i>0:     a[i] = tL
            if i<nz-1:  c[i] = tR
        rhs[0]  = -T_lam[0]  * Patm
        rhs[-1] = -T_lam[-1] * Pbot

        c_=np.zeros(nz); d_=np.zeros(nz)
        c_[0] = c[0]/b[0] if abs(b[0])>1e-40 else 0.0
        d_[0] = rhs[0]/b[0] if abs(b[0])>1e-40 else Patm
        for i in range(1, nz):
            denom = b[i] - a[i]*c_[i-1]
            c_[i] = c[i]/denom if abs(denom)>1e-40 else 0.0
            d_[i] = (rhs[i]-a[i]*d_[i-1])/denom if abs(denom)>1e-40 else Patm
        Pg_ss = np.zeros(nz)
        Pg_ss[-1] = d_[-1]
        for i in range(nz-2,-1,-1):
            Pg_ss[i] = d_[i] - c_[i]*Pg_ss[i+1]
        P_lo = min(Patm,Pbot) - 10.0
        P_hi = max(Patm,Pbot) + 10.0
        return np.clip(Pg_ss, P_lo, P_hi)

    def _Pin_steady(self, Patm, Tin, Tout):
        g    = self.p.get("g", 9.81)
        H_s  = self.p.get("H_stack", 2.5)
        Mg   = self.p.get("M_g", 0.02897)
        Rg   = self.p.get("R_g", 8.314)
        rho_i = Patm * Mg / (Rg * max(Tin,  1.0))
        rho_o = Patm * Mg / (Rg * max(Tout, 1.0))
        dP_stack = -g * H_s * (rho_o - rho_i) * 0.5
        return float(np.clip(Patm + dP_stack, Patm-20.0, Patm+5.0))

    def simulate_fast(self, days=3.0, dt_out=3600.0, n_sub=None):
        nz    = self.nz
        y     = self.build_y0()
        T_end = days * 86400.0
        n_out = int(round(T_end / dt_out)) + 1
        t_arr = np.linspace(0.0, T_end, n_out)
        Y     = np.zeros((self.nstate, n_out))
        Y[:, 0] = y.copy()
        t     = 0.0
        phi_m = self.p["phi_m"]

        h = self._cfl_dt(y[:nz], y[nz:2*nz], 0.0)

        for i_out in range(1, n_out):
            t_next = t_arr[i_out]
            while t < t_next - 1e-9:
                h = min(h, t_next - t)

                theta   = np.clip(y[0:nz], 1e-8, 0.95*phi_m)
                Pg      = np.clip(y[nz:2*nz], 5e4, 2e5)
                Cm      = np.maximum(y[2*nz:3*nz], 0.0)
                Cf      = np.maximum(y[3*nz:4*nz], 0.0)
                Cm220   = np.maximum(y[7*nz:8*nz], 0.0)
                Cf220   = np.maximum(y[8*nz:9*nz], 0.0)
                Ar      = np.clip(y[6*nz:7*nz], 0.0, 1.0)
                eps_mem = np.clip(y[9*nz:10*nz], 0.01, 0.80)

                M_state = y[4*nz:5*nz]
                Cs_state = np.maximum(y[5*nz:6*nz], 0.0)
                Cm_new, Cf_new, Cm20_new, Cf20_new, Cs_new = self._advance_Cm_implicit(
                    Cm, Cf, Cm220, Cf220, theta, Pg, Ar, eps_mem, t, h,
                    M=M_state, Cs=Cs_state)

                self._J_entry_frozen = None
                if self.p.get("entry_mass_coupling_on", True):
                    C_bs_now = float(y[10*nz])
                    Pin_now  = float(y[10*nz+2])
                    z_wall_s, z_slab_s = self._interface_depths()
                    Pg0_s    = self._at_depth(Pg, z_wall_s)
                    Pg_bot_s = self._at_depth(Pg, z_slab_s)
                    Pg_eff_s = float(np.sqrt(Pg0_s*Pg_bot_s)) if Pg_bot_s > Pg0_s else Pg0_s
                    Cf_eff_s = float(0.3*self._at_depth(Cf_new, z_wall_s)
                                     + 0.7*self._at_depth(Cf_new, z_slab_s))
                    T0_s     = self._at_depth(self.T_field(t), z_slab_s)
                    Tin_s    = self.T_indoor(t)
                    theta0_s = self._at_depth(theta, z_slab_s)
                    Jc = self.crack_network_flux(Cf_eff_s, C_bs_now, Pg_eff_s, Pin_now, theta0_s, T0_s, Tin_s)
                    Jf = self.foundation_2d_network_flux(Cf_eff_s, C_bs_now, Pg_eff_s, Pin_now, theta0_s)
                    J_entry_s  = Jc + Jf
                    A_fp_s     = self.p.get("A_footprint", self.p.get("V_bs", 50.0)/2.5)
                    L_infl_s   = max(float(self.p.get("L_entry_influence", 0.15*self.H)), self.dz)
                    wt_s = np.exp(-0.5*((self.z - z_wall_s)/L_infl_s)**2)
                    wb_s = np.exp(-0.5*((self.z - z_slab_s)/L_infl_s)**2)
                    wt_s /= max(float(np.sum(wt_s)), 1e-30)
                    wb_s /= max(float(np.sum(wb_s)), 1e-30)
                    w_s  = 0.30*wt_s + 0.70*wb_s
                    _Swf_s = np.clip(self.p.get("gamma_sw_f", 0.2)
                                     * (theta / max(self.p["phi_m"], 1e-12)), 1e-6, 0.9999)
                    _thf_s = np.clip(_Swf_s * self.p["phi_f"], 1e-12, 0.9999*self.p["phi_f"])
                    _beta_f_s  = self._beta_partition(_thf_s, self.T_field(t), "f")
                    vol_cell_s = np.maximum(A_fp_s*self.dz*_beta_f_s, 1e-30)
                    S222_s   = self.source_rn222(theta, self.T_field(t), Ar, eps_mem)
                    _cms, _cfs = self._deep_equilibrium_conc(
                        S222_s[-1], theta[-1], Pg[-1],
                        float(np.atleast_1d(self.T_field(t))[-1]), self.lam_rn)
                    Cf_eq_s  = float(np.atleast_1d(_cfs)[0])
                    C_soft_s = max(1e-4 * Cf_eq_s, 1e-6)
                    if J_entry_s > 0.0:
                        Cf_old_s = Cf_new
                        a_s      = h*w_s*J_entry_s/vol_cell_s
                        b_s      = Cf_old_s - C_soft_s - a_s
                        Cf_new   = 0.5*(b_s + np.sqrt(b_s*b_s
                                                      + 4.0*C_soft_s*Cf_old_s))
                        self._J_entry_frozen = float(
                            np.sum(vol_cell_s * (Cf_old_s - Cf_new)) / h)
                    else:
                        Cf_new = np.maximum(Cf_new - h*w_s*J_entry_s/vol_cell_s, 0.0)

                y[2*nz:3*nz] = Cm_new
                y[3*nz:4*nz] = Cf_new
                y[5*nz:6*nz] = Cs_new
                y[7*nz:8*nz] = Cm20_new
                y[8*nz:9*nz] = Cf20_new

                k1 = self._rhs_no_Cm(t, y)
                y2 = np.clip(y + (h/2)*k1, -1e12, 1e12)
                k2 = self._rhs_no_Cm(t + h/2, y2)
                y3 = np.clip(y + (h/2)*k2, -1e12, 1e12)
                k3 = self._rhs_no_Cm(t + h/2, y3)
                y4 = np.clip(y + h*k3, -1e12, 1e12)
                k4 = self._rhs_no_Cm(t + h, y4)
                y  = y + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
                t += h
                self._J_entry_frozen = None

                y[0:nz]            = np.clip(y[0:nz],    1e-8,  0.95*phi_m)
                theta_now = np.clip(y[0:nz], 1e-8, 0.95*phi_m)
                y[nz:2*nz]  = self._Pg_steady(theta_now, t)
                Patm_now    = self._Patm(t)
                Tin_now  = self.T_indoor(t)
                Tout_now = self.T_outdoor(t)
                y[10*nz+2] = self._Pin_steady(Patm_now, Tin_now, Tout_now)
                y[2*nz:3*nz]       = np.maximum(y[2*nz:3*nz],  0.0)
                y[3*nz:4*nz]       = np.maximum(y[3*nz:4*nz],  0.0)
                y[4*nz:5*nz]       = np.nan_to_num(y[4*nz:5*nz], nan=0.0,
                                                   posinf=0.0, neginf=0.0)
                y[5*nz:6*nz]       = np.maximum(y[5*nz:6*nz],  0.0)
                y[6*nz:7*nz]       = np.clip(y[6*nz:7*nz],  0.0, 1.0)
                y[7*nz:8*nz]       = np.maximum(y[7*nz:8*nz],  0.0)
                y[8*nz:9*nz]       = np.maximum(y[8*nz:9*nz],  0.0)
                y[9*nz:10*nz]      = np.clip(y[9*nz:10*nz],  0.01, 0.80)
                y[10*nz]           = np.clip(y[10*nz],    0.0, 5e4)
                y[10*nz+1]         = np.clip(y[10*nz+1],  0.0, 5e4)
                y[10*nz+2]         = np.clip(y[10*nz+2],  5e4, 2e5)
                y[10*nz+3:10*nz+7] = np.maximum(y[10*nz+3:10*nz+7], 0.0)
                _lv_p  = self._air_exchange_rate(t)
                _dep_p = self.p.get("dep_progeny", 1.5/3600.0)
                y[10*nz+6] = (self.lam_po214
                              / max(self.lam_po214 + _lv_p + _dep_p, 1e-30)
                              ) * y[10*nz+5]

                h = self._cfl_dt(y[:nz], y[nz:2*nz], t)

            Y[:, i_out] = y.copy()

        class _Sol:
            pass
        sol         = _Sol()
        sol.t       = t_arr
        sol.y       = Y
        sol.success = True
        sol.message = 'simulate_fast (operator-split semi-implicit) OK'
        return sol

    def extract_results(self, sol):
        nz  = self.nz
        t   = sol.t
        res = {
            "t":        t,
            "theta":    sol.y[0:nz],
            "Pg":       sol.y[nz:2*nz],
            "Cm":       sol.y[2*nz:3*nz],
            "Cf":       sol.y[3*nz:4*nz],
            "M":        sol.y[4*nz:5*nz],
            "Cs":       sol.y[5*nz:6*nz],
            "Ar":       sol.y[6*nz:7*nz],
            "Cm220":    sol.y[7*nz:8*nz],
            "Cf220":    sol.y[8*nz:9*nz],
            "eps_mem":  sol.y[9*nz:10*nz],
            "C_bs":     sol.y[10*nz],
            "C_gt":     sol.y[10*nz+1],
            "Pin":      sol.y[10*nz+2],
            "C_Po218":  sol.y[10*nz+3],
            "C_Pb214":  sol.y[10*nz+4],
            "C_Bi214":  sol.y[10*nz+5],
            "C_Po214":  sol.y[10*nz+6],
        }
        return res


    def emanation_vesicular(self, theta, T_mean=None):
        theta = np.asarray(theta, dtype=float)
        phi_m = self.p["phi_m"]
        Sw    = np.clip(theta / max(phi_m, 1e-6), 0.0, 1.0)

        r_grain = self.p["grain_size_mm"] * 1e-3 / 2.0
        R_g     = 63e-6
        R_w     = 35e-9
        C_s     = 0.25

        f_v   = self.p.get("vesicle_fraction", 0.0)
        r_ves = r_grain * (f_v**(1./3.)) if f_v > 0 else r_grain

        if T_mean is None:
            T_mean = self.p.get("T_ref", 293.15)
        H_T = self._henry_rn(T_mean)

        eps_surf = C_s * min(1.0, R_g / max(r_grain, 1e-12))

        if f_v > 0:
            eps_ves = f_v * C_s * min(1.0, R_g / max(r_ves, 1e-12))
        else:
            eps_ves = 0.0

        eps_wet = ((1.0 - f_v) * C_s
                   * min(1.0, R_w / max(r_grain, 1e-12))
                   / (1.0 + H_T))

        eps_total = eps_surf * (1.0 - Sw) + eps_ves + eps_wet * Sw
        return np.clip(eps_total, 0.0, 1.0)


    def geothermal_T_field(self, t=0.0, T_surface_t=None):
        import math
        if T_surface_t is None:
            T_surface_t = self.p.get("T0", 288.0)

        kappa = 5e-7

        dTdz      = self.p.get("T_grad", 0.025)
        T_geo     = T_surface_t + dTdz * self.z

        T_day    = self.p.get("T_atm", 86400.0)
        A_day    = self.p.get("T_surf_amp", 8.0)
        phi_day  = 0.0
        omega_d  = 2.0 * math.pi / T_day
        z0_d     = math.sqrt(2.0 * kappa / omega_d)
        T_diurnal = (A_day * np.exp(-self.z / z0_d)
                     * np.sin(omega_d * t - self.z / z0_d + phi_day))

        T_year   = 3.1536e7
        A_season = 12.0
        phi_seas  = 0.0
        omega_s  = 2.0 * math.pi / T_year
        z0_s     = math.sqrt(2.0 * kappa / omega_s)
        T_seasonal = (A_season * np.exp(-self.z / z0_s)
                      * np.sin(omega_s * t - self.z / z0_s + phi_seas))

        return T_geo + T_diurnal + T_seasonal

    def geothermal_pe_enhancement(self, theta=None, Pg=None, t=0.0):
        if theta is None:
            theta = np.full(self.nz, 0.4 * self.p["phi_m"])
        if Pg is None:
            Pg = np.full(self.nz, self.p.get("P_atm0", 101325.0))

        theta = np.asarray(theta, float)
        Pg    = np.asarray(Pg,    float)

        T_hot  = self.geothermal_T_field(t)
        T_mean = float(np.mean(T_hot))
        T_ref  = self.p.get("T_ref", 293.15)

        phi_m  = self.p["phi_m"]
        Sg     = np.maximum(phi_m - theta, 0.0) / phi_m
        phi_g  = np.maximum(phi_m * Sg, 1e-12)
        D0     = self.p.get("D0", 1.1e-5)
        mq_fac = phi_g**(10./3.) / phi_m**2
        D_hot  = D0 * (T_mean / T_ref)**1.75 * mq_fac
        D_ref_ = D0 * mq_fac
        D_mean     = float(np.mean(D_hot))
        D_mean_ref = float(np.mean(D_ref_))

        mu_hot = self.p.get("mu", 1.82e-5) * (T_mean / T_ref)**0.7
        mu_ref = self.p.get("mu", 1.82e-5)

        k0    = self.p.get("k0", 1e-13)
        H     = self.p.get("H", 3.0)
        dP    = self.p.get("dP_init", 5.0)
        dPdz  = dP / max(H, 1e-3)

        u_hot = k0 / mu_hot  * dPdz
        u_ref = k0 / mu_ref  * dPdz

        Pe_hot = u_hot * H / max(D_mean, 1e-12)
        Pe_ref = u_ref * H / max(D_mean_ref, 1e-12)
        Da_H   = self.lam_rn * H**2 / max(D_mean, 1e-12)

        Pe_enh_theory = (T_mean / T_ref)**(-2.45)
        Pe_enh_actual = Pe_hot / max(Pe_ref, 1e-30)

        return {
            "Pe_H":                  Pe_hot,
            "Da_H":                  Da_H,
            "Pe_H_cold":             Pe_ref,
            "Pe_enhancement_actual": Pe_enh_actual,
            "Pe_enhancement_theory": Pe_enh_theory,
            "T_mean":                T_mean,
            "mu_hot":                mu_hot,
            "D_eff_mean":            D_mean,
            "regime": "advection" if Pe_hot > 1.0 else "diffusion",
            "scaling_law": f"Pe_H(hot)/Pe_H(cold) = (T/T_ref)^(-2.45) = {Pe_enh_theory:.3f}",
        }


    def barometric_pumping(self, t, theta, Pg_ambient=None):
        import math
        theta = np.asarray(theta, float)
        if Pg_ambient is None:
            Pg_ambient = self._Patm(t)

        phi_m = self.p["phi_m"]
        Sg    = np.maximum(phi_m - theta, 0.0) / phi_m
        phi_g = np.maximum(phi_m * Sg, 1e-3)

        omega   = 2.0 * math.pi / self.p.get("T_atm", 86400.0)
        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        phi_g_m = float(np.mean(phi_g))

        z_p = math.sqrt(2.0 * k0 * Pg_ambient
                        / max(omega * mu * phi_g_m, 1e-30))
        z_p = max(z_p, 1e-3)

        dP_amp  = self.p.get("P_atm_amp", 150.0)
        phi_baro = 0.0

        Pg_baro = (Pg_ambient
                   + dP_amp * np.exp(-self.z / z_p)
                   * np.cos(omega * t - self.z / z_p + phi_baro))

        H  = self.p.get("H", 3.0)
        BE = math.exp(-H / z_p)

        krg = self._kr_gas_vg(self._Se_drainage(theta))
        dPbaro_dz = np.zeros(self.nz)
        dPbaro_dz[1:-1] = (Pg_baro[2:] - Pg_baro[:-2]) / (2 * self.dz)
        dPbaro_dz[0]    = (Pg_baro[1]  - Pg_baro[0])   / self.dz
        dPbaro_dz[-1]   = (Pg_baro[-1] - Pg_baro[-2])  / self.dz

        u_baro = -(k0 * krg / mu) * dPbaro_dz

        dP_surface = dP_amp * math.cos(omega * t + phi_baro)

        return {
            "Pg_field":        Pg_baro,
            "u_baro":          u_baro,
            "z_p_pneumatic":   z_p,
            "BE":              BE,
            "dP_surface":      dP_surface,
            "z_p_scaling":
                f"z_p = √(2k₀P_atm/ωμφ) = {z_p:.3f} m  (k₀={k0:.1e} m²)",
            "omega":           omega,
            "phi_g_mean":      phi_g_m,
        }


    def capillary_entry_pressure(self, theta, dtheta_dt=None):
        theta = np.asarray(theta, float)
        phi_m  = self.p["phi_m"]
        theta_r = self.p.get("theta_r", 0.03)
        Se    = np.clip((theta - theta_r) / max(phi_m - theta_r, 1e-6),
                        0.0, 1.0)

        P_e_mac = self.p.get("P_entry", 500.0)
        P_e_mic = P_e_mac * 50.0
        phi_mac = self.p.get("phi_f", 0.05)
        f_mac   = min(phi_mac / max(phi_m, 1e-6), 1.0)

        P_c_eq = (P_e_mac * np.power(np.maximum(Se, 1e-12), -1./1.5) * (1.0 - f_mac)
                + P_e_mic * np.power(np.maximum(Se, 1e-12), -1./3.0) * f_mac)
        P_c_eq = np.clip(P_c_eq, 0.0, 1e6)

        tau_cap = self.p.get("tau_cap", 1800.0)
        if dtheta_dt is not None:
            phi_m_ref = max(phi_m, 1e-6)
            dSe_dt = np.asarray(dtheta_dt, float) / phi_m_ref
            P_c_dyn = P_c_eq + tau_cap * dSe_dt
        else:
            P_c_dyn = P_c_eq.copy()

        Se_vg = self._Se_drainage(theta)
        n_vg  = self.p.get("soil_n", 1.5)
        m_vg  = 1.0 - 1.0/n_vg
        krg_eq = np.power(np.maximum(1.0 - Se_vg, 0.0), 0.5) * \
                 np.power(np.maximum(1.0 - Se_vg**( 1.0/m_vg), 0.0), 2.0*m_vg)
        krg_eq = np.clip(krg_eq, 0.0, 1.0)

        S_trg   = np.maximum(0.0, 0.25 * (1.0 - Se))
        krg_wet = np.maximum(krg_eq - S_trg, 0.0)

        return {
            "P_c_eq":  P_c_eq,
            "P_c_dyn": P_c_dyn,
            "krg_eq":  krg_eq,
            "krg_wet": krg_wet,
            "S_trg":   S_trg,
            "Se":      Se,
            "f_mac":   f_mac,
            "tau_cap": tau_cap,
        }


    def fracture_aperture_climate(self, theta_mean, T_mean, Pg_mean):
        phi_m  = self.p["phi_m"]
        Sw     = float(np.clip(theta_mean / max(phi_m, 1e-6), 0.0, 1.0))
        T_ref  = self.p.get("T_ref", 293.15)

        b0      = self.p.get("b0_fracture_mean", 2e-4)
        c_sat   = 0.6
        c_T     = 2e-4
        c_sig   = 5e-8
        b_min   = 1e-7

        sigma_tot = self.p.get("sigma_tot_frac", 3e5)
        biot      = 0.7
        sigma_eff = sigma_tot - biot * (float(Pg_mean) - self.p.get("P_atm0", 101325.0))
        sigma_eff = max(sigma_eff, 0.0)

        thermal  = max(1.0 + c_T * (T_mean - T_ref), 0.05)
        b_eff    = b0 * np.exp(-c_sat * Sw) * thermal * np.exp(-c_sig * sigma_eff)
        b_eff    = max(b_eff, b_min)

        k_f_eff = b_eff**2 / 12.0

        chi0  = 1.0
        a_sat = 2.5
        a_sig = 1e-7
        chi_eff = (chi0 * np.exp(-a_sat * Sw)
                   * np.exp(-a_sig * sigma_eff))
        chi_eff = max(chi_eff, 1e-12)

        return {
            "b_eff":      b_eff,
            "k_f_eff":    k_f_eff,
            "chi_eff":    chi_eff,
            "sigma_eff":  sigma_eff,
            "Sw":         Sw,
            "thermal_factor": thermal,
        }


    def transport_regime(self, Pe_H, Da_H, phi_tot=None):
        if phi_tot is None:
            phi_tot = self.p["phi_m"] + self.p.get("phi_f", 0.05)

        phi_mac  = self.p.get("phi_f", 0.05)
        f_mac    = phi_mac / max(phi_tot, 1e-6)
        baro_num = self.p.get("P_atm_amp", 150.0) / self.p.get("P_atm0", 101325.0)

        if Pe_H > 1.0:
            regime = 1
            name   = "Avvettivo (Pe > 1)"
            L_star = "H — trasporto sull'intera colonna"
            C_scaling = "C_in ~ Pe_H · Π_src · Π_geo"
            insight = (
                "Regime Pe > 1: l'advection controlla il trasporto. "
                "C_in ∝ k₀·ΔP/(μ·ACH·V). Aumentare k₀ di 10× → C_in×10. "
                "Questo regime è RARO nei suoli non-vulcanici (Pe << 1). "
                "Spiega perché gli edifici su tufo saldato hanno C_in 10-100× "
                "superiore agli edifici su terreno sabbioso."
            )
        elif Da_H > 100.0:
            regime = 2
            name   = "Limitato da decadimento (Da > 100)"
            L_star = "√(D/λ) — lunghezza di penetrazione Rn"
            C_scaling = "C_in ~ Da_H^{-1/2} · Π_src"
            insight = (
                "Il decadimento controlla. Il Rn decade prima di raggiungere "
                "la superficie. C_in ∝ (D/λ)^0.5 · S_vol. "
                "Materiali vulcanici: D_eff > D_argilla → penetrazione maggiore "
                "→ C_in più alta a parità di sorgente."
            )
        elif Da_H < 0.01:
            regime = 3
            name   = "Limitato da sorgente (Da << 1)"
            L_star = "H — l'intera colonna contribuisce"
            C_scaling = "C_in ~ Π_src · φ_tot · H"
            insight = "C_in ∝ A_Ra · ρ_b · ε · H. Dipende quasi solo dalla sorgente."
        elif f_mac > 0.3 and Pe_H > 0.1:
            regime = 4
            name   = "Bypass di frattura (φ_mac > 0.3)"
            L_star = "b_eff³/12 · frac_density"
            C_scaling = "C_in ~ χ · f_frac · Pe_H · Π_src"
            insight = (
                "Il canalizzamento attraverso fratture domina. "
                "C_in amplificato dal fattore χ = exp(3σ²_b). "
                "Rocce vulcaniche: σ_b > 0.5 → χ > 3. "
                "Spiega i valori anomali nelle costruzioni su tufo saldato fratturato."
            )
        elif Pe_H < 0.1 and baro_num > 0.001:
            regime = 5
            name   = "Dominato da pompaggio barometrico"
            L_star = "z_p = √(2k₀P_atm/ωμφ)"
            C_scaling = "C_in ~ BE · Π_baro · Π_src"
            insight = (
                "Il pompaggio barometrico guida il trasporto. "
                "C_in oscilla con periodo diurno. BE = exp(-H/z_p). "
                "Suoli vulcanici: z_p >> H → BE alto → segnale barometrico forte."
            )
        elif Pe_H < 0.1 and Da_H < 10.0:
            regime = 0
            name   = "Diffusivo (Pe < 0.1)"
            L_star = "√(D/λ) — lunghezza di penetrazione Rn"
            C_scaling = "C_in ~ Da_H^{-1/2} · Π_src · Π_geo"
            insight = (
                "Regime diffusivo standard. Come i suoli non-vulcanici. "
                "D_eff maggiore nei materiali vulcanici → penetrazione maggiore "
                "→ C_in superiore a parità di sorgente."
            )
        else:
            regime = 6
            name   = "Misto / Transizione"
            L_star = "Pe_H · L_diff (scala mista)"
            C_scaling = "C_in ~ Pe_H^0.5 · Da_H^{-0.5} · Π_src"
            insight = (
                "Zona di transizione. Diffusione e advection co-dominano. "
                "Sensibile a piccole variazioni di k₀ o ΔP. "
                "Monitorare Pe_H nel tempo per rilevare transizioni di regime."
            )

        return {
            "regime_id":        regime,
            "regime_name":      name,
            "L_star":           L_star,
            "C_scaling":        C_scaling,
            "physical_insight": insight,
            "Pe_H":             Pe_H,
            "Da_H":             Da_H,
            "phi_tot":          phi_tot,
            "f_macropore":      f_mac,
            "baro_number":      baro_num,
        }


    def lithotype_comparison(self):
        results = {}

        for lit_name, lit_data in VOLCANIC_LITHOTYPES.items():
            try:
                m = build_model(lit_name, override={"nz": 15})

                sol = m.simulate_fast(days=2.0, dt_out=3600.0)
                out = m.extract_results(sol)

                t   = np.array(out["t"])
                Cbs = np.array(out["C_bs"])
                Cgt = np.array(out["C_gt"])
                Ar  = np.array(out["Ar"])
                eps = np.array(out["eps_mem"])
                Pg  = np.array(out["Pg"])

                mask = t > t[-1] - 6*3600
                def m_last(a):
                    a2 = np.asarray(a)
                    if a2.ndim == 2: return float(np.nanmean(a2[:, mask]))
                    return float(np.nanmean(a2[mask]))

                Cbs_m = m_last(Cbs)
                theta_f = np.array(out["theta"])[:, -1]
                Pg_f    = Pg[:, -1]
                T_f     = m.T_field(sol.t[-1])

                pi = m.buckingham_pi(theta_f, Pg_f, T_f)

                Cf0   = float(np.array(out["Cf"])[0, -1])
                Cf_bot= float(np.array(out["Cf"])[-1, -1])
                Cf_eff= 0.3*Cf0 + 0.7*Cf_bot
                Pg0   = float(Pg_f[0]); Pg_b = float(Pg_f[-1])
                Pg_eff= np.sqrt(Pg0 * Pg_b)
                Pin   = m._Pin_steady(m._Patm(sol.t[-1]),
                                      m.p.get("T_in0", 293.15),
                                      m.p.get("T_out0", 283.15))
                J_c = m.crack_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                            float(theta_f[0]),
                                            float(T_f[0]),
                                            m.p.get("T_in0", 293.15))
                J_f = m.foundation_2d_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                                    float(theta_f[0]))
                J_tot = J_c + J_f
                A_fp_lc = m.p.get("A_footprint", m.p.get("V_bs", 50.0)/2.5)
                J_surface_lc = J_tot / max(A_fp_lc, 1e-12)

                regime = m.transport_regime(pi["Pe_H"], pi["DaDiff"])

                frac_props = m.fracture_aperture_climate(
                    float(np.mean(theta_f)),
                    float(np.mean(T_f)),
                    float(np.mean(Pg_f))
                )

                eps_ves = m.emanation_vesicular(theta_f, float(np.mean(T_f)))

                results[lit_name] = {
                    "label":       lit_data["label"],
                    "Cin_Bq_m3":   Cbs_m,
                    "C_gt_Bq_m3":  m_last(Cgt),
                    "Pe_H":        pi["Pe_H"],
                    "Da_H":        pi["Da_H"],
                    "J_surface":   J_surface_lc,
                    "J_entry_Bq_s": J_tot,
                    "eps_mean":    m_last(eps),
                    "eps_vesicular": float(np.mean(eps_ves)),
                    "Ar_mean":     m_last(Ar),
                    "regime":      regime["regime_name"],
                    "b_eff_um":    frac_props["b_eff"] * 1e6,
                    "k_f_eff":     frac_props["k_f_eff"],
                    "phi_tot":     m.p["phi_m"] + m.p.get("phi_f", 0.05),
                    "k0":          m.p["k0"],
                    "A_Ra":        m.p["Ra226_Bqkg"],
                    "rho_bulk":    m.p["rho_bulk"],
                }
            except Exception as e:
                results[lit_name] = {"label": lit_data["label"], "error": str(e)}

        valid_res  = {k: v for k, v in results.items() if "Cin_Bq_m3" in v}
        sorted_res = dict(sorted(valid_res.items(),
                                  key=lambda x: x[1]["Cin_Bq_m3"],
                                  reverse=True))
        sorted_res.update({k: v for k, v in results.items() if "error" in v})
        return sorted_res


    def co2_magmatic_transport(self, Cm, x_CO2_in, Pg, theta, T_z, dt=0.0):
        import math

        Cm    = np.asarray(Cm,       dtype=float).copy()
        x_CO2 = np.asarray(x_CO2_in, dtype=float).copy()
        Pg    = np.asarray(Pg,        dtype=float)
        theta = np.asarray(theta,     dtype=float)
        T_z   = np.asarray(T_z,       dtype=float)

        nz    = self.nz
        dz    = self.dz
        phi_m = self.p["phi_m"]
        phi_f = self.p.get("phi_f", 0.05)
        phi_tot = phi_m + phi_f
        phi_g = np.maximum(phi_tot - theta, 1e-6)

        D0_CO2   = self.p.get("D0_CO2", 1.6e-5)
        T_mean   = float(np.mean(T_z))
        T_ref    = self.p.get("T_ref", 293.15)
        P_mean   = float(np.mean(Pg))
        D_CO2_ref = D0_CO2 * (T_mean / T_ref)**1.75 * (101325.0 / P_mean)

        D_CO2_eff = D_CO2_ref * phi_g**(10./3.) / max(phi_tot**2, 1e-12)

        M_Rn  = 0.222
        M_CO2 = 0.044
        M_N2  = 0.028
        D0_Rn = self.p.get("D0", 1.1e-5)
        D_Rn_CO2 = D0_Rn * math.sqrt((1.0/M_Rn + 1.0/M_CO2) /
                                     (1.0/M_Rn + 1.0/M_N2))

        D_Rn_eff = D0_Rn * phi_g**(10./3.) / max(phi_tot**2, 1e-12)

        S_CO2   = self.p.get("S_CO2", 1e-5)
        S_CO2_field = np.zeros(nz)
        n_src = max(1, nz // 3)
        for i in range(n_src):
            S_CO2_field[nz - 1 - i] = S_CO2 * math.exp(-3.0 * i / max(n_src, 1))

        x_surf        = self.p.get("x_CO2_surface", 4e-4)
        lap_CO2 = np.zeros(nz)
        if nz == 1:
            lap_CO2[0] = 2.0*(x_surf - x_CO2[0]) / dz**2
        else:
            lap_CO2[1:-1] = (x_CO2[2:] - 2.0*x_CO2[1:-1] + x_CO2[:-2]) / dz**2
            lap_CO2[0]    = (x_CO2[1] - 3.0*x_CO2[0] + 2.0*x_surf) / dz**2
            lap_CO2[-1]   = (x_CO2[-2] - x_CO2[-1]) / dz**2

        dx_CO2_dt = (D_CO2_eff * lap_CO2
                     + S_CO2_field * self.p.get("R_g", 8.314) * T_mean /
                       max(P_mean, 1e-3))

        dx_dz = np.zeros(nz)
        dx_dz[1:-1] = (x_CO2[2:] - x_CO2[:-2]) / (2.0 * dz)
        dx_dz[0]    = (x_CO2[1] - x_CO2[0]) / dz
        dx_dz[-1]   = (x_CO2[-1] - x_CO2[-2]) / dz

        x_safe    = np.clip(x_CO2, 0.0, 0.95)
        factor_SM = D_Rn_CO2 / np.maximum(1.0 - x_safe, 0.05)
        J_Rn_SM   = -phi_g * Cm * factor_SM * dx_dz

        div_J_SM = np.zeros(nz)
        div_J_SM[1:-1] = (J_Rn_SM[2:] - J_Rn_SM[:-2]) / (2.0 * dz)
        div_J_SM[0]    = (J_Rn_SM[1] - J_Rn_SM[0]) / dz
        div_J_SM[-1]   = (J_Rn_SM[-1] - J_Rn_SM[-2]) / dz
        dC_carrier = -div_J_SM

        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        H       = self.p.get("H", 3.0)
        g       = self.p.get("g", 9.81)
        Rgas    = self.p.get("R_g", 8.314)
        x_max   = float(np.max(np.clip(x_CO2, 0.0, 0.95)))
        x_min   = float(np.min(np.clip(x_CO2, 0.0, 0.95)))
        rho_CO2mix = P_mean / (Rgas * T_mean) * (
            M_CO2 * x_max + 0.02897 * (1.0 - x_max))
        rho_air    = P_mean * 0.02897 / (Rgas * T_mean)
        Delta_rho  = max(rho_air - rho_CO2mix, 0.0)
        D_CO2_mean  = float(np.mean(D_CO2_eff))
        Ra_volcanic = (k0 * Delta_rho * g * H
                       / max(mu * D_CO2_mean, 1e-30))

        Ra_crit   = 40.0
        convective = Ra_volcanic > Ra_crit

        if convective:
            u_conv   = D_CO2_mean * (Ra_volcanic - Ra_crit) / H
            chi_conv = math.sqrt(Ra_volcanic / Ra_crit)
        else:
            u_conv   = 0.0
            chi_conv = 1.0

        C_Rn_eff = Cm * (1.0 - x_safe)

        return {
            "x_CO2":       x_CO2,
            "dx_CO2_dt":   dx_CO2_dt,
            "J_Rn_SM":     J_Rn_SM,
            "dC_carrier":  dC_carrier,
            "C_Rn_eff":    C_Rn_eff,
            "Ra_volcanic": float(Ra_volcanic),
            "Ra_critical": Ra_crit,
            "convective":  bool(convective),
            "chi_conv":    float(chi_conv),
            "u_conv":      float(u_conv),
            "D_CO2_eff":   D_CO2_eff,
            "D_Rn_CO2":    D_Rn_CO2,
        }


    def bifurcation_analysis(self, dP_array=None, n_points=40,
                              theta_sw=None, T_mean=None,
                              return_regime_map=False):
        import math

        if dP_array is None:
            _mu_b   = self.p.get("mu", 1.82e-5)
            _k0_b   = max(self.p.get("k0", 1e-13), 1e-30)
            _th_b   = (theta_sw if theta_sw is not None
                       else 0.35 * self.p["phi_m"])
            _De_b   = float(np.mean(np.atleast_1d(
                          self._D_eff(np.full(self.nz, _th_b), "m"))))
            _dPc    = _mu_b * _De_b / _k0_b
            _lo, _hi = np.log10(max(_dPc, 1e-6)) - 2.0, np.log10(max(_dPc, 1e-6)) + 2.0
            dP_array = np.logspace(_lo, _hi, n_points)
        dP_array = np.asarray(dP_array, dtype=float)
        n = len(dP_array)

        if theta_sw is None:
            theta_sw = 0.35 * self.p["phi_m"]
        if T_mean is None:
            T_mean = self.p.get("T_ref", 293.15)

        phi_m   = self.p["phi_m"]
        phi_f   = self.p.get("phi_f", 0.05)
        phi_tot = phi_m + phi_f
        Sg      = max(phi_tot - theta_sw, 0.0) / max(phi_tot, 1e-12)
        phi_g   = max(phi_tot * Sg, 1e-4)
        D0      = self.p.get("D0", 1.1e-5)
        D_eff   = D0 * phi_g**(10./3.) / max(phi_tot**2, 1e-12)
        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        H       = self.p.get("H", 3.0)
        lam_rn  = self.lam_rn

        dP_crit_anal = mu * D_eff / max(k0, 1e-30)
        Da_H_ref     = lam_rn * H**2 / max(D_eff, 1e-30)

        nz     = self.nz
        T_arr  = np.full(nz, T_mean)
        theta  = np.full(nz, theta_sw)
        Ar0    = np.full(nz, 0.75)
        eps0   = np.full(nz, self.p.get("eps_max", 0.3) * 0.6)
        S222   = self.source_rn222(theta, T_arr, Ar0, eps0)
        Cm_eq  = S222 / max(lam_rn, 1e-30)
        _cm_de, _cf_de = self._deep_equilibrium_conc(
            S222, theta, np.full(nz, self.p.get("P_atm0", 101325.0)),
            T_arr, lam_rn)
        Cf_eq  = Cm_eq * (np.asarray(_cf_de)
                          / np.maximum(np.asarray(_cm_de), 1e-30))

        S_vol_mean = float(np.mean(S222))
        V_bs  = self.p.get("V_bs", 50.0)
        ach   = self.p.get("ACH_closed", 0.1) / 3600.0
        lam_v = lam_rn + ach
        A_eff = self.p.get("A_crack_m2", 0.05)
        Cm_eq_mean   = S_vol_mean / max(lam_rn, 1e-30)
        _phi_g_bif   = max(float(self.p["phi_m"] - float(theta_sw)), 1e-12)
        L_d_diff     = float(np.sqrt(D_eff / max(lam_rn * _phi_g_bif, 1e-30)))
        J_diff_lim   = Cm_eq_mean * float(np.sqrt(lam_rn * _phi_g_bif * D_eff)) * math.tanh(H / max(L_d_diff, 1e-9))
        C_diff_limit = J_diff_lim * A_eff / max(V_bs * lam_v, 1e-30)

        Patm  = self.p.get("P_atm0", 101325.0)
        Tin   = self.p.get("T_in0", 293.15)
        T0s   = T_mean

        Cin_arr  = np.zeros(n)
        Pe_arr   = np.zeros(n)
        Da_arr   = np.full(n, Da_H_ref)
        J_tot_arr = np.zeros(n)
        J_diff_arr= np.zeros(n)
        J_adv_arr = np.zeros(n)

        Cf0 = float(np.mean(Cf_eq[:3]))

        for i, dP in enumerate(dP_array):
            Pg_bot = Patm + float(dP)
            Pg_top = Patm
            Pg_eff = math.sqrt(Pg_top * Pg_bot)
            Pin    = Patm - 1.5

            u_Darcy = k0 * float(dP) / (mu * H)
            Pe_i    = u_Darcy * H / max(D_eff, 1e-30)

            J_d  = D_eff * float(np.mean(Cm_eq)) * A_eff / H

            Cf_bot = float(np.mean(Cf_eq[-3:]))
            Cf_eff = 0.3 * Cf0 + 0.7 * Cf_bot
            J_a    = (u_Darcy * Cf_eff * phi_g * A_eff
                      * max(Pg_eff - Pin, 0.0) / max(float(dP), 0.1))

            J_c = self.crack_network_flux(Cf_eff, Cin_arr[i-1] if i>0 else 0.0,
                                           Pg_eff, Pin, theta_sw, T0s, Tin)
            J_f = self.foundation_2d_network_flux(Cf_eff, Cin_arr[i-1] if i>0 else 0.0,
                                                    Pg_eff, Pin, theta_sw)
            J_net = J_c + J_f
            if J_net < 0:
                J_net = max(J_d, 0.0)

            J_diff_arr[i]  = float(J_d)
            J_adv_arr[i]   = float(max(J_a, 0.0))
            J_tot_arr[i]   = float(J_net)

            C_ss       = J_net / max(V_bs * lam_v, 1e-30)
            Cin_arr[i] = max(C_ss, 0.0)
            Pe_arr[i]  = Pe_i

        dCin_ddP = np.gradient(Cin_arr, dP_array)
        idx_star = int(np.argmax(np.abs(dCin_ddP)))
        dP_num   = float(dP_array[idx_star])

        mask_adv = Pe_arr > 1.0
        alpha_adv = np.nan
        if np.sum(mask_adv) >= 3:
            lx = np.log(dP_array[mask_adv])
            ly = np.log(np.maximum(Cin_arr[mask_adv], 1e-10))
            c  = np.polyfit(lx, ly, 1)
            alpha_adv = float(c[0])

        regime_map = None
        if return_regime_map:
            n_th = 15
            theta_grid = np.linspace(0.05*phi_m, 0.95*phi_m, n_th)
            Pe_grid    = np.zeros((n_th, n))
            for it, th in enumerate(theta_grid):
                Sg_t   = max(phi_tot - th, 0.0) / max(phi_tot, 1e-12)
                phi_gt = max(phi_tot * Sg_t, 1e-4)
                D_t    = D0 * phi_gt**(10./3.) / max(phi_tot**2, 1e-12)
                for j, dP in enumerate(dP_array):
                    u_t         = k0 * dP / (mu * H)
                    Pe_grid[it, j] = u_t * H / max(D_t, 1e-30)
            regime_map = {
                "theta_grid": theta_grid,
                "dP_array":   dP_array,
                "Pe_grid":    Pe_grid,
                "levels":     [0.1, 1.0, 10.0],
            }

        bif_str = (
            f"Bifurcation point: ΔP* = {dP_crit_anal:.3f} Pa (analytical), "
            f"{dP_num:.3f} Pa (numerical). "
            f"Advection scaling: C_in ∝ ΔP^{alpha_adv:.2f} "
            f"(theoretical: 1.0). Pe_H range: {Pe_arr.min():.3f}–{Pe_arr.max():.3f}."
        )

        return {
            "dP":              dP_array,
            "C_in":            Cin_arr,
            "Pe_H":            Pe_arr,
            "Da_H":            Da_arr,
            "J_total":         J_tot_arr,
            "J_diff":          J_diff_arr,
            "J_adv":           J_adv_arr,
            "dP_critical":     dP_crit_anal,
            "dP_numerical":    dP_num,
            "alpha_adv":       alpha_adv,
            "C_diffusion":     C_diff_limit,
            "D_eff":           D_eff,
            "regime_map":      regime_map,
            "bifurcation_str": bif_str,
        }


    def buckingham_pi_paper(self, n_samples=512, n_boot=200, seed=42,
                             return_samples=False):
        try:
            from scipy.stats import qmc as _qmc_mod
            _HAS_QMC = True
        except ImportError:
            _HAS_QMC = False
            _qmc_mod = None

        rng = np.random.default_rng(seed)

        param_ranges = {
            "k0":          (5e-16, 5e-10),
            "phi_m":       (0.08,  0.70),
            "phi_f":       (0.005, 0.35),
            "Ra226_Bqkg":  (5.0,   250.0),
            "rho_bulk":    (400.,  2800.),
            "H":           (0.3,   6.0),
            "dP_init":     (0.1,   100.0),
            "ACH_closed":  (0.02,  5.0),
            "A_crack_m2":  (1e-5,  0.1),
            "P_atm_amp":   (30.,   600.),
            "T_grad":      (0.005, 0.20),
            "grain_size_mm":(0.005, 2.0),
        }
        keys = list(param_ranges.keys())
        ndim = len(keys)

        if _HAS_QMC and _qmc_mod is not None:
            sampler = _qmc_mod.LatinHypercube(d=ndim, seed=int(seed))
            lhs_raw = sampler.random(n=n_samples)
        else:
            lhs_raw = rng.uniform(0.0, 1.0, (n_samples, ndim))

        lo_log = np.array([np.log10(max(v[0], 1e-30)) for v in param_ranges.values()])
        hi_log = np.array([np.log10(max(v[1], 1e-30)) for v in param_ranges.values()])
        X_log  = lhs_raw * (hi_log - lo_log) + lo_log
        X      = 10.0**X_log

        lam     = self.lam_rn
        mu_gas  = self.p.get("mu", 1.82e-5)
        Patm    = self.p.get("P_atm0", 101325.0)
        V_in    = self.p.get("V", 170.0)

        Cin_arr = np.zeros(n_samples)
        pi_arr  = np.zeros((n_samples, 7))

        for j in range(n_samples):
            pj = dict(zip(keys, X[j]))
            k0_j   = pj["k0"]
            phi_m_j= pj["phi_m"]
            phi_f_j= pj["phi_f"]
            phi_tot= phi_m_j + phi_f_j
            H_j    = pj["H"]
            dP_j   = pj["dP_init"]
            Ra_j   = pj["Ra226_Bqkg"]
            rb_j   = pj["rho_bulk"]
            ACH_j  = pj["ACH_closed"] / 3600.0
            A_j    = pj["A_crack_m2"]
            dPb_j  = pj["P_atm_amp"]
            dT_j   = pj["T_grad"]
            gs_j   = pj["grain_size_mm"]

            Sg_j   = max(phi_tot - 0.35*phi_m_j, 1e-3) / max(phi_tot, 1e-6)
            phi_g_j= max(phi_tot * Sg_j, 1e-4)
            D0_j   = 1.1e-5
            D_eff_j= D0_j * phi_g_j**(10./3.) / max(phi_tot**2, 1e-12)

            u_j  = k0_j * dP_j / (mu_gas * H_j)
            Pe_j = u_j * H_j / max(D_eff_j, 1e-30)

            r_grain = gs_j * 1e-3 / 2.0
            R_g     = 63e-6
            eps_j   = min(1.0, 0.25 * R_g / max(r_grain, 1e-10)) * 0.7

            S_vol_j = Ra_j * rb_j * lam * eps_j
            theta_w_j = 0.35 * phi_m_j
            beta_j    = max(phi_g_j + float(self._henry_rn(293.15)) * theta_w_j,
                            1e-6)
            Cm_j    = S_vol_j / (lam * beta_j)
            lam_eff = lam + ACH_j
            J_j     = (D_eff_j * Cm_j / max(H_j, 1e-3)
                       + u_j * Cm_j) * A_j
            Cin_j   = J_j / max(V_in * lam_eff, 1e-30)
            Cin_arr[j] = max(Cin_j, 1e-9)

            C_ref_j = 1000.0
            t_mix_j = 1.0 / max(lam_eff, 1e-30)
            pi_arr[j, 0] = lam * H_j**2 / max(D_eff_j, 1e-30)
            pi_arr[j, 1] = k0_j * dP_j / (mu_gas * lam * H_j**2)
            pi_arr[j, 2] = S_vol_j / max(lam * C_ref_j, 1e-30)
            pi_arr[j, 3] = A_j / max(H_j * V_in**(1.0/3.0), 1e-30)
            pi_arr[j, 4] = D_eff_j * t_mix_j / max(H_j**2, 1e-30)
            pi_arr[j, 5] = dPb_j / max(dP_j, 1e-1)
            pi_arr[j, 6] = dT_j * H_j / 293.15

        valid  = (Cin_arr > 1e-6) & np.all(np.isfinite(pi_arr) & (pi_arr > 0), axis=1)
        n_valid= int(np.sum(valid))
        Cin_v  = Cin_arr[valid]
        pi_v   = pi_arr[valid]

        if n_valid < 20:
            return {"error": f"Troppo pochi campioni validi ({n_valid})."}

        C_ref_global = np.median(Cin_v)
        Cin_star = Cin_v / max(C_ref_global, 1e-6)

        log_pi = np.log10(np.maximum(pi_v, 1e-30))
        log_C  = np.log10(np.maximum(Cin_star, 1e-30))
        A_mat  = np.column_stack([np.ones(n_valid), log_pi])

        coef, _, _, _ = np.linalg.lstsq(A_mat, log_C, rcond=None)
        log_K     = float(coef[0])
        exponents = coef[1:]

        log_C_pred = A_mat @ coef
        SS_res = np.sum((log_C - log_C_pred)**2)
        SS_tot = np.sum((log_C - log_C.mean())**2)
        R2     = float(1.0 - SS_res / max(SS_tot, 1e-30))

        boot_exp = np.zeros((n_boot, 7))
        boot_R2  = np.zeros(n_boot)
        boot_rng = np.random.default_rng(seed + 1)
        for b in range(n_boot):
            idx_b = boot_rng.integers(0, n_valid, size=n_valid)
            A_b   = A_mat[idx_b]
            C_b   = log_C[idx_b]
            cb, _, _, _ = np.linalg.lstsq(A_b, C_b, rcond=None)
            boot_exp[b] = cb[1:]
            pred_b = A_b @ cb
            ss_r = np.sum((C_b - pred_b)**2)
            ss_t = np.sum((C_b - C_b.mean())**2)
            boot_R2[b] = float(1.0 - ss_r / max(ss_t, 1e-30))

        ci95_lo = np.percentile(boot_exp, 2.5,  axis=0)
        ci95_hi = np.percentile(boot_exp, 97.5, axis=0)
        exponents_ci95 = np.column_stack([ci95_lo, ci95_hi])
        R2_ci95 = (float(np.percentile(boot_R2, 2.5)),
                   float(np.percentile(boot_R2, 97.5)))

        pi_std     = np.std(log_pi, axis=0)
        importance = np.abs(exponents) * pi_std
        boot_importance = np.abs(boot_exp) * pi_std[None, :]
        imp_ci95_lo = np.percentile(boot_importance, 2.5,  axis=0)
        imp_ci95_hi = np.percentile(boot_importance, 97.5, axis=0)
        importance_ci95 = np.column_stack([imp_ci95_lo, imp_ci95_hi])
        rank_idx   = np.argsort(importance)[::-1]

        pi_names = ["Da_H", "Π_Darcy-decay", "Π_src",
                    "Π_geo", "Π_Fourier", "Π_baro", "Π_geotherm"]

        terms_tex = " \\cdot ".join(
            f"{name}^{{{exp:+.3f}_{{{lo:+.2f}}}^{{{hi:+.2f}}}}}"
            for name, exp, lo, hi
            in zip(pi_names, exponents, ci95_lo, ci95_hi)
        )
        law_str_latex = (
            f"C_{{\\rm in}}^* \\approx "
            f"{10**log_K:.3e} \\cdot {terms_tex}"
            f"\\quad (R^2 = {R2:.3f})"
        )
        law_str_plain = (
            f"C_in* ≈ {10**log_K:.3e} · " +
            " · ".join(
                f"{n}^{e:+.3f}[{l:+.2f},{h:+.2f}]"
                for n,e,l,h in zip(pi_names, exponents, ci95_lo, ci95_hi)
            ) +
            f"  (R²={R2:.3f}, n={n_valid})"
        )

        out = {
            "exponents":       exponents,
            "exponents_ci95":  exponents_ci95,
            "log_K":           log_K,
            "R2":              R2,
            "R2_ci95":         R2_ci95,
            "pi_names":        pi_names,
            "importance":      importance,
            "importance_ci95": importance_ci95,
            "rank":            [pi_names[i] for i in rank_idx],
            "law_string":      law_str_plain,
            "law_latex":       law_str_latex,
            "n_valid":         n_valid,
            "n_samples":       n_samples,
            "n_boot":          n_boot,
            "C_ref":           float(C_ref_global),
        }
        if return_samples:
            out["Cin_samples"] = Cin_v
            out["pi_samples"]  = pi_v
        return out


    def indoor_box_model_validated(self, J_surface, Cin0=50.0, t_end=86400.0,
                                    n_mc=500, seed=43, zone="basement",
                                    use_progeny=True):
        import math

        if zone == "basement":
            V    = self.p.get("V_bs", 50.0)
            ACH  = self.p.get("ACH_closed", 0.10)
        else:
            V    = self.p.get("V_gt", 120.0)
            ACH  = self.p.get("ACH_closed", 0.10) * 0.5

        A_c      = self.p.get("A_crack_m2", 0.05)
        lam_rn   = self.lam_rn
        lam_dep  = self.p.get("dep_bs" if zone=="basement" else "dep_gt", 0.0)
        lam_v    = ACH / 3600.0
        lam_eff  = lam_rn + lam_v + lam_dep

        J_eff  = float(J_surface) * A_c / max(V, 1e-3)
        C_ss   = J_eff / max(lam_eff, 1e-30)
        tau    = 1.0 / max(lam_eff, 1e-30)
        t_90   = math.log(10.0) / max(lam_eff, 1e-30)

        n_t    = 200
        t_arr  = np.linspace(0.0, float(t_end), n_t)
        Cin_t  = C_ss + (float(Cin0) - C_ss) * np.exp(-lam_eff * t_arr)

        rng_mc = np.random.default_rng(seed)
        ACH_mc  = rng_mc.lognormal(math.log(ACH),  0.30, n_mc)
        V_mc    = rng_mc.normal(V,   V*0.10, n_mc)
        dep_mc  = rng_mc.lognormal(math.log(max(lam_dep, 1e-7)*3600.0), 0.40, n_mc) / 3600.0
        V_mc    = np.maximum(V_mc, 1.0)

        lam_eff_mc = lam_rn + ACH_mc/3600.0 + dep_mc
        C_ss_mc    = J_eff * V / np.maximum(V_mc * lam_eff_mc, 1e-30)
        Cin_end_mc = C_ss_mc + (float(Cin0) - C_ss_mc) * np.exp(-lam_eff_mc * t_end)

        lam_Po218 = DECAY_CONSTANTS["Po218"]
        lam_Pb214 = DECAY_CONSTANTS["Pb214"]
        lam_Bi214 = DECAY_CONSTANTS["Bi214"]
        lam_Po214 = DECAY_CONSTANTS.get("Po214", 4.23e3)
        lam_dep_p = self.p.get("dep_progeny", 1.5/3600.0)
        lam_eff_p = lam_v + lam_dep_p

        Crn_ss    = float(C_ss)
        C_Po218   = Crn_ss * lam_Po218 / max(lam_Po218 + lam_eff_p, 1e-30)
        C_Pb214   = C_Po218 * lam_Pb214 / max(lam_Pb214 + lam_eff_p, 1e-30)
        C_Bi214   = C_Pb214 * lam_Bi214 / max(lam_Bi214 + lam_eff_p, 1e-30)
        C_Po214   = C_Bi214 * lam_Po214 / max(lam_Po214 + lam_eff_p, 1e-30)

        F_eq = ((0.105 * C_Po218 + 0.516 * C_Pb214 + 0.379 * C_Bi214)
                / max(Crn_ss, 1e-6))

        T_occ      = 7000.0
        dose_mSv_yr= Crn_ss * F_eq * 9e-6 * T_occ

        return {
            "C_in_t":      Cin_t,
            "t_arr":       t_arr,
            "C_ss":        float(C_ss),
            "lambda_eff":  float(lam_eff),
            "tau":         float(tau),
            "t_90":        float(t_90),
            "C_mc_p5":     float(np.percentile(Cin_end_mc, 5)),
            "C_mc_p95":    float(np.percentile(Cin_end_mc, 95)),
            "C_mc_mean":   float(np.mean(Cin_end_mc)),
            "C_mc_std":    float(np.std(Cin_end_mc)),
            "progeny": {
                "C_Po218": float(C_Po218),
                "C_Pb214": float(C_Pb214),
                "C_Bi214": float(C_Bi214),
                "C_Po214": float(C_Po214),
            },
            "F_eq":        float(F_eq),
            "dose_mSv_yr": float(dose_mSv_yr),
            "zone":        zone,
            "V":           float(V),
            "ACH":         float(ACH),
            "A_crack_m2":  float(A_c),
        }


    def sensitivity_screen(self, theta_mean=None, delta_frac=0.05,
                            n_morris=10, seed=77):
        import math

        if theta_mean is None:
            theta_mean = 0.35 * self.p["phi_m"]

        def compute_Cbs_fast(model):
            nz    = model.nz
            th    = np.full(nz, theta_mean)
            T_arr = model.T_field(0.0)
            Ar0   = np.full(nz, 0.75)
            eps0  = np.full(nz, model.p.get("eps_max", 0.3) * 0.6)
            S222  = model.source_rn222(th, T_arr, Ar0, eps0)
            Cm_eq = S222 / max(model.lam_rn, 1e-30)
            _cm_ss, _cf_ss = model._deep_equilibrium_conc(
                S222, th, np.full(nz, model.p.get("P_atm0", 101325.0)),
                T_arr, model.lam_rn)
            Cf_eq = Cm_eq * (np.asarray(_cf_ss)
                             / np.maximum(np.asarray(_cm_ss), 1e-30))
            Pg    = np.full(nz, model.p.get("P_atm0", 101325.0))
            Pg[-1]+= model.p.get("dP_init", 5.0)
            Pg_eff = math.sqrt(Pg[0] * Pg[-1])
            Pin    = model.p.get("P_atm0", 101325.0) - 1.5
            T0s    = float(model.T_field(0.0)[0])
            Tin    = model.p.get("T_in0", 293.15)
            Cf_eff = 0.3*float(Cf_eq[0]) + 0.7*float(Cf_eq[-1])
            J_c    = model.crack_network_flux(Cf_eff, 0.0, Pg_eff, Pin, theta_mean, T0s, Tin)
            J_f    = model.foundation_2d_network_flux(Cf_eff, 0.0, Pg_eff, Pin, theta_mean)
            V_bs   = model.p.get("V_bs", 50.0)
            ach    = model.p.get("ACH_closed", 0.1) / 3600.0
            lam_v  = model.lam_rn + ach
            return (J_c + J_f) / max(V_bs * lam_v, 1e-30)

        params_key = [
            "k0", "Ra226_Bqkg", "rho_bulk", "phi_m", "phi_f",
            "dP_init", "ACH_closed", "H", "T_grad",
            "eps_max", "grain_size_mm", "A_crack_m2"
        ]
        params_key = [p for p in params_key if p in self.p]

        C0 = compute_Cbs_fast(self)
        if C0 < 1e-8:
            return {"error": "C_bs nominale troppo basso", "C_nominal": float(C0)}

        n_evals = 0

        elasticities = {}
        for pk in params_key:
            p0 = float(self.p[pk])
            if abs(p0) < 1e-30:
                continue
            try:
                self.p[pk] = p0 * (1.0 + delta_frac)
                C_plus  = compute_Cbs_fast(self); n_evals += 1
                self.p[pk] = p0 * (1.0 - delta_frac)
                C_minus = compute_Cbs_fast(self); n_evals += 1
                elasticities[pk] = (C_plus - C_minus) / (2.0 * delta_frac * C0)
            except Exception:
                elasticities[pk] = 0.0
            finally:
                self.p[pk] = p0

        rng_m = np.random.default_rng(seed)
        morris_EE = {pk: [] for pk in params_key}

        Delta_frac = 0.10

        for _ in range(n_morris):
            p_base = {pk: float(self.p[pk]) for pk in params_key}
            perm = rng_m.permutation(len(params_key)).tolist()
            p_curr = p_base.copy()

            for pk in params_key:
                self.p[pk] = p_curr[pk]
            try:
                C_curr = compute_Cbs_fast(self); n_evals += 1
            except Exception:
                for pk in params_key:
                    self.p[pk] = p_base[pk]
                continue

            for idx in perm:
                pk = params_key[idx]
                p_orig = p_curr[pk]
                direction = rng_m.choice([-1.0, 1.0])
                p_new  = p_orig * (1.0 + direction * Delta_frac)
                p_curr[pk] = p_new
                for k2 in params_key:
                    self.p[k2] = p_curr[k2]
                try:
                    C_new = compute_Cbs_fast(self); n_evals += 1
                    EE_i  = (C_new - C_curr) / (direction * Delta_frac * max(abs(p_orig), 1e-30))
                    morris_EE[pk].append(EE_i)
                    C_curr = C_new
                except Exception:
                    p_curr[pk] = p_orig
                    for k2 in params_key:
                        self.p[k2] = p_base[k2]
                    break

            for pk in params_key:
                self.p[pk] = p_base[pk]

        morris_mu_star = {}
        morris_sigma   = {}
        for pk in params_key:
            ee = np.asarray(morris_EE.get(pk, [0.0]))
            morris_mu_star[pk] = float(np.mean(np.abs(ee)))
            morris_sigma[pk]   = float(np.std(ee))

        mu_max = max(v for v in morris_mu_star.values()) or 1.0
        score  = {}
        for pk in params_key:
            eps_i  = abs(elasticities.get(pk, 0.0))
            mu_i   = morris_mu_star.get(pk, 0.0) / mu_max
            score[pk] = 0.6 * eps_i + 0.4 * mu_i

        ranking = sorted(
            [(pk, elasticities.get(pk, 0.0),
              morris_mu_star.get(pk, 0.0),
              morris_sigma.get(pk, 0.0),
              score.get(pk, 0.0))
             for pk in params_key],
            key=lambda x: x[4], reverse=True
        )

        return {
            "elasticities":   elasticities,
            "morris_mu_star": morris_mu_star,
            "morris_sigma":   morris_sigma,
            "score":          score,
            "ranking":        ranking,
            "C_nominal":      float(C0),
            "n_evals":        n_evals,
        }


    def co2_rn_cotransport(self, Cm, x_CO2_profile, Pg, theta, T_z):
        Cm    = np.asarray(Cm,            dtype=float).copy()
        xCO2  = np.asarray(x_CO2_profile, dtype=float).copy()
        Pg_   = np.asarray(Pg,            dtype=float)
        theta_ = np.asarray(theta,        dtype=float)
        T_z_  = np.asarray(T_z,           dtype=float)

        nz    = self.nz
        dz    = self.dz
        phi_m = self.p["phi_m"]
        phi_f = self.p.get("phi_f", 0.05)
        phi_t = phi_m + phi_f
        D0    = self.p.get("D0", 1.1e-5)
        k0    = self.p.get("k0", 1e-13)
        mu    = self.p.get("mu", 1.82e-5)
        H     = self.p.get("H", 3.0)

        Sg_arr  = np.clip((phi_t - theta_) / np.maximum(phi_t, 1e-12), 0.0, 1.0)
        phi_g   = phi_t * Sg_arr

        T_ref = 293.15; P_ref = 101325.0
        D_local = D0 * (T_z_ / T_ref)**1.75 * (P_ref / np.maximum(Pg_, 1.0))
        phi_g_eff = np.maximum(phi_g, 1e-4)
        D_eff = D_local * phi_g_eff**(10./3.) / np.maximum(phi_t**2, 1e-12)

        v_gas = np.zeros(nz)
        for i in range(1, nz - 1):
            dPdz = (Pg_[i+1] - Pg_[i-1]) / (2.0 * dz)
            v_gas[i] = -k0 / mu * dPdz
        v_gas[0] = v_gas[1]; v_gas[nz-1] = v_gas[nz-2]

        u_carrier = v_gas * xCO2

        J_diff = np.zeros(nz)
        for i in range(1, nz - 1):
            dCdz = (Cm[i+1] - Cm[i-1]) / (2.0 * dz)
            J_diff[i] = -D_eff[i] * phi_g_eff[i] * dCdz
        J_diff[0]    = -D_eff[0] * phi_g_eff[0] * (Cm[0] - 0.0) / dz
        J_diff[nz-1] = J_diff[nz-2]

        J_carrier = u_carrier * Cm * phi_g_eff

        C_eff = Cm * np.maximum(1.0 - xCO2, 0.0)

        u_mean = float(np.mean(np.abs(u_carrier)))
        D_mean = float(np.mean(D_eff))
        Pe_CO2 = u_mean * H / max(D_mean, 1e-30)

        J_total = J_diff + J_carrier
        enh_arr = np.abs(J_total) / np.maximum(np.abs(J_diff), 1e-30)
        enhancement = float(np.mean(enh_arr))

        return {
            "J_carrier":   J_carrier,
            "J_diffusion": J_diff,
            "C_effective": C_eff,
            "enhancement": enhancement,
            "Pe_CO2":      Pe_CO2,
            "D_eff":       D_eff,
            "u_carrier":   u_carrier,
        }

    def bifurcation_dP(self, dP_array=None, n_points=40, theta_sw=None,
                       T_mean=None):
        bif = self.bifurcation_analysis(
            dP_array=dP_array,
            n_points=n_points,
            theta_sw=theta_sw,
            T_mean=T_mean,
            return_regime_map=False,
        )
        bif["Cin"] = bif["C_in"]
        return bif

    def buckingham_pi_extended(self, n_samples=512, seed=42):
        from scipy.stats import qmc

        rng = np.random.default_rng(int(seed))

        log_lo = np.array([-18., -1., -2.0, -2.5, -3.5, -1.5, -3., -7.])
        log_hi = np.array([-11.,  3., -0.3, -0.7, -1.0, -0.3,  1., -3.])

        sampler  = qmc.LatinHypercube(d=8, seed=int(seed))
        unit_lhs = sampler.random(n=n_samples)
        log_samp = qmc.scale(unit_lhs, log_lo, log_hi)
        raw      = 10.0 ** log_samp

        k0_s    = raw[:, 0]; dP_s  = raw[:, 1]; theta_s = raw[:, 2]
        phi_m_s = raw[:, 3]; phi_f_s = raw[:, 4]; eps_s = raw[:, 5]
        ach_s   = raw[:, 6]; sCO2_s  = raw[:, 7]

        D0    = self.p.get("D0", 1.1e-5)
        D_CO2 = self.p.get("D0_CO2", 1.6e-5)
        mu    = self.p.get("mu", 1.82e-5)
        H     = self.p.get("H", 3.0)
        lam   = self.lam_rn
        V_bs  = self.p.get("V_bs", 50.0)
        C_ref = 100.0

        phi_t = phi_m_s + phi_f_s
        Sg    = np.clip((phi_t - theta_s) / np.maximum(phi_t, 1e-12), 0.0, 1.0)
        phi_g = phi_t * Sg
        D_eff = D0 * phi_g**(10./3.) / np.maximum(phi_t**2, 1e-12)
        u_adv = k0_s / mu * dP_s / H
        C_ref_soil = 1e4

        pi1 = u_adv * H / np.maximum(D_eff, 1e-30)
        pi2 = lam * H**2 / np.maximum(D_eff, 1e-30)
        pi3 = theta_s / np.maximum(phi_m_s, 1e-12)
        pi4 = k0_s * dP_s / (mu * max(D0, 1e-30))
        pi5 = lam * V_bs / np.maximum(ach_s * V_bs, 1e-30)
        pi6 = H * np.sqrt(k0_s / mu) * np.sqrt(np.maximum(dP_s * 1.2, 0.0)) / mu
        pi7 = eps_s * phi_m_s / np.maximum(phi_f_s, 1e-12)
        pi8 = sCO2_s * H**2 / max(D_CO2 * C_ref_soil, 1e-30)

        pi_mat   = np.column_stack([pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8])
        pi_names = ["Pe_H", "Da_H", "theta/phi_m", "k0*dP/mu*D0",
                    "lambda*V/ACH", "Re_p", "eps/phi_f", "pi_CO2"]

        S_rn  = (eps_s * phi_m_s
                 * self.p.get("rho_bulk", 1600.0)
                 * self.p.get("Ra226_Bqkg", 20.0)
                 * lam)
        Cm_eq = S_rn / max(lam, 1e-30)
        J_diff_s = D_eff * Cm_eq / H
        J_adv_s  = u_adv * Cm_eq
        A_bs     = self.p.get("A_bs", 50.0)
        lam_v_s  = ach_s / 3600.0
        lam_tot_s = lam + lam_v_s
        C_in_s   = (J_diff_s + J_adv_s) * A_bs / np.maximum(lam_tot_s * V_bs, 1e-12)
        C_in_s   = np.clip(C_in_s, 1.0, 1e6)

        eps_safe = 1e-30
        Y       = np.log10(C_in_s / C_ref)
        Pi_log  = np.log10(np.maximum(np.abs(pi_mat), eps_safe))
        X       = np.column_stack([np.ones(n_samples), Pi_log])
        try:
            beta_all = np.linalg.solve(X.T @ X, X.T @ Y)
        except np.linalg.LinAlgError:
            beta_all = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta0 = float(beta_all[0]); beta = beta_all[1:]
        Y_hat = X @ beta_all
        ss_res = np.sum((Y - Y_hat)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        R2 = 1.0 - ss_res / max(ss_tot, 1e-30)

        n_boot = 200
        beta_boot = np.zeros((n_boot, 8))
        rng_b = np.random.default_rng(int(seed) + 1000)
        for b in range(n_boot):
            idx_b = rng_b.integers(0, n_samples, size=n_samples)
            Xb = X[idx_b]; Yb = Y[idx_b]
            try:
                bb = np.linalg.solve(Xb.T @ Xb, Xb.T @ Yb)
            except np.linalg.LinAlgError:
                bb = np.linalg.lstsq(Xb, Yb, rcond=None)[0]
            beta_boot[b] = bb[1:]
        beta_ci_lo = np.percentile(beta_boot, 2.5,  axis=0)
        beta_ci_hi = np.percentile(beta_boot, 97.5, axis=0)

        terms = [f"pi_{i+1}^{{{b:.3f}}}" for i, b in enumerate(beta)]
        law = (f"C_in/C_ref = 10^{{{beta0:.3f}}} * " + " * ".join(terms))

        pi_stats = {
            pi_names[i]: {
                "mean":   float(np.mean(pi_mat[:, i])),
                "median": float(np.median(pi_mat[:, i])),
                "p5":     float(np.percentile(pi_mat[:, i], 5)),
                "p95":    float(np.percentile(pi_mat[:, i], 95)),
            }
            for i in range(8)
        }

        return {
            "pi_names":     pi_names,
            "samples":      pi_mat,
            "C_in_samples": C_in_s,
            "beta":         beta,
            "beta0":        beta0,
            "beta_ci_lo":   beta_ci_lo,
            "beta_ci_hi":   beta_ci_hi,
            "R2":           float(R2),
            "law":          law,
            "pi_stats":     pi_stats,
            "n_samples":    n_samples,
        }

    def buckingham_pi_extended_ode(self, n_samples=40, seed=42, nz=15,
                                    days=1.5, lits=None):
        import warnings
        if lits is None:
            lits = list(GEOLITHOTYPES.keys())
        rng = np.random.default_rng(int(seed))

        X_ode = []; Y_ode = []; lits_run = []
        pi_names = ["Pe_H", "DaAdv", "Pi_src", "Pi_vent",
                    "Pi_frac", "theta_r", "eps_phi"]

        for _ in range(n_samples):
            lit_name = rng.choice(lits)
            lit_data = GEOLITHOTYPES[lit_name]
            Ra_base  = lit_data.get("Ra226_Bqkg", 40.0)
            override = {
                "nz":         nz,
                "dP_init":    float(rng.uniform(0.5, 25.0)),
                "Ra226_Bqkg": float(rng.uniform(Ra_base * 0.4,
                                                 Ra_base * 2.5)),
                "ACH_closed": float(rng.uniform(0.05, 0.5)),
            }
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    m   = build_model(lit_name, override=override)
                    sol = m.simulate_fast(days=days, dt_out=3600.0)
                out   = m.extract_results(sol)
                t_arr = np.array(out["t"])
                Cbs   = np.array(out["C_bs"])
                mask  = t_arr > t_arr[-1] - 4*3600
                C_in  = float(np.nanmean(Cbs[mask]))
                if C_in > 0:
                    theta_f = np.array(out["theta"])[:, -1]
                    Pg_f    = np.array(out["Pg"])[:, -1]
                    T_f     = m.T_field(sol.t[-1])
                    pi      = m.buckingham_pi(theta_f, Pg_f, T_f)
                    phi_tot = m.p["phi_m"] + m.p.get("phi_f", 0.0)
                    eps_phi = (m.p.get("eps_max", 0.2) * m.p["phi_m"]
                               / max(m.p.get("phi_f", 1e-4), 1e-12))
                    Sw_f    = float(np.mean(theta_f / max(m.p["phi_m"], 1e-12)))
                    row = [pi["Pe_H"], pi["DaAdv"],
                           pi["Pi_src"], pi["Pi_vent"], pi["Pi_frac"],
                           Sw_f, eps_phi]
                    X_ode.append(row)
                    Y_ode.append(C_in)
                    lits_run.append(lit_name)
            except Exception:
                pass

        n_conv = len(Y_ode)
        if n_conv < 6:
            return {"error": f"Solo {n_conv} simulazioni convergenti (<6)"}

        X_arr  = np.array(X_ode)
        Y_arr  = np.array(Y_ode)
        eps_   = 1e-30
        logY   = np.log10(np.maximum(Y_arr, eps_))
        logX   = np.log10(np.maximum(np.abs(X_arr), eps_))
        Xmat   = np.column_stack([np.ones(n_conv), logX])
        coef, _, _, _ = np.linalg.lstsq(Xmat, logY, rcond=None)
        Yhat   = Xmat @ coef
        R2     = float(1.0 - np.sum((logY - Yhat)**2)
                       / max(np.sum((logY - logY.mean())**2), eps_))
        terms  = " * ".join([f"{pi_names[i]}^{{{coef[i+1]:.3f}}}"
                              for i in range(len(pi_names))])
        law    = f"C_in/Cref = 10^{coef[0]:.3f} * " + terms

        return {
            "pi_names":    pi_names,
            "C_in_ode":    Y_arr,
            "lits_run":    lits_run,
            "n_converged": n_conv,
            "coef":        coef,
            "R2":          R2,
            "law":         law,
        }

    def q2_leave_one_out(self, bench=None, nz=15, days=1.5, verbose=True):
        import warnings

        lits_all = list(GEOLITHOTYPES.keys())
        N = len(lits_all)

        if bench is None:
            if verbose:
                print("  Q2_LOO: esecuzione benchmark su tutti i 20 litotipi...")
            bench = run_lithology_benchmark(nz=nz, days=days, verbose=verbose)

        Pi_all = []
        Cin_all = []
        valid_lits = []

        for lit_name in lits_all:
            r = bench.get(lit_name, {})
            if "error" in r or not np.isfinite(r.get("Cin_Bq_m3", float("nan"))):
                if verbose:
                    print(f"  [SKIP] {lit_name}: benchmark fallito")
                continue
            Pe   = max(r.get("Pe_H", 1e-10), 1e-10)
            Da   = max(r.get("Da_H", 1e-10), 1e-10)
            Cin  = max(r.get("Cin_Bq_m3", 0.0), 0.1)
            Pi_all.append([Pe, Da])
            Cin_all.append(Cin)
            valid_lits.append(lit_name)

        N_v = len(valid_lits)
        if N_v < 4:
            return {"Q2_LOO": float("nan"),
                    "error": f"Solo {N_v} litotipi validi (<4)"}

        eps_  = 1e-30
        logX  = np.log10(np.maximum(np.array(Pi_all), eps_))
        logY  = np.log10(np.array(Cin_all))

        Ypred_log = np.zeros(N_v)
        for i in range(N_v):
            mask_tr  = [j for j in range(N_v) if j != i]
            Xtr      = np.column_stack([np.ones(len(mask_tr)), logX[mask_tr]])
            Ytr      = logY[mask_tr]
            coef, _, _, _ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
            Xte          = np.hstack([[1.0], logX[i]])
            Ypred_log[i] = float(Xte @ coef)

        residuals  = logY - Ypred_log
        SS_pred    = float(np.sum(residuals**2))
        SS_tot     = float(np.sum((logY - np.mean(logY))**2))
        Q2_LOO     = float(1.0 - SS_pred / max(SS_tot, eps_))

        Cin_pred = 10.0**Ypred_log

        if Q2_LOO >= 0.70:
            interp = "ECCELLENTE: il modello predice litologie mai viste"
        elif Q2_LOO >= 0.50:
            interp = "ACCETTABILE: generalizzazione parziale"
        else:
            interp = "INSUFFICIENTE: possibile overfitting o dati insufficienti"

        summary = (
            "Q2_LOO = {:.3f} | {}\n"
            "N litotipi = {} | RMSE_log10 = {:.3f}\n"
            "Range C_in osservato: [{:.1f}, {:.1f}] Bq/m3"
        ).format(
            Q2_LOO, interp, N_v, float(np.sqrt(SS_pred / N_v)),
            min(Cin_all), max(Cin_all)
        )
        if verbose:
            for _line in summary.split("\n"):
                print("  " + _line)

        return {
            "Q2_LOO":        Q2_LOO,
            "N":             N_v,
            "lits":          valid_lits,
            "C_in_obs":      np.array(Cin_all),
            "C_in_pred":     Cin_pred,
            "residuals_log": residuals,
            "RMSE_log10":    float(np.sqrt(SS_pred / N_v)),
            "summary":       summary,
        }

    def indoor_box_model(self, J_surface, Cin0=50.0, dt=3600.0,
                         zone="basement", use_progeny=True):
        zone_params = {
            "basement": {"V": self.p.get("V_bs", 50.0),
                         "A": self.p.get("A_bs", 50.0),  "ach_mult": 1.0},
            "ground":   {"V": self.p.get("V_gt", 150.0),
                         "A": self.p.get("A_bs", 50.0)*0.3, "ach_mult": 1.5},
            "upper":    {"V": 200.0,
                         "A": self.p.get("A_bs", 50.0)*0.1, "ach_mult": 2.5},
        }
        zp      = zone_params.get(zone, zone_params["basement"])
        V       = zp["V"];   A = zp["A"]
        ACH     = self.p.get("ACH_closed", 0.10) * zp["ach_mult"]
        lam_v_s = ACH / 3600.0
        lam_tot = self.lam_rn + lam_v_s

        J_eff = float(J_surface) * A
        C_ss  = J_eff / (lam_tot * V)
        C_t   = C_ss + (float(Cin0) - C_ss) * np.exp(-lam_tot * float(dt))
        tau_eq = 1.0 / lam_tot

        progeny = {"C_Po218": 0.0, "C_Pb214": 0.0,
                   "C_Bi214": 0.0, "C_Po214": 0.0}
        if use_progeny:
            lam_Po218 = DECAY_CONSTANTS["Po218"]
            lam_Pb214 = DECAY_CONSTANTS["Pb214"]
            lam_Bi214 = DECAY_CONSTANTS["Bi214"]
            lam_Po214 = DECAY_CONSTANTS["Po214"]
            lam_eff_p = lam_v_s + self.p.get("dep_progeny", 1.5/3600.0)
            C_Po218_ss = C_ss * lam_Po218 / (lam_Po218 + lam_eff_p)
            C_Pb214_ss = C_Po218_ss * lam_Pb214 / (lam_Pb214 + lam_eff_p)
            C_Bi214_ss = C_Pb214_ss * lam_Bi214 / (lam_Bi214 + lam_eff_p)
            C_Po214_ss = C_Bi214_ss * lam_Po214 / (lam_Po214 + lam_eff_p)
            progeny = {"C_Po218": float(C_Po218_ss), "C_Pb214": float(C_Pb214_ss),
                       "C_Bi214": float(C_Bi214_ss), "C_Po214": float(C_Po214_ss)}

        if use_progeny and C_ss > 0:
            F_eq = self.equilibrium_factor(
                C_ss, progeny["C_Po218"], progeny["C_Pb214"],
                progeny["C_Bi214"], progeny["C_Po214"])
        else:
            F_eq = 0.4

        f_occ = 0.80; hours_y = 8760.0
        dose_mSv_y = C_ss * F_eq * 9.0e-6 * f_occ * hours_y

        return {
            "C_ss": float(C_ss), "C_t": float(C_t), "C0": float(Cin0),
            "tau_eq": float(tau_eq), "J_eff": float(J_eff), "ACH": float(ACH),
            "F_eq": float(F_eq), "dose_mSv_y": float(dose_mSv_y),
            "zone": zone, "progeny": progeny,
        }

    def sensitivity_indices_analytical(self, theta_mean=None, delta_frac=0.05):
        if theta_mean is None:
            theta_mean = 0.35 * self.p["phi_m"]

        if self.p.get("sensitivity_param_names_fixed", True):
            params_key = ["k0", "phi_m", "eps_max", "Ra226_Bqkg", "rho_bulk",
                          "D0", "ACH_closed", "H", "phi_f", "T_ref"]
        else:
            params_key = ["k0", "phi_m", "eps_max", "Ra_soil", "rho_s",
                          "D0", "lam_vent", "H", "phi_f", "T_ref"]
        p_base = {pk: self.p[pk] for pk in params_key if pk in self.p}

        nz = self.nz
        T_arr = np.full(nz, self.p.get("T_ref", 293.15))
        theta = np.full(nz, theta_mean)
        Ar0   = np.full(nz, 0.75)
        eps0  = np.full(nz, self.p.get("eps_max", 0.3) * 0.6)

        def _C_fast():
            eps0_loc = (np.full(nz, self.p.get("eps_max", 0.3) * 0.6)
                        if self.p.get("sensitivity_param_names_fixed", True) else eps0)
            S222   = self.source_rn222(theta, T_arr, Ar0, eps0_loc)
            Cm_eq  = S222 / max(self.lam_rn, 1e-30)
            phi_t  = self.p["phi_m"] + self.p.get("phi_f", 0.05)
            Sg     = max((phi_t - theta_mean) / max(phi_t, 1e-12), 0.0)
            phi_g  = phi_t * Sg
            D_eff  = self.p.get("D0", 1.1e-5) * phi_g**(10./3.) / max(phi_t**2, 1e-12)
            k0n    = self.p.get("k0", 1e-13)
            mu_n   = self.p.get("mu", 1.82e-5)
            H_n    = self.p.get("H", 3.0)
            dP_nom = self.p.get("dP_bottom", 5.0)
            u_adv  = k0n / mu_n * dP_nom / H_n
            _th_f_sf = np.clip(self.p.get("gamma_sw_f", 0.2)
                               * (theta / max(self.p["phi_m"], 1e-12))
                               * self.p.get("phi_f", 0.05),
                               1e-12, 0.9999*max(self.p.get("phi_f", 0.05), 1e-12))
            _alpha_sf = np.atleast_1d(self._mf_alpha(theta, _th_f_sf,
                                                     np.full(nz, 101325.0),
                                                     np.full(nz, 101325.0)))
            _bf_sf    = np.atleast_1d(self._beta_partition(_th_f_sf, T_arr, "f"))
            _num_sf   = _alpha_sf * self.p.get("eta_inst", 1.0)
            Cf_eq  = Cm_eq * np.clip(_num_sf / np.maximum(
                _num_sf + self.lam_rn*_bf_sf, 1e-30), 0.0, 1.0)
            J_diff = D_eff * float(np.mean(Cm_eq)) / H_n
            J_adv  = u_adv * float(np.mean(Cf_eq))
            A_bs   = self.p.get("A_bs", 50.0); V_bs = self.p.get("V_bs", 50.0)
            lam_v  = self.p.get("ACH_closed", 0.10) / 3600.0
            return (J_diff + J_adv) * A_bs / max((self.lam_rn + lam_v) * V_bs, 1e-30)

        C0 = _C_fast(); n_evals = 1
        elasticities = {}

        for pk in params_key:
            if pk not in self.p: continue
            p_orig = self.p[pk]
            try:
                self.p[pk] = p_orig * (1.0 + delta_frac)
                C_plus  = _C_fast(); n_evals += 1
                self.p[pk] = p_orig * (1.0 - delta_frac)
                C_minus = _C_fast(); n_evals += 1
            except Exception:
                elasticities[pk] = 0.0
                self.p[pk] = p_orig
                continue
            self.p[pk] = p_orig
            elasticities[pk] = (C_plus - C_minus) / (2.0 * delta_frac * max(abs(C0), 1e-30))

        for pk in p_base:
            self.p[pk] = p_base[pk]

        ranking = sorted(
            [(pk, elasticities.get(pk, 0.0)) for pk in params_key],
            key=lambda x: abs(x[1]), reverse=True
        )
        return {
            "elasticities": elasticities,
            "C_nominal":    float(C0),
            "ranking":      ranking,
            "n_evals":      n_evals,
        }


def build_model(lithotype="tuff_unwelded", override=None):
    if lithotype in GEOLITHOTYPES:
        lit = GEOLITHOTYPES[lithotype].copy()
    elif lithotype in VOLCANIC_LITHOTYPES:
        lit = VOLCANIC_LITHOTYPES[lithotype].copy()
    else:
        raise KeyError(
            f"Litotipo '{lithotype}' non trovato. Disponibili: "
            + str(list(GEOLITHOTYPES.keys()))
        )

    texture = lit.get("texture", None)
    if texture and texture in VG_TEXTURE_PARAMS:
        vg = VG_TEXTURE_PARAMS[texture]
        lit.setdefault("soil_alpha", vg[0])
        lit.setdefault("soil_n",     vg[1])
        lit.setdefault("theta_r",    vg[2])
        lit.setdefault("phi_m",      vg[3])
        lit.setdefault("Ksat0",      vg[4])
    p = {
        "H":             3.0,
        "nz":            50,
        "phi_m":         lit["phi_m"],
        "phi_f":         lit["phi_f"],
        "theta_r":       lit.get("theta_r", 0.02),
        "A_bs":          50.0 / 2.5,
        "k0":            lit["k0"],
        "k_frac":        lit["k_frac"],
        "Ksat0":         lit["Ksat0"],
        "Ksat_grad":     0.2,
        "soil_alpha":    lit["soil_alpha"],
        "soil_n":        lit["soil_n"],
        "Sg_crit":       0.08,
        "Sg_sharp":      0.02,
        "nk":            3.5,
        "kz_grad":       0.0,
        "k_min":         1e-18,
        "k_max":         1e-10,
        "mu":            1.82e-5,
        "M_g":           0.02897,
        "R_g":           8.314,
        "g":             9.81,
        "rho_air":       1.20,
        "P_atm0":        101325.0,
        "P_atm_amp":     150.0,
        "T_atm":         86400.0,
        "dP_init":       5.0,
        "dP_bottom":     5.0,
        "T0":            288.0,
        "T_grad":        0.025,
        "T_surf_amp":    8.0,
        "T_damp_depth":  0.12,
        "T_in0":         293.15,
        "T_in_amp":      2.0,
        "T_out0":        283.15,
        "T_out_amp":     5.0,
        "T_ref":         293.15,
        "seismic_recovery_on":  False,
        "M_w_unrest":           0.0,
        "seis_t0_s":            0.0,
        "k_seis_floor":         1.0,
        "moisture_emanation_on":   False,
        "Sw_peak_moist":           0.17,
        "Sw_sigma_moist":          0.12,
        "f_emanation_dry":         0.45,
        "Sw_block_moist":          0.88,
        "Sw_block_sharp_moist":    0.04,
        "moisture_diffusivity_on": False,
        "Sw_perc_gas":             0.95,
        "diff_choke_exp":          3.0,
        "diff_choke_min":          1e-4,
        "T_out_year_amp":   0.0,
        "T_out_year_phase": -np.pi/2,
        "T_in_year_amp":    0.0,
        "T_in_year_phase":  -np.pi/2,
        "T_year":           365.25*86400.0,
        "T_day":            86400.0,
        "Ra226_Bqkg":    lit["Ra226_Bqkg"],
        "Th232_Bqkg":    lit.get("Th232_Bqkg", lit["Ra226_Bqkg"] * 0.8),
        "rho_bulk":      lit["rho_bulk"],
        "D0":            lit["D0"],
        "Dmin":          1e-12,
        "Dmax":          1.1e-5,
        "nD":            2.5,
        "betaD":         2.0,
        "eps_max":       lit["eps_max"],
        "Sw_peak_eman":  lit["Sw_peak_eman"],
        "Sw_sigma_eman": 0.14,
        "eps_dry":       0.06,
        "Sw_block_eman": 0.90,
        "Sw_block_sharp":0.03,
        "vesicle_fraction": lit.get("vesicle_fraction", 0.0),
        "f_src_to_fracture": lit.get("f_src_to_fracture", 0.0),
        "P_entry":          500.0,
        "sigma_tot_frac":   3e5,
        "tau_cap":          1800.0,
        "D0_CO2":           1.6e-5,
        "S_CO2":            1e-5,
        "b0_fracture_mean": 2e-4,
        "beta_T_eman":   0.003,
        "grain_size_mm": lit["grain_size_mm"],
        "grain_ref_mm":  0.25,
        "grain_exp_eman":0.20,
        "eps_ratio_220_222": 1.0,
        "H_rn0":         0.23,
        "dH_rn":         2800.0,
        "A_k_open":      1.0/3600.0,
        "A_k_close":     0.35/3600.0,
        "A_ng":          2.0,
        "A_nw":          2.0,
        "A_beta_dry":    1.0e9,
        "A_dtheta_crit": 1e-8,
        "A_beta_frac":   1.5,
        "A_beta_rewet":  1e6,
        "A_beta_gradPg": 0.0,
        "A_gradPg_ref":  100.0,
        "a_mf0":         1.0,
        "d_mf":          0.50,
        "n_mf_sat":      2.0,
        "beta_mf_p":     1e-6,
        "mem_nu":        0.70,
        "mem_tau":       6*3600.0,
        "mem_alpha":     1.0/3600.0,
        "eta_inst":      1.0,
        "eta_mem":       0.5,
        "gamma_sw_f":    0.20,
        "k_ads_dry":     2e-6,
        "k_ads_wet":     5e-7,
        "k_des_dry":     2e-7,
        "k_des_wet":     8e-7,
        "n_cracks":      18,
        "crack_seed":    42,
        "n_frac_horiz":  20,
        "horiz_seed":    7,
        "Cd_crack":      0.61,
        "D_entry":       1e-6,
        "nD_entry":      2.4,
        "foundation_coupling": 0.35,
        "Sw_close":      0.85,
        "k_seis_max":    10.0,
        "seis_seed":     77,
        "V":             170.0,
        "V_bs":           50.0,
        "V_gt":          120.0,
        "H_stack":         2.5,
        "A_crack_m2":    lit["A_crack_m2"],
        "A_leak":        5e-3,
        "Cd_leak":       0.62,
        "C_wind":        0.60,
        "U_wind":        1.0,
        "ACH_closed":    0.10,
        "ACH_open":      1.20,
        "ACH_mech":      0.0,
        "lambda_bg":     0.2/3600.0,
        "dep_bs":        0.0,
        "dep_gt":        0.0,
        "dep_progeny":   1.5/3600.0,
        "rain0":         2e-7,
        "T_rain":        864000.0,
        "evap0":         1e-7,
        "T_evap":        86400.0,
        "evap_phase":    np.pi/2,
        "tau_hyst_dry":  2*3600.0,
        "tau_hyst_wet":  48*3600.0,
        "seismic_site":  lit["seismic_site"],
        "t_total_seis":  7*86400.0,
        "theta0":        0.35 * lit["phi_m"],
        "C0_m220":       0.0,
        "C0_f220":       0.0,
        "Cin0":          50.0,
        "Cin_gt0":       20.0,
        "C_ref":         100.0,
        "a_int0":        1200.0,
        "k_la0":         3.2e-5,
        "t_open1s":      7*3600,
        "t_open1e":      9*3600,
        "t_open2s":      19*3600,
        "t_open2e":      22*3600,
        "ACH_sharp":     300.0,
    }

    for geo_key in ("regime", "emanation_model", "eps_crystalline",
                    "D_matrix_crystalline", "k_frac_eff",
                    "fracture_density_factor", "eps_karst",
                    "f_karst_conduit", "D_conduit",
                    "k_karst_conduit", "f_karst_active"):
        if geo_key in lit:
            p.setdefault(geo_key, lit[geo_key])

    if override:
        p.update(override)
    return VolcanicRadonModelV2(p)


def run_validation():
    results = {}
    for name, lit in VOLCANIC_LITHOTYPES.items():
        print(f"  Simulating {name} ({lit['label']}) ...")
        m   = build_model(name)
        sol = m.simulate(days=3.0)
        if not sol.success:
            print(f"    [WARNING] solver failed: {sol.message}")
        res = m.extract_results(sol)
        t_i = -1
        theta_f = res["theta"][:, t_i]
        Pg_f    = res["Pg"][:, t_i]
        T_f     = m.T_field(sol.t[t_i])
        pi      = m.buckingham_pi(theta_f, Pg_f, T_f)
        C_bs_f  = float(res["C_bs"][t_i])
        Cf220_f = float(res["Cf220"][0, t_i]) if res["Cf220"].ndim == 2 else 0.0
        ratio_iso = Cf220_f / max(C_bs_f, 1.0)
        F_eq = m.equilibrium_factor(
            C_bs_f,
            float(res["C_Po218"][t_i]),
            float(res["C_Pb214"][t_i]),
            float(res["C_Bi214"][t_i]),
            float(res["C_Po214"][t_i]),
        )
        S_m_f = m.source_rn222(theta_f, T_f,
                                res["Ar"][:,t_i], res["eps_mem"][:,t_i])
        lv    = m.p.get("ACH_closed",0.1)/3600.0
        Cf0_v    = float(res["Cf"][0,  t_i])
        Cf_bot_v = float(res["Cf"][-1, t_i])
        Cf_eff_v = 0.3*Cf0_v + 0.7*Cf_bot_v
        Pg0_v    = float(Pg_f[0]); Pg_bot_v = float(Pg_f[-1])
        Pg_eff_v = float(np.sqrt(Pg0_v*Pg_bot_v)) if Pg_bot_v > Pg0_v else Pg0_v
        Pin_v    = float(res["Pin"][t_i])
        Tin_v    = float(m.T_indoor(sol.t[t_i]))
        J_c_v = m.crack_network_flux(Cf_eff_v, C_bs_f, Pg_eff_v, Pin_v,
                                     float(theta_f[0]), float(T_f[0]), Tin_v)
        J_f_v = m.foundation_2d_network_flux(Cf_eff_v, C_bs_f, Pg_eff_v, Pin_v,
                                             float(theta_f[0]))
        Jent  = float(J_c_v + J_f_v)
        mb_err= m.mass_balance_check(
            res["Cm"][:,t_i], res["Cf"][:,t_i],
            C_bs_f, float(res["C_gt"][t_i]),
            Jent, S_m_f, lv, sol.t[t_i], theta=theta_f
        )
        results[name] = {
            "sol":         sol,
            "res":         res,
            "pi":          pi,
            "F_eq":        F_eq,
            "ratio_iso":   ratio_iso,
            "mb_err":      mb_err,
            "C_bs_final":  C_bs_f,
            "Pin_final":   float(res["Pin"][t_i]),
            "eps_mem_mean":float(np.mean(res["eps_mem"][:,t_i])),
            "Ar_mean":     float(np.mean(res["Ar"][:,t_i])),
            "phi_tot":     m.p["phi_m"] + m.p.get("phi_f", 0.05),
        }
        print(f"    Pe_H={pi['Pe_H']:.3f}  C_bs={C_bs_f:.1f} Bq/m³  F_eq={F_eq:.3f}  mb_err={mb_err:.2e}")
    return results


def run_lithology_benchmark(lithotypes=None, days=2.0, nz=15, verbose=True):
    if lithotypes is None:
        lithotypes = list(GEOLITHOTYPES.keys())

    results = {}

    for lit_name in lithotypes:
        if lit_name not in GEOLITHOTYPES:
            if verbose:
                print(f"  [SKIP] {lit_name} non in GEOLITHOTYPES")
            continue

        lit_data = GEOLITHOTYPES[lit_name]
        if verbose:
            print(f"  Simulazione {lit_name} ({lit_data['label']}) ...")

        try:
            m = build_model(lit_name, override={"nz": nz})
            sol = m.simulate_fast(days=days, dt_out=3600.0)
            out = m.extract_results(sol)

            t   = np.array(out["t"])
            Cbs = np.array(out["C_bs"])
            Cgt = np.array(out["C_gt"])
            Ar  = np.array(out["Ar"])
            eps = np.array(out["eps_mem"])
            Pg  = np.array(out["Pg"])

            mask = t > t[-1] - 6*3600
            def m_last(a):
                a2 = np.asarray(a)
                if a2.ndim == 2:
                    return float(np.nanmean(a2[:, mask]))
                return float(np.nanmean(a2[mask]))

            Cbs_m   = m_last(Cbs)
            theta_f = np.array(out["theta"])[:, -1]
            Pg_f    = Pg[:, -1]
            T_f     = m.T_field(sol.t[-1])

            pi = m.buckingham_pi(theta_f, Pg_f, T_f)

            k_mat  = float(np.mean(m._k_gas_standard(theta_f, m.z)))
            k_tot  = float(np.mean(m._k_gas(theta_f, m.z)))
            Pi_frac = k_tot / max(k_mat, 1e-30)

            Cf0    = float(np.array(out["Cf"])[0,  -1])
            Cf_bot = float(np.array(out["Cf"])[-1, -1])
            Cf_eff = 0.3*Cf0 + 0.7*Cf_bot
            Pg0    = float(Pg_f[0]); Pg_b = float(Pg_f[-1])
            Pg_eff = np.sqrt(Pg0 * Pg_b)
            Pin    = m._Pin_steady(m._Patm(sol.t[-1]),
                                   m.p.get("T_in0", 293.15),
                                   m.p.get("T_out0", 283.15))
            J_c = m.crack_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                       float(theta_f[0]), float(T_f[0]),
                                       m.p.get("T_in0", 293.15))
            J_f = m.foundation_2d_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                               float(theta_f[0]))
            J_tot = J_c + J_f
            A_fp = m.p.get("A_footprint", m.p.get("V_bs", 50.0)/2.5)
            J_surface_val = J_tot / max(A_fp, 1e-12)

            regime_label = m.transport_regime(pi["Pe_H"], pi["DaDiff"])
            regime_geo   = m.p.get("regime", "porous_diffusive")

            C_bs_f  = Cbs_m
            Cf220_s = float(np.array(out["Cf220"])[0, -1]) if "Cf220" in out else 0.0
            ratio_iso = Cf220_s / max(C_bs_f, 1.0)

            F_eq = m.equilibrium_factor(
                C_bs_f,
                float(np.array(out["C_Po218"])[-1]),
                float(np.array(out["C_Pb214"])[-1]),
                float(np.array(out["C_Bi214"])[-1]),
                float(np.array(out["C_Po214"])[-1]),
            )

            eps_mean = m_last(eps)

            results[lit_name] = {
                "label":          lit_data["label"],
                "Cin_Bq_m3":      Cbs_m,
                "C_gt_Bq_m3":     m_last(Cgt),
                "Pe_H":           pi["Pe_H"],
                "Da_H":           pi["Da_H"],
                "DaDiff":         pi["DaDiff"],
                "Pi_frac":        Pi_frac,
                "regime_pd":      regime_label,
                "regime":         regime_geo,
                "J_surface":      J_surface_val,
                "J_entry_Bq_s":   J_tot,
                "phi_tot":        lit_data["phi_m"] + lit_data.get("phi_f", 0.05),
                "eps_mean":       eps_mean,
                "Ar_mean":        m_last(Ar),
                "F_eq":           F_eq,
                "ratio_iso":      ratio_iso,
                "Ra226_Bqkg":     lit_data["Ra226_Bqkg"],
                "k0":             lit_data["k0"],
                "phi_m":          lit_data["phi_m"],
                "emanation_model":lit_data.get("emanation_model", "volcanic_vesicular"),
            }

            if verbose:
                print(f"    Pe_H={pi['Pe_H']:.3f}  C_in={Cbs_m:.1f} Bq/m3  "
                      f"regime={regime_geo}  F_eq={F_eq:.3f}")

        except Exception as exc:
            if verbose:
                print(f"    [WARNING] {lit_name}: {exc}")
            results[lit_name] = {"label": lit_data["label"], "error": str(exc),
                                  "Cin_Bq_m3": float("nan")}

    results = dict(sorted(
        results.items(),
        key=lambda kv: (kv[1]["Cin_Bq_m3"] if np.isfinite(kv[1].get("Cin_Bq_m3", float("nan"))) else float("-inf")),
        reverse=True
    ))
    return results


def print_lithology_benchmark_table(bench):
    print()
    print(f"  {'Litotipo':<32} {'Regime':<22} "
          f"{'C_in':>9} {'Pe_H':>7} {'Da_H':>7} "
          f"{'Pi_frac':>9} {'F_eq':>7} {'eps':>7}")
    print("  " + "-"*100)
    for name, r in bench.items():
        if "error" in r:
            print(f"  {r['label']:<32} {'ERROR':22} {r.get('error','')}")
            continue
        print(f"  {r['label']:<32} {r['regime']:<22} "
              f"{r['Cin_Bq_m3']:9.1f} {r['Pe_H']:7.3f} {r['Da_H']:7.3f} "
              f"{r['Pi_frac']:9.2f} {r['F_eq']:7.3f} {r['eps_mean']:7.4f}")
    print()


def plot_lithology_benchmark(bench, save_path=None, dpi=150):
    REGIME_COLORS = {
        "geothermal_active":  "#e74c3c",
        "fracture_dominant":  "#8e44ad",
        "karst_conduit":      "#2980b9",
        "porous_advective":   "#27ae60",
        "porous_diffusive":   "#e67e22",
    }
    REGIME_LABELS = {
        "geothermal_active":  "Vulcanici attivi",
        "fracture_dominant":  "Rocce cristalline",
        "karst_conduit":      "Carsici",
        "porous_advective":   "Sabbie/Alluvioni",
        "porous_diffusive":   "Argille/Limi/Marne",
    }

    valid = {k: v for k, v in bench.items() if "error" not in v
             and np.isfinite(v.get("Cin_Bq_m3", float("nan")))}
    names    = list(valid.keys())
    labels   = [v["label"] for v in valid.values()]
    Cin      = np.array([v["Cin_Bq_m3"]   for v in valid.values()])
    Pe       = np.array([v["Pe_H"]         for v in valid.values()])
    Da       = np.array([v.get("DaDiff", v["Da_H"]) for v in valid.values()])
    k0_vals  = np.array([v["k0"]           for v in valid.values()])
    Ra_vals  = np.array([v["Ra226_Bqkg"]   for v in valid.values()])
    phi_vals = np.array([v["phi_m"]        for v in valid.values()])
    regimes  = [v["regime"]               for v in valid.values()]
    colors   = [REGIME_COLORS.get(r, "#95a5a6") for r in regimes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    ax = axes[0, 0]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, np.maximum(Cin, 0.1), color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, color="white")
    ax.set_xscale("log")
    ax.set_xlabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("A — Concentrazione indoor per litotipo", color="white",
                 fontsize=10, fontweight="bold")
    ax.axvline(75,  color="#f39c12", lw=1.2, ls="--", label="IT mean 75 Bq/m3")
    ax.axvline(100, color="#ffa657", lw=1.2, ls="--", label="WHO 100 Bq/m3 (ref. level)")
    ax.axvline(300, color="#e74c3c", lw=1.2, ls="--", label="IAEA 300 Bq/m3 (action level)")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white", loc="lower right")
    ax.grid(axis="x", alpha=0.2, color="white")

    ax = axes[0, 1]
    seen_regimes = set()
    for i, (pe, da, c, r) in enumerate(zip(Pe, Da, colors, regimes)):
        lbl = REGIME_LABELS.get(r, r) if r not in seen_regimes else None
        ax.scatter(pe, da, color=c, s=90, alpha=0.88, edgecolors="white",
                   linewidths=0.4, label=lbl, zorder=3)
        seen_regimes.add(r)
    ax.axvline(0.1,   color="#aaaaaa", lw=0.8, ls=":",  alpha=0.5)
    ax.axvline(1.0,   color="#aaaaaa", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(100.0, color="#aaaaaa", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Pe$_H$ [-]", color="white")
    ax.set_ylabel("Da$_{diff}$ = $\\lambda H^2/D_{pore}$ [-]", color="white")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("B — Mappa regime Pe-Da", color="white",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white",
              loc="upper left", ncol=1)
    ax.grid(alpha=0.2, color="white")
    ax.text(0.005, 5, "Diff. puro", color="#aaaaaa", fontsize=7, alpha=0.7)
    ax.text(3.0,   5, "Adv. dom. (Pe>1)", color="#aaaaaa", fontsize=7, alpha=0.7)
    ax.text(0.005, 0.2,"Sorgente",  color="#aaaaaa", fontsize=7, alpha=0.7)

    ax = axes[1, 0]
    seen_regimes = set()
    for i, (k, c_val, c, r) in enumerate(zip(k0_vals, Cin, colors, regimes)):
        lbl = REGIME_LABELS.get(r, r) if r not in seen_regimes else None
        ax.scatter(k, max(c_val, 0.1), color=c, s=90, alpha=0.88,
                   edgecolors="white", linewidths=0.4, label=lbl, zorder=3)
        seen_regimes.add(r)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k$_0$ [m$^2$]", color="white")
    ax.set_ylabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("C — C$_{in}$ vs permeabilita'", color="white",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, color="white")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white",
              loc="upper left", ncol=1)

    ax = axes[1, 1]
    sc = ax.scatter(Ra_vals, np.maximum(Cin, 0.1), c=phi_vals,
                    s=90, cmap="plasma", alpha=0.88,
                    edgecolors="white", linewidths=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("phi_m [-]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ra-226 [Bq kg$^{-1}$]", color="white")
    ax.set_ylabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("D — C$_{in}$ vs Ra-226 (dim. = phi_m)", color="white",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, color="white")

    plt.suptitle(
        ("Benchmark predittivo multi-litotipo - VolcanicRadonModel v5 | "
         "20 lithotypes | 5 regimes"),
        color="white", fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="#1a1a2e")
        print(f"  Salvato: {save_path}")
    return fig


def generate_causal_chain_figure(results, outpath="causal_chain_summary.png"):
    lits_plot = ["tuff_unwelded", "pumice", "sandy_soil"]
    colors    = ["#c0392b", "#2980b9", "#27ae60"]
    lw        = 2.0

    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    def t_h(sol):
        return sol.t / 3600.0

    ax1 = fig.add_subplot(gs[0, 0])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax1.plot(t_h(r["sol"]), r["res"]["theta"][0], color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax1.set_xlabel("Tempo [h]"); ax1.set_ylabel("θ [m³/m³]")
    ax1.set_title("Layer 1 — Umidità z=0\n(Richards, drenaggio)")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        Patm0 = r["res"]["Pg"].mean()
        ax2.plot(t_h(r["sol"]), r["res"]["Pg"][0]-101325, color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax2.set_xlabel("Tempo [h]"); ax2.set_ylabel("ΔPg [Pa]")
    ax2.set_title("Layer 1 — Pressione gas suolo\n(barometrico + geomeccanica)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax3.plot(t_h(r["sol"]), r["res"]["Ar"].mean(axis=0), color=c, lw=lw)
    ax3.set_xlabel("Tempo [h]"); ax3.set_ylabel("Ar [-]")
    ax3.set_title("Layer 2 — Accessibilità pori\n(variabile di stato Ar)")
    ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax4.plot(t_h(r["sol"]), r["res"]["eps_mem"].mean(axis=0), color=c, lw=lw)
    ax4.set_xlabel("Tempo [h]"); ax4.set_ylabel("ε_mem [-]")
    ax4.set_title("Layer 2 — Isteresi emanazione\n(memoria wetting/drying)")
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 0:2])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax5.plot(t_h(r["sol"]), r["res"]["Cm"].mean(axis=0), color=c, lw=lw, label=f"Cm {lab}")
        ax5.plot(t_h(r["sol"]), r["res"]["Cf"].mean(axis=0), color=c, lw=lw, ls="--")
    ax5.set_xlabel("Tempo [h]"); ax5.set_ylabel("C_Rn [Bq m⁻³]")
    ax5.set_title("Layer 3 — Radon matrice () e frattura (--)\n(memoria Mittag-Leffler + adsorbimento)")
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2:4])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax6.plot(t_h(r["sol"]), r["res"]["C_bs"], color=c, lw=lw,   label=f"Interrato {lab}")
        ax6.plot(t_h(r["sol"]), r["res"]["C_gt"], color=c, lw=lw, ls="--")
    ax6.set_xlabel("Tempo [h]"); ax6.set_ylabel("C_in [Bq m⁻³]")
    ax6.set_title("Layer 5 — Indoor Rn-222\n(multizona: interrato —, PT --)")
    ax6.legend(fontsize=7); ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0:2])
    name0 = "tuff_unwelded"
    r0    = results[name0]
    th0   = t_h(r0["sol"])
    ax7.plot(th0, r0["res"]["C_bs"],    color="k",        lw=lw, label="Rn-222")
    ax7.plot(th0, r0["res"]["C_Po218"], color="#e74c3c", lw=lw, label="Po-218")
    ax7.plot(th0, r0["res"]["C_Pb214"], color="#e67e22", lw=lw, label="Pb-214")
    ax7.plot(th0, r0["res"]["C_Bi214"], color="#9b59b6", lw=lw, label="Bi-214")
    ax7.plot(th0, r0["res"]["C_Po214"], color="#3498db", lw=lw, label="Po-214")
    ax7.set_xlabel("Tempo [h]"); ax7.set_ylabel("Concentrazione [Bq m⁻³]")
    ax7.set_title(f"— Catena progenie Rn-222\n({VOLCANIC_LITHOTYPES[name0]['label']})")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 2:4])
    for name, c in zip(lits_plot, colors):
        r   = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax8.plot(t_h(r["sol"]), r["res"]["Cm220"].mean(axis=0), color=c, lw=lw, label=lab)
    ax8.set_xlabel("Tempo [h]"); ax8.set_ylabel("C_Rn220 (matrice) [Bq m⁻³]")
    ax8.set_title("— Thoron (Rn-220) in matrice\n(diagnostica profondità sorgente)")
    ax8.legend(fontsize=7); ax8.grid(alpha=0.3)

    ax9 = fig.add_subplot(gs[3, 0])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax9.plot(t_h(r["sol"]), r["res"]["Pin"]-101325, color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax9.set_xlabel("Tempo [h]"); ax9.set_ylabel("ΔPin [Pa]")
    ax9.set_title("Layer 5 — Pressione indoor\n(stack + vento + barometrico)")
    ax9.legend(fontsize=7); ax9.grid(alpha=0.3)

    ax10 = fig.add_subplot(gs[3, 1])
    pi_names = ["Pe_H", "Da_H", "Pi_frac"]
    pi_vals  = {name: [results[name]["pi"][k] for k in pi_names] for name in lits_plot}
    x        = np.arange(len(pi_names))
    w        = 0.25
    for i, (name, c) in enumerate(zip(lits_plot, colors)):
        ax10.bar(x + i*w, np.log10(np.maximum(pi_vals[name], 1e-6)),
                 w, color=c, alpha=0.85, label=VOLCANIC_LITHOTYPES[name]["label"])
    ax10.set_xticks(x + w)
    ax10.set_xticklabels(pi_names, fontsize=8)
    ax10.set_ylabel("log₁₀(Π)")
    ax10.set_title("Layer 6 — Numeri di Buckingham π")
    ax10.legend(fontsize=7); ax10.grid(alpha=0.3, axis="y")

    ax11 = fig.add_subplot(gs[3, 2])
    Pe_vals = [results[n]["pi"]["Pe_H"]   for n in VOLCANIC_LITHOTYPES.keys()]
    C_vals  = [results[n]["C_bs_final"]   for n in VOLCANIC_LITHOTYPES.keys()]
    labels  = [VOLCANIC_LITHOTYPES[n]["label"] for n in VOLCANIC_LITHOTYPES.keys()]
    sc_c    = ["#c0392b","#e74c3c","#2980b9","#e67e22","#8e44ad","#27ae60","#7f8c8d"]
    for i, (pe, cv, lb, cc) in enumerate(zip(Pe_vals, C_vals, labels, sc_c)):
        ax11.scatter(pe, cv, s=80, color=cc, zorder=5, label=lb)
    Pe_fit = np.logspace(-1, 1, 50)
    valid  = [(pe, cv) for pe, cv in zip(Pe_vals, C_vals) if pe > 0 and cv > 0]
    if len(valid) >= 3:
        lpe   = np.log([v[0] for v in valid])
        lcv   = np.log([v[1] for v in valid])
        coef  = np.polyfit(lpe, lcv, 1)
        ax11.plot(Pe_fit, np.exp(coef[1]) * Pe_fit**coef[0], "k--", lw=1.5,
                  label=f"Power law: α={coef[0]:.2f}")
    ax11.set_xscale("log"); ax11.set_yscale("log")
    ax11.set_xlabel("Pe_H"); ax11.set_ylabel("C_bs final [Bq m⁻³]")
    ax11.set_title("Scaling law: C_indoor ~ Pe_H^α")
    ax11.legend(fontsize=6); ax11.grid(alpha=0.3, which="both")

    ax12 = fig.add_subplot(gs[3, 3])
    F_vals = [results[n]["F_eq"] for n in VOLCANIC_LITHOTYPES.keys()]
    ax12.barh(labels, F_vals, color=sc_c, edgecolor="k", linewidth=0.5)
    ax12.axvline(0.4, color="k", ls="--", lw=1.2, label="F=0.4 (tipico indoor)")
    ax12.set_xlabel("Fattore di equilibrio F")
    ax12.set_title("— Equilibrio progenie\n(F = dose efficace / dose a eq.)")
    ax12.legend(fontsize=8); ax12.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "VolcanicRadonModel v2.0 — Catena Causale Definitiva\n"
        "moisture/T/P/fractures → source → matrix-fracture → ingress → indoor → checks",
        fontsize=13, fontweight="bold", y=0.995
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figura salvata: {outpath}")
    return outpath


def compute_scaling_laws(n_samples=48, seed=42, nz=40):
    rng  = np.random.default_rng(seed)
    X    = []
    Y    = []
    lits_sampled = []
    lits = list(GEOLITHOTYPES.keys())
    for _ in range(n_samples):
        name = rng.choice(lits)
        Ra_min = GEOLITHOTYPES[name].get("Ra226_Bqkg", 40.0) * 0.5
        Ra_max = GEOLITHOTYPES[name].get("Ra226_Bqkg", 40.0) * 3.0
        Ra_max = max(Ra_max, Ra_min + 10.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dP_draw = float(rng.uniform(0.5, 30.0))
            m = build_model(name, override={
                "dP_init":   dP_draw,
                "dP_bottom": dP_draw,
                "Ra226_Bqkg":float(rng.uniform(Ra_min, Ra_max)),
                "ACH_closed":float(rng.uniform(0.05, 0.5)),
                "nz":        nz,
            })
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                sol = m.simulate(days=2.0)
            if sol.success and np.isfinite(sol.y[10*m.nz, -1]):
                res    = m.extract_results(sol)
                theta_f= res["theta"][:,  -1]
                Pg_f   = res["Pg"][:,    -1]
                T_f    = m.T_field(sol.t[-1])
                pi     = m.buckingham_pi(theta_f, Pg_f, T_f)
                C_in   = float(res["C_bs"][-1])
                if C_in > 0:
                    X.append([pi["Pe_H"], pi["DaAdv"], pi["Pi_src"], pi["Pi_vent"]])
                    Y.append(C_in)
                    lits_sampled.append(name)
        except Exception:
            pass
    if len(X) < 6:
        return None
    X   = np.array(X)
    Y   = np.array(Y)
    eps = 1e-30
    A   = np.column_stack([
        np.ones(len(X)),
        np.log(np.maximum(X[:,0], eps)),
        np.log(np.maximum(X[:,1], eps)),
        np.log(np.maximum(X[:,2], eps)),
        np.log(np.maximum(X[:,3], eps)),
    ])
    logY = np.log(np.maximum(Y, eps))
    coef, _, _, _ = np.linalg.lstsq(A, logY, rcond=None)
    Y_pred = np.exp(A @ coef)
    logYpred = np.log(np.maximum(Y_pred, eps))
    SS_res = np.sum((logY - logYpred)**2)
    SS_tot = np.sum((logY - np.mean(logY))**2)
    R2     = 1.0 - SS_res / max(SS_tot, 1e-30)
    return {
        "coef":    coef,
        "R2":      R2,
        "n":       len(Y),
        "lits":    lits_sampled,
        "law":     (f"C_in* ~ {np.exp(coef[0]):.3e} x "
                    f"Pe_H^{coef[1]:.3f} x DaAdv^{coef[2]:.3f} x "
                    f"Pi_src^{coef[3]:.3f} x Pi_vent^{coef[4]:.3f}"),
    }


def plot_volcanic_comparison_paper(results: dict, save_path: str = None,
                                    dpi: int = 150):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID
    MUTED  = "#8b949e"
    PAL    = ["#e84393","#58a6ff","#56d364","#ffa657",
              "#f0883e","#bc8cff","#39d353","#79c0ff"]

    fig = plt.figure(figsize=(20, 16), facecolor=DARK)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.44, wspace=0.36,
        left=0.06, right=0.97, top=0.93, bottom=0.08
    )

    def _ax(row, col, colspan=1):
        return fig.add_subplot(gs[row, col:col+colspan])

    def _style(ax, title, xl, yl, xsc="linear", ysc="linear"):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold",
                     wrap=True)
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.5, lw=0.6, ls="--")
        ax.set_xscale(xsc); ax.set_yscale(ysc)

    lit_names = list(results.keys())
    colors    = {n: PAL[i % len(PAL)] for i, n in enumerate(lit_names)}

    def _get(r, *keys, default=0.0):
        for k in keys:
            if k in r:
                v = r[k]
                if hasattr(v, "__len__"):
                    arr = np.asarray(v)
                    return float(arr[-1]) if arr.ndim == 1 else float(arr.mean())
                return float(v)
        return float(default)

    def _get_pi(r, key, default=1e-3):
        if "pi" in r: return float(r["pi"].get(key, default))
        return float(r.get(key, default))

    C_in_vals = [_get(r, "Cin_Bq_m3", "C_bs_final") for r in results.values()]
    Pe_vals   = [_get_pi(r, "Pe_H") for r in results.values()]
    Da_vals   = [_get_pi(r, "Da_H") for r in results.values()]
    Feq_vals  = [_get(r, "F_eq", default=0.4) for r in results.values()]
    J_vals    = [_get(r, "J_surface", default=1e-4) for r in results.values()]
    phi_vals  = [_get(r, "phi_tot", default=0.3) for r in results.values()]
    k0_vals   = [_get(r, "k0", default=1e-14) for r in results.values()]
    labels_lit= [results[n].get("label", n) for n in lit_names]

    ax_a = _ax(0, 0)
    x_pos = np.arange(len(lit_names))
    bars  = ax_a.bar(
        x_pos, C_in_vals,
        color=[colors[n] for n in lit_names],
        edgecolor=GRID, linewidth=0.8, alpha=0.90
    )
    ax_a.axhline(100, color="#ffa657", ls="--", lw=1.0, alpha=0.85,
                 label="100 Bq m⁻³ (WHO ref. level)")
    ax_a.axhline(300, color="#e84393", ls="--", lw=1.0, alpha=0.85,
                 label="300 Bq m⁻³ (IAEA action level)")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels_lit, rotation=38, ha="right", fontsize=7)
    for bar_obj, val in zip(bars, C_in_vals):
        if val > 0:
            ax_a.text(bar_obj.get_x() + bar_obj.get_width()/2,
                      val * 1.12, f"{val:.0f}",
                      ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax_a.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75,
                loc="upper right")
    _style(ax_a, "A — Radon Indoor C_in per Litotipo Vulcanico",
           "Litotipo", "C_in  [Bq m⁻³]", ysc="log")

    ax_b = _ax(0, 1)
    xmin, xmax, ymin, ymax = 1e-4, 1e3, 1e-2, 1e4
    ax_b.fill_betweenx([ymin, 100], xmin, 1,
                       color="#58a6ff", alpha=0.08, label="Diffusive")
    ax_b.fill_betweenx([100, ymax],  xmin, 1,
                       color="#58a6ff", alpha=0.05)
    ax_b.fill_betweenx([ymin, 100], 1, xmax,
                       color="#e84393", alpha=0.08, label="Advective")
    ax_b.fill_betweenx([100, ymax],  1, xmax,
                       color="#f0883e", alpha=0.06, label="Decay-limited")
    ax_b.axvline(1.0, color="#ffa657", ls="--", lw=1.2, alpha=0.8,
                 label="Pe_H = 1")
    ax_b.axhline(100, color="#56d364", ls="--", lw=1.2, alpha=0.8,
                 label="Da_H = 100")
    for i, (n, Pe, Da) in enumerate(zip(lit_names, Pe_vals, Da_vals)):
        ax_b.scatter(Pe, Da, color=colors[n], s=90, zorder=6,
                     edgecolors=TEXT, lw=0.7)
        ax_b.annotate(labels_lit[i], (Pe, Da),
                      textcoords="offset points", xytext=(5, 3),
                      fontsize=6, color=TEXT)
    ax_b.set_xlim(xmin, xmax); ax_b.set_ylim(ymin, ymax)
    ax_b.legend(fontsize=6, labelcolor=TEXT, facecolor=PANEL,
                framealpha=0.75, loc="upper left")
    _style(ax_b, "B — Diagramma di Regime Pe_H – Da_H",
           "Pe_H  (Péclet idraulico) [-]", "Da_H  (Damköhler idraulico) [-]",
           xsc="log", ysc="log")

    ax_c = _ax(0, 2)
    k0_arr = np.array(k0_vals)
    J_arr  = np.array(J_vals)
    for i, n in enumerate(lit_names):
        ax_c.scatter(k0_arr[i], J_arr[i], color=colors[n], s=90, zorder=5,
                     edgecolors=TEXT, lw=0.7, label=labels_lit[i])
    mask_c = (k0_arr > 0) & (J_arr > 0)
    if mask_c.sum() >= 3:
        c2 = np.polyfit(np.log10(k0_arr[mask_c]), np.log10(J_arr[mask_c]), 1)
        k_l  = np.logspace(np.log10(k0_arr.min()*0.5), np.log10(k0_arr.max()*2), 60)
        J_l  = 10**c2[1] * k_l**c2[0]
        ax_c.plot(k_l, J_l, "--", color="#bc8cff", lw=1.5,
                  label=f"J ∝ k₀^{c2[0]:.2f}")
    ax_c.legend(fontsize=5.8, labelcolor=TEXT, facecolor=PANEL,
                framealpha=0.75, ncol=2)
    _style(ax_c, "C — Flusso Superficiale J vs Permeabilità k₀",
           "k₀  [m²]", "J_surface  [Bq m⁻² s⁻¹]", xsc="log", ysc="log")

    ax_d = _ax(1, 0)
    phi_arr = np.array(phi_vals)
    Cin_arr2= np.array(C_in_vals)
    for i, n in enumerate(lit_names):
        ax_d.scatter(phi_arr[i], Cin_arr2[i], color=colors[n], s=90, zorder=5,
                     edgecolors=TEXT, lw=0.7, label=labels_lit[i])
    mask_d = (phi_arr > 0) & (Cin_arr2 > 0)
    if mask_d.sum() >= 3:
        c3 = np.polyfit(np.log10(phi_arr[mask_d]), np.log10(Cin_arr2[mask_d]), 1)
        phi_l = np.logspace(np.log10(phi_arr.min()*0.8), np.log10(phi_arr.max()*1.2), 60)
        Cin_l = 10**c3[1] * phi_l**c3[0]
        ax_d.plot(phi_l, Cin_l, "--", color="#39d353", lw=1.5,
                  label=f"C_in ∝ φ^{c3[0]:.2f}")
    ax_d.legend(fontsize=5.8, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_d, "D — Scaling C_in vs Porosità Totale φ_tot",
           "φ_tot = φ_m + φ_f  [-]", "C_in  [Bq m⁻³]", xsc="log", ysc="log")

    ax_e = _ax(1, 1, colspan=2)
    Feq_arr = np.array(Feq_vals)
    ax_e.axhspan(0.4, 0.5, color="#ffa657", alpha=0.10,
                 label="Range tipico indoor F_eq ∈ [0.4, 0.5] (WHO/UNSCEAR)")
    for i, n in enumerate(lit_names):
        if Cin_arr2[i] > 0 and Feq_arr[i] > 0:
            ax_e.scatter(Cin_arr2[i], Feq_arr[i], color=colors[n], s=100,
                         zorder=5, edgecolors=TEXT, lw=0.7)
            ax_e.annotate(labels_lit[i], (Cin_arr2[i], Feq_arr[i]),
                          textcoords="offset points", xytext=(6, 3),
                          fontsize=6.5, color=TEXT)
    ax_e.axvline(100, color="#ffa657", ls=":", lw=1.0, alpha=0.7)
    ax_e.axvline(300, color="#e84393", ls=":", lw=1.0, alpha=0.7)
    ax_e.set_xlim(left=1e0); ax_e.set_ylim(0.0, 1.0)
    ax_e.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_e, "E — Fattore di Equilibrio F_eq vs C_in Radon\n"
                 "(F_eq basso = ventilazione alta o breve tempo di sosta progenie)",
           "C_in  [Bq m⁻³]", "F_eq  (UNSCEAR 2000) [-]", xsc="log")

    fig.suptitle(
        "VolcanicRadonModel v2.0 — Confronto Multi-Litotipo Vulcanico\n"
        "Daniel de Siano  ·  Target: peer-reviewed journal",
        color=TEXT, fontsize=13, fontweight="bold", y=0.978
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_volcanic_comparison_paper] Salvato: {save_path}")
    return fig


def plot_bifurcation_paper(bif: dict, save_path: str = None,
                            lithotype_label: str = "", dpi: int = 150):
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    dP     = np.asarray(bif.get("dP",    bif.get("dP_range", [])))
    Cin    = np.asarray(bif.get("C_in",  bif.get("Cin", [])))
    Pe     = np.asarray(bif.get("Pe_H",  []))
    J_tot  = np.asarray(bif.get("J_total",   np.zeros_like(dP)))
    J_diff = np.asarray(bif.get("J_diff",    np.zeros_like(dP)))
    J_adv  = np.asarray(bif.get("J_adv",     np.zeros_like(dP)))
    dP_star_a = bif.get("dP_critical",  None)
    dP_star_n = bif.get("dP_numerical", None)
    alpha_adv = bif.get("alpha_adv",    np.nan)
    C_diff    = bif.get("C_diffusion",  None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK,
                              gridspec_kw={"wspace": 0.33})

    def _style(ax, title, xl, yl, xsc="log", ysc="log"):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold")
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.45, lw=0.6, ls="--")
        ax.set_xscale(xsc); ax.set_yscale(ysc)

    ax_l = axes[0]
    if dP_star_a is not None:
        ax_l.axvspan(dP.min(), dP_star_a, color="#58a6ff", alpha=0.10,
                     label="Regime diffusivo")
        ax_l.axvspan(dP_star_a, dP.max(), color="#e84393", alpha=0.10,
                     label="Regime avvettivo")
        ax_l.axvline(dP_star_a, color="#ffa657", ls="--", lw=1.4, alpha=0.9,
                     label=f"ΔP* (anal.) = {dP_star_a:.2f} Pa")
    if dP_star_n is not None and dP_star_n != dP_star_a:
        ax_l.axvline(dP_star_n, color="#39d353", ls=":", lw=1.2, alpha=0.8,
                     label=f"ΔP* (num.) = {dP_star_n:.2f} Pa")
    if C_diff is not None and C_diff > 0:
        ax_l.axhline(C_diff, color=GRID, ls=":", lw=1.0, alpha=0.7,
                     label=f"C_diff = {C_diff:.1f} Bq m⁻³")
    ax_l.plot(dP, Cin, color="#56d364", lw=2.0, marker="o",
              markersize=4, zorder=5, label="C_in (modello)")
    if not np.isnan(alpha_adv):
        mask_a = Pe > 1.0
        if mask_a.sum() >= 2:
            dP_fit = dP[mask_a]; Cin_fit = Cin[mask_a]
            C0_fit = Cin_fit[0] / (dP_fit[0]**alpha_adv + 1e-30)
            ax_l.plot(dP_fit, C0_fit * dP_fit**alpha_adv,
                      "--", color="#bc8cff", lw=1.4,
                      label=f"C_in ∝ ΔP^{alpha_adv:.2f}")
    ax_l.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_l, f"Biforcazione C_in vs ΔP\n{lithotype_label}",
           "ΔP_sub-slab  [Pa]", "C_in  [Bq m⁻³]")

    ax_c = axes[1]
    ax_c.plot(dP, Pe, color="#bc8cff", lw=2.0, marker="s",
              markersize=4, zorder=5, label="Pe_H (Péclet idraulico)")
    ax_c.axhline(1.0, color="#ffa657", ls="--", lw=1.2, alpha=0.8,
                 label="Pe_H = 1 (biforcazione)")
    ax_c.axhline(0.1, color="#58a6ff", ls=":", lw=1.0, alpha=0.7,
                 label="Pe_H = 0.1 (diff. puro)")
    ax_c.axhline(10,  color="#e84393", ls=":", lw=1.0, alpha=0.7,
                 label="Pe_H = 10 (adv. forte)")
    ax_c.fill_between(dP, Pe, 1.0, where=(Pe < 1.0),
                      alpha=0.12, color="#58a6ff")
    ax_c.fill_between(dP, Pe, 1.0, where=(Pe >= 1.0),
                      alpha=0.12, color="#e84393")
    ax_c.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_c, "Numero di Péclet Idraulico vs ΔP",
           "ΔP_sub-slab  [Pa]", "Pe_H  [-]")

    ax_r = axes[2]
    J_tot_pos = np.maximum(J_tot, 1e-12)
    sc = ax_r.scatter(J_diff, J_adv, c=dP, cmap="plasma",
                      s=60, zorder=5, edgecolors=TEXT, lw=0.5,
                      norm=plt.matplotlib.colors.LogNorm(
                          vmin=dP.min(), vmax=dP.max()))
    J_levels = np.logspace(np.log10(J_tot_pos.min()+1e-12),
                            np.log10(J_tot_pos.max()), 4)
    j_range  = np.logspace(-6, 0, 80)
    for Jlev in J_levels:
        ax_r.plot(j_range, Jlev - j_range, ":", color=GRID, lw=0.8, alpha=0.6)
    j_eq = np.logspace(np.log10(max(min(J_diff[J_diff>0], default=1e-8), 1e-8)),
                       np.log10(max(max(J_adv[J_adv>0], default=1e-5), 1e-5)), 60)
    ax_r.plot(j_eq, j_eq, "--", color="#ffa657", lw=1.2, alpha=0.7,
              label="J_diff = J_adv")
    cbar = fig.colorbar(sc, ax=ax_r, pad=0.02)
    cbar.set_label("ΔP [Pa]", color=TEXT, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=6.5)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT)
    ax_r.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_r, "J_diffusivo vs J_avvettivo per ogni ΔP\n(colore = ΔP [Pa])",
           "J_diff  [Bq s⁻¹]", "J_adv  [Bq s⁻¹]")

    fig.suptitle(
        f"Analisi Biforcazione Rn-222 — {lithotype_label}\n"
        "VolcanicRadonModel v2.0  ·  Daniel de Siano",
        color=TEXT, fontsize=12, fontweight="bold", y=1.01
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_bifurcation_paper] Salvato: {save_path}")
    return fig


def plot_buckingham_scaling(bpi: dict, save_path: str = None, dpi: int = 150):
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    pi_names = bpi.get("pi_names", [])
    exps     = np.asarray(bpi.get("exponents", []))
    ci95     = np.asarray(bpi.get("exponents_ci95", np.zeros((len(exps), 2))))
    imps     = np.asarray(bpi.get("importance", np.abs(exps)))
    R2       = bpi.get("R2", 0.0)
    R2_ci    = bpi.get("R2_ci95", (R2, R2))
    law_str  = bpi.get("law_string", "")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK,
                                      gridspec_kw={"wspace": 0.35})

    def _style(ax, title, xl, yl):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold")
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.5, lw=0.6, ls="--")

    rank_idx = np.argsort(imps)[::-1]
    names_r  = [pi_names[i] if i < len(pi_names) else f"π_{i}" for i in rank_idx]
    imps_r   = imps[rank_idx]
    exps_r   = exps[rank_idx]
    ci_lo_r  = ci95[rank_idx, 0] if ci95.shape[0] > 0 else np.zeros_like(exps_r)
    ci_hi_r  = ci95[rank_idx, 1] if ci95.shape[0] > 0 else np.zeros_like(exps_r)

    clrs     = ["#56d364" if e >= 0 else "#e84393" for e in exps_r]
    y_pos    = np.arange(len(names_r))
    ax_l.barh(y_pos, imps_r, color=clrs, edgecolor=GRID, alpha=0.85, height=0.6)
    for y_i, lo, hi in zip(y_pos, ci_lo_r, ci_hi_r):
        ax_l.errorbar(imps_r[y_i], y_i,
                      xerr=[[abs(exps_r[y_i]-lo)], [abs(hi-exps_r[y_i])]],
                      fmt="none", color=TEXT, capsize=3, lw=0.8)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(names_r, fontsize=7.5, color=TEXT)
    ax_l.axvline(0, color=GRID, lw=0.8)
    ax_l.invert_yaxis()
    _style(ax_l,
           f"A — Importanza Relativa Gruppi π\n|aᵢ|·σ(log πᵢ)",
           "Importanza relativa [-]", "Gruppo adimensionale")

    Cin_meas = bpi.get("Cin_samples", None)
    pi_samp  = bpi.get("pi_samples",  None)
    log_K    = bpi.get("log_K", 0.0)

    if Cin_meas is not None and pi_samp is not None and len(Cin_meas) > 0:
        C_ref = bpi.get("C_ref", np.median(Cin_meas))
        Cin_star = Cin_meas / max(C_ref, 1e-6)
        log_pi_s = np.log10(np.maximum(pi_samp, 1e-30))
        A_pred   = np.column_stack([np.ones(len(Cin_star)), log_pi_s])
        Cin_pred = 10**(log_K + log_pi_s @ exps)
        ax_r.scatter(Cin_star, Cin_pred,
                     c=np.log10(np.maximum(Cin_meas, 1e-6)),
                     cmap="viridis", s=25, alpha=0.6, edgecolors="none",
                     zorder=4)
        min_v = min(Cin_star.min(), Cin_pred.min())
        max_v = max(Cin_star.max(), Cin_pred.max())
        line_v = np.logspace(np.log10(max(min_v, 1e-6)), np.log10(max_v*1.05), 40)
        ax_r.plot(line_v, line_v, "--", color="#ffa657", lw=1.5,
                  label="Predetto = Simulato (1:1)")
        ax_r.plot(line_v, line_v*2, ":", color=GRID, lw=0.8, alpha=0.6,
                  label="Fattore 2×")
        ax_r.plot(line_v, line_v*0.5, ":", color=GRID, lw=0.8, alpha=0.6)
        ax_r.set_xscale("log"); ax_r.set_yscale("log")
        ax_r.text(0.05, 0.95, f"R² = {R2:.3f}\n[{R2_ci[0]:.3f}, {R2_ci[1]:.3f}]",
                  transform=ax_r.transAxes, color=TEXT, fontsize=8,
                  va="top", fontweight="bold")
        ax_r.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    else:
        ax_r.text(0.5, 0.5, "Dati campionamento non disponibili\n"
                  "(usa return_samples=True in buckingham_pi_paper)",
                  ha="center", va="center", color=TEXT, fontsize=9,
                  transform=ax_r.transAxes)

    _style(ax_r, f"B — Parity Plot  C_in* (R²={R2:.3f})\n"
                  + law_str[:60] + ("..." if len(law_str)>60 else ""),
           "C_in* simulato [-]", "C_in* predetto (OLS) [-]")

    fig.suptitle("Buckingham π Analysis — VolcanicRadonModel v2.0",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_buckingham_scaling] Salvato: {save_path}")
    return fig


def plot_sensitivity_ranking(sa: dict, save_path: str = None, dpi: int = 150):
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    ranking   = sa.get("ranking", [])
    elast     = sa.get("elasticities", {})
    mu_star   = sa.get("morris_mu_star", {})
    sigma_m   = sa.get("morris_sigma", {})
    score_d   = sa.get("score", {})
    C0        = sa.get("C_nominal", 1.0)

    if not ranking:
        return None

    names  = [r[0] for r in ranking]
    eps_v  = np.array([abs(elast.get(n, 0.0)) for n in names])
    mu_v   = np.array([mu_star.get(n, 0.0) for n in names])
    sig_v  = np.array([sigma_m.get(n, 0.0) for n in names])
    scr_v  = np.array([score_d.get(n, 0.0) for n in names])
    sign_v = np.array([np.sign(elast.get(n, 1.0)) for n in names])

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK)
    ax.set_facecolor(PANEL)

    eps_med = np.median(eps_v[eps_v > 0]) if np.any(eps_v > 0) else 0.5
    mu_med  = np.median(mu_v[mu_v > 0])   if np.any(mu_v > 0)  else 0.5
    ax.axvline(eps_med, color=GRID, ls="--", lw=0.8, alpha=0.7)
    ax.axhline(mu_med,  color=GRID, ls="--", lw=0.8, alpha=0.7)

    for txt, xy, ha_v in [
        ("Critico\n(lineare, alto impatto)", (eps_v.max()*0.7, mu_v.max()*0.85), "center"),
        ("Non-lineare\no interazione",       (eps_v.max()*0.1, mu_v.max()*0.85), "center"),
        ("Impatto locale",                   (eps_v.max()*0.7, mu_v.max()*0.10), "center"),
        ("Trascurabile",                     (eps_v.max()*0.1, mu_v.max()*0.10), "center"),
    ]:
        ax.text(*xy, txt, color=GRID, fontsize=7, ha="center", va="center",
                fontstyle="italic", alpha=0.8)

    sizes  = 100 + 400 * (scr_v / max(scr_v.max(), 1e-6))
    colors_s = ["#56d364" if s >= 0 else "#e84393" for s in sign_v]
    scatter  = ax.scatter(eps_v, mu_v, s=sizes, c=colors_s,
                           edgecolors=TEXT, lw=0.7, zorder=5, alpha=0.85)

    for i, name in enumerate(names):
        ax.annotate(
            name.replace("_", " "), (eps_v[i], mu_v[i]),
            textcoords="offset points",
            xytext=(7, 4 if i % 2 == 0 else -10),
            fontsize=7.5, color=TEXT, fontweight="bold"
        )

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#56d364",
               ms=9, label="ε > 0 (C_in aumenta con p_i)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#e84393",
               ms=9, label="ε < 0 (C_in diminuisce con p_i)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=TEXT,
               ms=6,  label="Dimensione ∝ score combinato"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, labelcolor=TEXT,
              facecolor=PANEL, framealpha=0.8, loc="upper left")

    ax.set_xlabel("|Elasticità| = |∂ log C_in / ∂ log pᵢ| [-]",
                  color=TEXT, fontsize=9)
    ax.set_ylabel("μ* Morris = ⟨|EE_i|⟩ [Bq m⁻³ / unità_i]",
                  color=TEXT, fontsize=9)
    ax.set_title(
        f"Sensitivity Screen — Morris + Elasticità  "
        f"(C_in nominale = {C0:.0f} Bq m⁻³)\n"
        "VolcanicRadonModel v2.0  ·  Daniel de Siano",
        color=TEXT, fontsize=10, fontweight="bold", pad=9
    )
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID, alpha=0.4, lw=0.6, ls="--")

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_sensitivity_ranking] Salvato: {save_path}")
    return fig

if __name__ == "__main__" and os.environ.get("RADON_RUN_DEMO") == "1":
    print("=" * 70)
    print("VolcanicRadonModel v2.0 -> v5  (GeoLithotypes Extension)")
    print(f"  {len(GEOLITHOTYPES)} litotipi | 5 regimi | 4 modelli di emanazione")
    print("=" * 70)

    print("\n[1] Benchmark predittivo su 20 litotipi geologici (GEOLITHOTYPES)...")
    bench = run_lithology_benchmark(nz=12, days=1.5, verbose=True)
    print_lithology_benchmark_table(bench)

    print("\n[2] Validazione litotipi vulcanici originali (retrocompatibilita')...")
    results = run_validation()

    print("\n[3] Scaling laws Monte Carlo su 20 litotipi (n=32, nz=40)...")
    sl = compute_scaling_laws(n_samples=32, seed=42, nz=40)
    if sl:
        print(f"  Scaling law: {sl['law']}")
        print(f"  R2 = {sl['R2']:.3f}  (n={sl['n']} campioni)")
        if "lits" in sl:
            from collections import Counter
            cnt = Counter(sl["lits"])
            top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            print(f"  Litotipi piu' campionati: {top}")
    else:
        print("  [WARNING] Scaling law non convergente -- aumentare n_samples")

    print("\n[4] Q2_LOO (leave-one-lithology-out) ...")
    m_ref = build_model("tuff_unwelded", override={"nz": 10})
    q2_res = m_ref.q2_leave_one_out(bench=bench, verbose=True)
    if "Q2_LOO" in q2_res:
        print(f"  Q2_LOO = {q2_res['Q2_LOO']:.3f}  "
              f"RMSE_log10 = {q2_res.get('RMSE_log10',float('nan')):.3f}")

    print("\n[5] Summary fisica -- 20 litotipi:")
    print(f"  {'Litotipo':<32} {'Regime':<22} {'C_in':>9} "
          f"{'Pe_H':>7} {'F_eq':>7} {'eps':>7}")
    print("  " + "-"*90)
    for name in GEOLITHOTYPES:
        r = bench.get(name, {})
        if "error" in r or not r:
            continue
        label  = r.get("label", name)
        regime = r.get("regime", "?")
        Cin    = r.get("Cin_Bq_m3",  float("nan"))
        Pe     = r.get("Pe_H",        float("nan"))
        F_eq   = r.get("F_eq",        float("nan"))
        eps    = r.get("eps_mean",     float("nan"))
        print(f"  {label:<32} {regime:<22} {Cin:9.1f} "
              f"{Pe:7.4f} {F_eq:7.3f} {eps:7.4f}")

    print("\n[6] Generazione figura catena causale (litotipi vulcanici)...")
    generate_causal_chain_figure(
        results,
        outpath="causal_chain_summary.png"
    )

    print("\nDone. File generati:")
    print("  causal_chain_summary.png")


def plot_volcanic_comparison(results: dict, save_path: str = None,
                             dpi: int = 150):
    return plot_volcanic_comparison_paper(
        results=results,
        save_path=save_path,
        dpi=dpi,
    )


def plot_bifurcation(bif: dict, save_path: str = None, dpi: int = 150):
    return plot_bifurcation_paper(
        bif=bif,
        save_path=save_path,
        dpi=dpi,
    )


import os as _os
import time as _time
import warnings as _warnings
from collections import defaultdict as _defaultdict

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def safe_log10(x, eps=1e-30):
    arr = np.maximum(np.asarray(x, dtype=float), eps)
    return np.log10(arr)


def build_design_matrix(df, predictors, add_intercept=True, log_transform=True):
    n = len(df)
    cols = []
    names = []
    if add_intercept:
        cols.append(np.ones(n))
        names.append("intercept")
    for pred in predictors:
        if pred not in df.columns:
            raise KeyError(
                "Predittore '{}' non trovato nel DataFrame. "
                "Colonne disponibili: {}".format(pred, list(df.columns))
            )
        vals = df[pred].values.astype(float)
        if log_transform:
            _n_bad = int(np.sum(~(np.isfinite(vals) & (vals > 0.0))))
            if _n_bad:
                raise ValueError(
                    "build_design_matrix: il predittore '{}' contiene {} valori "
                    "non positivi o non finiti su {}. safe_log10 li mapperebbe "
                    "a -30, falsandone il coefficiente: vanno scartati a monte."
                    .format(pred, _n_bad, len(vals)))
        if log_transform:
            vals = safe_log10(vals)
        cols.append(vals)
        if log_transform:
            names.append("log10_" + pred)
        else:
            names.append(pred)
    X = np.column_stack(cols)
    return X, names


def ols_fit(X, y):
    from scipy import stats as _stats
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat  = X @ coef
    resid = y - yhat
    sse   = float(np.dot(resid, resid))
    sigma2 = sse / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, coef / np.where(se > 0, se, 1.0), np.nan)
        p_val  = np.where(np.isfinite(t_stat),
                          2.0 * _stats.t.sf(np.abs(np.nan_to_num(t_stat)), df=max(n - k, 1)),
                          np.nan)
    return {
        "coef":   coef,
        "yhat":   yhat,
        "resid":  resid,
        "se":     se,
        "t_stat": t_stat,
        "p_val":  p_val,
        "n":      n,
        "k":      k,
        "SSE":    sse,
        "SST":    float(np.var(y) * n),
    }


def evaluate_scaling_model(y_true, y_pred, k):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n   = len(y_true)
    res = y_true - y_pred
    sse = float(np.dot(res, res))
    sst = float(np.var(y_true) * n)
    r2  = 1.0 - sse / max(sst, 1e-30)
    rmse = float(np.sqrt(sse / max(n, 1)))
    mae  = float(np.mean(np.abs(res)))
    sigma2_hat = sse / max(n, 1)
    k_aic = k + 1
    aic = n * np.log(max(sigma2_hat, 1e-30)) + 2.0 * k_aic
    bic = n * np.log(max(sigma2_hat, 1e-30)) + k_aic * np.log(max(n, 1))
    ratio = 10.0 ** y_pred / np.maximum(10.0 ** y_true, 1e-30)
    f2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    f3 = float(np.mean((ratio >= 1.0/3) & (ratio <= 3.0)))
    f5 = float(np.mean((ratio >= 0.2) & (ratio <= 5.0)))
    return {
        "R2": r2, "RMSE": rmse, "MAE": mae,
        "AIC": aic, "BIC": bic,
        "SSE": sse, "SST": sst,
        "n": n, "k": k,
        "F2": f2, "F3": f3, "F5": f5,
    }


def scaling_law_string(coef, feature_names, model_name="M0"):
    lines = ["=" * 60]
    lines.append("Scaling Law — {}".format(model_name))
    lines.append("=" * 60)
    lines.append("log10(C*) = ")
    terms = []
    for i, (c, name) in enumerate(zip(coef, feature_names)):
        if name == "intercept":
            terms.append("  {:+.4f}".format(c))
        else:
            terms.append("  {:+.4f} x {}".format(c, name))
    lines.append("\n".join(terms))
    lines.append("")
    lines.append("Forma moltiplicativa (scala lineare):")
    if feature_names[0] == "intercept":
        base = 10.0 ** coef[0]
        mult_parts = ["  C* = {:.5g}".format(base)]
        for c, name in zip(coef[1:], feature_names[1:]):
            var = name.replace("log10_", "")
            mult_parts.append(" x {}^({:+.4f})".format(var, c))
        lines.append("".join(mult_parts))
    lines.append("=" * 60)
    return "\n".join(lines)


_RA_PREDICTORS = ["PeH", "DaRn", "phi_g", "Sw", "Pi_src", "Pi_vent"]
_RA_REGIMES    = [
    "barometric_pumping",
    "fracture_bypass",
    "decay_limited",
    "advective",
    "mixed_transition",
]
_LAM_RN = 2.098e-6
_D0_REF = 1.1e-5
_MU_AIR = 1.82e-5
_C_REF  = 1000.0

_M0_BETA_FROZEN = {
    "Pe_H":    0.0297,
    "Da_Rn":  -0.8879,
    "phi_g":  -1.0848,
    "Sw":     -1.0316,
    "Pi_src":  0.5766,
    "Pi_vent": -0.1678,
}
_M0_INTERCEPT_FROZEN = -2.4250
_M0_RMSE_LOG_FROZEN  = 0.650

_M0_BETA_IDENTIFIABLE = {
    "Pe_H":    0.0545,
    "Da_Rn":  -0.2951,
    "Sw":     -1.1319,
    "Pi_src":  0.5569,
    "Pi_vent": -0.1662,
}
_M0_INTERCEPT_IDENTIFIABLE = -2.6263
_M0_RMSE_LOG_IDENTIFIABLE  = 0.654


def make_regimeaware_lhs_samples(n_samples=1700, seed=42, lits=None):
    if lits is None:
        lits = list(GEOLITHOTYPES.keys())
    n_lits = len(lits)

    low  = np.array([0.5, 5.0, 0.02, 0.10, 2.2])
    high = np.array([50.0, 500.0, 2.0, 0.90, 8.0])
    n_cont = 5

    rng = np.random.default_rng(seed)

    if _HAS_QMC:
        try:
            from scipy.stats.qmc import LatinHypercube, scale as _qmc_scale
            sampler = LatinHypercube(d=n_cont, seed=seed)
            unit    = sampler.random(n_samples)
            scaled  = _qmc_scale(unit, low, high)
        except Exception:
            scaled = rng.uniform(low, high, size=(n_samples, n_cont))
    else:
        scaled = rng.uniform(low, high, size=(n_samples, n_cont))

    lit_indices = np.floor(np.linspace(0, n_lits - 1e-9, n_samples)).astype(int)
    rng.shuffle(lit_indices)

    samples = []
    for i in range(n_samples):
        samples.append({
            "dP":         float(scaled[i, 0]),
            "Ra":         float(scaled[i, 1]),
            "ACH":        float(scaled[i, 2]),
            "theta_frac": float(scaled[i, 3]),
            "H":          float(scaled[i, 4]),
            "lit_name":   lits[lit_indices[i]],
        })
    return samples


def run_regimeaware_sample(sample, nz=15, days=1.5):
    lit_name = sample["lit_name"]
    try:
        lit_data = GEOLITHOTYPES[lit_name]
    except KeyError:
        return None

    theta_r = float(lit_data.get("theta_r", 0.02))
    theta_s = float(lit_data.get("phi_m", 0.35))
    frac    = float(sample["theta_frac"])
    theta_init = theta_r + frac * (theta_s - theta_r)
    theta_init = float(np.clip(theta_init, theta_r + 1e-4, 0.95 * theta_s))

    _A_fp_lhs = 20.0
    override = {
        "dP_init":    float(sample["dP"]),
        "dP_bottom":  float(sample["dP"]),
        "Ra226_Bqkg": float(sample["Ra"]),
        "ACH_closed": float(sample["ACH"]),
        "theta0":     theta_init,
        "theta_init": theta_init,
        "A_footprint": _A_fp_lhs,
        "V_bs":       _A_fp_lhs * float(sample["H"]),
        "H_room":     float(sample["H"]),
        "nz":         int(nz),
    }
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            mod = build_model(lit_name, override=override)
            sol = mod.simulate_fast(days=float(days), dt_out=7200.0)
    except Exception:
        return None

    return extract_regimeaware_features_from_solution(
        mod, sol, lit_name, sample_id=None
    )


def extract_regimeaware_features_from_solution(model, sol, lith_name,
                                                sample_id=None):
    try:
        res  = model.extract_results(sol)
        p    = model.p
        nz   = model.nz

        C_in = float(res["C_bs"][-1])
        if not np.isfinite(C_in) or C_in <= 0.0:
            return None

        phi_m  = float(p.get("phi_m",   0.35))
        phi_f  = float(p.get("phi_f",   0.01))
        k0     = float(p.get("k0",      1e-12))
        dP     = float(p.get("dP_init", 5.0))
        ACH    = float(p.get("ACH_closed", 0.5))
        H      = float(p.get("H", 3.0))
        Ra     = float(p.get("Ra226_Bqkg", 50.0))
        rho_b  = float(p.get("rho_bulk", 1500.0))
        Ef     = float(np.mean(np.asarray(res["eps_mem"])[:, -1]))

        theta_final = res["theta"][:, -1]
        theta_r = float(p.get("theta_r", 0.02))
        theta_s = phi_m
        _Sw_raw = float(np.mean((theta_final - theta_r) / max(theta_s - theta_r, 1e-6)))
        Sw = float(np.clip(_Sw_raw, 0.0, 1.0)) if _Sw_raw > 0.0 else float("nan")
        phi_g = float(np.clip(phi_m - float(np.mean(theta_final)), 1e-4, 1.0))

        D_bulk = float(p.get("D0", _D0_REF)) * (phi_g) ** (10.0 / 3.0) / max(phi_m ** 2, 1e-12)
        D_bulk = max(D_bulk, 1e-12)
        D_pore = max(D_bulk / max(phi_g, 1e-12), 1e-12)
        D_eff  = D_bulk

        v_darcy = k0 * dP / (_MU_AIR * max(H, 0.1))
        v_eff   = v_darcy / max(phi_g, 1e-12)
        Pe_H   = k0 * dP / (_MU_AIR * D_bulk)
        Da_Rn  = _LAM_RN * H ** 2 / D_pore
        Da_H   = _LAM_RN * H / max(v_eff, 1e-20)
        Pi_src = rho_b * Ra * Ef / _C_REF
        Pi_vent = ACH / (3600.0 * _LAM_RN)
        kDP    = k0 * dP / (_MU_AIR * _D0_REF)
        Cstar  = C_in / _C_REF

        Patm_amp = float(p.get("P_atm_amp", 150.0))
        Patm_ref = float(p.get("P_atm0",    101325.0))
        Pi_baro  = Patm_amp / max(Patm_ref * 0.01, 1e-10)

        regime_declared = str(GEOLITHOTYPES.get(lith_name, {}).get("regime", "unknown"))

        row = {
            "Cin":              C_in,
            "log10_Cin":        float(safe_log10(C_in)),
            "Cstar":            Cstar,
            "log10_Cstar":      float(safe_log10(Cstar)),
            "PeH":              Pe_H,
            "DaRn":             Da_Rn,
            "DaH":              Da_H,
            "phi_g":            phi_g,
            "Sw":               Sw,
            "Pi_src":           Pi_src,
            "Pi_vent":          Pi_vent,
            "kDP":              kDP,
            "phi_mac":          phi_f,
            "Pi_baro":          Pi_baro,
            "k0":               k0,
            "dP":               dP,
            "ACH":              ACH,
            "H":                H,
            "Ra226":            Ra,
            "lithotype":        lith_name,
            "regime_declared":  regime_declared,
        }
        row["regime_inferred"] = infer_physical_regime(row)
        if sample_id is not None:
            row["sample_id"] = sample_id
        return row

    except Exception:
        return None


def infer_physical_regime(row):
    PeH     = row.get("PeH",     np.nan)
    DaRn    = row.get("DaRn",    row.get("DaH", np.nan))
    phi_mac = row.get("phi_mac", row.get("phi_f", 0.0))
    Pi_baro = row.get("Pi_baro", 0.0)

    if np.isfinite(float(Pi_baro)) and float(Pi_baro) > 1.0:
        return "barometric_pumping"
    if np.isfinite(float(phi_mac)) and float(phi_mac) > 0.30:
        return "fracture_bypass"
    if np.isfinite(float(DaRn))   and float(DaRn) > 100.0:
        return "decay_limited"
    if np.isfinite(float(PeH))    and float(PeH)  > 1.0:
        return "advective"
    return "mixed_transition"


def generate_regimeaware_dataset(n_samples=1700, seed=42, nz=15, days=1.5,
                                  lits=None, min_Cin=0.5, min_PeH=1e-8,
                                  verbose=True):
    samples = make_regimeaware_lhs_samples(n_samples=n_samples, seed=seed, lits=lits)
    rows    = []
    n_fail  = 0
    n_degen = 0
    t0      = _time.time()

    for i, samp in enumerate(samples):
        row = run_regimeaware_sample(samp, nz=nz, days=days)
        if row is None:
            n_fail += 1
            continue
        if row.get("Cin",  0.0) < min_Cin:
            n_fail += 1
            continue
        if row.get("PeH",  0.0) < min_PeH:
            n_fail += 1
            continue
        _bad = [q for q in _RA_PREDICTORS
                if not (np.isfinite(row.get(q, np.nan)) and row.get(q, 0.0) > 0.0)]
        if _bad:
            n_fail += 1
            n_degen += 1
            continue
        row["run_id"] = len(rows)
        rows.append(row)
        if verbose and (i + 1) % 100 == 0:
            elapsed = _time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (n_samples - i - 1) / max(rate, 1e-6)
            print("  [{}/{}]  validi={:4d}  falliti={:3d}  "
                  "{:.0f}s elapsed  ETA {:.0f}s".format(
                      i + 1, n_samples, len(rows), n_fail, elapsed, eta))

    total_elapsed = _time.time() - t0
    if verbose:
        print("Dataset generato: {}/{} validi  ({:.0f}s)".format(
            len(rows), n_samples, total_elapsed))
        if n_degen:
            print("  di cui SCARTATI perche' degeneri (predittore non positivo "
                  "o non finito, p.es. Sw indefinita in colonna drenata sotto "
                  "theta_r): {} ({:.1f}%)".format(
                      n_degen, 100.0*n_degen/max(n_samples, 1)))

    if _HAS_PANDAS:
        return _pd.DataFrame(rows)
    return rows


def dataset_regime_summary(df):
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        records = df.to_dict("records")
    else:
        records = list(df)

    by_regime = _defaultdict(list)
    for row in records:
        by_regime[row.get("regime_inferred", "unknown")].append(row)

    n_total = max(len(records), 1)
    summary = []
    for regime in _RA_REGIMES + ["unknown"]:
        rows_r = by_regime.get(regime, [])
        if not rows_r:
            continue
        Cin_vals   = [r["Cin"]  for r in rows_r if np.isfinite(r.get("Cin",  np.nan))]
        PeH_vals   = [r["PeH"]  for r in rows_r if np.isfinite(r.get("PeH",  np.nan))]
        DaRn_vals  = [r["DaRn"] for r in rows_r if np.isfinite(r.get("DaRn", np.nan))]
        summary.append({
            "regime":      regime,
            "n":           len(rows_r),
            "percent":     100.0 * len(rows_r) / n_total,
            "Cin_median":  float(np.median(Cin_vals))  if Cin_vals  else np.nan,
            "Cin_mean":    float(np.mean(Cin_vals))    if Cin_vals  else np.nan,
            "PeH_median":  float(np.median(PeH_vals))  if PeH_vals  else np.nan,
            "DaRn_median": float(np.median(DaRn_vals)) if DaRn_vals else np.nan,
        })
    return summary


def _df_to_arrays(df, col):
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        return df[col].values.astype(float)
    return np.array([r[col] for r in df], dtype=float)


def _get_regime_col(df):
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        if "regime_inferred" in df.columns:
            return df["regime_inferred"].values
        return df["regime"].values
    col = "regime_inferred" if "regime_inferred" in df[0] else "regime"
    return np.array([r.get(col, "unknown") for r in df])


def fit_scaling_M0_global(df):
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        sub = df.copy()
    else:
        sub = df

    X, feat_names = build_design_matrix(sub, _RA_PREDICTORS,
                                         add_intercept=True, log_transform=True)
    y = safe_log10(_df_to_arrays(sub, "Cstar"))
    fit    = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M0")
    return {
        "model":         "M0",
        "coef":          fit["coef"],
        "feature_names": feat_names,
        "se":            fit["se"],
        "t_stat":        fit["t_stat"],
        "p_val":         fit["p_val"],
        "metrics":       metrics,
        "yhat":          fit["yhat"],
        "resid":         fit["resid"],
        "law_string":    law_str,
        "X":             X,
        "y":             y,
    }


def fit_scaling_M1_regime_dummies(df, reference_regime="advective"):
    regimes_arr = _get_regime_col(df)
    all_regimes = [r for r in _RA_REGIMES if r != reference_regime]

    X_base, feat_names_base = build_design_matrix(
        df, _RA_PREDICTORS, add_intercept=True, log_transform=True
    )
    dummy_cols = []
    dummy_names = []
    for reg in all_regimes:
        col = (regimes_arr == reg).astype(float)
        if float(np.sum(col)) < 1.0:
            continue
        dummy_cols.append(col)
        dummy_names.append("D_" + reg)

    X = np.column_stack([X_base] + dummy_cols) if dummy_cols else X_base
    feat_names = feat_names_base + dummy_names
    y = safe_log10(_df_to_arrays(df, "Cstar"))

    fit     = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M1")

    return {
        "model":              "M1",
        "coef":               fit["coef"],
        "feature_names":      feat_names,
        "se":                 fit["se"],
        "t_stat":             fit["t_stat"],
        "p_val":              fit["p_val"],
        "metrics":            metrics,
        "yhat":               fit["yhat"],
        "resid":              fit["resid"],
        "law_string":         law_str,
        "dummy_cols":         dummy_names,
        "reference_regime":   reference_regime,
        "X":                  X,
        "y":                  y,
    }


def fit_scaling_M2_interaction(df, reference_regime="advective"):
    res_m1 = fit_scaling_M1_regime_dummies(df, reference_regime=reference_regime)
    X_base = res_m1["X"]
    feat_names_base = res_m1["feature_names"]
    y      = res_m1["y"]

    i_peh = feat_names_base.index("log10_PeH")
    i_sw  = feat_names_base.index("log10_Sw")
    interaction = X_base[:, i_peh] * X_base[:, i_sw]

    X         = np.column_stack([X_base, interaction])
    feat_names = feat_names_base + ["log10_PeH_x_log10_Sw"]

    fit     = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M2")

    return {
        "model":           "M2",
        "coef":            fit["coef"],
        "feature_names":   feat_names,
        "se":              fit["se"],
        "t_stat":          fit["t_stat"],
        "p_val":           fit["p_val"],
        "metrics":         metrics,
        "yhat":            fit["yhat"],
        "resid":           fit["resid"],
        "law_string":      law_str,
        "X":               X,
        "y":               y,
    }


def fit_scaling_M3_by_regime(df):
    regimes_arr  = _get_regime_col(df)
    y_global     = safe_log10(_df_to_arrays(df, "Cstar"))
    yhat_global  = np.full_like(y_global, np.nan)

    per_regime = {}
    coef_table = []
    _n_min_regime = len(_RA_PREDICTORS) + 1 + 3
    for reg in _RA_REGIMES:
        mask = (regimes_arr == reg)
        if mask.sum() < _n_min_regime:
            continue
        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            sub = df[mask].copy()
        else:
            sub = [r for r, m in zip(df, mask) if m]

        X, feat_names = build_design_matrix(
            sub, _RA_PREDICTORS, add_intercept=True, log_transform=True
        )
        y_sub = safe_log10(_df_to_arrays(sub, "Cstar"))
        fit   = ols_fit(X, y_sub)
        metr  = evaluate_scaling_model(y_sub, fit["yhat"], k=fit["k"])

        yhat_global[mask] = fit["yhat"]
        per_regime[reg] = {
            "coef":         fit["coef"],
            "feature_names":feat_names,
            "se":           fit["se"],
            "t_stat":       fit["t_stat"],
            "p_val":        fit["p_val"],
            "metrics":      metr,
            "n":            int(mask.sum()),
        }
        for i, (c, fn) in enumerate(zip(fit["coef"], feat_names)):
            coef_table.append({
                "regime":   reg,
                "feature":  fn,
                "coef":     float(c),
                "se":       float(fit["se"][i]),
                "p_val":    float(fit["p_val"][i]),
            })

    _w = np.isfinite(yhat_global)
    metrics_global = evaluate_scaling_model(y_global[_w], yhat_global[_w],
                                            k=7 * len(per_regime))
    try:
        metrics_global["AIC"] = float(sum(pr["metrics"]["AIC"] for pr in per_regime.values()))
        metrics_global["BIC"] = float(sum(pr["metrics"]["BIC"] for pr in per_regime.values()))
    except (KeyError, TypeError):
        pass
    return {
        "model":          "M3",
        "per_regime":     per_regime,
        "metrics_global": metrics_global,
        "yhat_global":    yhat_global,
        "y_global":       y_global,
        "coef_table":     coef_table,
    }


def fit_scaling_M4_piecewise(df, pe_break=0.5):
    PeH_arr     = _df_to_arrays(df, "PeH")
    mask_hi     = PeH_arr > pe_break
    mask_lo     = ~mask_hi
    y_global    = safe_log10(_df_to_arrays(df, "Cstar"))
    yhat_global = np.full_like(y_global, np.nan)

    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        sub_hi = df[mask_hi].copy()
        sub_lo = df[mask_lo].copy()
    else:
        sub_hi = [r for r, m in zip(df, mask_hi) if m]
        sub_lo = [r for r, m in zip(df, mask_lo) if m]

    fit_hi = None
    if mask_hi.sum() >= 5:
        X_hi, fn_hi = build_design_matrix(sub_hi, _RA_PREDICTORS,
                                           add_intercept=True, log_transform=True)
        PeH_hi_log = safe_log10(_df_to_arrays(sub_hi, "PeH"))
        X_hi  = np.column_stack([X_hi, PeH_hi_log ** 2])
        fn_hi = fn_hi + ["log10_PeH_sq"]
        y_hi  = y_global[mask_hi]
        fit_hi = ols_fit(X_hi, y_hi)
        fit_hi["metrics"] = evaluate_scaling_model(y_hi, fit_hi["yhat"], k=fit_hi["k"])
        yhat_global[mask_hi] = fit_hi["yhat"]

    fit_lo = None
    if mask_lo.sum() >= 5:
        X_lo, fn_lo = build_design_matrix(sub_lo, _RA_PREDICTORS,
                                           add_intercept=True, log_transform=True)
        y_lo  = y_global[mask_lo]
        fit_lo = ols_fit(X_lo, y_lo)
        fit_lo["metrics"] = evaluate_scaling_model(y_lo, fit_lo["yhat"], k=fit_lo["k"])
        yhat_global[mask_lo] = fit_lo["yhat"]

    k_tot = ((fit_hi["k"] if fit_hi else 0) +
             (fit_lo["k"] if fit_lo else 0))
    _w4 = np.isfinite(yhat_global)
    metrics_global = evaluate_scaling_model(y_global[_w4], yhat_global[_w4], k=k_tot)
    _aic_parts = [f["metrics"]["AIC"] for f in (fit_hi, fit_lo)
                  if f is not None and isinstance(f.get("metrics"), dict)]
    _bic_parts = [f["metrics"]["BIC"] for f in (fit_hi, fit_lo)
                  if f is not None and isinstance(f.get("metrics"), dict)]
    if _aic_parts:
        metrics_global["AIC"] = float(sum(_aic_parts))
        metrics_global["BIC"] = float(sum(_bic_parts))

    return {
        "model":          "M4",
        "fit_hi":         fit_hi,
        "fit_lo":         fit_lo,
        "n_hi":           int(mask_hi.sum()),
        "n_lo":           int(mask_lo.sum()),
        "metrics_global": metrics_global,
        "yhat_global":    yhat_global,
        "y_global":       y_global,
        "pe_break":       pe_break,
    }


def compare_scaling_models_M0_M4(df):
    print("  Fitting M0 (globale)...")
    M0 = fit_scaling_M0_global(df)
    print("  Fitting M1 (+ dummy regime)...")
    M1 = fit_scaling_M1_regime_dummies(df)
    print("  Fitting M2 (+ interazione PeH*Sw)...")
    M2 = fit_scaling_M2_interaction(df)
    print("  Fitting M3 (per regime)...")
    M3 = fit_scaling_M3_by_regime(df)
    print("  Fitting M4 (piecewise PeH)...")
    M4 = fit_scaling_M4_piecewise(df)

    def _row(name, fit_result):
        if name == "M3":
            m = fit_result["metrics_global"]
            k = 7 * max(len(fit_result["per_regime"]), 1)
        elif name == "M4":
            m = fit_result["metrics_global"]
            k = m["k"]
        else:
            m = fit_result["metrics"]
            k = m["k"]
        return {
            "model": name,
            "n": int(m["n"]),
            "k": int(k),
            "R2":   round(m["R2"],   4),
            "RMSE": round(m["RMSE"], 4),
            "AIC":  round(m["AIC"],  1),
            "BIC":  round(m["BIC"],  1),
            "F2":   round(m["F2"],   4),
        }

    table = []
    for name, fit in [("M0", M0), ("M1", M1), ("M2", M2), ("M3", M3), ("M4", M4)]:
        table.append(_row(name, fit))

    def _finite(rows, key):
        return [r for r in rows if isinstance(r.get(key), (int, float))
                and np.isfinite(r[key])]
    _by_aic = _finite(table, "AIC")
    _by_r2  = _finite(table, "R2")
    best_aic = min(_by_aic, key=lambda r: r["AIC"])["model"] if _by_aic else None
    best_r2  = max(_by_r2,  key=lambda r: r["R2"])["model"]  if _by_r2  else None
    _m0_aic = next((r["AIC"] for r in table if r["model"] == "M0"), None)
    for row in table:
        tags = []
        if row["model"] == best_aic: tags.append("AIC minimo")
        if row["model"] == best_r2:  tags.append("R2 massimo")
        if (_m0_aic is not None and isinstance(row.get("AIC"), (int, float))
                and np.isfinite(row["AIC"]) and row["model"] != "M0"):
            d = row["AIC"] - _m0_aic
            tags.append("dAIC vs M0 = {:+.1f}".format(d))
        row["note"] = " / ".join(tags)

    return {"M0": M0, "M1": M1, "M2": M2, "M3": M3, "M4": M4, "table": table}


def nested_f_test(model_reduced, model_full):
    from scipy.stats import f as _f_dist

    def _extract(m):
        if "metrics" in m:
            return m["metrics"]["SSE"], m["metrics"]["k"], m["metrics"]["n"]
        elif "metrics_global" in m:
            return m["metrics_global"]["SSE"], m["metrics_global"]["k"], m["metrics_global"]["n"]
        else:
            return m["SSE"], m["k"], m["n"]

    SSE_r, k_r, n = _extract(model_reduced)
    SSE_f, k_f, _ = _extract(model_full)

    delta_k   = int(k_f - k_r)
    delta_sse = float(SSE_r - SSE_f)
    df2       = int(n - k_f)

    if delta_k <= 0 or SSE_f <= 0 or df2 <= 0:
        return {
            "F_stat": np.nan, "p_val": np.nan,
            "df1": delta_k, "df2": df2,
            "SSE_r": SSE_r, "SSE_f": SSE_f,
            "k_r": k_r, "k_f": k_f, "n": n,
            "delta_k": delta_k, "delta_sse": delta_sse,
        }
    F_stat = (delta_sse / delta_k) / (SSE_f / df2)
    p_val  = float(1.0 - _f_dist.cdf(F_stat, delta_k, df2))

    return {
        "F_stat":    float(F_stat),
        "p_val":     p_val,
        "df1":       delta_k,
        "df2":       df2,
        "SSE_r":     SSE_r,
        "SSE_f":     SSE_f,
        "k_r":       k_r,
        "k_f":       k_f,
        "n":         n,
        "delta_k":   delta_k,
        "delta_sse": delta_sse,
    }


def compute_predictor_vif(df, predictors=None):
    if predictors is None:
        predictors = _RA_PREDICTORS + ["kDP"]

    X_dict = {}
    for pred in predictors:
        try:
            vals = _df_to_arrays(df, pred)
            X_dict[pred] = safe_log10(vals)
        except (KeyError, Exception):
            pass
    preds_ok = list(X_dict.keys())
    X_mat = np.column_stack([X_dict[p] for p in preds_ok])

    vif = {}
    for j, pred in enumerate(preds_ok):
        y_j   = X_mat[:, j]
        X_oth = np.delete(X_mat, j, axis=1)
        Xd    = np.column_stack([np.ones(len(y_j)), X_oth])
        try:
            coef_j = np.linalg.lstsq(Xd, y_j, rcond=None)[0]
            yhat_j = Xd @ coef_j
            ss_res = np.sum((y_j - yhat_j) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2_j   = 1.0 - ss_res / max(ss_tot, 1e-30)
            vif[pred] = float(1.0 / max(1.0 - r2_j, 1e-10))
        except Exception:
            vif[pred] = np.nan

    if "PeH" in X_dict and "kDP" in X_dict:
        r = float(np.corrcoef(X_dict["PeH"], X_dict["kDP"])[0, 1])
        vif["_corr_PeH_kDP"] = r

    return vif


def test_kDP_redundancy(df):
    others = ["DaRn", "phi_g", "Sw", "Pi_src", "Pi_vent"]

    def _fit_variant(preds):
        X, _ = build_design_matrix(df, preds, add_intercept=True, log_transform=True)
        y    = safe_log10(_df_to_arrays(df, "Cstar"))
        fit  = ols_fit(X, y)
        return evaluate_scaling_model(y, fit["yhat"], k=fit["k"])

    V1 = _fit_variant(["PeH", "kDP"] + others)
    V2 = _fit_variant(["PeH"]        + others)
    V3 = _fit_variant(["kDP"]        + others)

    dR2_V1_V2 = abs(V1["R2"] - V2["R2"])
    dR2_V2_V3 = V2["R2"] - V3["R2"]

    if dR2_V1_V2 < 0.002:
        verdict = "ridondante"
        concl = ("kDP e' ridondante rispetto a PeH (delta_R2 = {:.4f}). "
                 "PeH e' il predittore fisicamente fondato. "
                 "kDP va rimosso dal modello finale.".format(dR2_V1_V2))
    else:
        verdict = "informativo"
        concl = ("kDP apporta informazione aggiuntiva (delta_R2 = {:.4f}). "
                 "Verificare la definizione dei gruppi adimensionali.".format(dR2_V1_V2))

    return {
        "V1": V1, "V2": V2, "V3": V3,
        "delta_R2_V1_V2": dR2_V1_V2,
        "delta_R2_V2_V3": dR2_V2_V3,
        "verdict": verdict,
        "conclusion": concl,
    }


def crossval_kfold_scaling(df, model="M0", kfold=10, seed=42):
    y_all   = safe_log10(_df_to_arrays(df, "Cstar"))
    regimes = _get_regime_col(df)
    n       = len(y_all)

    rng        = np.random.default_rng(seed)
    fold_idx   = np.arange(n) % kfold
    fold_idx   = rng.permutation(fold_idx)

    yhat_oot = np.empty(n)
    for k in range(kfold):
        te_mask = fold_idx == k
        tr_mask = ~te_mask

        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_tr = df[tr_mask].copy()
            df_te = df[te_mask].copy()
        else:
            df_tr = [r for r, m in zip(df, tr_mask) if m]
            df_te = [r for r, m in zip(df, te_mask) if m]

        if model == "M1":
            fit_k = fit_scaling_M1_regime_dummies(df_tr)
        else:
            fit_k = fit_scaling_M0_global(df_tr)

        X_te, _ = build_design_matrix(
            df_te, _RA_PREDICTORS, add_intercept=True, log_transform=True
        ) if model == "M0" else (
            _build_M1_design(df_te, fit_k), None
        )
        if model == "M1":
            X_te = _build_M1_design(df_te, fit_k)
        yhat_oot[te_mask] = X_te @ fit_k["coef"]

    sse_oot = float(np.sum((y_all - yhat_oot) ** 2))
    sst_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    Q2      = 1.0 - sse_oot / max(sst_tot, 1e-30)
    rmse    = float(np.sqrt(sse_oot / n))

    return {
        "Q2":          Q2,
        "RMSE_oot":    rmse,
        "kfold":       kfold,
        "model":       model,
        "predictions": yhat_oot,
        "true_values": y_all,
    }


def _build_M1_design(df, m1_fit_ref):
    X_base, _ = build_design_matrix(df, _RA_PREDICTORS,
                                     add_intercept=True, log_transform=True)
    regimes_arr = _get_regime_col(df)
    ref = m1_fit_ref.get("reference_regime", "advective")
    dummy_names = m1_fit_ref.get("dummy_cols", [])
    dummy_cols  = []
    for dname in dummy_names:
        reg = dname.replace("D_", "")
        dummy_cols.append((regimes_arr == reg).astype(float))
    if dummy_cols:
        return np.column_stack([X_base] + dummy_cols)
    return X_base


def crossval_leave_one_regime_out(df, model="M1"):
    regimes_arr = _get_regime_col(df)
    regimes     = [r for r in _RA_REGIMES if (regimes_arr == r).sum() >= 5]
    results     = {}

    for reg in regimes:
        tr_mask = regimes_arr != reg
        te_mask = regimes_arr == reg

        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_tr = df[tr_mask].copy()
            df_te = df[te_mask].copy()
        else:
            df_tr = [r for r, m in zip(df, tr_mask) if m]
            df_te = [r for r, m in zip(df, te_mask) if m]

        y_te = safe_log10(_df_to_arrays(df_te, "Cstar"))

        ref_tr = "advective" if reg != "advective" else \
                 next(r for r in regimes if r != reg)

        try:
            if model == "M1":
                fit_tr = fit_scaling_M1_regime_dummies(df_tr, reference_regime=ref_tr)
                X_te   = _build_M1_design(df_te, fit_tr)
            else:
                fit_tr = fit_scaling_M0_global(df_tr)
                X_te, _ = build_design_matrix(df_te, _RA_PREDICTORS,
                                               add_intercept=True, log_transform=True)
            yhat_te = X_te @ fit_tr["coef"]
        except Exception:
            results[reg] = {"Q2": np.nan, "RMSE": np.nan, "bias": np.nan,
                            "n_test": int(te_mask.sum()), "predictions": None}
            continue

        res    = y_te - yhat_te
        sse_te = float(np.dot(res, res))
        sst_te = float(np.var(y_te) * len(y_te))
        Q2     = 1.0 - sse_te / max(sst_te, 1e-30)
        rmse   = float(np.sqrt(sse_te / max(len(y_te), 1)))
        bias   = float(np.mean(res))

        results[reg] = {
            "Q2":          Q2,
            "RMSE":        rmse,
            "bias":        bias,
            "n_test":      int(te_mask.sum()),
            "predictions": yhat_te,
        }
    return results


def bootstrap_scaling_coefficients(df, fit_func, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(df) if not (_HAS_PANDAS and isinstance(df, _pd.DataFrame)) \
          else len(df)

    ref_fit = fit_func(df)
    k_coef  = len(ref_fit["coef"])
    coefs_boot = np.empty((n_boot, k_coef))
    r2_boot    = np.empty(n_boot)

    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        idx_all = df.index.to_numpy()
    else:
        idx_all = np.arange(n)

    for b in range(n_boot):
        idx_b = rng.choice(idx_all, size=n, replace=True)
        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_b = df.loc[idx_b].copy()
        else:
            df_b = [df[i] for i in idx_b]
        try:
            fit_b = fit_func(df_b)
            coefs_boot[b] = fit_b["coef"]
            r2_boot[b]    = fit_b["metrics"]["R2"] \
                            if "metrics" in fit_b else np.nan
        except Exception:
            coefs_boot[b] = np.nan
            r2_boot[b]    = np.nan

    return {
        "coef_mean":     np.nanmean(coefs_boot, axis=0),
        "coef_ci95_lo":  np.nanpercentile(coefs_boot, 2.5, axis=0),
        "coef_ci95_hi":  np.nanpercentile(coefs_boot, 97.5, axis=0),
        "R2_mean":       float(np.nanmean(r2_boot)),
        "R2_ci95_lo":    float(np.nanpercentile(r2_boot, 2.5)),
        "R2_ci95_hi":    float(np.nanpercentile(r2_boot, 97.5)),
        "feature_names": ref_fit["feature_names"],
        "n_boot":        n_boot,
    }


def verify_exponent_signs(M3_result):
    EXPECTED = {
        "log10_PeH":     +1,
        "log10_DaRn":    -1,
        "log10_phi_g":   +1,
        "log10_Sw":      -1,
        "log10_Pi_src":  +1,
        "log10_Pi_vent": -1,
    }
    PHYSICAL_INTERPRETATIONS = {
        ("log10_Sw", +1, "mixed_transition"):
            "Sw positivo nel Misto: compensazione termica/diffusiva",
        ("log10_Sw", +1, "fracture_bypass"):
            "Sw positivo in Frattura: fratture mantengono conduttivita' ad alta Sw",
        ("log10_phi_g", -1, "barometric_pumping"):
            "phi_g negativo in Barometrico: pompaggio dominante su diffusione",
    }

    results = {}
    for regime, fit_r in M3_result["per_regime"].items():
        feat_names = fit_r["feature_names"]
        coef       = fit_r["coef"]
        p_vals     = fit_r["p_val"]
        n_correct  = 0
        n_total    = 0
        ok_per_coef = {}
        status_per_coef = {}
        anomalies   = []

        for i, (fn, c) in enumerate(zip(feat_names, coef)):
            if fn not in EXPECTED:
                continue
            n_total  += 1
            exp_sign  = EXPECTED[fn]
            act_sign  = int(np.sign(c)) if abs(c) > 1e-6 else 0
            p_val     = float(p_vals[i]) if i < len(p_vals) else np.nan
            if not np.isfinite(p_val):
                status = "indeterminato"
            elif act_sign == exp_sign:
                status = "corretto" if p_val < 0.05 else "corretto_ns"
            else:
                status = "sbagliato" if p_val < 0.05 else "sbagliato_ns"
            sign_ok = status.startswith("corretto")
            status_per_coef[fn] = status
            ok_per_coef[fn] = sign_ok
            if sign_ok:
                n_correct += 1
            else:
                key = (fn, act_sign, regime)
                interp = PHYSICAL_INTERPRETATIONS.get(
                    key, "Segno anomalo: p={:.3g}, beta={:.3f}".format(p_val, c)
                )
                anomalies.append("{}: atteso {:+d}, trovato {:+d} — {}".format(
                    fn, exp_sign, act_sign, interp))

        results[regime] = {
            "expected_signs": {fn: EXPECTED[fn] for fn in EXPECTED},
            "coef_values":    {fn: float(c) for fn, c in zip(feat_names, coef)
                               if fn in EXPECTED},
            "physically_ok":  ok_per_coef,
            "sign_status":    dict(status_per_coef),
            "n_correct":      n_correct,
            "n_total":        n_total,
            "anomalies":      anomalies,
        }
    return results


def make_regimeaware_scaling_report(df, model_results, validation_results,
                                     vif_results):
    import datetime as _dt
    summary = dataset_regime_summary(df)
    M1 = model_results["M1"]
    M3 = model_results["M3"]

    M3_laws = {}
    for reg, fit_r in M3["per_regime"].items():
        M3_laws[reg] = scaling_law_string(
            fit_r["coef"], fit_r["feature_names"], model_name="M3_" + reg
        )

    loro = validation_results.get("loro", {})
    LORO_table = [
        {"regime": reg, "Q2": v.get("Q2", np.nan),
         "RMSE": v.get("RMSE", np.nan), "n_test": v.get("n_test", 0)}
        for reg, v in loro.items()
    ]

    vif_table = [
        {"predictor": k, "VIF": v}
        for k, v in vif_results.get("vif", {}).items()
        if not k.startswith("_")
    ]

    def _best_by_aic(mr):
        best, best_aic = None, float("inf")
        for row in (mr.get("table", []) or []):
            try:
                a = float(row.get("AIC", np.inf)); nm = row.get("model")
            except (TypeError, ValueError):
                continue
            if np.isfinite(a) and a < best_aic:
                best, best_aic = nm, a
        return best

    def _best_by_r2(mr):
        best, best_r2 = None, -float("inf")
        for row in (mr.get("table", []) or []):
            try:
                r = float(row.get("R2", -np.inf)); nm = row.get("model")
            except (TypeError, ValueError):
                continue
            if np.isfinite(r) and r > best_r2:
                best, best_r2 = nm, r
        return best

    report = {
        "timestamp":                _dt.datetime.now().isoformat(),
        "dataset_summary":          summary,
        "model_comparison_table":   model_results.get("table", []),
        "best_predictive_model":    _best_by_aic(model_results),
        "best_predictive_metrics":  M1["metrics"],
        "best_interpretable_model": _best_by_r2(model_results),
        "best_interpretable_metrics": M3["metrics_global"],
        "M1_law":                   M1["law_string"],
        "M3_laws_by_regime":        M3_laws,
        "VIF_table":                vif_table,
        "LORO_table":               LORO_table,
        "Q2_kfold":                 validation_results.get("kfold", {}).get("Q2", np.nan),
        "exponent_signs":           validation_results.get("exponent_signs", {}),
        "kDP_redundancy":           vif_results.get("kDP_test", {}),
        "recommendation": (
            "Modello predittivo con AIC minimo: {}. Modello interpretativo con "
            "R2 massimo: {}. Ridondanza di kDP: {}.".format(
                _best_by_aic(model_results),
                _best_by_r2(model_results),
                (vif_results.get("kDP_test", {}) or {}).get("verdict", "non valutata"),
            )
        ),
    }
    return report


def export_regimeaware_results_csv(df, results, outdir="./regimeaware_output"):
    _os.makedirs(outdir, exist_ok=True)
    paths = {}

    if _HAS_PANDAS:
        p = _os.path.join(outdir, "regimeaware_dataset.csv")
        if isinstance(df, _pd.DataFrame):
            df.to_csv(p, index=False)
        else:
            _pd.DataFrame(df).to_csv(p, index=False)
        paths["dataset"] = p

        if "model_results" in results and "table" in results["model_results"]:
            p = _os.path.join(outdir, "model_comparison.csv")
            _pd.DataFrame(results["model_results"]["table"]).to_csv(p, index=False)
            paths["model_comparison"] = p

        if "model_results" in results and "M1" in results["model_results"]:
            M1 = results["model_results"]["M1"]
            rows_m1 = [
                {"feature": fn, "coef": float(c), "se": float(s), "p_val": float(p2)}
                for fn, c, s, p2 in zip(
                    M1["feature_names"], M1["coef"], M1["se"], M1["p_val"]
                )
            ]
            p = _os.path.join(outdir, "M1_coefficients.csv")
            _pd.DataFrame(rows_m1).to_csv(p, index=False)
            paths["M1_coef"] = p

        if "model_results" in results and "M3" in results["model_results"]:
            p = _os.path.join(outdir, "M3_coefficients_by_regime.csv")
            _pd.DataFrame(results["model_results"]["M3"]["coef_table"]).to_csv(
                p, index=False
            )
            paths["M3_coef"] = p

        vif_data = results.get("vif_results", {}).get("vif", {})
        if vif_data:
            p = _os.path.join(outdir, "VIF_table.csv")
            rows_vif = [{"predictor": k, "VIF": v}
                        for k, v in vif_data.items() if not k.startswith("_")]
            _pd.DataFrame(rows_vif).to_csv(p, index=False)
            paths["VIF"] = p

        loro = results.get("validation_results", {}).get("loro", {})
        if loro:
            p = _os.path.join(outdir, "LORO_results.csv")
            _pd.DataFrame([
                {"regime": r, "Q2": v["Q2"], "RMSE": v["RMSE"], "n_test": v["n_test"]}
                for r, v in loro.items()
            ]).to_csv(p, index=False)
            paths["LORO"] = p

    return paths


def plot_regimeaware_parity(df, model_result, savepath=None, dpi=180):
    REGIME_C = {
        "barometric_pumping": "#e67e22",
        "fracture_bypass":    "#27ae60",
        "decay_limited":      "#8e44ad",
        "advective":          "#e74c3c",
        "mixed_transition":   "#2980b9",
        "unknown":            "#aaaaaa",
    }
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PLOT_DARK)
    ax.set_facecolor(PLOT_PANEL)

    y_obs  = model_result.get("y",    safe_log10(_df_to_arrays(df, "Cstar")))
    y_pred = model_result.get("yhat", y_obs)
    regimes_arr = _get_regime_col(df)

    for reg in list(REGIME_C.keys()):
        mask = regimes_arr == reg
        if not mask.any():
            continue
        ax.scatter(y_obs[mask], y_pred[mask],
                   c=REGIME_C[reg], s=16, alpha=0.65,
                   label=reg.replace("_", " "), linewidths=0)

    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "w-",  lw=1.5, alpha=0.85)
    ax.plot(lims, [v + np.log10(2) for v in lims], "w--", lw=0.9, alpha=0.4)
    ax.plot(lims, [v - np.log10(2) for v in lims], "w--", lw=0.9, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)

    model_name = model_result.get("model", "M?")
    m = model_result.get("metrics", {})
    ax.set_title(
        "Parity Plot — {model}   "
        "R2={r2:.3f}  RMSE={rmse:.3f}  F2={f2:.1%}".format(
            model=model_name,
            r2=m.get("R2", np.nan),
            rmse=m.get("RMSE", np.nan),
            f2=m.get("F2", np.nan),
        ),
        color=PLOT_TEXT, pad=8
    )
    ax.set_xlabel("log10(C*) simulato",  color=PLOT_TEXT)
    ax.set_ylabel("log10(C*) predetto",  color=PLOT_TEXT)
    ax.tick_params(colors=PLOT_TEXT)
    ax.grid(True, color=PLOT_GRID, alpha=0.4)
    ax.legend(framealpha=0.3, fontsize=8, labelcolor=PLOT_TEXT)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor=PLOT_DARK)
    return fig


def plot_regimeaware_coefficients(M3_result, savepath=None, dpi=180):
    EXPECTED_SIGNS = {
        "log10_PeH":     +1, "log10_DaRn":    -1,
        "log10_phi_g":   +1, "log10_Sw":      -1,
        "log10_Pi_src":  +1, "log10_Pi_vent": -1,
    }
    regimes  = list(M3_result["per_regime"].keys())
    feats    = [fn for fn in ["log10_PeH","log10_DaRn","log10_phi_g",
                               "log10_Sw","log10_Pi_src","log10_Pi_vent"]]
    feat_lbl = ["Pe_H","Da_Rn","phi_g","S_w","Pi_src","Pi_vent"]

    n_reg  = len(regimes)
    n_feat = len(feats)
    fig, axes = plt.subplots(1, n_reg, figsize=(3.5 * n_reg, 5),
                              facecolor=PLOT_DARK, sharey=False)
    if n_reg == 1:
        axes = [axes]

    for ax, reg in zip(axes, regimes):
        ax.set_facecolor(PLOT_PANEL)
        fit_r = M3_result["per_regime"][reg]
        coef  = fit_r["coef"]
        se    = fit_r["se"]
        fnames = fit_r["feature_names"]

        betas = []
        errs  = []
        clrs  = []
        for fn in feats:
            if fn in fnames:
                idx = list(fnames).index(fn)
                b   = float(coef[idx])
                s   = float(se[idx])
                exp = EXPECTED_SIGNS.get(fn, 0)
                c   = "#27ae60" if (np.sign(b) == exp or abs(b) < 0.05) \
                               else "#e74c3c"
            else:
                b, s, c = 0.0, 0.0, "#aaaaaa"
            betas.append(b)
            errs.append(s)
            clrs.append(c)

        x = np.arange(n_feat)
        ax.bar(x, betas, color=clrs, alpha=0.85, width=0.65)
        ax.errorbar(x, betas, yerr=errs, fmt="none", color=PLOT_TEXT,
                    capsize=4, lw=1.5)
        ax.axhline(0, color=PLOT_TEXT, lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_lbl, rotation=30, ha="right",
                           color=PLOT_TEXT, fontsize=8)
        ax.tick_params(colors=PLOT_TEXT)
        ax.set_title(reg.replace("_", "\n"), color=PLOT_TEXT, fontsize=9)
        ax.set_ylabel("Esponente beta", color=PLOT_TEXT) if ax == axes[0] else None
        ax.grid(True, axis="y", color=PLOT_GRID, alpha=0.35)

    fig.suptitle("Esponenti M3 per Regime   "
                 "(verde=fisico, rosso=anomalo)",
                 color=PLOT_TEXT, fontsize=11, y=1.01)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor=PLOT_DARK)
    return fig


def run_regimeaware_scaling_pipeline(
    n_samples=1700,
    seed=42,
    nz=15,
    days=1.5,
    outdir="./regimeaware_output",
    run_M2=True,
    run_M4=True,
    verbose=True,
    skip_dataset_generation=False,
    prebuilt_df=None,
):
    _os.makedirs(outdir, exist_ok=True)
    t0_pipeline = _time.time()

    if skip_dataset_generation and prebuilt_df is not None:
        df = prebuilt_df
        if verbose:
            n_df = len(df) if not (_HAS_PANDAS and isinstance(df, _pd.DataFrame)) \
                   else len(df)
            print("[1] Dataset precostruito: n={}".format(n_df))
    else:
        if verbose:
            print("[1] Generazione dataset regime-aware "
                  "(n_samples={}, nz={}, days={})...".format(n_samples, nz, days))
        df = generate_regimeaware_dataset(
            n_samples=n_samples, seed=seed, nz=nz, days=days, verbose=verbose
        )

    if verbose:
        print("[2] Riepilogo dataset per regime...")
    summary = dataset_regime_summary(df)
    if verbose:
        print("  {:25s} {:5s} {:6s} {:12s} {:10s}".format(
            "Regime", "n", "%", "Cin_median", "PeH_median"))
        for row in summary:
            print("  {:25s} {:5d} {:5.1f}% {:12.1f} {:10.4g}".format(
                row["regime"], row["n"], row["percent"],
                row["Cin_median"] if np.isfinite(row["Cin_median"]) else -1,
                row["PeH_median"] if np.isfinite(row["PeH_median"]) else -1))

    if verbose:
        print("[3-7] Fit modelli M0–M4...")
    model_results = compare_scaling_models_M0_M4(df)
    if verbose:
        print("  {:5s} {:5s} {:5s} {:7s} {:7s} {:8s} {:8s} {:6s}".format(
            "Model", "n", "k", "R2", "RMSE", "AIC", "BIC", "F2"))
        for row in model_results["table"]:
            print("  {:5s} {:5d} {:5d} {:7.4f} {:7.4f} {:8.1f} {:8.1f} {:5.1%}".format(
                row["model"], row["n"], row["k"],
                row["R2"], row["RMSE"], row["AIC"], row["BIC"], row["F2"]))

    if verbose:
        print("[8b] F-test nested M0 vs M1/M2...")
    ft_M0_M1 = nested_f_test(model_results["M0"], model_results["M1"])
    ft_M0_M2 = nested_f_test(model_results["M0"], model_results["M2"]) \
               if run_M2 else None
    ft_M1_M2 = nested_f_test(model_results["M1"], model_results["M2"]) \
               if run_M2 else None
    if verbose:
        print("  M0 vs M1: F={:.2f}  p={:.3g}  dk={}".format(
            ft_M0_M1["F_stat"], ft_M0_M1["p_val"], ft_M0_M1["delta_k"]))
        if ft_M0_M2:
            print("  M0 vs M2: F={:.2f}  p={:.3g}".format(
                ft_M0_M2["F_stat"], ft_M0_M2["p_val"]))

    if verbose:
        print("[9] VIF e ridondanza kDP...")
    vif_dict = compute_predictor_vif(df, predictors=list(_RA_PREDICTORS))
    kDP_test  = test_kDP_redundancy(df)
    vif_results = {"vif": vif_dict, "kDP_test": kDP_test}
    if verbose:
        for pred, v in vif_dict.items():
            if not pred.startswith("_"):
                flag = " *** ALTO" if v > 5 else ""
                print("  VIF({}) = {:.2f}{}".format(pred, v, flag))
        print("  Conclusione kDP: {}".format(kDP_test["conclusion"]))

    if verbose:
        print("[11] Validazione LORO...")
    loro_res = crossval_leave_one_regime_out(df, model="M1")
    if verbose:
        for reg, v in loro_res.items():
            print("  LORO {}: Q2={:.3f}  RMSE={:.3f}  n_test={}".format(
                reg, v["Q2"] if np.isfinite(v["Q2"]) else -1,
                v["RMSE"], v["n_test"]))

    if verbose:
        print("[12] Validazione 10-fold...")
    kfold_res = crossval_kfold_scaling(df, model="M0", kfold=10, seed=seed)
    if verbose:
        print("  Q2_10fold (M0) = {:.4f}".format(kfold_res["Q2"]))

    if verbose:
        print("[13] Bootstrap CI 95% (M1, n_boot=500)...")
    boot_res = bootstrap_scaling_coefficients(
        df, fit_scaling_M1_regime_dummies, n_boot=500, seed=seed
    )
    if verbose:
        print("  R2 (bootstrap): {:.4f} [{:.4f}, {:.4f}]".format(
            boot_res["R2_mean"], boot_res["R2_ci95_lo"], boot_res["R2_ci95_hi"]))

    if verbose:
        print("[14] Verifica segni fisici esponenti M3...")
    signs_res = verify_exponent_signs(model_results["M3"])
    if verbose:
        for reg, v in signs_res.items():
            print("  {}: {}/{} segni corretti{}".format(
                reg, v["n_correct"], v["n_total"],
                "" if not v["anomalies"] else " | " + v["anomalies"][0]))

    validation_results = {
        "kfold":          kfold_res,
        "loro":           loro_res,
        "bootstrap":      boot_res,
        "exponent_signs": signs_res,
        "f_tests":        {
            "M0_vs_M1": ft_M0_M1,
            "M0_vs_M2": ft_M0_M2,
            "M1_vs_M2": ft_M1_M2,
        },
    }

    if verbose:
        print("[15] Generazione report strutturato...")
    report = make_regimeaware_scaling_report(df, model_results,
                                              validation_results, vif_results)

    if verbose:
        print("[16] Export CSV in {}...".format(outdir))
    full_results = {
        "model_results":      model_results,
        "validation_results": validation_results,
        "vif_results":        vif_results,
    }
    export_paths = export_regimeaware_results_csv(df, full_results, outdir=outdir)
    if verbose:
        for k, p in export_paths.items():
            print("  {} -> {}".format(k, p))

    if verbose:
        print("[17] Plot parity M1...")
    parity_path = _os.path.join(outdir, "regimeaware_parity_M1.png")
    plot_regimeaware_parity(
        df, model_results["M1"], savepath=parity_path, dpi=180
    )
    if verbose:
        print("[18] Plot esponenti M3...")
    coef_path = _os.path.join(outdir, "regimeaware_coef_M3.png")
    plot_regimeaware_coefficients(
        model_results["M3"], savepath=coef_path, dpi=180
    )

    elapsed = _time.time() - t0_pipeline
    if verbose:
        print("=" * 60)
        print("Pipeline regime-aware completata in {:.1f}s".format(elapsed))
        print("Modello predittivo: M1 (R2={:.4f}, AIC={:.1f})".format(
            model_results["M1"]["metrics"]["R2"],
            model_results["M1"]["metrics"]["AIC"]))
        print("Modello interpretativo: M3 (R2_global={:.4f})".format(
            model_results["M3"]["metrics_global"]["R2"]))
        print("Output in: {}".format(outdir))
        print("=" * 60)

    return {
        "df":                 df,
        "dataset_summary":    summary,
        "model_results":      model_results,
        "validation_results": validation_results,
        "vif_results":        vif_results,
        "report":             report,
        "export_paths":       export_paths,
    }


def run_site_validation(site_name, Ra226, phi_m, k0, Sw, ACH,
                        H=2.0, dP=5.0, eps_rn=0.25, phi_f=0.0,
                        lit_type='tuff_unwelded'):
    K0_PHYS_FLOOR = 1e-16
    if (k0 is None) or (not np.isfinite(k0)) or (k0 < K0_PHYS_FLOOR):
        k0 = K0_PHYS_FLOOR
    override = {
        'Ra226_Bqkg': Ra226,
        'phi_m':      phi_m,
        'k0':         k0,
        'theta0':     Sw * phi_m,
        'ACH_closed': ACH,
        'H':          H,
        'dP_init':    dP,
        'dP_bottom':  dP,
        'eps_max':    eps_rn,
        'phi_f':      phi_f,
    }
    m   = build_model(lit_type, override=override)
    sol = m.simulate_fast(days=10.0, dt_out=3600.0)
    out = m.extract_results(sol)

    t   = np.asarray(out['t'])
    C_bs = np.asarray(out['C_bs'])
    mask = t > (t[-1] - 24 * 3600.0)
    C_ode = float(np.nanmean(C_bs[mask]))

    theta_f = np.asarray(out['theta'])[:, -1]
    Pg_f    = np.asarray(out['Pg'])[:, -1]
    T_f     = m.T_field(sol.t[-1])
    pi      = m.buckingham_pi(theta_f, Pg_f, T_f)

    reg = m.transport_regime(pi.get('Pe_H', np.nan), pi.get('DaDiff', np.nan))

    return {
        'site':    site_name,
        'C_ode':   C_ode,
        'Pi_src':  float(pi.get('Pi_src', np.nan)),
        'regime':  reg.get('regime_name', 'unknown'),
        'Pe_H':    float(pi.get('Pe_H', np.nan)),
    }


def run_gold_validation(site_table, src_correction=1.0):
    import pandas as pd
    rows = []
    for s in site_table:
        res = run_site_validation(
            s['site'],
            s['Ra226'] * src_correction,
            s['phi_m'],
            s['k0'],
            s['Sw'],
            s['ACH'],
            s.get('H', 2.0),
            s.get('dP', 5.0),
            s.get('eps_rn', 0.25),
            s.get('phi_f', 0.0),
            s.get('lit_type', 'tuff_unwelded'),
        )
        C_obs  = s['C_meas']
        C_pred = res['C_ode']
        log_ratio = np.log10(C_pred / max(C_obs, 0.1))
        rows.append({
            'site':        s['site'],
            'domain':      s.get('domain', 'unknown'),
            'C_meas':      C_obs,
            'C_ode_pred':  C_pred,
            'src_correction_used': float(src_correction),
            'log10_ratio': log_ratio,
            'F2_pass':     abs(log_ratio) <= np.log10(2),
            'F5_pass':     abs(log_ratio) <= np.log10(5),
            'regime':      res['regime'],
        })
    df = pd.DataFrame(rows)

    for domain in ['all', 'volcanic', 'sedimentary', 'granite_outlimit']:
        sub = df if domain == 'all' else df[df['domain'] == domain]
        if len(sub) == 0:
            continue
        F2  = sub['F2_pass'].mean() * 100
        F5  = sub['F5_pass'].mean() * 100
        MBE = sub['log10_ratio'].mean()
        print(f"  {domain:20s} N={len(sub):2d} "
              f"F2={F2:.1f}% F5={F5:.1f}% MBE={MBE:+.2f}")
    return df


def run_holdout_validation(csv_path='data/site_validation_master.csv',
                           src_calib_Pi=5.6e7, freeze_split=True,
                           verbose=True):
    import pandas as pd
    df = pd.read_csv(csv_path)

    if 'holdout' not in df.columns:
        raise KeyError("CSV must contain a 'holdout' column (0=derivation, 1=holdout)")

    df_deriv = df[df['holdout'] == 0].copy()
    df_hold  = df[df['holdout'] == 1].copy()

    rho_b = 2400.0
    pi_src_field = (rho_b * df_deriv['Ra226'] * df_deriv['eps_rn']
                    / (_LAM_RN * _C_REF))
    pi_src_field_median = float(np.median(pi_src_field))
    src_correction = src_calib_Pi / pi_src_field_median

    if verbose:
        print(f"[holdout] derivation N={len(df_deriv)}  holdout N={len(df_hold)}")
        print(f"[holdout] median Pi_src_field (derivation only) = "
              f"{pi_src_field_median:.3e}")
        print(f"[holdout] derived src_correction = x{src_correction:.2f}")

    def _subset_metrics(df_subset, label):
        if len(df_subset) == 0:
            return None
        site_table = df_subset.to_dict('records')
        res = run_gold_validation(site_table, src_correction=src_correction)
        rows = []
        for domain in ['all', 'volcanic', 'sedimentary', 'granite_outlimit']:
            sub = res if domain == 'all' else res[res['domain'] == domain]
            if len(sub) == 0:
                continue
            rows.append({
                'split':   label,
                'subset':  domain,
                'N':       int(len(sub)),
                'F2_pct':  round(sub['F2_pass'].mean() * 100, 1),
                'F5_pct':  round(sub['F5_pass'].mean() * 100, 1),
                'MBE_log': round(sub['log10_ratio'].mean(), 3),
            })
        return pd.DataFrame(rows)

    if verbose:
        print("\n[holdout] === HOLDOUT-SET metrics (independent) ===")
    metrics_hold = _subset_metrics(df_hold, 'holdout')
    if verbose:
        print("\n[holdout] === DERIVATION-SET metrics (in-sample) ===")
    metrics_deriv = _subset_metrics(df_deriv, 'derivation')

    return {
        'src_correction': src_correction,
        'df_deriv':       df_deriv,
        'df_hold':        df_hold,
        'metrics_hold':   metrics_hold,
        'metrics_deriv':  metrics_deriv,
    }


def sobol_sensitivity_scaling(beta=None, intercept=_M0_INTERCEPT_FROZEN, n_base=1024,
                              calc_second_order=False, seed=42, verbose=True):
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    if beta is None:
        beta = [0.0297, -0.8879, -1.0848, -1.0316, 0.5766, -0.1678]

    problem = {
        'num_vars': 6,
        'names': ['log_Pe_H', 'log_Da_Rn', 'log_phi_g',
                  'log_Sw', 'log_Pi_src', 'log_Pi_vent'],
        'bounds': [
            [-2.95,  1.50],
            [ 1.14,  3.59],
            [-1.93, -0.41],
            [-1.50, -0.48],
            [ 0.75,  2.58],
            [ 0.72,  2.42],
        ],
    }

    try:
        param_values = saltelli.sample(problem, n_base,
                                       calc_second_order=calc_second_order,
                                       seed=seed)
    except TypeError:
        np.random.seed(seed)
        param_values = saltelli.sample(problem, n_base,
                                       calc_second_order=calc_second_order)
    Y = intercept + param_values.dot(np.asarray(beta))

    try:
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                           print_to_console=False, seed=seed)
    except TypeError:
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                           print_to_console=False)
    Si['names'] = problem['names']

    if verbose:
        print("[sobol] First-order indices (S1):")
        for name, s1, s1c in zip(problem['names'], Si['S1'], Si['S1_conf']):
            print(f"  {name:12s} S1 = {s1:+.3f} +/- {s1c:.3f}")
        print("[sobol] Total-order indices (ST):")
        for name, st, stc in zip(problem['names'], Si['ST'], Si['ST_conf']):
            print(f"  {name:12s} ST = {st:+.3f} +/- {stc:.3f}")
    return Si


def predict_with_interval(Pi_groups, beta=None, intercept=_M0_INTERCEPT_FROZEN,
                          rmse_log=_M0_RMSE_LOG_FROZEN, c_ref=_C_REF, z=1.96):
    if beta is None:
        beta = [0.0297, -0.8879, -1.0848, -1.0316, 0.5766, -0.1678]
    log_C_star = intercept + sum(b * np.log10(max(p, 1e-30))
                                 for b, p in zip(beta, Pi_groups))
    C_in    = 10 ** log_C_star * c_ref
    C_in_lo = 10 ** (log_C_star - z * rmse_log) * c_ref
    C_in_hi = 10 ** (log_C_star + z * rmse_log) * c_ref
    return C_in, C_in_lo, C_in_hi


def nazaroff_1992(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5):
    phi_g     = phi_m * (1.0 - Sw)
    D_eff     = D0 * phi_g ** (7.0 / 3.0) / phi_m ** 2
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    S         = rho_b * Ra226 * eps_rn * lambda_Rn
    C_soil    = S / lambda_Rn
    H_room    = 2.5
    transfer  = (D_eff / H) / (H_room * (ACH / 3600.0 + lambda_Rn))
    C_in      = C_soil * transfer
    return C_in


def jobbagy_2017(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5):
    phi_g     = phi_m * (1.0 - Sw)
    D_eff     = D0 * phi_g ** (7.0 / 3.0) / phi_m ** 2
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    L_d       = np.sqrt(D_eff / lambda_Rn)
    E_surface = rho_b * Ra226 * eps_rn * lambda_Rn * L_d
    H_room    = 2.5
    C_in      = (E_surface / H_room) / (ACH / 3600.0 + lambda_Rn)
    return C_in


def sainz_2018(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5, k_adv=0.15):
    phi_g     = phi_m * (1.0 - Sw)
    D_eff     = D0 * phi_g ** (7.0 / 3.0) / phi_m ** 2
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    L_d       = np.sqrt(D_eff / lambda_Rn)
    E_diff    = rho_b * Ra226 * eps_rn * lambda_Rn * L_d
    E_total   = E_diff * (1.0 + k_adv)
    H_room    = 2.5
    C_in      = (E_total / H_room) / (ACH / 3600.0 + lambda_Rn)
    return C_in


def _metrics_logspace(C_pred, C_meas):
    C_pred = np.asarray(C_pred, dtype=float)
    C_meas = np.asarray(C_meas, dtype=float)
    ok = np.isfinite(C_pred) & np.isfinite(C_meas) & (C_pred > 0) & (C_meas > 0)
    C_pred, C_meas = C_pred[ok], C_meas[ok]
    if len(C_pred) == 0:
        return dict(N=0, F2_pct=np.nan, F5_pct=np.nan,
                    MBE_log=np.nan, R2_log=np.nan)
    log_ratio = np.log10(C_pred / C_meas)
    F2  = float(np.mean(np.abs(log_ratio) <= np.log10(2)) * 100)
    F5  = float(np.mean(np.abs(log_ratio) <= np.log10(5)) * 100)
    MBE = float(np.mean(log_ratio))
    y      = np.log10(C_meas)
    yhat   = np.log10(C_pred)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(N=int(len(C_pred)), F2_pct=round(F2, 1), F5_pct=round(F5, 1),
                MBE_log=round(MBE, 3), R2_log=round(R2, 3))


def run_literature_benchmark(csv_path='data/site_validation_master.csv',
                             m0_src_correction=7.0, verbose=True):
    import pandas as pd
    df = pd.read_csv(csv_path)

    C_meas = df['C_meas'].to_numpy(dtype=float)

    naz = np.array([nazaroff_1992(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])
    job = np.array([jobbagy_2017(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])
    sai = np.array([sainz_2018(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])


    C_m0 = np.array([
        _apply_M0_simple({
            'phi_m': r.phi_m, 'Sw': r.Sw, 'ACH': r.ACH,
            'Ra226': r.Ra226, 'eps_rn': r.eps_rn, 'H': r.H,
            'k0': getattr(r, 'k0', 1e-15), 'dP': getattr(r, 'dP', 5.0),
        })
        for r in df.itertuples()
    ])

    per_site = pd.DataFrame({
        'site':           df['site'],
        'C_meas':         C_meas,
        'C_nazaroff1992': naz,
        'C_jobbagy2017':  job,
        'C_sainz2018':    sai,
        'C_M0_corr':      C_m0,
    })

    rows = []
    for label, pred in [('Nazaroff (1992)', naz),
                        ('Jobbagy et al. (2017)', job),
                        ('Sainz et al. (2018)', sai),
                        ('M0 (stationary, regime-aware)', C_m0)]:
        m = _metrics_logspace(pred, C_meas)
        m['model'] = label
        rows.append(m)
    summary = pd.DataFrame(rows)[['model', 'N', 'F2_pct', 'F5_pct',
                                  'MBE_log', 'R2_log']]

    if verbose:
        print("[literature] Comparative benchmark on field sites:")
        print(summary.to_string(index=False))

    return {'per_site': per_site, 'summary': summary}


def _soil_column_steady(D_eff, lambda_Rn, S, H, N_z, C_top=0.0):
    dz = H / (N_z - 1)
    a  = D_eff / dz ** 2
    main = np.full(N_z, -(2.0 * a + lambda_Rn))
    low  = np.full(N_z, a)
    up   = np.full(N_z, a)
    rhs  = np.full(N_z, -S)

    main[0] = 1.0
    up[0]   = 0.0
    rhs[0]  = C_top
    main[-1] = -(2.0 * a + lambda_Rn)
    low[-1]  = 2.0 * a
    rhs[-1]  = -S

    n = N_z
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = up[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        m = main[i] - low[i] * cp[i - 1]
        cp[i] = up[i] / m if i < n - 1 else 0.0
        dp[i] = (rhs[i] - low[i] * dp[i - 1]) / m
    C = np.zeros(n)
    C[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        C[i] = dp[i] - cp[i] * C[i + 1]
    return C


def solve_1D_column(N_z, dz, D_eff, lambda_Rn, S, params):
    H   = params['H']
    N_z = int(round(H / dz)) + 1 if dz > 0 else N_z
    C   = _soil_column_steady(D_eff, lambda_Rn, S, H, N_z, C_top=0.0)
    dz_eff = H / (N_z - 1)
    J_entry = D_eff * (C[1] - C[0]) / dz_eff
    ACH = params['ACH']
    H_room = float(params.get('H_room', 2.5))
    C_indoor = (J_entry / H_room) / (ACH / 3600.0 + lambda_Rn)
    return float(C_indoor)


def solve_coupled_1D_columns(N_x, N_z, dz, dx, D_eff, D_eff_x,
                             lambda_Rn, S, params, crack_cols):
    H = params['H']
    N_z = int(round(H / dz)) + 1
    dz_eff = H / (N_z - 1)
    D_enh = float(params.get("crack_D_enhance", 3.0))
    D_col = [D_eff * D_enh if i in crack_cols else D_eff for i in range(N_x)]
    cols = [_soil_column_steady(D_col[i], lambda_Rn, S, H, N_z, C_top=0.0)
            for i in range(N_x)]
    cols = np.array(cols)

    G_max = max(float(np.mean(_soil_column_steady(D_col[i], lambda_Rn,
                                                  1.0, H, N_z, C_top=0.0)))
                for i in range(N_x))
    rho_bound = 4.0 * D_eff_x / dx ** 2 * max(G_max, 0.0)
    omega = 1.0 if rho_bound <= 1.0 else 1.0 / rho_bound

    TOL, MAX_IT = 1e-6, 2000
    converged, resid = False, np.inf
    for _ in range(MAX_IT):
        cols_tgt = cols.copy()
        for i in range(N_x):
            left  = cols[i - 1] if i > 0 else cols[i]
            right = cols[i + 1] if i < N_x - 1 else cols[i]
            lap_x = D_eff_x * (left - 2.0 * cols[i] + right) / dx ** 2
            S_eff = S + lap_x
            S_scalar = float(np.mean(S_eff))
            cols_tgt[i] = _soil_column_steady(D_col[i], lambda_Rn,
                                              S_scalar, H, N_z, C_top=0.0)
        resid = np.max(np.abs(cols_tgt - cols)) / (np.max(np.abs(cols)) + 1e-30)
        cols = cols + omega * (cols_tgt - cols)
        if not np.all(np.isfinite(cols)):
            raise RuntimeError(
                "solve_coupled_1D_columns: soluzione non finita "
                "(N_x={}, dx={:.4g}, D_eff_x={:.4g}, rho_bound={:.4g})".format(
                    N_x, dx, D_eff_x, rho_bound))
        if resid < TOL:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "solve_coupled_1D_columns: non convergente "
            "(resid={:.3e} > TOL={:.1e}, N_x={}, dx={:.4g}, omega={:.4g}, "
            "rho_bound={:.4g})".format(resid, TOL, N_x, dx, omega, rho_bound))
    if np.any(cols < 0.0):
        raise RuntimeError(
            "solve_coupled_1D_columns: concentrazioni NEGATIVE nella soluzione "
            "laterale (min={:.3e}): geometria incompatibile".format(
                float(np.min(cols))))

    ACH = params['ACH']
    J_total = 0.0
    for i in range(N_x):
        J_i = D_col[i] * (cols[i][1] - cols[i][0]) / dz_eff
        weight = 1.0 if i in crack_cols else 0.5
        J_total += weight * J_i
    J_mean = J_total / N_x
    H_room = float(params.get('H_room', 2.5))
    C_indoor_2D = (J_mean / H_room) / (ACH / 3600.0 + lambda_Rn)
    return float(C_indoor_2D)


def quasi2D_radon_model(params, N_x=10, L_slab=10.0, crack_spacing=2.0):
    H   = params['H']
    N_z = 20
    dz  = H / N_z
    dx  = L_slab / N_x

    phi_g     = params['phi_m'] * (1.0 - params['Sw'])
    D_eff     = 1.1e-5 * phi_g ** (7.0 / 3.0) / params['phi_m'] ** 2
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    S         = rho_b * params['Ra226'] * params['eps_rn'] * lambda_Rn

    D_eff_x   = D_eff * 0.5

    crack_positions = np.arange(crack_spacing / 2.0, L_slab, crack_spacing)
    crack_cols = [int(x / dx) for x in crack_positions]

    C_indoor_2D = solve_coupled_1D_columns(
        N_x, N_z, dz, dx, D_eff, D_eff_x, lambda_Rn, S, params, crack_cols)
    C_indoor_1D = solve_1D_column(N_z, dz, D_eff, lambda_Rn, S, params)

    ratio_2D_1D = C_indoor_2D / C_indoor_1D if C_indoor_1D > 0 else np.nan
    return C_indoor_2D, C_indoor_1D, ratio_2D_1D


def run_quasi2d_campi_flegrei(verbose=True):
    params = {
        'Ra226':  140.0,
        'eps_rn': 0.30,
        'phi_m':  0.38,
        'Sw':     0.38,
        'H':      2.0,
        'ACH':    0.5,
        'dP':     5.0,
    }
    C_meas = 276.0
    C_2D, C_1D, ratio = quasi2D_radon_model(
        params, N_x=10, L_slab=10.0, crack_spacing=2.0)
    factor_1D = C_1D / C_meas if C_meas > 0 else np.nan
    factor_2D = C_2D / C_meas if C_meas > 0 else np.nan
    out = {
        'site':        'Campi_Flegrei_tuff',
        'C_meas':      C_meas,
        'C_1D':        C_1D,
        'C_2D':        C_2D,
        'ratio_2D_1D': ratio,
        'factor_1D':   factor_1D,
        'factor_2D':   factor_2D,
    }
    if verbose:
        print("[quasi-2D] Campi Flegrei tuff site:")
        print(f"  C_1D  = {C_1D:8.1f} Bq/m^3")
        print(f"  C_2D  = {C_2D:8.1f} Bq/m^3")
        print(f"  C_meas= {C_meas:8.1f} Bq/m^3")
        print(f"  ratio C_2D/C_1D = {ratio:.3f}  "
              f"(within factor 2: {abs(np.log10(ratio)) <= np.log10(2)})")
    return out


def _apply_M0_simple(params):
    rho_b     = 1600.0
    lambda_Rn = _LAM_RN
    phi_m = params['phi_m']; Sw = params['Sw']
    phi_g = phi_m * (1.0 - Sw)
    D0    = 1.1e-5
    D_eff = D0 * phi_g ** (7.0 / 3.0) / phi_m ** 2
    ACH   = params['ACH']
    H_room = params.get('H_room', 2.5)
    L_d   = np.sqrt(D_eff / lambda_Rn)
    E_diff = rho_b * params['Ra226'] * params['eps_rn'] * lambda_Rn * L_d
    k0  = params.get('k0', 1e-15)
    dP  = params.get('dP', 5.0)
    H   = params.get('H', 2.0)
    PeH = (k0 * dP) / (1.82e-5 * max(D_eff, 1e-30))
    k_adv = 0.15 * float(np.tanh(PeH / 0.1))
    E_total = E_diff * (1.0 + k_adv)
    C_in = (E_total / H_room) / (ACH / 3600.0 + lambda_Rn)
    return float(C_in)


def _synthetic_ingv_monthly(annual_mean=276.0, seed=42):
    import pandas as pd
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2011-01-01', periods=84, freq='MS')
    month = dates.month.to_numpy()
    season = np.cos(2 * np.pi * (month - 1) / 12.0)
    Sw  = 0.38 + 0.10 * season
    ACH = 0.50 - 0.20 * season
    C_meas = annual_mean * (1.0 + 0.30 * season) \
        * (1.0 + 0.05 * rng.standard_normal(84))
    return pd.DataFrame({'date': dates, 'Sw_monthly': Sw,
                         'ACH_monthly': ACH, 'C_in': C_meas})


def temporal_validation(monthly_csv=None, params_base=None, verbose=True):
    import os
    import pandas as pd

    if params_base is None:
        params_base = {'Ra226': 140.0, 'eps_rn': 0.30, 'phi_m': 0.38,
                       'k0': 1e-15, 'H': 2.0, 'dP': 5.0}

    source = 'INGV file'
    if monthly_csv and os.path.exists(monthly_csv):
        df = pd.read_csv(monthly_csv, parse_dates=['date'])
        if 'Sw_monthly' not in df.columns:
            df['Sw_monthly'] = params_base.get('Sw', 0.38)
        if 'ACH_monthly' not in df.columns:
            df['ACH_monthly'] = 0.5
    else:
        source = 'synthetic (Sabbarese et al. 2020 seasonal cycle)'
        df = _synthetic_ingv_monthly(annual_mean=276.0)

    C_pred = []
    for _, row in df.iterrows():
        p = params_base.copy()
        p['Sw']  = float(row['Sw_monthly'])
        p['ACH'] = float(row['ACH_monthly'])
        C_pred.append(_apply_M0_simple(p))
    df['C_pred'] = C_pred

    C_obs  = df['C_in'].to_numpy(dtype=float)
    C_hat  = np.asarray(C_pred, dtype=float)

    R2_temporal  = 1.0 - float(np.sum((C_obs - C_hat)**2)) / max(float(np.sum((C_obs - np.mean(C_obs))**2)), 1e-30)
    MBE_temporal = float(np.mean(np.log10(C_hat / C_obs)))
    F2_temporal  = float(np.mean(np.abs(np.log10(C_hat / C_obs)) < np.log10(2)))

    C_mean_measured  = float(C_obs.mean())
    C_mean_predicted = float(C_hat.mean())
    ratio_annual     = C_mean_predicted / C_mean_measured

    peak_trough = float(C_obs.max() / C_obs.min())

    out = {
        'source':           source,
        'n_months':         int(len(df)),
        'R2_temporal':      round(float(R2_temporal), 3),
        'MBE_temporal':     round(MBE_temporal, 3),
        'F2_temporal':      round(F2_temporal, 3),
        'C_mean_measured':  round(C_mean_measured, 1),
        'C_mean_predicted': round(C_mean_predicted, 1),
        'ratio_annual':     round(ratio_annual, 3),
        'peak_trough':      round(peak_trough, 3),
        'monthly':          df,
    }
    if verbose:
        print(f"[temporal] source = {source}  (N = {len(df)} months)")
        print(f"[temporal] R2(temporal)  = {out['R2_temporal']}")
        print(f"[temporal] MBE(temporal) = {out['MBE_temporal']} log10")
        print(f"[temporal] F2(temporal)  = {out['F2_temporal']:.1%}")
        print(f"[temporal] annual mean measured  = {out['C_mean_measured']} Bq/m^3")
        print(f"[temporal] annual mean predicted = {out['C_mean_predicted']} Bq/m^3")
        print(f"[temporal] ratio (pred/meas)     = {out['ratio_annual']}")
        print(f"[temporal] seasonal peak/trough  = {out['peak_trough']}")
    return out


import os
import json
import math
import hashlib
import datetime as _dt

import numpy as np


_RESPONSIVE_REGIMES = ("advective", "fracture_bypass", "barometric_pumping",
                       "mixed_transition")
_INERT_REGIMES = ("decay_limited",)

_FORBIDDEN_INPUT_KEYS = (
    "c_meas", "c_radon", "rn_meas", "radon_meas", "c_in_meas", "measured",
    "c_unrest", "c_quiet", "c_obs", "rn222_meas", "activity_meas",
    "r_meas", "lag_days_meas",
)


def _assert_no_measurement_leak(records):
    for rec in records:
        for key in rec.keys():
            kl = str(key).strip().lower()
            for bad in _FORBIDDEN_INPUT_KEYS:
                if bad in kl:
                    raise ValueError(
                        "BLIND-VALIDATION VIOLATION: input record contains a "
                        f"measurement-like column '{key}'. Tranche A must hold "
                        "only input parameters (site + deformation + barometric "
                        "forcing). Remove all measured-radon columns and retry."
                    )
    return True


def _seismic_permeability_factor(M_w, site=None, k_seis_max=None,
                                 A_seis=None, gamma_seis=None, M_ref=None):
    site_p = {}
    if site is not None:
        _key = str(site).replace("_", " ").strip().lower()
        for _k, _v in SEISMIC_SITES.items():
            if str(_k).strip().lower() == _key:
                site_p = _v
                break
    A     = float(A_seis)     if A_seis     is not None else float(site_p.get("A_seis", 0.02))
    gamma = float(gamma_seis) if gamma_seis is not None else float(site_p.get("gamma_seis", 1.8))
    Mref  = float(M_ref)      if M_ref      is not None else float(site_p.get("M_ref", 2.5))
    cap   = float(k_seis_max) if k_seis_max is not None else 10.0
    k_seis = 1.0 + A * math.exp(gamma * (float(M_w) - Mref))
    return float(np.clip(k_seis, 1.0, cap))


_TAU_SEIS_RECOVERY_DAYS = {
    "Campi Flegrei":   18.0,
    "Etna":             9.0,
    "Catalonia":        7.0,
    "Iceland":          6.0,
    "default":         12.0,
}


def _tau_seis_recovery_seconds(site=None, tau_days=None):
    if tau_days is not None and np.isfinite(float(tau_days)):
        return float(tau_days) * 86400.0
    d = _TAU_SEIS_RECOVERY_DAYS["default"]
    if site is not None:
        _key = str(site).replace("_", " ").strip().lower()
        for _k, _v in _TAU_SEIS_RECOVERY_DAYS.items():
            if str(_k).strip().lower() == _key:
                d = _v
                break
    return float(d) * 86400.0


def seismic_recovery_factor(t, t0, k_seis_peak, tau, k_floor=1.0):
    t   = np.asarray(t, dtype=float)
    dt  = t - float(t0)
    amp = max(float(k_seis_peak) - 1.0, 0.0)
    k   = k_floor + amp * np.exp(-np.clip(dt, 0.0, None) / max(float(tau), 1e-9))
    k   = np.where(dt < 0.0, k_floor, k)
    return float(k) if np.ndim(k) == 0 else k


def _time_to_recovery_peak_days(tau, u_eff_days):
    tau_d = max(float(tau) / 86400.0, 1e-6)
    t_tr  = max(float(u_eff_days), 1e-6)
    if abs(tau_d - t_tr) < 1e-6:
        return float(t_tr)
    t_peak = (tau_d * t_tr) / (tau_d - t_tr) * math.log(tau_d / t_tr)
    return float(max(t_peak, 0.0))


_SE_EXP_EFF = {
    "advective":          0.046,
    "fracture_bypass":    0.046,
    "barometric_pumping": 0.046,
    "mixed_transition":   0.017,
    "decay_limited":      0.0,
}
_SIGMA_LOG10_KSEIS_DEFAULT = 0.12


_SIGMA_LOG10_RMEAS_OBS = 0.11


def _amplitude_band_R(R_pred, k_seis, regime, z=1.96,
                      sigma_log10_kseis=_SIGMA_LOG10_KSEIS_DEFAULT,
                      R_inert_tol=1.10,
                      sigma_log10_Rmeas_obs=_SIGMA_LOG10_RMEAS_OBS):
    if regime in _INERT_REGIMES:
        sig = math.log10(max(R_inert_tol, 1.0 + 1e-6)) / max(z, 1e-9)
        R_lo = 10 ** (0.0 - z * sig)
        R_hi = 10 ** (0.0 + z * sig)
        return float(R_lo), float(R_hi), float(sig)
    E = _effective_exponent(regime)
    SE_E = _SE_EXP_EFF.get(regime, 0.060)
    log_k = math.log10(max(float(k_seis), 1e-30))
    var_logR = ((log_k ** 2) * (SE_E ** 2)
                + (E ** 2) * (sigma_log10_kseis ** 2)
                + sigma_log10_Rmeas_obs ** 2)
    sig = math.sqrt(max(var_logR, 0.0))
    log_R = math.log10(max(float(R_pred), 1e-30))
    R_lo = 10 ** (log_R - z * sig)
    R_hi = 10 ** (log_R + z * sig)
    return float(R_lo), float(R_hi), float(sig)


def _effective_exponent(regime, beta=_M0_BETA_FROZEN):
    beta_PeH_eff = {
        "advective":          0.4212,
        "fracture_bypass":    0.4212,
        "barometric_pumping": 0.4212,
        "mixed_transition":   0.1741,
        "decay_limited":      0.0,
    }.get(regime, beta["Pe_H"])
    return beta_PeH_eff


def _sigma_log10_R_physics(rec, k_seis, regime,
                           sigma_log10_kseis=_SIGMA_LOG10_KSEIS_DEFAULT,
                           sigma_log10_Rmeas_obs=_SIGMA_LOG10_RMEAS_OBS,
                           h=1e-3, **kw):
    R_hi_ = _amplitude_factor_R_physics(rec, k_seis * 10.0 ** h, regime, **kw)
    R_lo_ = _amplitude_factor_R_physics(rec, k_seis * 10.0 ** (-h), regime, **kw)
    S = (math.log10(max(R_hi_, 1e-30))
         - math.log10(max(R_lo_, 1e-30))) / (2.0 * h)
    var_logR = (S * sigma_log10_kseis) ** 2 + sigma_log10_Rmeas_obs ** 2
    return float(math.sqrt(max(var_logR, 0.0))), float(S)


def _amplitude_factor_R(k_seis, regime, beta=_M0_BETA_FROZEN):
    beta_PeH_eff = {
        "advective":          0.4212,
        "fracture_bypass":    0.4212,
        "barometric_pumping": 0.4212,
        "mixed_transition":   0.1741,
        "decay_limited":      0.0,
    }.get(regime, beta["Pe_H"])
    exp_eff = beta_PeH_eff
    if exp_eff < 0.0:
        exp_eff = 0.0
    if regime in _INERT_REGIMES:
        return 1.0
    R = float(k_seis) ** exp_eff
    return max(R, 1.0)


_LAMBDA_RN = math.log(2.0) / (3.8235 * 86400.0)


_RHO_CO2_SOIL = 1.80
_F_CO2_TO_MS = 1.0 / (86400.0 * _RHO_CO2_SOIL * 1000.0)


def _co2_carrier_pore_velocity(F_CO2, phi_g):
    phi_g = max(float(phi_g), 1e-4)
    q_gas = max(float(F_CO2), 0.0) * _F_CO2_TO_MS
    return q_gas / phi_g


_F_CO2_STAR = 300.0


def _carrier_dilution_factor(F_CO2, F_star=_F_CO2_STAR):
    x = max(float(F_CO2), 0.0) / max(float(F_star), 1e-9)
    return 2.0 * x / (1.0 + x * x)


_MU_WATER   = 1.0e-3
_S_STORAGE  = 1.0e-9
_L_SOURCE_DEFAULT = 1000.0


def _pressure_gradient_length(rec, H):
    L = rec.get("L_source", None) if isinstance(rec, dict) else None
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = None
    if L is None or not math.isfinite(L) or L <= 0.0:
        L = _L_SOURCE_DEFAULT
    return max(L, float(H), 1e-3)


def _pore_darcy_velocity(k, dP, H, phi_g, mu=1.82e-5):
    phi_g = max(float(phi_g), 1e-4)
    v_darcy = (float(k) / mu) * (float(dP) / max(float(H), 1e-3))
    return abs(v_darcy) / phi_g


def _migration_length(v_pore, De, lam=_LAMBDA_RN):
    v = max(float(v_pore), 0.0)
    De = max(float(De), 1e-12)
    disc = math.sqrt(v * v + 4.0 * De * lam)
    denom = disc - v
    if denom < 1e-30:
        return v / lam
    return 2.0 * De / denom


def _quasi_steady_surface_flux(v, De, H, lam, G):
    De = max(float(De), 1e-12)
    v = max(float(v), 0.0)
    Cparticular = float(G) / max(float(lam), 1e-30)
    disc = math.sqrt(v ** 2 + 4.0 * De * lam)
    r1 = 2.0 * lam / (v + disc)
    r2 = -(v + disc) / (2.0 * De)
    e1_0 = math.exp(np.clip(r1 * (0.0 - H), -700, 0.0))
    e2_0 = math.exp(np.clip(r2 * (0.0 - H), -700, 700))
    Msys = np.array([[e1_0, e2_0], [r1, r2]])
    rhs_sys = np.array([-Cparticular, 0.0])
    try:
        Asol, Bsol = np.linalg.solve(Msys, rhs_sys)
    except np.linalg.LinAlgError:
        return 0.0
    dCdx_0 = Asol * r1 * e1_0 + Bsol * r2 * e2_0
    return max(float(De * dCdx_0), 0.0)


def _amplitude_factor_R_physics(rec, k_seis, regime,
                                g_dP_seis=None, z_sample=None,
                                De_quiet=None, co2_carrier_flux_on=True,
                                carrier_dilution_on=False,
                                ):
    if regime in _INERT_REGIMES:
        return 1.0
    eta = {
        "advective":          1.00,
        "fracture_bypass":    1.00,
        "barometric_pumping": 0.60,
        "mixed_transition":   0.35,
        "decay_limited":      0.0,
    }.get(regime, 0.30)

    k0   = float(rec.get("k0", 1e-13))
    H    = float(rec.get("H", 3.0))
    Sw   = float(rec.get("Sw", 0.4))
    phi  = float(rec.get("phi_m", 0.3))
    dP   = float(rec.get("dP", 1.0e5))
    phi_g = max(phi * (1.0 - Sw), 1e-3)
    D0 = _D0_REF
    De = (De_quiet if De_quiet is not None
          else D0 * (phi_g ** (7.0/3.0)) / max(phi ** 2, 1e-6))
    De = max(De, 1e-8)
    z_s = float(z_sample) if z_sample is not None else H
    g_dP = float(g_dP_seis) if g_dP_seis is not None else math.sqrt(max(k_seis, 1.0))


    if co2_carrier_flux_on and (rec.get("CO2_flux", None) is not None):
        F_CO2 = float(rec.get("CO2_flux"))
        v_q = _co2_carrier_pore_velocity(F_CO2, phi_g)
        v_u = v_q * (1.0 + eta * (k_seis - 1.0))
    else:
        dP_quiet = max(dP, 1.0e3)
        L_grad   = _pressure_gradient_length(rec, H)
        v_q = _pore_darcy_velocity(k0, dP_quiet, L_grad, phi_g)
        k_un  = k0 * (1.0 + eta * (k_seis - 1.0))
        dP_un = dP_quiet * (1.0 + eta * (g_dP - 1.0))
        v_u = _pore_darcy_velocity(k_un, dP_un, L_grad, phi_g)


    v_q = max(v_q, 1e-30)
    v_u = max(v_u, 1e-30)
    rho_b_ = 1600.0
    G = rho_b_ * float(rec.get("Ra226", 100.0)) * float(rec.get("eps_rn", 0.3)) * _LAMBDA_RN
    q_q = _quasi_steady_surface_flux(v_q, De, H, _LAMBDA_RN, G)
    q_u = _quasi_steady_surface_flux(v_u, De, H, _LAMBDA_RN, G)
    R = q_u / max(q_q, 1e-30)


    if carrier_dilution_on and (rec.get("CO2_flux", None) is not None):
        A = _carrier_dilution_factor(float(rec.get("CO2_flux")))
        R = 1.0 + (R - 1.0) * A


    return float(max(R, 1.0))


def _pore_pressure_diffusion_lag_days(rec, k_seis):
    k0 = float(rec.get("k0", 1e-13))
    k_eff = max(k0 * max(float(k_seis), 1.0), 1e-25)
    L = rec.get("L_source", None)
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = None
    if L is None or not math.isfinite(L) or L <= 0.0:
        L = _L_SOURCE_DEFAULT
    c_hyd = k_eff / (_MU_WATER * _S_STORAGE)
    if c_hyd <= 0.0:
        return 0.0
    k_deep = rec.get("k_deep", None)
    try:
        k_deep = float(k_deep)
        if math.isfinite(k_deep) and k_deep > 0:
            c_hyd = (k_deep * max(float(k_seis), 1.0)) / (_MU_WATER * _S_STORAGE)
    except (TypeError, ValueError):
        pass
    if not (1e-2 <= c_hyd <= 1e2):
        warnings.warn(
            "_pore_pressure_diffusion_lag_days: c = {:.3g} m^2/s fuori dal range "
            "validato Talwani (1997) [0.1, 10] m^2/s (k0={:.2e} m^2, L={:.0f} m). "
            "Il k0 superficiale non e' rappresentativo del percorso profondo: "
            "fornire 'k_deep' nella Tranche-A per un risultato affidabile."
            .format(c_hyd, k0, L), RuntimeWarning, stacklevel=2)
        return None
    t_s = (L * L) / (4.0 * math.pi * c_hyd)
    return float(t_s / 86400.0)


def _advective_transit_lag_days(rec, k_seis, regime, g_dP_seis=None,
                               co2_carrier_flux_on=True,
                               pore_pressure_lag_on=False,
                               ):
    if regime in _INERT_REGIMES:
        return None
    eta = {
        "advective":          1.00,
        "fracture_bypass":    1.00,
        "barometric_pumping": 0.60,
        "mixed_transition":   0.35,
        "decay_limited":      0.0,
    }.get(regime, 0.30)
    k0   = float(rec.get("k0", 1e-13))
    H    = float(rec.get("H", 3.0))
    Sw   = float(rec.get("Sw", 0.4))
    phi  = float(rec.get("phi_m", 0.3))
    dP   = float(rec.get("dP", 1.0e5))
    phi_g = max(phi * (1.0 - Sw), 1e-3)
    g_dP = float(g_dP_seis) if g_dP_seis is not None else math.sqrt(max(k_seis, 1.0))
    if co2_carrier_flux_on and (rec.get("CO2_flux", None) is not None):
        F_CO2 = float(rec.get("CO2_flux"))
        v_q = _co2_carrier_pore_velocity(F_CO2, phi_g)
        v_u = v_q * (1.0 + eta * (k_seis - 1.0))
    else:
        k_un  = k0 * (1.0 + eta * (k_seis - 1.0))
        dP_un = max(dP, 1.0e3) * (1.0 + eta * (g_dP - 1.0))
        L_grad = _pressure_gradient_length(rec, H)
        v_u = _pore_darcy_velocity(k_un, dP_un, L_grad, phi_g)
    if v_u <= 1e-15:
        return None
    lag_adv = H / v_u
    lag_cap = 1.0 / _LAMBDA_RN
    lag = min(lag_adv, lag_cap)
    lag_days = lag / 86400.0


    if pore_pressure_lag_on:
        lag_press_days = _pore_pressure_diffusion_lag_days(rec, k_seis)
        if lag_press_days is not None:
            lag_days = max(lag_days, lag_press_days)

    return float(lag_days)


def _time_lag_days(k0, phi_g, H, P_atm=1.0132e5, mu=1.82e-5,
                   T_atm=86400.0, rho_g=1.2):
    omega = 2.0 * math.pi / T_atm
    phi_g = max(float(phi_g), 1e-3)
    z_p = math.sqrt(2.0 * float(k0) * P_atm / max(omega * mu * phi_g, 1e-30))
    z_p = max(z_p, 1e-3)
    dP_baro = 150.0
    u_eff = (float(k0) / mu) * (dP_baro / z_p)
    u_eff = max(u_eff, 1e-12)
    dt_sec = float(H) / u_eff
    return dt_sec / 86400.0


def decompose_seismic_vs_meteo(t, radon, stack_driver,
                               extra_drivers=None,
                               unrest_t0=None, tau_seis=None,
                               log_radon=True):
    t      = np.asarray(t, dtype=float)
    y_raw  = np.asarray(radon, dtype=float)
    N      = y_raw.size
    y      = np.log(np.clip(y_raw, 1e-12, None)) if log_radon else y_raw.copy()

    cols   = {"const": np.ones(N)}
    sd     = np.asarray(stack_driver, dtype=float)
    cols["stack"] = (sd - sd.mean()) / (sd.std() + 1e-12)
    if extra_drivers:
        for name, v in extra_drivers.items():
            v = np.asarray(v, dtype=float)
            cols[name] = (v - v.mean()) / (v.std() + 1e-12)
    has_seis = (unrest_t0 is not None and tau_seis is not None)
    if has_seis:
        env = np.where(t >= float(unrest_t0),
                       np.exp(-np.clip(t - float(unrest_t0), 0.0, None)
                              / max(float(tau_seis), 1e-9)),
                       0.0)
        cols["seismic"] = (env - env.mean()) / (env.std() + 1e-12)

    names = list(cols.keys())
    X     = np.column_stack([cols[n] for n in names])

    def _ols(Xm, yv):
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        yhat = Xm @ beta
        ss_res = float(np.sum((yv - yhat) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) + 1e-30
        return beta, yhat, 1.0 - ss_res / ss_tot

    beta_full, yhat_full, R2_full = _ols(X, y)
    coef = {n: float(b) for n, b in zip(names, beta_full)}

    meteo_names = [n for n in names if n != "seismic"]
    Xm = np.column_stack([cols[n] for n in meteo_names])
    _, yhat_meteo, R2_meteo = _ols(Xm, y)

    if has_seis:
        seismic_partial = max(R2_full - R2_meteo, 0.0)
        denom = max(1.0 - R2_meteo, 1e-9)
        seismic_fraction = seismic_partial / denom
    else:
        seismic_partial = 0.0
        seismic_fraction = 0.0

    residual = y - yhat_meteo
    return {
        "regressors":          names,
        "coef":                coef,
        "R2_full":             float(R2_full),
        "R2_meteo_only":       float(R2_meteo),
        "R2_seismic_partial":  float(seismic_partial),
        "seismic_fraction":    float(seismic_fraction),
        "log_space":           bool(log_radon),
        "fitted_full":         yhat_full,
        "fitted_meteo":        yhat_meteo,
        "residual_after_meteo": residual,
        "note": (
            "seismic_fraction = partial R2 of the ADD-1 recovery envelope after "
            "removing the ADD-3 stack-effect/seasonal regressors. High value => "
            "the radon transient cannot be explained by weather/season alone."
        ),
    }


_FRACTURE_SPACING_S_DEFAULT = 1.0
_ELASTIC_APERTURE_ALPHA_DEFAULT = 5.0e-4

_YOUNGS_MODULUS_GPA_BY_LITHOLOGY = {
    "tuff":         2.0,
    "pumice":       2.0,
    "basalt":       40.0,
    "andesite":     25.0,
    "dacite":       25.0,
    "rhyolite":     20.0,
    "phonolite":    20.0,
    "carbonate":    50.0,
    "hydrothermal": 10.0,
    "default":      15.0,
}


def _youngs_modulus_pa(lit_type):
    lt = str(lit_type).lower() if lit_type is not None else ""
    for key, e_gpa in _YOUNGS_MODULUS_GPA_BY_LITHOLOGY.items():
        if key != "default" and key in lt:
            return e_gpa * 1.0e9
    return _YOUNGS_MODULUS_GPA_BY_LITHOLOGY["default"] * 1.0e9


def _elastic_aperture_alpha(rec, b0, c_sig_ref=5.0e-8):
    E_site = _youngs_modulus_pa(rec.get("lit_type", None))
    return float(b0) * float(c_sig_ref) * E_site


def _mogi_areal_strain_from_uplift(uplift_m, source_depth_m, station_radius_m=0.0):
    uplift_m = np.asarray(uplift_m, dtype=float)
    f = max(float(source_depth_m), 1.0)
    r = max(float(station_radius_m), 0.0)
    denom = f * (f ** 2 + r ** 2)
    numer = (2.0 * f ** 2 - r ** 2)
    return uplift_m * (numer / denom)


def _co2_flux_time_series(rec, eta_t, F0=None):
    F0 = float(F0) if F0 is not None else float(rec.get("CO2_flux", 50.0))
    F0 = max(F0, 1e-6)
    F_esc_max = 5.0
    logF0 = math.log(F0)
    logF1 = math.log(F0 * F_esc_max)
    logF_t = logF0 + np.asarray(eta_t, dtype=float) * (logF1 - logF0)
    return np.exp(logF_t)


def _deformation_time_series(rec, n_points=200):
    t_days = rec.get("defo_time_days", None)
    eps_series = rec.get("defo_strain_series", None)
    uplift_series = rec.get("defo_uplift_series", None)
    if t_days is not None and eps_series is not None:
        try:
            t_arr = np.asarray(t_days, dtype=float)
            e_arr = np.asarray(eps_series, dtype=float)
            if t_arr.shape[0] == e_arr.shape[0] and t_arr.shape[0] >= 2:
                order = np.argsort(t_arr)
                return {"t_s": t_arr[order] * 86400.0, "strain": e_arr[order],
                        "fidelity": "series"}
        except (TypeError, ValueError):
            pass
    if t_days is not None and uplift_series is not None:
        try:
            t_arr = np.asarray(t_days, dtype=float)
            u_arr = np.asarray(uplift_series, dtype=float)
            if t_arr.shape[0] == u_arr.shape[0] and t_arr.shape[0] >= 2:
                order = np.argsort(t_arr)
                f_source = rec.get("source_depth_m", rec.get("L_source", None))
                if f_source is not None:
                    r_station = float(rec.get("station_radius_m", 0.0))
                    strain_arr = _mogi_areal_strain_from_uplift(
                        u_arr[order], float(f_source), r_station)
                    return {"t_s": t_arr[order] * 86400.0, "strain": strain_arr,
                            "fidelity": "series"}
        except (TypeError, ValueError):
            pass

    def _valid_num(x):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return xf if math.isfinite(xf) else None

    r = _valid_num(rec.get("defo_rate", None))
    dur = _valid_num(rec.get("unrest_duration", None))
    if r is None or dur is None or dur <= 0:
        return None
    t_peak = _valid_num(rec.get("defo_onset_to_peak", None))
    if t_peak is None:
        t_peak = dur
    t_peak = min(max(t_peak, 1e-3), dur)

    peak_uplift_m = r * t_peak * 1.0e-3
    f_source = rec.get("source_depth_m", rec.get("L_source", None))
    try:
        f_source = float(f_source)
    except (TypeError, ValueError):
        f_source = None
    if f_source is None or not math.isfinite(f_source) or f_source <= 0:
        f_source = _L_SOURCE_DEFAULT
    peak_strain = float(_mogi_areal_strain_from_uplift(peak_uplift_m, f_source))
    t_grid = np.linspace(0.0, dur, n_points)
    ramp_up = np.clip(t_grid / max(t_peak, 1e-9), 0.0, 1.0)
    ramp_dn = np.clip((dur - t_grid) / max(dur - t_peak, 1e-9), 0.0, 1.0)
    strain = peak_strain * np.minimum(ramp_up, ramp_dn)
    return {"t_s": t_grid * 86400.0, "strain": strain, "fidelity": "ramp"}


def _fracture_aperture_from_strain(strain, b0, alpha):
    b = b0 + alpha * np.asarray(strain, dtype=float)
    return np.clip(b, 1.0e-7, 5.0e-3)


def _permeability_from_aperture(b, s=_FRACTURE_SPACING_S_DEFAULT):
    b = np.asarray(b, dtype=float)
    return (b ** 3) / (12.0 * max(float(s), 1e-6))


def _solve_transient_radon_column(rec, k_t_func, t_eval_s,
                                  nx=60, De=None, lam=_LAMBDA_RN,
                                  dP_t_func=None, v_extra_func=None):
    H = float(rec.get("H", 3.0))
    Sw = float(rec.get("Sw", 0.4))
    phi = float(rec.get("phi_m", 0.3))
    dP0 = float(rec.get("dP", 1.0e5))
    mu = 1.82e-5
    phi_g = max(phi * (1.0 - Sw), 1e-3)

    def dP_wrap(t_s):
        if dP_t_func is not None:
            return max(float(dP_t_func(t_s)), 1e3)
        return dP0

    D0 = _D0_REF
    L_grad_t = _pressure_gradient_length(rec, H)
    De_eff = De if De is not None else D0 * (phi_g ** (7.0 / 3.0)) / max(phi ** 2, 1e-6)
    De_eff = max(De_eff, 1e-8)

    rho_b = 1600.0
    Ra226 = float(rec.get("Ra226", 100.0))
    eps_rn = float(rec.get("eps_rn", 0.3))
    G = rho_b * Ra226 * eps_rn * lam
    Cparticular = G / lam

    def k_wrap(t_s):
        return max(float(k_t_func(t_s)), 1e-19)

    def q_surf_steady(v):
        return _quasi_steady_surface_flux(v, De_eff, H, lam, G)

    def v_of_t(t_s):
        k_now = k_wrap(t_s)
        dP_now = dP_wrap(t_s)
        v = (k_now / mu) * (dP_now / L_grad_t) / phi_g
        if v_extra_func is not None:
            v = v + max(float(v_extra_func(t_s)), 0.0)
        return v

    _mode_rate_diff = (math.pi ** 2) * De_eff / (H ** 2)

    def tau_of_v(v):
        v = max(float(v), 0.0)
        rate_adv = (v ** 2) / (4.0 * De_eff)
        rate = lam + _mode_rate_diff + rate_adv
        return 1.0 / max(rate, 1e-12)

    def dqdt(t_s, q):
        v = v_of_t(t_s)
        q_ss = q_surf_steady(v)
        tau = tau_of_v(v)
        return (q_ss - q) / tau

    t_arr = np.asarray(t_eval_s, dtype=float)
    n = t_arr.shape[0]
    q_out = np.empty(n, dtype=float)

    v0 = v_of_t(float(t_arr[0]))
    q_out[0] = q_surf_steady(v0)

    for i in range(1, n):
        t_prev = float(t_arr[i - 1])
        t_next = float(t_arr[i])
        dt_total = t_next - t_prev
        if dt_total <= 0:
            q_out[i] = q_out[i - 1]
            continue
        v_here = v_of_t(0.5 * (t_prev + t_next))
        tau_here = tau_of_v(v_here)
        n_sub = max(1, int(math.ceil(4.0 * dt_total / max(tau_here, 1e-6))))
        n_sub = min(n_sub, 64)
        h = dt_total / n_sub
        q = q_out[i - 1]
        t_cur = t_prev
        for _ in range(n_sub):
            t_mid = t_cur + 0.5 * h
            v_mid = v_of_t(t_mid)
            tau_mid = max(tau_of_v(v_mid), 1e-9)
            q_ss_mid = q_surf_steady(v_mid)
            q = q_ss_mid + (q - q_ss_mid) * math.exp(-h / tau_mid)
            q = max(q, 0.0)
            t_cur += h
        q_out[i] = q

    return {"t_s": t_arr, "C_surface": q_out, "success": True,
            "G_norm": G, "De_eff": De_eff}

def transient_amplitude_and_lag(rec, k_seis, regime,
                                b0=None, alpha=None, s_frac=None,
                                n_points=200, overpressure_gain_on=True,
                                co2_carrier_on=True):
    if regime in _INERT_REGIMES:
        return 1.0, None, None

    defo = _deformation_time_series(rec, n_points=n_points)
    if defo is None:
        return None, None, None

    s_frac = float(s_frac) if s_frac is not None else _FRACTURE_SPACING_S_DEFAULT
    if b0 is not None:
        b0 = float(b0)
    else:
        k0_site = float(rec.get("k0", 1e-13))
        b0 = (12.0 * s_frac * max(k0_site, 1e-25)) ** (1.0 / 3.0)
    alpha = float(alpha) if alpha is not None else _elastic_aperture_alpha(rec, b0)

    b_t = _fracture_aperture_from_strain(defo["strain"], b0, alpha)
    k_t = _permeability_from_aperture(b_t, s_frac)
    t_s = defo["t_s"]

    def k_of_t(t_query):
        return float(np.interp(t_query, t_s, k_t))

    strain_arr = np.asarray(defo["strain"], dtype=float)
    smax = max(float(np.max(np.abs(strain_arr))), 1e-30)
    eta_t = np.clip(np.abs(strain_arr) / smax, 0.0, 1.0)

    dP0 = float(rec.get("dP", 1.0e5))
    g_dP_max = math.sqrt(max(float(k_seis), 1.0)) if overpressure_gain_on else 1.0
    dP_t = dP0 * (1.0 + eta_t * (g_dP_max - 1.0))

    def dP_of_t(t_query):
        return float(np.interp(t_query, t_s, dP_t))

    v_co2_of_t = None
    if co2_carrier_on and (rec.get("CO2_flux", None) is not None):
        phi = float(rec.get("phi_m", 0.3))
        Sw = float(rec.get("Sw", 0.4))
        phi_g = max(phi * (1.0 - Sw), 1e-3)
        F_t = _co2_flux_time_series(rec, eta_t, F0=rec.get("CO2_flux"))
        v_co2_t = np.array([_co2_carrier_pore_velocity(f, phi_g) for f in F_t])

        def v_co2_of_t(t_query):
            return float(np.interp(t_query, t_s, v_co2_t))

    sol = _solve_transient_radon_column(rec, k_of_t, t_s, dP_t_func=dP_of_t,
                                        v_extra_func=v_co2_of_t)
    if not sol["success"] or np.any(~np.isfinite(sol["C_surface"])):
        return None, None, None

    C_surf = sol["C_surface"]
    if not (np.isfinite(C_surf[0]) and C_surf[0] > 0):
        return None, None, None
    C_quiet = C_surf[0]

    i_peak = int(np.nanargmax(C_surf))
    C_peak = float(C_surf[i_peak])
    i_peak_defo = int(np.argmax(np.abs(strain_arr)))
    lag_days = float((t_s[i_peak] - t_s[i_peak_defo]) / 86400.0)

    R_pred = max(C_peak / C_quiet, 1.0)
    return float(R_pred), float(lag_days), defo["fidelity"]


def predict_bradyseismic_blind(tranche_A,
                               beta=_M0_BETA_FROZEN,
                               R_inert_tol=1.10,
                               outfile="predictions_frozen.json",
                               physics_amplitude_lag=True,
                               co2_carrier_flux_on=True,
                               carrier_dilution_on=False,
                               transient_kt_on=False,
                               pore_pressure_lag_on=False):
    if isinstance(tranche_A, str):
        import pandas as pd
        records = pd.read_csv(tranche_A).to_dict("records")
    else:
        records = list(tranche_A)
    _assert_no_measurement_leak(records)

    rmse_log = _M0_RMSE_LOG_FROZEN
    z = 1.96
    sites_out = []
    for rec in records:
        site = rec.get("site", "UNKNOWN")
        regime = rec.get("regime")
        if regime is None:
            regime = infer_physical_regime(rec)
        expected = regime in _RESPONSIVE_REGIMES

        M_w = float(rec.get("M_w_unrest", 0.0))
        k_seis = _seismic_permeability_factor(M_w, site=rec.get("seismic_site", site))

        R_transient, lag_transient, transient_fidelity = (None, None, None)
        S_kseis = None
        if transient_kt_on:
            R_transient, lag_transient, transient_fidelity = transient_amplitude_and_lag(
                rec, k_seis, regime)

        if transient_kt_on and R_transient is not None:
            R_pred = R_transient
            sigma_logR = (math.sqrt(_SIGMA_LOG10_KSEIS_DEFAULT ** 2
                                    + (0.5 * _SIGMA_LOG10_KSEIS_DEFAULT) ** 2)
                          if regime not in _INERT_REGIMES
                          else math.log10(max(R_inert_tol, 1.0 + 1e-6)) / max(z, 1e-9))
            logR = math.log10(max(R_pred, 1e-30))
            R_lo = 10 ** (logR - z * sigma_logR)
            R_hi = 10 ** (logR + z * sigma_logR)
        elif physics_amplitude_lag:
            R_pred = _amplitude_factor_R_physics(rec, k_seis, regime,
                                                 co2_carrier_flux_on=co2_carrier_flux_on,
                                                 carrier_dilution_on=carrier_dilution_on,
)
            if regime in _INERT_REGIMES:
                sigma_logR = math.log10(max(R_inert_tol, 1.0 + 1e-6)) / max(z, 1e-9)
            else:
                sigma_logR, S_kseis = _sigma_log10_R_physics(
                    rec, k_seis, regime,
                    co2_carrier_flux_on=co2_carrier_flux_on,
                    carrier_dilution_on=carrier_dilution_on)
            logR = math.log10(max(R_pred, 1e-30))
            R_lo = 10 ** (logR - z * sigma_logR)
            R_hi = 10 ** (logR + z * sigma_logR)
        else:
            R_pred = _amplitude_factor_R(k_seis, regime, beta=beta)
            R_lo, R_hi, sigma_logR = _amplitude_band_R(R_pred, k_seis, regime, z=z,
                                                       R_inert_tol=R_inert_tol)

        phi_g = max(float(rec.get("phi_m", 0.3)) * (1.0 - float(rec.get("Sw", 0.4))), 1e-3)
        tau_seis = _tau_seis_recovery_seconds(
            site=rec.get("seismic_site", site),
            tau_days=rec.get("tau_seis_days"),
        )
        if expected:
            lag_transport = _time_lag_days(rec.get("k0", 1e-13), phi_g,
                                           rec.get("H", 3.0))
            lag_recovery  = _time_to_recovery_peak_days(tau_seis, lag_transport)
            if transient_kt_on and lag_transient is not None:
                lag = round(lag_transient, 3)
                lag_transport = round(lag_transport, 3)
            elif physics_amplitude_lag:
                lag_adv = _advective_transit_lag_days(rec, k_seis, regime,
                                                      co2_carrier_flux_on=co2_carrier_flux_on,
                                                      pore_pressure_lag_on=pore_pressure_lag_on)
                lag = round(lag_adv, 3) if lag_adv is not None else None
                lag_transport = round(lag_transport, 3)
            else:
                lag = round(lag_recovery, 3)
                lag_transport = round(lag_transport, 3)
        else:
            lag = None
            lag_transport = None

        sites_out.append({
            "site":            site,
            "lit_type":        rec.get("lit_type"),
            "regime":          regime,
            "expected_effect": bool(expected),
            "k_seis_peak":     round(k_seis, 4),
            "R_pred":          round(R_pred, 4),
            "R_lo95":          round(R_lo, 4),
            "R_hi95":          round(R_hi, 4),
            "sigma_log10_R":   round(sigma_logR, 4),
            "dlog10R_dlog10kseis": (None if S_kseis is None else round(S_kseis, 4)),
            "lag_days_pred":   lag,
            "lag_transport_days": lag_transport,
            "tau_recovery_days":  round(tau_seis / 86400.0, 3),
            "R_inert_tolerance": R_inert_tol,
            "transient_kt_fidelity": transient_fidelity,
            "R_meas":          None,
            "lag_days_meas":   None,
            "amplitude_pass":  None,
            "lag_pass":        None,
            "regime_pass":     None,
        })

    payload = {
        "candidate":        "C1_bradyseismic_radon_forcing_Campi_Flegrei",
        "model_version":    "volcanic_radon_model_v11_corrected",
        "frozen_beta_M0":   beta,
        "rmse_log":         rmse_log,
        "amplitude_lag_method": ("physics_advection_length" if physics_amplitude_lag
                                  else "legacy_powerlaw_recovery"),
        "band_method":      "physical_ratio_propagation",
        "band_method_note": (
            "La banda 95% su R = C_unrest/C_quiet e' derivata per propagazione "
            "del primo ordine in log10 da SE(esponente efficace) e "
            "sigma_log10(k_seis); NON dall'RMSE globale di M0, perche' gli errori "
            "sistematici di sito (Ra-226, porosita', geometria) si cancellano nel "
            "rapporto sullo stesso sito."
        ),
        "sigma_log10_kseis": _SIGMA_LOG10_KSEIS_DEFAULT,
        "SE_exp_eff":        _SE_EXP_EFF,
        "responsive_regimes": list(_RESPONSIVE_REGIMES),
        "inert_regimes":     list(_INERT_REGIMES),
        "n_sites":          len(sites_out),
        "created_utc":      _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sites":            sites_out,
        "success_criteria": {
            "amplitude":  "measured R within [R_lo95, R_hi95] for >= 70% of "
                          "responsive sites",
            "lag":        "measured lag within +/- 1 day of lag_days_pred for "
                          ">= 60% of responsive sites",
            "specificity": "regime classification correct (responsive shows "
                           "R>tol, inert shows R<=tol) for >= 80% of all sites",
            "STRONG_discovery": "amplitude AND lag AND specificity all pass",
            "multiple_testing_caveat": (
                "Il verdetto STRONG_DISCOVERY e' la CONGIUNZIONE di tre test "
                "confrontati ciascuno con la propria soglia fissa, senza "
                "correzione per confronti multipli e senza tener conto della "
                "loro non-indipendenza (le tre metriche condividono gli stessi "
                "15 siti). Va letto in modo DESCRITTIVO, non come una "
                "affermazione formale di significativita' congiunta. Con n=15 "
                "gli intervalli di confidenza di Wilson (riportati in "
                "verify_against_measured) sono ampi ~25 punti percentuali: e' "
                "quella l'incertezza da citare, non la frazione puntuale."
            ),
            "PARTIAL":          "specificity passes AND (amplitude OR lag) passes",
            "NOT_confirmed":    "specificity fails",
        },
    }

    with open(outfile, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[C1] Blind prediction written -> {outfile} "
          f"({len(sites_out)} sites, {sum(s['expected_effect'] for s in sites_out)} "
          f"responsive).")
    return payload


def freeze_prediction(prediction_file="predictions_frozen.json",
                      code_file=__file__,
                      manifest_out="C1_FREEZE_MANIFEST.json"):
    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "candidate":            "C1_bradyseismic_radon_forcing_Campi_Flegrei",
        "frozen_utc":           _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "prediction_file":      os.path.basename(prediction_file),
        "prediction_sha256":    _sha256(prediction_file),
        "code_file":            os.path.basename(code_file),
        "code_sha256":          _sha256(code_file),
        "git_tag_expected":     "v5.1-frozen-blind",
        "integrity_note": "This manifest must be deposited with a trusted "
                          "timestamp (OSF/preprint) before Tranche B (measured "
                          "radon) is received. The code must not be modified "
                          "after this point. Any post-hoc change invalidates "
                          "the blind validation.",
    }
    with open(manifest_out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[C1] FREEZE manifest written -> {manifest_out}")
    print(f"      prediction SHA-256: {manifest['prediction_sha256']}")
    print(f"      code       SHA-256: {manifest['code_sha256']}")
    return manifest


def verify_against_measured(prediction_file="predictions_frozen.json",
                            measured_csv="measured_tranche_B.csv",
                            report_out="C1_VERIFICATION_REPORT.json"):
    import pandas as pd

    with open(prediction_file) as fh:
        pred = json.load(fh)
    meas = pd.read_csv(measured_csv).set_index("site").to_dict("index")

    n_resp = n_resp_amp = n_resp_lag = 0
    n_lag_nopred = 0
    n_all = n_regime_ok = 0
    n_meas = 0
    detail = []
    for s in pred["sites"]:
        site = s["site"]
        m = meas.get(site)
        n_all += 1
        if m is None:
            detail.append({"site": site, "status": "NO_MEASUREMENT"})
            continue
        n_meas += 1

        R_meas = float(m.get("R_meas", np.nan))
        lag_meas = float(m.get("lag_days_meas", np.nan))
        s["R_meas"] = R_meas
        s["lag_days_meas"] = lag_meas

        if s["expected_effect"]:
            regime_ok = R_meas > s["R_inert_tolerance"]
        else:
            regime_ok = R_meas <= s["R_inert_tolerance"]
        s["regime_pass"] = bool(regime_ok)
        n_regime_ok += int(regime_ok)

        if s["expected_effect"]:
            n_resp += 1
            amp_ok = (s["R_lo95"] <= R_meas <= s["R_hi95"])
            try:
                lag_pred = float(s.get("lag_days_pred"))
            except (TypeError, ValueError):
                lag_pred = float("nan")
            if not np.isfinite(lag_pred):
                lag_ok = False
                s["lag_status"] = "NO_PREDICTION"
            elif not np.isfinite(lag_meas):
                lag_ok = False
                s["lag_status"] = "NO_LAG_MEASUREMENT"
            else:
                lag_ok = (abs(lag_meas - lag_pred) <= 1.0)
                s["lag_status"] = "TESTED"
            s["amplitude_pass"] = bool(amp_ok)
            s["lag_pass"] = bool(lag_ok)
            n_resp_amp += int(amp_ok)
            n_resp_lag += int(lag_ok)
            n_lag_nopred += int(s["lag_status"] == "NO_PREDICTION")

        detail.append(s)

    def _wilson_ci(k_succ, n_tot, z=1.96):
        if not n_tot:
            return (float("nan"), float("nan"))
        ph = k_succ / n_tot
        den = 1.0 + z*z/n_tot
        ctr = (ph + z*z/(2*n_tot)) / den
        hw  = z*math.sqrt(ph*(1.0-ph)/n_tot + z*z/(4*n_tot*n_tot)) / den
        return (max(0.0, ctr-hw), min(1.0, ctr+hw))

    amp_frac = (n_resp_amp / n_resp) if n_resp else 0.0
    lag_frac = (n_resp_lag / n_resp) if n_resp else 0.0
    reg_frac = (n_regime_ok / n_meas) if n_meas else 0.0

    amplitude_pass   = amp_frac >= 0.70
    lag_pass         = lag_frac >= 0.60
    specificity_pass = reg_frac >= 0.80

    if specificity_pass and amplitude_pass and lag_pass:
        verdict = "STRONG_DISCOVERY_CONFIRMED"
    elif specificity_pass and (amplitude_pass or lag_pass):
        verdict = "PARTIAL_CONFIRMATION"
    else:
        verdict = "NOT_CONFIRMED"

    report = {
        "candidate":          pred["candidate"],
        "verified_utc":       _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_sites_total":      n_all,
        "n_sites_measured":   n_meas,
        "n_responsive":       n_resp,
        "amplitude_fraction": round(amp_frac, 3),
        "lag_fraction":       round(lag_frac, 3),
        "n_lag_not_predicted": n_lag_nopred,
        "specificity_fraction": round(reg_frac, 3),
        "amplitude_CI95":      [round(v, 3) for v in _wilson_ci(n_resp_amp, n_resp)],
        "lag_CI95":            [round(v, 3) for v in _wilson_ci(n_resp_lag, n_resp)],
        "specificity_CI95":    [round(v, 3) for v in _wilson_ci(n_regime_ok, n_meas)],
        "n_inert_sites":       int(sum(1 for x in pred["sites"]
                                       if not x.get("expected_effect", True))),
        "n_responsive_sites":  int(sum(1 for x in pred["sites"]
                                       if x.get("expected_effect", True))),
        "specificity_caveat": (
            "Se n_inert_sites == 0 la metrica misura la SENSIBILITA' (quota di "
            "siti responsivi che mostrano davvero un segnale), non la "
            "specificita': senza negativi non e' possibile stimare i falsi "
            "allarmi. Per una vera specificita' servono siti pre-registrati "
            "come inerti."
        ),
        "amplitude_pass":     amplitude_pass,
        "lag_pass":           lag_pass,
        "specificity_pass":   specificity_pass,
        "VERDICT":            verdict,
        "per_site":           detail,
    }
    with open(report_out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 70)
    print(f"  CANDIDATE 1 — BLIND VERIFICATION RESULT")
    print("=" * 70)
    print(f"  Specificity (regime) : {reg_frac*100:5.1f}%  -> "
          f"{'PASS' if specificity_pass else 'FAIL'}")
    print(f"  Amplitude (R in band): {amp_frac*100:5.1f}%  -> "
          f"{'PASS' if amplitude_pass else 'FAIL'}")
    print(f"  Lag (+/- 1 day)      : {lag_frac*100:5.1f}%  -> "
          f"{'PASS' if lag_pass else 'FAIL'}")
    print("-" * 70)
    print(f"  VERDICT: {verdict}")
    print("=" * 70)
    return report


def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description='volcanicradonmodel v5 - regime-aware ODE radon transport')
    parser.add_argument('--mode',
                        choices=['benchmark', 'validate', 'pipeline',
                                 'holdout', 'sobol',
                                 'literature', 'quasi2d', 'temporal',
                                 'c1_predict', 'c1_freeze', 'c1_verify'],
                        default='benchmark',
                        help='Run mode: benchmark (lithology sweep), '
                             'validate (gold-standard field sites), '
                             'pipeline (full scaling law), '
                             'holdout (frozen derivation/holdout split), '
                             'sobol (global Sobol sensitivity), '
                             'literature (Nazaroff/Jobbagy/Sainz vs M0 benchmark), '
                             'quasi2d (1D vs quasi-2D geometric comparison), '
                             'temporal (seasonal/annual-mean validation), '
                             'c1_predict (Candidate-1 blind prediction from '
                             'Tranche-A inputs), c1_freeze (hash+timestamp '
                             'pre-registration), c1_verify (compare frozen '
                             'prediction vs measured Tranche-B))')
    parser.add_argument('--tranche_a', default='data/c1_tranche_A_inputs.csv',
                        help='Input-only CSV for Candidate-1 blind prediction')
    parser.add_argument('--measured', default='data/c1_tranche_B_measured.csv',
                        help='Measured CSV (Tranche B) for Candidate-1 verify')
    parser.add_argument('--frozen', default='predictions_frozen.json',
                        help='Frozen prediction JSON file (Candidate 1)')
    parser.add_argument('--outdir', default='./output',
                        help='Output directory for CSV and figures')
    parser.add_argument('--nsamples', type=int, default=40,
                        help='Number of ODE samples (pipeline mode)')
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == 'benchmark':
        results = run_lithology_benchmark(nz=15, days=1.5)
        plot_lithology_benchmark(results, save_path=f'{args.outdir}/benchmark.png')
    elif args.mode == 'validate':
        import pandas as pd
        df = pd.read_csv('data/site_validation_master.csv')
        site_table = df.to_dict('records')
        for corr in [1.0, 7.0]:
            print(f"\n[validate] src_correction = {corr}")
            out = run_gold_validation(site_table, src_correction=corr)
            out.to_csv(f'{args.outdir}/validation_corr{int(corr)}.csv', index=False)
    elif args.mode == 'pipeline':
        run_regimeaware_scaling_pipeline(
            n_samples=args.nsamples, outdir=args.outdir)
    elif args.mode == 'holdout':
        res = run_holdout_validation(csv_path='data/site_validation_master.csv')
        if res['metrics_hold'] is not None:
            res['metrics_hold'].to_csv(
                f'{args.outdir}/holdout_metrics.csv', index=False)
        if res['metrics_deriv'] is not None:
            res['metrics_deriv'].to_csv(
                f'{args.outdir}/derivation_metrics.csv', index=False)
    elif args.mode == 'sobol':
        Si = sobol_sensitivity_scaling(n_base=1024)
        import pandas as pd
        pd.DataFrame({
            'name': Si['names'],
            'S1':   Si['S1'], 'S1_conf': Si['S1_conf'],
            'ST':   Si['ST'], 'ST_conf': Si['ST_conf'],
        }).to_csv(f'{args.outdir}/sobol_indices.csv', index=False)
    elif args.mode == 'literature':
        res = run_literature_benchmark(
            csv_path='data/site_validation_master.csv', m0_src_correction=7.0)
        res['summary'].to_csv(
            f'{args.outdir}/literature_benchmark_summary.csv', index=False)
        res['per_site'].to_csv(
            f'{args.outdir}/literature_benchmark_per_site.csv', index=False)
    elif args.mode == 'quasi2d':
        out = run_quasi2d_campi_flegrei()
        import pandas as pd
        pd.DataFrame([out]).to_csv(
            f'{args.outdir}/quasi2d_campi_flegrei.csv', index=False)
    elif args.mode == 'temporal':
        out = temporal_validation(
            monthly_csv='data/campi_flegrei_monthly_2011_2017.csv')
        out['monthly'].to_csv(
            f'{args.outdir}/temporal_monthly.csv', index=False)
    elif args.mode == 'c1_predict':
        import pandas as pd
        tranche_A = pd.read_csv(args.tranche_a).to_dict('records')
        predict_bradyseismic_blind(
            tranche_A, outfile=f'{args.outdir}/{os.path.basename(args.frozen)}')
    elif args.mode == 'c1_freeze':
        freeze_prediction(
            prediction_file=f'{args.outdir}/{os.path.basename(args.frozen)}',
            code_file=__file__,
            manifest_out=f'{args.outdir}/C1_FREEZE_MANIFEST.json')
    elif args.mode == 'c1_verify':
        verify_against_measured(
            prediction_file=f'{args.outdir}/{os.path.basename(args.frozen)}',
            measured_csv=args.measured,
            report_out=f'{args.outdir}/C1_VERIFICATION_REPORT.json')


if __name__ == '__main__':
    _cli_main()
