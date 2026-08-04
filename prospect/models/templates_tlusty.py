"""
templates_tlusty.py — Prospector parameter template for TLUSTY LRD photosphere fitting.

Provides build_tlusty_model(zred, logL_init, ...) which returns a SpecModel configured
for single-component quasi-star / LRD photosphere fitting with:
  - Liu+2026 TLUSTY atmospheric templates via TLUSTYBasis
  - Calzetti+2000 dust extinction
  - No nebular emission

Free parameters
---------------
teff      : Teff [K]; Gaussian prior centred on mesa_teff  (default 5000 K)
logg      : log g;    Gaussian prior centred on mesa_logg  (default -2.0)
mh_idx    : continuous metallicity index [0,2] → [M/H]=[-2,-1,0]; Uniform(0,2)
logL_star : log10(L*/Lsun); Uniform prior over log_l_range (default [5,14])
dust2     : Calzetti E(B-V) equivalent; Uniform(0, dust2_max)
"""

import numpy as np
from astropy.cosmology import WMAP9 as cosmo

from .parameters import ProspectorParams

__all__ = ["build_tlusty_model", "TemplateLibrary"]


def build_tlusty_model(zred,
                       mesa_teff=5000., mesa_teff_sigma=800.,
                       mesa_logg=-2.0,  mesa_logg_sigma=0.5,
                       logL_init=10.,   log_l_range=(5., 14.),
                       dust2_max=4.0,
                       **kwargs):
    """
    Build a ProspectorParams instance for TLUSTY photosphere fitting.

    Parameters
    ----------
    zred : float
        Spectroscopic redshift (fixed).
    mesa_teff, mesa_teff_sigma : float
        Centre and 1-sigma width of the Teff Gaussian prior (K).
    mesa_logg, mesa_logg_sigma : float
        Centre and 1-sigma width of the log g Gaussian prior.
    logL_init : float
        Initial value (and Uniform prior centre) for log10(L_star/L_sun).
    log_l_range : tuple
        (min, max) for the Uniform prior on logL_star.
    dust2_max : float
        Upper bound of the Uniform prior on dust2 (Calzetti E(B-V)).

    Returns
    -------
    model : ProspectorParams
    """
    from prospect.models.priors import Normal, Uniform, TopHat

    lumdist = cosmo.luminosity_distance(zred).to('Mpc').value

    model_params = {
        # --- atmospheric grid parameters ---
        "teff": {
            "N": 1, "isfree": True,
            "init": mesa_teff,
            "prior": Normal(mean=mesa_teff, sigma=mesa_teff_sigma),
            "units": "K",
        },
        "logg": {
            "N": 1, "isfree": True,
            "init": mesa_logg,
            "prior": Normal(mean=mesa_logg, sigma=mesa_logg_sigma),
            "units": "log10(cm/s2)",
        },
        "mh_idx": {
            "N": 1, "isfree": True,
            "init": 1.0,
            "prior": Uniform(mini=0.0, maxi=2.0),
            "units": "index; 0=[M/H]=-2, 1=-1, 2=0",
        },

        # --- normalisation ---
        "logL_star": {
            "N": 1, "isfree": True,
            "init": logL_init,
            "prior": Uniform(mini=log_l_range[0], maxi=log_l_range[1]),
            "units": "log10(L_star/L_sun)",
        },

        # --- dust (Calzetti+2000) ---
        "dust_type": {
            "N": 1, "isfree": False,
            "init": 2,   # 2 = Calzetti+2000
        },
        "dust2": {
            "N": 1, "isfree": True,
            "init": 1.0,
            "prior": Uniform(mini=0.0, maxi=dust2_max),
            "units": "Calzetti E(B-V)",
        },

        # --- fixed bookkeeping ---
        "mass": {
            "N": 1, "isfree": False,
            "init": 1.0,
            "units": "Msun (dummy; normalization absorbed in logL_star)",
        },
        "zred": {
            "N": 1, "isfree": False,
            "init": float(zred),
        },
        "lumdist": {
            "N": 1, "isfree": False,
            "init": float(lumdist),
            "units": "Mpc",
        },

        # --- nebular emission off ---
        "add_neb_emission": {
            "N": 1, "isfree": False,
            "init": False,
        },
        "nebemlineinspec": {
            "N": 1, "isfree": False,
            "init": True,   # lines pre-absorbed in model; keeps _need_lines=False
        },
    }

    model = ProspectorParams(model_params, **kwargs)
    return model
