#!/usr/bin/env python
"""
tlusty_quickstart.py — minimal working example for the TLUSTY LRD photosphere
basis of Curtis et al. (in preparation).

What it does
------------
1. Loads TLUSTYBasis from the R = 10,000 cache shipped with this fork
   (prospect/sources/tlusty_data/tlusty_cache_r10k_40k.npz), so no external data or
   FSPS installation is required.  Only numpy, scipy, and matplotlib are used.
2. Evaluates two Little Red Dot photospheres from the paper:
     - a hot, bare photosphere at The Egg's best-fit parameters
       (Teff = 5293 K, log g = -2.255, [M/H] = -1), showing the Ca II
       triplet and alkali absorption that anchor the fits, and
     - a cold, water-bearing photosphere like the Water Dot cold components
       (Teff = 2000 K, log g = -3.0), showing the 1.28-1.45 um H2O band.
3. Plots both, marks the diagnostic features, and saves
   demo/tlusty_quickstart.png.

The composite photosphere + host-galaxy model used for the actual fits is
TLUSTYPlusGalaxyBasis in the same module.  It additionally needs python-fsps
with SPS_HOME set; construct it exactly like TLUSTYBasis, e.g.

    from prospect.sources.tlusty_basis import TLUSTYPlusGalaxyBasis
    sps = TLUSTYPlusGalaxyBasis(spec_dir=DATA, target_R=10000,
                                cache_file=CACHE, phot_dust_mode='shared',
                                phot_dust_law='smc')

and call sps.get_spectra_components(**params) to receive the photosphere and
galaxy spectra separately (the paper smooths only the photosphere, so the two
components must stay separate until after the line-of-sight convolution).

Usage
-----
    python demo/tlusty_quickstart.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the repository importable when the script runs from a bare checkout.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:
    # The canonical import when python-fsps is installed with SPS_HOME set.
    from prospect.sources.tlusty_basis import TLUSTYBasis
except Exception:
    # prospect.sources imports FSPS-dependent modules at package-import time.
    # Without FSPS, load the basis module directly from its file; TLUSTYBasis
    # itself needs only numpy and scipy.
    import importlib.util
    _path = os.path.join(REPO, "prospect", "sources", "tlusty_basis.py")
    _spec = importlib.util.spec_from_file_location("tlusty_basis", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    TLUSTYBasis = _mod.TLUSTYBasis
    print("note: FSPS unavailable, loaded TLUSTYBasis directly from file")

DATA = os.path.join(REPO, "prospect", "sources", "tlusty_data")
CACHE = os.path.join(DATA, "tlusty_cache_r10k_40k.npz")

# spec_dir is only consulted when the cache file is missing; with the shipped
# cache this example is fully self-contained.
sps = TLUSTYBasis(spec_dir=DATA, target_R=10000, cache_file=CACHE, verbose=True)

# Two photospheres from the paper.  mh_idx indexes the metallicity slices
# (0 -> [M/H] = -2, 1 -> -1, 2 -> 0) and logL_star sets the luminosity in
# log10 L_sun, which fixes the radius through the Stefan-Boltzmann law.
egg = dict(teff=5293.0, logg=-2.255, mh_idx=1.0, logL_star=9.9)
cold = dict(teff=2000.0, logg=-3.0, mh_idx=1.0, logL_star=9.4)

wave, spec_egg, _ = sps.get_galaxy_spectrum(**egg)
_, spec_cold, _ = sps.get_galaxy_spectrum(**cold)

# Convert L_nu [Lsun/Hz] to a lambda*F_lambda shape for plotting.
C_AA = 2.998e18  # speed of light in Angstrom/s
lfl_egg = spec_egg * C_AA / wave
lfl_cold = spec_cold * C_AA / wave

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(wave / 1e4, lfl_egg, color="#228833", lw=1.0, alpha=0.9,
        label=r"The Egg fit: $T=5293$ K, $\log g=-2.26$")
ax.plot(wave / 1e4, lfl_cold, color="#4477AA", lw=1.0, alpha=0.9,
        label=r"Water Dot cold component: $T=2000$ K, $\log g=-3.0$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.3, 3.0)
ymax = 3.0 * max(lfl_egg.max(), lfl_cold.max())
ax.set_ylim(ymax / 1e7, ymax)
ax.set_xlabel(r"rest wavelength ($\mu$m)")
ax.set_ylabel(r"$\lambda L_\lambda$ (arbitrary)")

features = [(0.5893, "Na D"), (0.7682, "K I"), (0.8579, r"Ca II"),
            (1.365, r"H$_2$O")]
for lam, name in features:
    ax.axvline(lam, color="0.6", lw=0.8, alpha=0.7, zorder=0)
    ax.text(lam, 0.97, name, transform=ax.get_xaxis_transform(),
            rotation=90, fontsize=8, ha="right", va="top", color="0.35")

ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()

out = os.path.join(REPO, "demo", "tlusty_quickstart.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
