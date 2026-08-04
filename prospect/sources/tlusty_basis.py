"""
tlusty_basis.py — SPS interface for Liu+2026 TLUSTY LRD photosphere templates.

Written for "Weighing Little Red Dots by Their Atmospheres: A Spectral
Framework for Testing the Quasi-Star Hypothesis in Little Red Dots"
(Curtis et al., in preparation).  If you use this basis, please cite that
paper and Liu et al. (2026) for the underlying TLUSTY model grid.  This
module lives beside Prospector's stock source bases and the AGN
accretion-disk basis of Wang et al. (2024) (agnssp_basis.py) and follows
the same SPS-object API, so either photosphere class can be passed
anywhere Prospector expects an sps object.

Provides two classes:

  TLUSTYBasis
      Single-component photosphere SPS compatible with Prospector's SpecModel.
      Reads Liu+2026 .spec files, converts their air wavelengths to vacuum,
      convolves to a target resolving power, and
      builds separable bilinear interpolation tables over (Teff, logg) per
      [M/H] slice (brackets T first, interpolates logg independently per T —
      avoids the 2D convex-hull restriction of LinearNDInterpolator).
      Supports SMC, Calzetti+Noll+09, and MW (CCM+O'Donnell) dust extinction,
      selected by the dust_law constructor argument.

  TLUSTYPlusGalaxyBasis
      Two-component composite SPS: TLUSTYBasis photosphere + FastStepBasis
      (FSPS/C3K) host galaxy.  Exposes get_spectra_components() so that the
      calling code (_logl) can apply sigma_smooth only to the photosphere —
      which Prospector's SpecModel cannot do when both components are summed
      before smoothing.

      Photosphere dust is handled at this level (not inside TLUSTYBasis) so
      that the dust law and parameters are shared with, or independent of, the
      host-galaxy dust law.  Two modes are supported:

        phot_dust_mode='shared' (default)
            The photosphere sees the same foreground ISM dust as the host galaxy.
            dust2_gal (diffuse A_V) and dust_index (Noll+09 slope modifier) are
            applied to the photosphere via the Calzetti+Noll+09 curve, exactly
            as FSPS applies them to the galaxy internally.  No extra free
            parameter is needed.

        phot_dust_mode='independent'
            The photosphere has its own dust parameter av_phot (A_V in
            magnitudes), with the law chosen by phot_dust_law ('smc',
            'calzetti', or 'mw').  When phot_dust_law='calzetti', the slope
            modifier is read from dust_index_phot (default 0 = pure Calzetti).

      To use independent mode: set phot_dust_mode='independent' in build_sps(),
      add av_phot as a free parameter in build_model(), and (for 'calzetti')
      optionally add dust_index_phot.

=============================================================================
Dust extinction laws
=============================================================================

  SMC  — Gordon+2003 (Table 1): A(λ)/A_V tabulated.
         Motivated by ΔBIC evidence from early single-component Egg fits.

  Calzetti+Noll+09  — Calzetti+2000 base curve with power-law slope modifier δ.
         A(λ)/A_V = k_cal(λ)/R_V × (λ/5500 Å)^δ
         k_cal is the Calzetti+2000 starburst attenuation curve; R_V = 4.05.
         Matches FSPS dust_type=4 (Noll+09) with no UV bump.
         δ=0 → pure Calzetti.  δ<0 → steeper (more UV attenuation).
         This is the law used by the host-galaxy component internally.

  MW   — Cardelli+1989 + O'Donnell 1994 optical update: A(λ)/A_V polynomial.
         Default R_V = 3.1; overridden by params.get('r_v_phot', 3.1).

=============================================================================
Free parameters consumed by TLUSTYBasis.get_galaxy_spectrum()
=============================================================================

  teff      : float, K         — effective temperature
  logg      : float            — log10(g / cm s^-2)
  mh_idx    : float in [0, 2]  — metallicity index; 0=[M/H]=-2, 1=-1, 2=0
  logL_star : float            — log10(L_star / L_sun)
  dust2     : float            — A_V in magnitudes; applied with self.dust_law.
                                 Ignored when dust_law='none'.

=============================================================================
Normalization convention
=============================================================================
TLUSTY .spec files store Eddington flux H_λ [erg/s/cm²/Å].  Integral satisfies:
  σ_SB × T_eff⁴ = 4π ∫ H_λ dλ

get_galaxy_spectrum converts to total L_sun/Hz using the *interpolated* H_lam
so that R is self-consistent with the actual spectrum (not an assumed blackbody):
  H_bol      = ∫ H_λ dλ  (trapezoidal, from interpolated spectrum)
  4πR²       = L_star / (4π × H_bol)
  H_ν        = H_λ × λ²/c
  L_ν        = 4πR² × 4π H_ν
  spec [L_sun/Hz] = L_ν / L_sun

Using σ_SB × T_stated⁴ in the denominator instead of 4π H_bol would introduce a
factor-of-several normalization error whenever T is clipped by the sparse grid
(e.g. [M/H]=0 only has T={4000,5000} K, so any sampled T>5000 K clips to 5000 K
while the old formula evaluated at the stated T, causing a 3× error at T=6750 K).

=============================================================================
Out-of-bounds handling
=============================================================================
Teff is clipped to the grid's T range before bracketing.  Logg outside the
per-T range triggers linear extrapolation (matching Liu+2026's original
_liu_tlogg code).  T and logg values inside the grid are always valid;
there is no 2D convex hull that can exclude physically valid combinations
like (T=5293 K, logg=−2.255).
"""

import os
import re
import glob
import time
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["TLUSTYBasis", "TLUSTYPlusGalaxyBasis"]

# ── Physical constants (CGS) ──────────────────────────────────────────────────
_lsun     = 3.846e33       # erg/s
_c_AA     = 2.998e18       # Å/s
_sigma_SB = 5.6704e-5      # erg/s/cm²/K⁴
_CKMS     = 2.998e5        # km/s

_SPEC_PATTERN = re.compile(r't(\d+)g([+-]\d+)z([+-]\d+)v(\d+)\.spec')


def _air2vac(wave_air):
    """
    Convert standard-air wavelengths [Å] to vacuum.

    SYNSPEC writes the Liu+2026 .spec files on the usual stellar-atmosphere
    convention (air wavelengths above 2000 Å), whereas FSPS, the sedpy
    filters, and reduced JWST/DESI/PypeIt spectra are all on vacuum
    wavelengths.  The library is therefore converted to vacuum once, when
    the cache is built, so that every component of the composite model and
    the data share one frame.  Formula from Allen's Astrophysical Quantities
    (as used by sedpy.observate.air2vac); the correction is +2.3 Å (82 km/s)
    at 8500 Å.
    """
    ss = 1e4 / wave_air
    return wave_air * (1.0 + 6.4328e-5
                       + 2.94981e-2 / (146.0 - ss ** 2)
                       + 2.5540e-4 / (41.0 - ss ** 2))


# =============================================================================
# Extinction / attenuation curves
# =============================================================================

# ── SMC (Gordon+2003, G03_SMCBar) ────────────────────────────────────────────
# A(λ)/A_V vs x = 1/λ [µm⁻¹].  Values from dust_extinction.G03_SMCBar
# (maintained by Karl Gordon, the original paper author).
# NIR (x < 1.1): CCM power law with R_V=2.74.  Optical+UV: direct measurements.
# Previous table had NIR values 1.7–4× too high, biasing dust2_gal downward and
# distorting the NIR (FIRE) slope.  Far-UV also corrected.
_SMC_X   = np.array([0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
                      2.27, 2.70, 3.33, 3.70, 4.55, 5.88])
_SMC_RAV = np.array([0.1082, 0.1705, 0.2500, 0.4652, 0.7210, 1.0013,
                      1.3719, 1.6703, 2.0153, 2.2751, 3.0899, 4.0707])


def _smc_transmission(wave_aa, a_v):
    """Flux transmission (0–1) for SMC extinction (Gordon+2003) with A_V = a_v."""
    if a_v <= 0.0:
        return np.ones(len(wave_aa))
    x   = 1.0 / (wave_aa * 1e-4)        # 1/µm
    rav = np.interp(x, _SMC_X, _SMC_RAV,
                    left=_SMC_RAV[0], right=_SMC_RAV[-1])
    return 10.0 ** (-0.4 * np.clip(rav, 0.0, None) * a_v)


# ── Calzetti+2000 + Noll+09 slope modifier (FSPS dust_type=4 base law) ────────
# k(λ) = A(λ)/E(B-V)_star; R_V = 4.05; so A(λ)/A_V = k(λ)/R_V.
# Valid range: 0.12–2.2 µm (1200–22000 Å).  k → 0 outside (no far-IR attenuation).
_R_V_CAL = 4.05


def _calzetti_k(wave_aa):
    """Calzetti+2000 attenuation curve k(λ) = A(λ)/E(B-V)_star."""
    lam = wave_aa * 1e-4          # wavelength in µm
    k   = np.zeros(len(wave_aa), dtype=float)

    # 0.12–0.63 µm  (UV/optical)
    sel = (wave_aa >= 1200.) & (wave_aa <= 6300.)
    l   = lam[sel]
    k[sel] = (2.659 * (-2.156 + 1.509/l - 0.198/l**2 + 0.011/l**3)
              + _R_V_CAL)

    # 0.63–2.2 µm  (optical/NIR)
    sel = (wave_aa > 6300.) & (wave_aa <= 22000.)
    l   = lam[sel]
    k[sel] = 2.659 * (-1.857 + 1.040/l) + _R_V_CAL

    # Beyond 2.2 µm: k = 0 (no dust attenuation in far-NIR/IR)
    return np.clip(k, 0.0, None)


def _calzetti_transmission(wave_aa, a_v, dust_index=0.0):
    """
    Flux transmission for Calzetti+2000 + Noll+09 slope modifier.

    Matches FSPS dust_type=4 with no UV bump:
        A(λ) = a_v × k(λ)/R_V × (λ/5500 Å)^dust_index

    Parameters
    ----------
    wave_aa    : wavelength array, Å
    a_v        : A_V in magnitudes (same convention as FSPS dust2)
    dust_index : slope modifier δ.  0 = pure Calzetti.  δ<0 steepens UV.
    """
    if a_v <= 0.0:
        return np.ones(len(wave_aa))
    k    = _calzetti_k(wave_aa)
    tilt = (wave_aa / 5500.0) ** dust_index
    return 10.0 ** (-0.4 * a_v * k / _R_V_CAL * tilt)


# ── Milky Way — Cardelli+1989 + O'Donnell 1994 optical update ─────────────────
def _ccm_transmission(wave_aa, a_v, r_v=3.1):
    """
    Flux transmission for CCM+O'Donnell MW extinction.

    Parameters
    ----------
    wave_aa : wavelength array, Å
    a_v     : A_V in magnitudes
    r_v     : total-to-selective extinction ratio (default 3.1)
    """
    if a_v <= 0.0:
        return np.ones(len(wave_aa))

    x = 1.0 / (wave_aa * 1e-4)   # 1/µm
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # IR: 0.3 ≤ x ≤ 1.1 (0.9–3.3 µm)
    s = (x >= 0.3) & (x <= 1.1)
    a[s] =  0.574 * x[s] ** 1.61
    b[s] = -0.527 * x[s] ** 1.61

    # Optical/NIR: 1.1 < x ≤ 3.3 (0.3–0.9 µm) — O'Donnell 1994 coefficients
    s = (x > 1.1) & (x <= 3.3)
    y = x[s] - 1.82
    a[s] = (1          + 0.17699*y  - 0.50447*y**2 - 0.02427*y**3
            + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[s] = (1.41338*y  + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)

    # UV: 3.3 < x ≤ 8.0 (0.125–0.3 µm)
    s  = (x > 3.3) & (x <= 8.0)
    xv = x[s]
    Fa = np.where(xv >= 5.9, -0.04473*(xv-5.9)**2 - 0.009779*(xv-5.9)**3, 0.)
    Fb = np.where(xv >= 5.9,  0.2130 *(xv-5.9)**2 + 0.1207  *(xv-5.9)**3, 0.)
    a[s] = 1.752 - 0.316*xv - 0.104/((xv-4.67)**2 + 0.341) + Fa
    b[s] = -3.090 + 1.825*xv + 1.206/((xv-4.62)**2 + 0.263) + Fb

    # Far-UV: 8 < x ≤ 10 (0.1–0.125 µm)
    s = (x > 8.0) & (x <= 10.0)
    y = x[s] - 8.0
    a[s] = -1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3
    b[s] =  13.670 + 4.257*y - 0.420*y**2 + 0.374*y**3

    a_lam = a + b / r_v     # = A(λ)/A_V
    return 10.0 ** (-0.4 * a_v * np.clip(a_lam, 0.0, None))


# =============================================================================
# TLUSTYBasis
# =============================================================================

class TLUSTYBasis:
    """
    SPS interface for the Liu+2026 TLUSTY photosphere spectral library.

    Compatible with Prospector's SpecModel as a drop-in SPS class.  On
    construction, all .spec files are read, converted from air to vacuum
    wavelengths (the frame of FSPS and of the data), convolved from native
    R~430,000 to target_R, and stored for fast trilinear interpolation in
    (Teff, logg, [M/H]).

    Parameters
    ----------
    spec_dir : str
        Path to the directory containing the Liu+2026 .spec files.
    target_R : int or float, optional
        Target resolving power for the in-memory cache.  Default: 10000.
    wave_lo, wave_hi : float, optional
        Rest-frame wavelength range in Å for the output grid.
        Default: 2000–40000 Å.
    verbose : bool, optional
        Print progress during cache construction.  Default: True.
    cache_file : str, optional
        Pre-built .npz cache path.  Loaded if exists; built and saved if not.
    dust_law : str, optional
        Extinction law applied to the photosphere in get_galaxy_spectrum().
        Options: 'none' (default, no attenuation), 'smc' (Gordon+2003),
        'calzetti' (Calzetti+2000 + Noll+09 slope via dust_index param),
        'mw' (CCM+O'Donnell, R_V from params.get('r_v', 3.1)).
        The dust amount is always read from params.get('dust2', 0.0) [A_V mag].

        Note: when TLUSTYBasis is used inside TLUSTYPlusGalaxyBasis, set
        dust_law='none' (the default).  TLUSTYPlusGalaxyBasis handles dust at
        the composite level so it can share parameters with the host galaxy.
    """

    def __init__(self,
                 spec_dir,
                 target_R=10000,
                 wave_lo=2000.0,
                 wave_hi=40000.0,
                 verbose=True,
                 cache_file=None,
                 xi10_cache_file=None,
                 dust_law='none',
                 **kwargs):

        self.target_R = int(target_R)
        self.wave_lo  = float(wave_lo)
        self.wave_hi  = float(wave_hi)
        self.dust_law = dust_law.lower()

        _valid_laws = ('none', 'smc', 'calzetti', 'mw')
        if self.dust_law not in _valid_laws:
            raise ValueError(f"dust_law must be one of {_valid_laws}; got '{dust_law}'")

        # spectral_resolution: library σ in km/s, used by SpecModel as libres
        self.spectral_resolution = _CKMS / (self.target_R * 2.355)

        # Build output wavelength grid (log-spaced at target_R)
        dloglam = 1.0 / self.target_R
        w_edges = np.exp(np.arange(np.log(self.wave_lo),
                                   np.log(self.wave_hi) + dloglam,
                                   dloglam))
        self.wave = 0.5 * (w_edges[:-1] + w_edges[1:])    # (N_pix,) Å

        # Load or build the convolved cache
        if cache_file is not None and os.path.exists(cache_file):
            if verbose:
                print(f"Loading cached spectra from {cache_file}", flush=True)
            self._load_npz(cache_file)
        else:
            if verbose:
                print(f"Building R={self.target_R} cache from {spec_dir} …",
                      flush=True)
            self._build_from_spec_dir(spec_dir, verbose=verbose)
            if cache_file is not None:
                np.savez_compressed(cache_file,
                                    wave=self.wave,
                                    flux=self._flux.astype(np.float32),
                                    meta=np.array(self._meta, dtype=object))
                if verbose:
                    print(f"  Saved cache to {cache_file}", flush=True)

        # Build per-[M/H], per-T logg→flux lookup tables for separable interpolation.
        # Separable bilinear interpolation (bracket in T, then interpolate in logg
        # for each T independently) matches Liu+2026's original _liu_tlogg approach
        # and avoids the 2D convex-hull restriction of LinearNDInterpolator, which
        # wrongly excluded physically valid points like (T=5293, logg=-2.255).
        self._MH_vals = sorted(set(self._MH.tolist()))
        self._n_models_per_MH = {}
        # _T_grids[mh]  : sorted unique T values for that [M/H]
        # _logg_flux[mh][T] : (sorted_logg_array, flux_array) for that (mh, T)
        self._T_grids   = {}
        self._logg_flux = {}
        for mh in self._MH_vals:
            sel = self._MH == mh
            self._n_models_per_MH[mh] = int(sel.sum())
            T_u = np.sort(np.unique(self._T[sel]))
            self._T_grids[mh] = T_u
            self._logg_flux[mh] = {}
            for Tv in T_u:
                mask = sel & (self._T == Tv)
                lv   = self._logg[mask]
                fv   = self._flux[mask]
                oi   = np.argsort(lv)
                self._logg_flux[mh][Tv] = (lv[oi], fv[oi])

        # Optional ξ_mtb=10 km/s (v10) grid for microturbulence interpolation.
        # v10 models only exist for T=4000–5000 K, [M/H]=−1.
        # When xi10_cache_file is provided, _logg_flux_v10[mh][T] holds the
        # same (sorted_logg, flux) tuples as _logg_flux but for ξ=10 km/s.
        self._has_xi10 = xi10_cache_file is not None and os.path.exists(xi10_cache_file)
        if self._has_xi10:
            self._load_xi10_npz(xi10_cache_file, verbose=verbose)
        else:
            self._logg_flux_v10 = {}

        if verbose:
            print(f"  {len(self._T)} models | "
                  f"wave: {self.wave[0]:.0f}–{self.wave[-1]:.0f} Å | "
                  f"R={self.target_R} ({len(self.wave)} pix) | "
                  f"[M/H] slices: {self._n_models_per_MH} | "
                  f"dust_law={self.dust_law}", flush=True)

    # ------------------------------------------------------------------
    def _build_from_spec_dir(self, spec_dir, verbose=True):
        """Read, convolve, and store all .spec files from spec_dir."""
        files   = sorted(glob.glob(os.path.join(spec_dir, 't*.spec')))
        records = []
        seen    = set()
        for fp in files:
            m = _SPEC_PATTERN.match(os.path.basename(fp))
            if not m:
                continue
            T    = int(m.group(1))
            logg = int(m.group(2)) / 100.0
            MH   = int(m.group(3))
            key  = (T, logg, MH)
            if key in seen:
                continue
            seen.add(key)
            records.append({'T': T, 'logg': logg, 'MH': MH, 'file': fp})

        if not records:
            raise FileNotFoundError(f"No .spec files found in {spec_dir}")

        N_pix    = len(self.wave)
        flux_out = np.zeros((len(records), N_pix), dtype=np.float64)
        dloglam_out = 1.0 / self.target_R
        log_w_out   = np.log(self.wave)
        t0 = time.time()

        for i, rec in enumerate(records):
            raw   = np.loadtxt(rec['file'])
            w_raw = _air2vac(raw[:, 0].astype(np.float64))   # .spec files are air; cache is vacuum
            f_raw = raw[:, 1].astype(np.float64)

            buf = 0.05 * (self.wave_hi - self.wave_lo)
            sel = (w_raw >= self.wave_lo - buf) & (w_raw <= self.wave_hi + buf)
            w_r = w_raw[sel]
            f_r = f_raw[sel]
            if len(w_r) < 100:
                warnings.warn(f"Too few points for {os.path.basename(rec['file'])}")
                flux_out[i] = np.nan
                continue

            dloglam_native = np.median(np.diff(w_r)) / np.median(w_r)
            dloglam_resamp = min(dloglam_native * 5.0, dloglam_out / 8.0)
            log_w_r  = np.arange(np.log(w_r[0]), np.log(w_r[-1]), dloglam_resamp)
            f_resamp = np.interp(log_w_r, np.log(w_r), f_r)

            sigma_logpix = (1.0 / (self.target_R * 2.355)) / dloglam_resamp
            f_conv = gaussian_filter1d(f_resamp, sigma_logpix)

            flux_out[i] = np.interp(log_w_out, log_w_r, f_conv,
                                    left=f_conv[0], right=f_conv[-1])

            if verbose and ((i + 1) % 25 == 0 or i == len(records) - 1):
                print(f"  {i+1}/{len(records)}  ({time.time()-t0:.0f}s)  "
                      f"T={rec['T']}, logg={rec['logg']}, MH={rec['MH']}",
                      flush=True)

        self._meta = records
        self._flux = flux_out
        self._T    = np.array([r['T']    for r in records], dtype=np.float64)
        self._logg = np.array([r['logg'] for r in records], dtype=np.float64)
        self._MH   = np.array([r['MH']   for r in records], dtype=int)

    # ------------------------------------------------------------------
    def _load_npz(self, cache_file):
        """Load a previously saved .npz cache."""
        data = np.load(cache_file, allow_pickle=True)
        self.wave  = data['wave'].astype(np.float64)
        flux_raw   = data['flux']
        meta_arr   = data['meta']
        self._meta = list(meta_arr)
        self._flux = flux_raw.astype(np.float64)
        self._T    = np.array([m['T']    for m in meta_arr], dtype=np.float64)
        self._logg = np.array([m['logg'] for m in meta_arr], dtype=np.float64)
        self._MH   = np.array([m['MH']   for m in meta_arr], dtype=int)

    # ------------------------------------------------------------------
    def _load_xi10_npz(self, cache_file, verbose=True):
        """Load the ξ_mtb=10 km/s (v10) cache and build _logg_flux_v10 lookup tables."""
        data     = np.load(cache_file, allow_pickle=True)
        flux_v10 = data['flux'].astype(np.float64)
        meta_v10 = list(data['meta'])
        T_v10    = np.array([m['T']    for m in meta_v10], dtype=np.float64)
        logg_v10 = np.array([m['logg'] for m in meta_v10], dtype=np.float64)
        MH_v10   = np.array([m['MH']   for m in meta_v10], dtype=int)

        self._logg_flux_v10 = {}
        n_v10 = {}
        for mh in np.unique(MH_v10):
            sel = MH_v10 == mh
            T_u = np.sort(np.unique(T_v10[sel]))
            self._logg_flux_v10[int(mh)] = {}
            n_v10[int(mh)] = int(sel.sum())
            for Tv in T_u:
                mask = sel & (T_v10 == Tv)
                lv   = logg_v10[mask]
                fv   = flux_v10[mask]
                oi   = np.argsort(lv)
                self._logg_flux_v10[int(mh)][float(Tv)] = (lv[oi], fv[oi])

        if verbose:
            print(f"  v10 (ξ=10 km/s) cache loaded: {len(meta_v10)} models | "
                  f"[M/H] slices: {n_v10}", flush=True)

    # ------------------------------------------------------------------
    def get_galaxy_spectrum(self, **params):
        """
        Interpolate to (teff, logg, mh_idx), apply L_star normalization,
        and optionally apply dust extinction.

        Parameters (read from **params)
        ---------------------------------
        teff      : float, K
        logg      : float
        mh_idx    : float in [0, 2]
        logL_star : float, log10(L_star/L_sun)
        dust2     : float, A_V in magnitudes; applied with self.dust_law.
                    Ignored when dust_law='none'.
        dust_index: float; used only when dust_law='calzetti' (default 0.0).
        r_v       : float; used only when dust_law='mw' (default 3.1).

        Returns
        -------
        wave  : ndarray (N_pix,), vacuum Å
        spec  : ndarray (N_pix,), Lsun/Hz
        mfrac : float = 1.0
        """
        teff      = float(np.atleast_1d(params['teff'])[0])
        logg      = float(np.atleast_1d(params['logg'])[0])
        mh_idx    = float(np.atleast_1d(params['mh_idx'])[0])
        logL_star = float(np.atleast_1d(params['logL_star'])[0])
        xi_mtb    = float(np.atleast_1d(params.get('xi_mtb', 2.0))[0])
        xi_wt     = float(np.clip((xi_mtb - 2.0) / 8.0, 0.0, 1.0))

        H_lam = self._interpolate(teff, logg, mh_idx, xi_wt=xi_wt)

        L_star_cgs = 10.0 ** logL_star * _lsun
        H_bol      = np.trapz(H_lam, self.wave)
        four_pi_R2 = L_star_cgs / (4.0 * np.pi * H_bol)

        H_nu = H_lam * self.wave ** 2 / _c_AA
        L_nu = four_pi_R2 * (4.0 * np.pi) * H_nu
        spec = L_nu / _lsun                        # Lsun/Hz

        # Dust attenuation dispatched by self.dust_law
        a_v = float(np.atleast_1d(params.get('dust2', 0.0))[0])
        if self.dust_law == 'smc' and a_v > 0.0:
            spec = spec * _smc_transmission(self.wave, a_v)
        elif self.dust_law == 'calzetti' and a_v > 0.0:
            d_idx = float(np.atleast_1d(params.get('dust_index', 0.0))[0])
            spec  = spec * _calzetti_transmission(self.wave, a_v, d_idx)
        elif self.dust_law == 'mw' and a_v > 0.0:
            r_v  = float(np.atleast_1d(params.get('r_v', 3.1))[0])
            spec = spec * _ccm_transmission(self.wave, a_v, r_v)
        # dust_law='none': no attenuation applied

        return self.wave, spec, 1.0

    # ------------------------------------------------------------------
    def get_galaxy_elines(self):
        """Return empty emission line arrays (no nebular emission)."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    def _interp_one_mh(self, teff, logg, mh, xi_wt=0.0):
        """
        Separable bilinear interpolation for one [M/H] slice.

        Brackets teff between the two nearest T grid values, then interpolates
        in logg independently for each T bracket using the models available at
        that T.  When logg falls outside the logg range at a given T, the
        interpolation weight gF is CLAMPED to [0, 1] — i.e. the nearest grid-edge
        spectrum is used, with NO extrapolation (previously gF was left unclamped,
        which produced unbounded linear extrapolation off the grid edge, e.g. a
        ~3x lever arm for logg well below the v10 logg floor).  T is likewise
        clipped to the grid's T range before bracketing.

        When xi_wt > 0 and v10 models exist for this mh and the requested T
        range, the result is blended: (1-xi_wt)*v02 + xi_wt*v10.  The v10
        grid is clamped to its own T range; outside it xi_wt is forced to 0.
        """
        T_u = self._T_grids[mh]
        teff_c = np.clip(teff, T_u[0], T_u[-1])

        Ti  = np.clip(np.searchsorted(T_u, teff_c, side='right') - 1,
                      0, len(T_u) - 2)
        T_lo, T_hi = T_u[Ti], T_u[Ti + 1]
        tF = (teff_c - T_lo) / (T_hi - T_lo) if T_hi > T_lo else 0.0

        result_v02 = np.zeros(len(self.wave))
        for T_v, tf in ((T_lo, 1.0 - tF), (T_hi, tF)):
            if tf == 0.0:
                continue
            ls, fs = self._logg_flux[mh][T_v]
            gi  = np.clip(np.searchsorted(ls, logg, side='right') - 1,
                          0, len(ls) - 2)
            g_lo, g_hi = ls[gi], ls[gi + 1]
            gF  = (logg - g_lo) / (g_hi - g_lo) if g_hi > g_lo else 0.0
            gF  = min(1.0, max(0.0, gF))  # clamp: no logg extrapolation beyond grid edge
            result_v02 += tf * ((1.0 - gF) * fs[gi] + gF * fs[gi + 1])

        if xi_wt == 0.0 or mh not in self._logg_flux_v10:
            return result_v02

        # v10 interpolation — same bilinear logic, clamped to v10 T range
        v10_grid = self._logg_flux_v10[mh]
        T_u_v10  = np.array(sorted(v10_grid.keys()))
        if teff > T_u_v10[-1]:
            return result_v02   # above v10 ceiling: no microturbulence effect

        teff_v10 = np.clip(teff, T_u_v10[0], T_u_v10[-1])
        Ti10 = np.clip(np.searchsorted(T_u_v10, teff_v10, side='right') - 1,
                       0, len(T_u_v10) - 2)
        T_lo10, T_hi10 = T_u_v10[Ti10], T_u_v10[Ti10 + 1]
        tF10 = ((teff_v10 - T_lo10) / (T_hi10 - T_lo10)
                if T_hi10 > T_lo10 else 0.0)

        result_v10 = np.zeros(len(self.wave))
        for T_v, tf in ((T_lo10, 1.0 - tF10), (T_hi10, tF10)):
            if tf == 0.0:
                continue
            ls, fs = v10_grid[T_v]
            gi  = np.clip(np.searchsorted(ls, logg, side='right') - 1,
                          0, len(ls) - 2)
            g_lo, g_hi = ls[gi], ls[gi + 1]
            gF  = (logg - g_lo) / (g_hi - g_lo) if g_hi > g_lo else 0.0
            gF  = min(1.0, max(0.0, gF))  # clamp: no logg extrapolation beyond grid edge
            result_v10 += tf * ((1.0 - gF) * fs[gi] + gF * fs[gi + 1])

        return (1.0 - xi_wt) * result_v02 + xi_wt * result_v10

    # ------------------------------------------------------------------
    def _interpolate(self, teff, logg, mh_idx, xi_wt=0.0):
        """
        Trilinear interpolation over (Teff, logg, [M/H]) using separable
        bilinear interpolation in (T, logg) for each [M/H] bracket.

        Replaces the old LinearNDInterpolator approach, which imposed a 2D
        convex hull on the scattered (T, logg) points and wrongly excluded
        physically valid combinations such as (T=5293 K, logg=−2.255) that
        fall between two T slices with different logg coverage.  The separable
        approach brackets T first, then interpolates (or extrapolates) in logg
        independently for each T bracket — matching Liu+2026's original code.

        xi_wt is forwarded to _interp_one_mh; it has no effect unless
        _logg_flux_v10 was populated via _load_xi10_npz.
        """
        mh_idx = np.clip(mh_idx, 0.0, float(len(self._MH_vals) - 1))
        lo_i   = int(np.floor(mh_idx))
        hi_i   = min(lo_i + 1, len(self._MH_vals) - 1)
        frac   = mh_idx - lo_i

        F_lo = self._interp_one_mh(teff, logg, self._MH_vals[lo_i], xi_wt=xi_wt)
        if frac == 0.0 or lo_i == hi_i:
            return F_lo
        F_hi = self._interp_one_mh(teff, logg, self._MH_vals[hi_i], xi_wt=xi_wt)
        return (1.0 - frac) * F_lo + frac * F_hi

    # ------------------------------------------------------------------
    @property
    def wavelengths(self):
        """Alias for compatibility with Prospector utilities."""
        return self.wave


# =============================================================================
# TLUSTYPlusGalaxyBasis
# =============================================================================

class TLUSTYPlusGalaxyBasis:
    """
    Composite SPS: TLUSTYBasis photosphere + FastStepBasis (FSPS/C3K) galaxy.

    Why this class exists
    ---------------------
    Prospector's SpecModel.predict() applies a single sigma_smooth LOSVD to
    the entire model spectrum.  The LRD sigma_smooth (electron-scattering
    broadening in the quasi-star atmosphere) must NOT be applied to the host
    galaxy continuum.  This class exposes get_spectra_components() so that
    _logl can smooth each component independently.

    Dust modes  (phot_dust_mode × phot_dust_law)
    --------------------------------------------
    phot_dust_law selects the attenuation curve — 'smc', 'calzetti', or 'mw'
    — applied to BOTH components.

    phot_dust_mode='shared'
        One free A_V parameter (dust2_gal) attenuates both the photosphere and
        the galaxy with the same law and the same value.  No extra free
        parameter is needed.

    phot_dust_mode='independent'
        Each component has its own free A_V: av_phot for the photosphere and
        dust2_gal for the galaxy.  The same phot_dust_law is used for both.

    In both modes, birth-cloud dust (dust1, Charlot & Fall 2000) remains inside
    FSPS and is applied only to young galaxy stars.  For calzetti mode, FSPS
    handles the galaxy's diffuse dust internally (dust2=dust2_gal); for smc/mw,
    FSPS receives dust2=0 and external attenuation is applied after.

    Component 1 — LRD photosphere (TLUSTYBasis, Liu+2026)
    -------------------------------------------------------
    ~167 TLUSTY non-LTE models at [M/H] = −2, −1, 0.  Trilinear interpolation
    in (Teff, logg, mh_idx).  NaN returned for out-of-bounds points.
    Stefan-Boltzmann normalisation via logL_star.

    Component 2 — Host galaxy (FastStepBasis, FSPS/C3K)
    -----------------------------------------------------
    Continuity non-parametric SFH (Leja+2019), N_BINS_SFH age bins.
    zcontinuous=1 for continuous metallicity interpolation via logzsol.

    Parameters
    ----------
    spec_dir : str
    target_R : int or float
    cache_file : str
    phot_dust_mode : str
        'shared' or 'independent'.  Default: 'shared'.
    phot_dust_law : str
        Law for independent photosphere dust: 'smc', 'calzetti', or 'mw'.
        Ignored in shared mode.  Default: 'smc'.
    verbose : bool
    """

    def __init__(self, spec_dir, target_R, cache_file,
                 xi10_cache_file=None,
                 phot_dust_mode='shared', phot_dust_law='smc',
                 verbose=True):

        _valid_modes = ('shared', 'independent')
        _valid_laws  = ('smc', 'calzetti', 'mw')
        if phot_dust_mode not in _valid_modes:
            raise ValueError(f"phot_dust_mode must be one of {_valid_modes}")
        if phot_dust_law not in _valid_laws:
            raise ValueError(f"phot_dust_law must be one of {_valid_laws}")

        self.phot_dust_mode = phot_dust_mode
        self.phot_dust_law  = phot_dust_law

        self.tlusty = TLUSTYBasis(
            spec_dir=spec_dir,
            target_R=target_R,
            wave_lo=2000.0,
            wave_hi=40000.0,
            verbose=verbose,
            cache_file=cache_file,
            xi10_cache_file=xi10_cache_file,
            dust_law='none',
        )
        self.wave = self.tlusty.wave      # master log-spaced grid (Å)

        from prospect.sources import FastStepBasis
        self.galaxy = FastStepBasis(zcontinuous=1, compute_vega_mags=False)
        if verbose:
            print(f'  FastStepBasis (FSPS/C3K, zcontinuous=1) initialised.  '
                  f'dust_mode={phot_dust_mode}  dust_law={phot_dust_law}',
                  flush=True)

        self.spectral_resolution = self.tlusty.spectral_resolution

    # ------------------------------------------------------------------
    @property
    def wavelengths(self):
        """Alias for Prospector utility compatibility."""
        return self.wave

    # ------------------------------------------------------------------
    def _dust_trans(self, a_v, params):
        """
        Transmission array for self.phot_dust_law at A_V = a_v.

        dust_index (Calzetti slope modifier) and r_v (MW R_V) are read from
        params with defaults of 0.0 and 3.1 respectively.
        """
        if self.phot_dust_law == 'smc':
            return _smc_transmission(self.wave, a_v)
        elif self.phot_dust_law == 'calzetti':
            return _calzetti_transmission(
                self.wave, a_v, float(params.get('dust_index', 0.0)))
        else:  # 'mw'
            return _ccm_transmission(
                self.wave, a_v, float(params.get('r_v', 3.1)))

    # ------------------------------------------------------------------
    def get_spectra_components(self, **params):
        """
        Return photosphere and galaxy spectra separately on self.wave (Lsun/Hz).

        Dust behaviour
        --------------
        shared mode   — one A_V (dust2_gal) attenuates both components with
                        phot_dust_law.  For calzetti, FSPS handles the galaxy
                        internally (dust2=dust2_gal); for smc/mw the galaxy is
                        attenuated externally (FSPS receives dust2=0).
        independent   — photosphere uses av_phot, galaxy uses dust2_gal,
                        both via phot_dust_law applied externally (FSPS dust2=0
                        for smc/mw; FSPS dust2=dust2_gal for calzetti so its
                        internal Calzetti matches).

        Birth-cloud dust (dust1) always stays inside FSPS in both modes.

        Required params
        ---------------
        teff, logg, mh_idx, logL_star, mass, agebins, logzsol, dust2_gal,
        dust1, sfh, imf_type, dust_type, dust1_index,
        add_neb_emission, add_neb_continuum, add_dust_emission, zred
        shared mode:      dust_index (calzetti), r_v (mw)
        independent mode: av_phot, + same slope/r_v params as above

        Returns
        -------
        wave, spec_tl, spec_gal, mfrac
        """
        # ── Photosphere ──────────────────────────────────────────────────────
        _, spec_tl, _ = self.tlusty.get_galaxy_spectrum(**params)

        a_v_phot = float(params.get(
            'dust2_gal' if self.phot_dust_mode == 'shared' else 'av_phot', 0.0))
        if a_v_phot > 0.0:
            spec_tl = spec_tl * self._dust_trans(a_v_phot, params)

        # ── Galaxy ───────────────────────────────────────────────────────────
        galaxy_params  = dict(params)
        a_v_gal        = float(params.get('dust2_gal', 0.0))
        use_fsps_dust  = (self.phot_dust_law == 'calzetti')

        if use_fsps_dust:
            # FSPS Calzetti matches phot_dust_law='calzetti'; let FSPS handle it.
            galaxy_params['dust2'] = a_v_gal
        else:
            # smc/mw: bypass FSPS diffuse dust; apply externally below.
            galaxy_params['dust2'] = 0.0

        wave_gal, spec_per_mass, mfrac = self.galaxy.get_galaxy_spectrum(
            **galaxy_params)
        mtot     = np.atleast_1d(params.get('mass', np.array([1.0]))).sum()
        spec_gal = np.interp(self.wave, wave_gal, spec_per_mass * mtot,
                             left=0.0, right=0.0)

        if not use_fsps_dust and a_v_gal > 0.0:
            spec_gal = spec_gal * self._dust_trans(a_v_gal, params)

        return self.wave, spec_tl, spec_gal, mfrac

    # ------------------------------------------------------------------
    def get_galaxy_spectrum(self, **params):
        """
        Summed spectrum — preserves the Prospector SPS API.

        Returns NaN if the photosphere is OOB.  The primary inference path
        uses get_spectra_components() to keep components separate.
        """
        wave, spec_tl, spec_gal, mfrac = self.get_spectra_components(**params)
        if np.any(np.isnan(spec_tl)):
            return wave, np.full(len(wave), np.nan), 1.0
        return wave, spec_tl + spec_gal, mfrac

    # ------------------------------------------------------------------
    def get_galaxy_elines(self):
        """Return empty arrays — neither component produces emission lines."""
        return np.array([]), np.array([])
