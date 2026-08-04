#!/usr/bin/env python3
"""
Build the TLUSTY R=5000 cache from Liu+2026 raw .spec files.

Reads 233 ASCII two-column files (wavelength AA, surface F_lambda erg/s/cm2/AA)
at R~430,000, convolves each to R=5000 on a uniform log-lambda grid, and saves:

  tlusty_r5000.npz
    wave  : (N_pix,)         Angstrom, log-spaced at R=5000
    flux  : (N_models, N_pix) float32, surface F_lambda [erg/s/cm2/AA]
    meta  : (N_models,)       object array of dicts: {T, logg, MH}

Usage:
    python build_tlusty_cache.py [/path/to/specs/dir]

Default SPEC_DIR is set below and can be overridden on the command line.
"""
import os, re, sys, glob, time
import numpy as np
from scipy.ndimage import gaussian_filter1d

SPEC_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../../../work/lrdmesa/liu2026_library/'
    'LRD_synthetic_spectral_library-main/specs'
)
if len(sys.argv) > 1:
    SPEC_DIR = sys.argv[1]

OUT_PATH = os.path.join(os.path.dirname(__file__), 'tlusty_r5000.npz')

# Output grid: full TLUSTY wavelength range at R=5000
R_OUT  = 5000
W_LO   = 2000.0    # AA
W_HI   = 40000.0   # AA

w_edges = np.exp(np.arange(np.log(W_LO), np.log(W_HI) + 1.0/R_OUT, 1.0/R_OUT))
w_cen   = 0.5 * (w_edges[:-1] + w_edges[1:])
log_w_out = np.log(w_cen)
N_PIX   = len(w_cen)
print(f'Output grid: R={R_OUT}, {N_PIX} pixels, {W_LO:.0f}–{W_HI:.0f} AA')

# Parse model files
files   = sorted(glob.glob(os.path.join(SPEC_DIR, 't*.spec')))
pattern = re.compile(r't(\d+)g([+-]\d+)z([+-]\d+)v(\d+)\.spec')
records = []
seen = set()
for fp in files:
    m = pattern.match(os.path.basename(fp))
    if not m:
        continue
    T    = int(m.group(1))
    logg = int(m.group(2)) / 100.0
    MH   = int(m.group(3))
    key  = (T, logg, MH)
    if key in seen:
        continue  # skip duplicate turbulence variants (v10, etc.)
    seen.add(key)
    records.append({'T': T, 'logg': logg, 'MH': MH, 'file': fp})

print(f'Found {len(records)} unique (T, logg, MH) models')
print(f'T:    {sorted(set(r["T"]    for r in records))}')
print(f'logg: {sorted(set(r["logg"] for r in records))}')
print(f'MH:   {sorted(set(r["MH"]   for r in records))}')

# Convolve and resample each model
flux_out = np.zeros((len(records), N_PIX), dtype=np.float32)
dloglam_out = 1.0 / R_OUT
t0 = time.time()

for i, rec in enumerate(records):
    raw   = np.loadtxt(rec['file'])
    w_raw = raw[:, 0].astype(np.float64)
    f_raw = raw[:, 1].astype(np.float64)

    # Buffer slightly beyond output range to avoid edge ringing
    sel = (w_raw >= 1800.0) & (w_raw <= 42000.0)
    w_r = w_raw[sel]
    f_r = f_raw[sel]
    if len(w_r) < 100:
        print(f'  WARNING: too few points for {os.path.basename(rec["file"])}')
        flux_out[i, :] = np.nan
        continue

    # Resample native linear-lambda grid onto uniform log-lambda grid.
    # Native: R~430,000 (dloglam ≈ 2.3e-6). Intermediate: 5× native but
    # at least 8× finer than output R to keep convolution Gaussian.
    dloglam_native = np.median(np.diff(w_r)) / np.median(w_r)
    dloglam_resamp = min(dloglam_native * 5.0, dloglam_out / 8.0)
    log_w_r  = np.arange(np.log(w_r[0]), np.log(w_r[-1]), dloglam_resamp)
    f_resamp = np.interp(log_w_r, np.log(w_r), f_r)

    # Constant-R Gaussian in log-lambda space
    sigma_logpix = (1.0 / (R_OUT * 2.355)) / dloglam_resamp
    f_conv = gaussian_filter1d(f_resamp, sigma_logpix)

    flux_out[i] = np.interp(log_w_out, log_w_r, f_conv,
                            left=np.nan, right=np.nan).astype(np.float32)

    if (i + 1) % 25 == 0 or i == len(records) - 1:
        print(f'  {i+1}/{len(records)}  ({time.time()-t0:.0f}s)  '
              f'T={rec["T"]}, logg={rec["logg"]}, MH={rec["MH"]}', flush=True)

# Save
meta = np.array(records, dtype=object)
np.savez_compressed(OUT_PATH, wave=w_cen, flux=flux_out, meta=meta)
print(f'\nSaved {OUT_PATH}')
print(f'  flux shape: {flux_out.shape}  wave shape: {w_cen.shape}')
print(f'  Elapsed: {time.time()-t0:.0f} s')
