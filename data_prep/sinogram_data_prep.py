"""
projection_data_prep.py

Physics-informed paired dataset generation for CT sinogram-domain denoising.

Noise is applied to raw DICOM helical projections BEFORE helix2fan rebinning —
the physically correct order, since Poisson shot noise originates in the
raw photon counts, not in the rebinned/filtered sinogram.

Pipeline
--------
    DICOM (raw helical projections)
        ├─ add Poisson / ring / motion noise ──► noisy helical projections
        │                                                 │
        │                                         helix2fan rebinning
        │                                         (curved→flat + helical→fan)
        │                                                 │
        │                                  noisy fan-beam .tif  ← training INPUT
        │
        └─ helix2fan rebinning ──────────────────► FD fan-beam .tif  ← training TARGET

Training:  diffCT (differentiable FBP) end-to-end
Inference: torch-radon

Noise models from published CT literature:
  [1] Chen et al., IEEE TMI 2017          – Poisson + electronic noise
  [2] Zabic et al., Eur Radiol 2013       – electronic noise parameters
  [3] Sijbers & Postnov, PMB 2004         – ring artifacts
  [4] Rashid et al., ESA 2010             – ring artifact model
  [5] Lell & Kachelriess, IR 2020         – motion artifacts

Note: Metal artifacts are not applied here because they require fan-beam
geometry (sinusoidal metal trace).  Apply them as a post-processing step
on the rebinned .tif if needed.
"""

import ast
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# helix2fan integration  (vendored under extern/helix2fan, Apache 2.0)
# ---------------------------------------------------------------------------

_HELIX2FAN_DIR = Path(__file__).parent.parent / 'extern' / 'helix2fan'
if str(_HELIX2FAN_DIR) not in sys.path:
    sys.path.insert(0, str(_HELIX2FAN_DIR))

try:
    from read_data import read_projections, unpack_tag                  # noqa: E402
    from rebinning_functions import (                                   # noqa: E402
        _rebin_curved_to_flat_detector_core,
        rebin_curved_to_flat_detector_multiprocessing,
        rebin_helical_to_fan_beam_trajectory,
    )
    from helper import save_to_tiff_stack_with_metadata                 # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"Cannot import helix2fan modules from {_HELIX2FAN_DIR}. "
        f"Expected files: read_data.py, rebinning_functions.py, helper.py"
    ) from exc


# ---------------------------------------------------------------------------
# 1.  Geometry extraction  (no argparse dependency)
# ---------------------------------------------------------------------------

def _extract_geometry(data_headers, indices):
    """Extract CT scanner geometry from DICOM-CT-PD headers.

    Replicates helix2fan/read_data.py::read_dicom() geometry extraction
    without the argparse dependency, returning a plain SimpleNamespace
    with the same attribute names that rebinning_functions.py expects.

    Parameters
    ----------
    data_headers : list of pydicom.Dataset
        Headers returned by read_projections().
    indices : slice
        Projection index range that was loaded (stored in metadata).

    Returns
    -------
    args : SimpleNamespace
    """
    angles = np.array([unpack_tag(d, 0x70311001) for d in data_headers]) + (np.pi / 2)
    angles = -np.unwrap(angles) - np.pi

    nu  = data_headers[0].Rows
    nv  = data_headers[0].Columns
    du  = unpack_tag(data_headers[0], 0x70291002)   # DetectorElementTransverseSpacing
    dv  = unpack_tag(data_headers[0], 0x70291006)   # DetectorElementAxialSpacing
    dv_rebinned = 1.0                               # mm — virtual rebinned pixel spacing

    det_central_element = list(struct.unpack('2f', data_headers[0][0x70311033].value))
    dso = unpack_tag(data_headers[0], 0x70311003)   # source-to-isocenter
    dsd = unpack_tag(data_headers[0], 0x70311031)   # source-to-detector
    ddo = dsd - dso

    z_positions = np.array([unpack_tag(d, 0x70311002) for d in data_headers])
    pitch = (
        (unpack_tag(data_headers[-1], 0x70311002) - unpack_tag(data_headers[0], 0x70311002))
        / ((np.max(angles) - np.min(angles)) / (2 * np.pi))
    )
    nz_rebinned = int((z_positions[-1] - z_positions[0]) / dv_rebinned)
    hu_factor   = float(data_headers[0][0x70411001].value)
    rotview     = int(len(angles) / ((angles[-1] - angles[0]) / (2 * np.pi)))

    dangles = np.array([unpack_tag(d, 0x7033100B) for d in data_headers])
    dz      = np.array([unpack_tag(d, 0x7033100C) for d in data_headers])
    drho    = np.array([unpack_tag(d, 0x7033100D) for d in data_headers])

    return SimpleNamespace(
        indices=[indices.start, indices.stop],
        nu=nu, nv=nv,
        du=float(du), dv=float(dv), dv_rebinned=float(dv_rebinned),
        det_central_element=det_central_element,
        dso=float(dso), dsd=float(dsd), ddo=float(ddo),
        pitch=float(pitch),
        z_positions=z_positions,
        nz_rebinned=int(nz_rebinned),
        hu_factor=float(hu_factor),
        rotview=int(rotview),
        angles=angles,
        dangles=dangles, dz=dz, drho=drho,
    )


# ---------------------------------------------------------------------------
# 2.  Helix2fan rebinning pipeline
# ---------------------------------------------------------------------------

def _run_helix2fan(args, raw_projections, n_jobs=8):
    """Run the full helix2fan rebinning on raw helical projections.

    Step 1 — Curved detector → flat detector  (parallelised with joblib)
    Step 2 — Helical trajectory → fan-beam trajectory  (Noo et al. 1999)

    Parameters
    ----------
    args : SimpleNamespace
        Scanner geometry from _extract_geometry().
    raw_projections : np.ndarray, shape (num_proj, nv, nu), float32
        Raw helical projections in log-attenuation units.
    n_jobs : int
        Parallel workers for curved→flat rebinning.

    Returns
    -------
    proj_fan : np.ndarray, shape (rotview, nu, nz_rebinned), float32
        Fan-beam sinogram stack ready for FBP reconstruction.
    """
    data = (args, raw_projections)

    print('  Rebinning: curved detector → flat detector ...')
    proj_flat = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(rebin_curved_to_flat_detector_multiprocessing)(data, col)
            for col in range(raw_projections.shape[0])
        )
    )

    print('  Rebinning: helical trajectory → fan-beam trajectory ...')
    proj_fan = rebin_helical_to_fan_beam_trajectory(args, proj_flat)

    return proj_fan


def _save_tif(proj_fan, path, args):
    """Save fan-beam projections as TIFF with embedded geometry metadata.

    The metadata dict uses the same keys as helix2fan so the saved .tif is
    directly loadable by helix2fan, diffCT, and torch-radon pipelines.
    """
    metadata = {
        'nu':   args.nu,   'nv':  args.nv,
        'du':   args.du,   'dv':  args.dv,
        'dv_rebinned':        args.dv_rebinned,
        'det_central_element': args.det_central_element,
        'dso':  args.dso,  'dsd': args.dsd,  'ddo': args.ddo,
        'pitch':       args.pitch,
        'nz_rebinned': args.nz_rebinned,
        'hu_factor':   args.hu_factor,
        'rotview':     args.rotview,
        'angles':      args.angles.tolist(),
        'z_positions': args.z_positions.tolist(),
        'dangles':     args.dangles.tolist(),
        'dz':          args.dz.tolist(),
        'drho':        args.drho.tolist(),
    }
    save_to_tiff_stack_with_metadata(proj_fan, path, metadata=metadata)


# ---------------------------------------------------------------------------
# 3.  Physics-based noise for raw helical projections
#     Input shape throughout: (num_proj, nv, nu)
#       num_proj — number of helical projections
#       nv       — detector axial rows
#       nu       — detector transverse columns
# ---------------------------------------------------------------------------

def add_poisson_noise(projections, dose_fraction=0.25, I0_FD=1e5,
                      sigma_e=10.0, seed=None):
    """Simulate low-dose CT quantum and electronic noise  [1][2].

    Applied element-wise to the full raw helical projection array:

        T      = exp(-clip(p, 0))          # transmission
        I_LD   = dose_fraction * I0_FD * T # expected LD photon count
        I~     = Poisson(I_LD) + N(0, σ_e²)
        p_noisy= -log(clip(I~, 1) / I0_LD)

    Parameters
    ----------
    projections : np.ndarray, shape (num_proj, nv, nu), float32
        Full-dose log-attenuation raw helical projections.
    dose_fraction : float
        Fraction of FD photon count used for LD acquisition [1].
        0.25 = quarter dose.
    I0_FD : float
        Reference FD incident photon count per detector element [1].
    sigma_e : float
        Electronic noise std in photon-count units [1][2].
    seed : int or None

    Returns
    -------
    np.ndarray, same shape and dtype
    """
    rng = np.random.default_rng(seed)
    p    = projections.astype(np.float64)
    I0_LD = dose_fraction * I0_FD

    transmission = np.exp(-np.clip(p, 0, None))
    I_expected   = I0_LD * transmission

    I_noisy  = rng.poisson(I_expected).astype(np.float64)
    I_noisy += rng.normal(0.0, sigma_e, size=I_noisy.shape)
    I_noisy  = np.clip(I_noisy, 1.0, None)

    return (-np.log(I_noisy / I0_LD)).astype(np.float32)


def add_ring_artifacts(projections, bad_detector_fraction=0.01,
                       amplitude_std=None, seed=None):
    """Simulate ring artifacts from miscalibrated detector columns  [3][4].

    A real detector defect produces a constant gain offset in a fixed
    column across ALL projections — in the raw helical array this means
    a fixed offset along axis=2 (nu).  After rebinning and reconstruction
    these columns produce concentric rings.

    Parameters
    ----------
    projections : np.ndarray, shape (num_proj, nv, nu), float32
    bad_detector_fraction : float
        Fraction of the nu transverse columns that are miscalibrated [3].
    amplitude_std : float or None
        Std of the gain offset.  Defaults to 5 % of the mean abs value [3][4].
    seed : int or None

    Returns
    -------
    np.ndarray, same shape and dtype
    """
    rng = np.random.default_rng(seed)
    nu  = projections.shape[2]

    if amplitude_std is None:
        amplitude_std = 0.05 * float(np.mean(np.abs(projections)))

    num_bad  = max(1, int(bad_detector_fraction * nu))
    bad_cols = rng.choice(nu, size=num_bad, replace=False)
    offsets  = rng.normal(0.0, amplitude_std, size=num_bad).astype(np.float32)

    out = projections.copy()
    for col, offset in zip(bad_cols, offsets):
        out[:, :, col] += offset   # all projections, all axial rows, this column
    return out


def add_motion_artifacts(projections, motion_fraction=0.15,
                         max_shift_pixels=3.0, seed=None):
    """Simulate rigid lateral patient motion during a contiguous acquisition span  [5].

    Shifts a contiguous block of raw projections in the transverse (nu)
    direction using linear interpolation, with a trapezoidal ramp profile.

    Parameters
    ----------
    projections : np.ndarray, shape (num_proj, nv, nu), float32
    motion_fraction : float
        Fraction of total projections affected by motion [5].
    max_shift_pixels : float
        Peak lateral shift in detector pixels [5].
    seed : int or None

    Returns
    -------
    np.ndarray, same shape and dtype
    """
    rng = np.random.default_rng(seed)
    num_proj, nv, nu = projections.shape

    motion_length = max(1, int(motion_fraction * num_proj))
    start      = rng.integers(0, num_proj - motion_length)
    peak_shift = rng.uniform(-max_shift_pixels, max_shift_pixels)

    half   = motion_length // 2
    shifts = np.concatenate([
        np.linspace(0, peak_shift, half),
        np.linspace(peak_shift, 0, motion_length - half),
    ])

    det_idx = np.arange(nu, dtype=np.float32)
    out = projections.copy()

    for i, shift in enumerate(shifts):
        proj_idx    = start + i
        shifted_idx = det_idx - shift
        left  = np.clip(np.floor(shifted_idx).astype(int), 0, nu - 1)
        right = np.clip(left + 1,                          0, nu - 1)
        frac  = (shifted_idx - left).astype(np.float32)  # shape (nu,)

        # projections[proj_idx] has shape (nv, nu)
        # projections[proj_idx][:, left/right] has shape (nv, nu) — safe 2D indexing
        out[proj_idx] = (
            (1.0 - frac) * projections[proj_idx][:, left] +
            frac         * projections[proj_idx][:, right]
        )

    return out


def apply_noise(projections, config):
    """Apply all enabled noise types to raw helical projections.

    Parameters
    ----------
    projections : np.ndarray, shape (num_proj, nv, nu), float32
    config : dict
        Noise configuration (see default_noise_config()).

    Returns
    -------
    np.ndarray, same shape and dtype
    """
    noisy = projections.copy()

    if config.get('poisson', {}).get('enabled', True):
        cfg   = config['poisson']
        noisy = add_poisson_noise(
            noisy,
            dose_fraction=cfg.get('dose_fraction', 0.25),
            I0_FD=cfg.get('I0_FD', 1e5),
            sigma_e=cfg.get('sigma_e', 10.0),
            seed=cfg.get('seed', None),
        )

    if config.get('ring', {}).get('enabled', False):
        cfg   = config['ring']
        noisy = add_ring_artifacts(
            noisy,
            bad_detector_fraction=cfg.get('bad_detector_fraction', 0.01),
            amplitude_std=cfg.get('amplitude_std', None),
            seed=cfg.get('seed', None),
        )

    if config.get('motion', {}).get('enabled', False):
        cfg   = config['motion']
        noisy = add_motion_artifacts(
            noisy,
            motion_fraction=cfg.get('motion_fraction', 0.15),
            max_shift_pixels=cfg.get('max_shift_pixels', 3.0),
            seed=cfg.get('seed', None),
        )

    return noisy


def default_noise_config():
    """Return the default physics-informed noise configuration."""
    return {
        'poisson': {
            'enabled':       True,
            'dose_fraction': 0.25,   # quarter dose [1]
            'I0_FD':         1e5,    # reference FD photon count [1]
            'sigma_e':       10.0,   # electronic noise std [1][2]
            'seed':          None,
        },
        'ring': {
            'enabled':               False,
            'bad_detector_fraction': 0.01,  # 1 % of columns [3]
            'amplitude_std':         None,  # default: 5 % of mean value
            'seed':                  None,
        },
        'motion': {
            'enabled':           False,
            'motion_fraction':   0.15,  # 15 % of projections affected [5]
            'max_shift_pixels':  3.0,   # up to 3 detector pixels [5]
            'seed':              None,
        },
    }


# ---------------------------------------------------------------------------
# 4.  Dataset preparation
# ---------------------------------------------------------------------------

def prep_projection_dataset(dicom_scans, save_dir, noise_config=None,
                             idx_start=12000, idx_stop=16000, n_jobs=8):
    """Generate paired (noisy, FD) fan-beam sinogram .tif files.

    For each scan:
      1. Load raw helical projections from DICOM-CT-PD files.
      2. Extract scanner geometry from DICOM headers.
      3. Apply physics-based noise to the raw projections.
      4. Run helix2fan rebinning on both FD and noisy projections.
      5. Save paired .tif files with full embedded geometry metadata.

    Output layout
    -------------
        save_dir/
            FD/     <scan_id>_FD_flat_fan_projections.tif
            noisy/  <scan_id>_noisy_flat_fan_projections.tif
            manifest.json

    The saved .tif files are directly usable by diffCT for end-to-end
    training and by torch-radon for inference — same format as helix2fan
    output.

    Parameters
    ----------
    dicom_scans : list of dict
        Each entry: {'scan_id': str, 'dicom_dir': str}
        dicom_dir should contain the raw DICOM-CT-PD projection files (.dcm).
    save_dir : str or Path
    noise_config : dict or None
        Noise configuration; uses default_noise_config() if None.
    idx_start : int
        Index of the first DICOM projection to load (default 12000,
        following helix2fan convention for the Mayo LDCT dataset).
    idx_stop : int
        Index of the last DICOM projection to load (default 16000).
    n_jobs : int
        Parallel workers for the curved→flat rebinning step.
    """
    if noise_config is None:
        noise_config = default_noise_config()

    save_dir  = Path(save_dir)
    fd_dir    = save_dir / 'FD'
    noisy_dir = save_dir / 'noisy'
    fd_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'noise_config': noise_config,
        'idx_range': [idx_start, idx_stop],
        'scans': [],
    }

    for scan in dicom_scans:
        scan_id   = scan['scan_id']
        dicom_dir = scan['dicom_dir']
        indices   = slice(idx_start, idx_stop)

        print(f'\n[{scan_id}]  Loading DICOM from: {dicom_dir}')
        data_headers, raw_projections = read_projections(dicom_dir, indices)
        # raw_projections: (num_proj, nv, nu), float32, log-attenuation units
        print(f'[{scan_id}]  Loaded projections: {raw_projections.shape}')

        args = _extract_geometry(data_headers, indices)
        print(f'[{scan_id}]  Geometry: rotview={args.rotview}, '
              f'nu={args.nu}, nz_rebinned={args.nz_rebinned}')

        # --- FD fan-beam sinogram (target) ---
        print(f'[{scan_id}]  Rebinning FD projections ...')
        fd_fan   = _run_helix2fan(args, raw_projections, n_jobs=n_jobs)
        fd_path  = fd_dir / f'{scan_id}_FD_flat_fan_projections.tif'
        _save_tif(fd_fan, fd_path, args)
        print(f'[{scan_id}]  FD saved → {fd_path}  shape={fd_fan.shape}')

        # --- Apply noise to raw helical projections (before rebinning) ---
        print(f'[{scan_id}]  Applying noise to raw helical projections ...')
        noisy_raw = apply_noise(raw_projections, noise_config)

        # --- Noisy fan-beam sinogram (input) ---
        print(f'[{scan_id}]  Rebinning noisy projections ...')
        noisy_fan  = _run_helix2fan(args, noisy_raw, n_jobs=n_jobs)
        noisy_path = noisy_dir / f'{scan_id}_noisy_flat_fan_projections.tif'
        _save_tif(noisy_fan, noisy_path, args)
        print(f'[{scan_id}]  Noisy saved → {noisy_path}  shape={noisy_fan.shape}')

        manifest['scans'].append({
            'scan_id':   scan_id,
            'fd_tif':    str(fd_path),
            'noisy_tif': str(noisy_path),
            'shape':     list(fd_fan.shape),   # [rotview, nu, nz_rebinned]
        })

    manifest_path = save_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nManifest saved: {manifest_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    config_path = 'config/projection_prep.json'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = json.load(f)

    if not cfg.get('dicom_scans'):
        raise ValueError(
            "'dicom_scans' is empty in the config. "
            "Add at least one entry: {\"scan_id\": \"...\", \"dicom_dir\": \"...\"}"
        )

    prep_projection_dataset(
        dicom_scans=cfg['dicom_scans'],
        save_dir=cfg['save_dir'],
        noise_config=cfg.get('noise_config', None),
        idx_start=cfg.get('idx_start', 12000),
        idx_stop=cfg.get('idx_stop', 16000),
        n_jobs=cfg.get('n_jobs', 8),
    )
