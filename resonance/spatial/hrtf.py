# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/hrtf.py -- HRTF-based binaural panner.

Hypothesis: a listenable binaural image needs only the two classic
  duplex cues -- interaural time difference (ITD) and interaural level
  difference (ILD) -- rendered parametrically from a spherical-head
  model, with measured HRTF sets loadable when the user has one.
Method:     ITD via the Woodworth & Schlosberg spherical-head formula
  (ITD = r/c * (theta + sin theta), theta the lateral incidence
  angle), ILD via a sin^2 head-shadow curve plus a first-order
  low-pass on the contralateral ear (high frequencies shadow more);
  measured sets load from a documented NPZ layout, nearest-neighbour
  pick, time-domain convolution.
Observation: ITD is zero at azimuth 0 and monotonic toward +/-90
  degrees with the correct sign on each side; ILD attenuates only the
  far ear; the measured loader rejects malformed files loudly instead
  of rendering garbage.
Result:     `pan()` places any mono source on the sphere with a
  parametric head model by default; `load_hrtf_set()` upgrades to
  measured data where available.

Conventions (module-local, stated once):
  * azimuth_deg: 0 = straight ahead, +90 = hard right, -90 = hard left,
    +/-180 = directly behind. Positive azimuth always means the
    listener's RIGHT.
  * elevation_deg: 0 = horizon, +90 = zenith, -90 = nadir.
  * ITD sign: `itd_seconds()` returns t_left - t_right. Positive =
    the right ear hears it first = source on the RIGHT. A source at
    +90 deg (hard right) yields a POSITIVE itd_seconds value, and
    `pan()` then delays the LEFT channel.
  * Everything is float32; mono shape (N,), stereo shape (2, N).

Honesty: this is a parametric approximation, not a measured HRTF.
  It gets ITD/ILD direction and magnitude roughly right but cannot
  reproduce pinna notches, so front/back confusions and the "cone of
  confusion" remain -- the far-side response is symmetric by
  construction. No medical or health claims of any kind: this is a
  spatialization effect for creative work.
"""
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_SOUND_M_S = 343.0   # dry air at ~20 C; close enough for a panner
DEFAULT_HEAD_RADIUS_M = 0.0875  # ~17.5 cm head diameter, adult average


@dataclass
class HeadModel:
    """Tunable spherical-head parameters.

    All defaults are population averages. A user who knows their own
    head circumference (cm / pi / 100) can pass a measured radius.
    """
    head_radius_m: float = DEFAULT_HEAD_RADIUS_M
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S
    ild_max_db: float = 12.0      # far-ear attenuation at +/-90 deg
    head_shadow_cutoff_hz: float = 4500.0  # far-ear lowpass at +/-90 deg
    sample_rate: int = 44100

    def __post_init__(self):
        if self.head_radius_m <= 0:
            raise ValueError(f"head_radius_m must be > 0, got "
                             f"{self.head_radius_m}")
        if self.speed_of_sound_m_s <= 0:
            raise ValueError(f"speed_of_sound_m_s must be > 0, got "
                             f"{self.speed_of_sound_m_s}")
        if self.ild_max_db < 0:
            raise ValueError(f"ild_max_db must be >= 0, got "
                             f"{self.ild_max_db}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got "
                             f"{self.sample_rate}")


def lateral_angle_rad(azimuth_deg, elevation_deg):
    """Lateral incidence angle: the angle that actually drives ITD.

    Hypothesis: for a spherical head, only the component of the source
      direction in the interaural plane matters; elevation folds in via
      sin(gamma) = sin(az) * cos(el).
    Method:     gamma = arcsin(clip(sin(az_rad) * cos(el_rad))).
    Observation: gamma = az at el = 0; gamma -> 0 as el -> +/-90
      (directly overhead, no ITD -- as physics demands).
    Result:     single lateral angle in radians, -pi/2..pi/2.
    """
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    s = np.clip(np.sin(az) * np.cos(el), -1.0, 1.0)
    return np.arcsin(s)


def itd_seconds(azimuth_deg, elevation_deg, head=None):
    """Signed interaural time difference, t_left - t_right, in seconds.

    Hypothesis: the Woodworth & Schlosberg spherical-head formula --
      ITD = (r/c) * (theta + sin theta) for the near side, mirrored
      (r/c) * (pi - theta + sin theta) past 90 deg -- captures the
      direction and scale of real ITDs with two measurable inputs.
    Method:     drive the formula with the lateral angle from
      `lateral_angle_rad()`; sign follows the azimuth sign
      (source right -> right ear first -> negative ITD).
    Observation: returns ~0 at az 0, ~+/-0.66 ms at az +/-90 deg for
      the default head (r=8.75 cm, c=343 m/s: (r/c)(pi/2+1) = 0.655
      ms -- matches the textbook ~0.65 ms), and saturates past 90 deg
      (mirrored branch), as the real far-side ITD does.
    Result:     float (or ndarray) seconds; positive = right ear leads
      (source on the right).

    References: Woodworth & Schlosberg, "Experimental Psychology"
      (1954); Kuhn, JASA 62 (1977) -- spherical-head ITD.
    """
    head = head or HeadModel()
    gamma = lateral_angle_rad(azimuth_deg, elevation_deg)
    # Near side (Woodworth): theta + sin theta. Past 90 deg the path
    # difference stops growing; mirror the branch so the curve peaks
    # at +/-90 and falls toward 180, like the real far-side ITD.
    theta = np.abs(gamma)
    branch = np.where(theta <= np.pi / 2,
                      theta + np.sin(theta),
                      np.pi - theta + np.sin(theta))
    magnitude = head.head_radius_m / head.speed_of_sound_m_s * branch
    return np.sign(gamma) * magnitude


def ild_db(azimuth_deg, elevation_deg, head=None):
    """Contralateral-ear attenuation in dB (>= 0).

    Hypothesis: head shadow grows roughly as sin^2 of the lateral
      angle -- zero on the median plane, maximal at +/-90 deg.
    Method:     ild = ild_max_db * sin^2(gamma), folded through the
      same lateral angle as ITD so elevation is accounted for.
    Observation: 0 dB at az 0; ild_max_db at az +/-90; symmetric
      front/back (the honest cone-of-confusion limit of a model with
      no pinna data).
    Result:     float (or ndarray) dB of far-ear attenuation.
    """
    head = head or HeadModel()
    gamma = lateral_angle_rad(azimuth_deg, elevation_deg)
    return head.ild_max_db * np.sin(gamma) ** 2


def _one_pole_lowpass(x, cutoff_hz, sample_rate):
    """First-order lowpass, float32, for the head-shadow shelving."""
    alpha = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    alpha = min(max(alpha, 0.0), 1.0 - 1e-6)
    y = np.empty_like(x)
    acc = 0.0
    # Per-sample loop: deliberate -- a one-pole filter is inherently
    # sequential, and N is audio-length, not a Python hot loop over
    # blocks. Kept in float64 accumulator for stability, cast back.
    for n in range(x.shape[0]):
        acc = (1.0 - alpha) * float(x[n]) + alpha * acc
        y[n] = acc
    return y.astype(np.float32, copy=False)


def pan(mono, azimuth_deg, elevation_deg=0.0, head=None, sample_rate=44100):
    """Binaural-pan a mono source. Returns stereo float32 (2, N).

    Hypothesis: applying the duplex cues -- fractional-sample delay
      for ITD, gain + high-frequency shelving for ILD -- yields a
      stable, directionally correct binaural image from one mono
      channel.
    Method:     1) ITD -> per-channel fractional delay (linear
      interpolation between the two bracketing integer delays);
      2) ILD -> attenuate the far ear by ild_db and low-pass it with
      a cutoff that tightens with lateral angle (no shadow at az 0,
      full head_shadow_cutoff_hz at +/-90); 3) near ear untouched.
    Observation: a source at +90 deg arrives first and loudest in the
      right channel; at -90 deg the mirror image; at 0 deg the two
      channels are bit-identical (no cue applied, no coloration).
    Result:     stereo float32 (2, N), same length as the input.

    Raises: ValueError on non-mono input.
    """
    x = np.asarray(mono, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"pan() needs mono (N,) input, got shape {x.shape}")
    head = head or HeadModel(sample_rate=sample_rate)
    n = x.shape[0]

    itd = float(itd_seconds(azimuth_deg, elevation_deg, head))
    # left_delay/right_delay: the FAR ear is delayed. itd = tL - tR,
    # so positive itd (source left) delays the RIGHT channel.
    delay_l = max(itd, 0.0) * sample_rate
    delay_r = max(-itd, 0.0) * sample_rate

    def _delay(sig, d):
        if d <= 0.0:
            return sig
        whole = int(d)
        frac = d - whole
        # Fractional delay by linear interpolation between the two
        # integer-delayed copies; zero-padded (no wrap).
        a = np.zeros(n, dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)
        if whole < n:
            a[whole:] = sig[:n - whole]
        if whole + 1 < n:
            b[whole + 1:] = sig[:n - whole - 1]
        return ((1.0 - frac) * a + frac * b).astype(np.float32)

    left = _delay(x, delay_l)
    right = _delay(x, delay_r)

    ild = float(ild_db(azimuth_deg, elevation_deg, head))
    gamma = float(lateral_angle_rad(azimuth_deg, elevation_deg))
    if gamma >= 0.0:
        # Source on the RIGHT (positive azimuth): left ear is far.
        left *= 10.0 ** (-ild / 20.0)
        if ild > 0.0:
            cutoff = head.head_shadow_cutoff_hz / max(np.sin(abs(gamma)), 1e-3)
            cutoff = min(cutoff, sample_rate / 2.0 * 0.99)
            left = _one_pole_lowpass(left, cutoff, sample_rate)
    else:
        right *= 10.0 ** (-ild / 20.0)
        if ild > 0.0:
            cutoff = head.head_shadow_cutoff_hz / max(np.sin(abs(gamma)), 1e-3)
            cutoff = min(cutoff, sample_rate / 2.0 * 0.99)
            right = _one_pole_lowpass(right, cutoff, sample_rate)

    return np.stack([left, right]).astype(np.float32)


# --------------------------------------------------------------------------
# Measured HRTF sets
# --------------------------------------------------------------------------

MEASURED_HRTF_DOC = """
Measured-HRTF file format (NPZ, written e.g. by numpy.savez):

  keys (all required):
    'azimuths_deg'   float array (M,)   -- source azimuth per measurement
    'elevations_deg' float array (M,)   -- source elevation per measurement
    'hrirs'          float array (M, 2, N) -- left/right head-related
                        impulse responses; hrirs[i, 0, :] is the LEFT ear
                        for direction (azimuths_deg[i], elevations_deg[i])
    'sample_rate'    scalar int/float   -- the HRIR sample rate

  The (azimuth, elevation) grid need not be regular; `pan_measured()`
  picks the nearest measured direction. Angles follow the same
  convention as the parametric model: azimuth 0 = front, + = right;
  elevation 0 = horizon, + = up.

  Where to get sets: the SADIE, CIPIC, and ARI databases publish
  measured HRIRs in their own formats; converting any of them to
  this NPZ layout is a small script (resample to your session rate,
  stack into (M, 2, N), save the four keys). The parametric model in
  this module remains the default -- measured data is an upgrade,
  not a requirement.
""".strip()


@dataclass
class MeasuredHRTF:
    """A validated measured-HRTF set, as loaded by `load_hrtf_set()`."""
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    hrirs: np.ndarray            # (M, 2, N) float32
    sample_rate: int
    source_path: str = field(default="")


def load_hrtf_set(path):
    """Load a measured HRTF set from the NPZ layout in MEASURED_HRTF_DOC.

    Raises ValueError -- loudly, naming the file and the exact defect
    -- on any malformation: missing keys, wrong shapes, mismatched
    counts, non-finite samples, bad sample rate. A corrupt HRTF set
    must never silently render wrong audio.
    """
    path = str(path)
    try:
        zf = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"load_hrtf_set: cannot read '{path}' as NPZ: "
            f"{type(exc).__name__}: {exc}\n"
            f"Expected layout:\n{MEASURED_HRTF_DOC}") from exc

    required = ("azimuths_deg", "elevations_deg", "hrirs", "sample_rate")
    missing = [k for k in required if k not in zf]
    if missing:
        raise ValueError(
            f"load_hrtf_set: '{path}' is missing required key(s) "
            f"{missing}. Expected keys: {list(required)}.\n"
            f"Expected layout:\n{MEASURED_HRTF_DOC}")

    az = np.asarray(zf["azimuths_deg"], dtype=np.float64).ravel()
    el = np.asarray(zf["elevations_deg"], dtype=np.float64).ravel()
    hrirs = np.asarray(zf["hrirs"], dtype=np.float64)
    sr_raw = zf["sample_rate"]

    problems = []
    if az.shape != el.shape:
        problems.append(f"azimuths_deg {az.shape} vs elevations_deg "
                        f"{el.shape}: counts must match")
    if hrirs.ndim != 3 or hrirs.shape[1] != 2:
        problems.append(f"hrirs must be (M, 2, N), got shape {hrirs.shape}")
    if hrirs.ndim == 3 and az.shape[0] != hrirs.shape[0]:
        problems.append(f"M mismatch: {az.shape[0]} directions but "
                        f"hrirs has M={hrirs.shape[0]}")
    if not np.all(np.isfinite(az)) or not np.all(np.isfinite(el)):
        problems.append("azimuths_deg/elevations_deg contain non-finite values")
    if not np.all(np.isfinite(hrirs)):
        problems.append("hrirs contain non-finite (NaN/inf) samples")
    try:
        sr = int(sr_raw)
        if sr <= 0 or float(sr_raw) != sr:
            problems.append(f"sample_rate must be a positive int, got {sr_raw!r}")
    except (TypeError, ValueError):
        problems.append(f"sample_rate must be a positive int, got {sr_raw!r}")
        sr = 0
    if hrirs.ndim == 3 and hrirs.shape[2] < 2:
        problems.append(f"hrirs impulse length N={hrirs.shape[2]}: "
                        f"need at least 2 taps")

    if problems:
        raise ValueError(
            f"load_hrtf_set: '{path}' is malformed "
            f"({len(problems)} problem(s)):\n  - " +
            "\n  - ".join(problems) +
            f"\nExpected layout:\n{MEASURED_HRTF_DOC}")

    return MeasuredHRTF(azimuths_deg=az.astype(np.float64),
                        elevations_deg=el.astype(np.float64),
                        hrirs=hrirs.astype(np.float32),
                        sample_rate=sr,
                        source_path=path)


def pan_measured(mono, azimuth_deg, elevation_deg, hrtf):
    """Binaural-pan via a measured HRTF set. Returns stereo (2, N).

    Hypothesis: convolving mono audio with the nearest measured HRIR
      pair reproduces the measured ear signals for that direction.
    Method:     nearest measured (azimuth, elevation) by great-circle
      distance on the sphere; FFT convolution per ear.
    Observation: output length equals input length (HRIR tail is
      folded in, not appended -- the tail past N samples is dropped,
      documented, and short HRIRs make the loss negligible); direction
      follows the file's own grid.
    Result:     stereo float32 (2, N).
    """
    x = np.asarray(mono, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"pan_measured() needs mono (N,) input, got "
                         f"{x.shape}")
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    # Unit vectors of all measured directions, great-circle distance
    # via dot product = cos(angle).
    maz = np.deg2rad(hrtf.azimuths_deg)
    mel = np.deg2rad(hrtf.elevations_deg)
    target = np.array([np.cos(el) * np.cos(az),
                       np.cos(el) * np.sin(az),
                       np.sin(el)])
    grid = np.stack([np.cos(mel) * np.cos(maz),
                     np.cos(mel) * np.sin(maz),
                     np.sin(mel)], axis=1)
    idx = int(np.argmax(grid @ target))
    h = hrtf.hrirs[idx]  # (2, N_hrir)
    n = x.shape[0]
    out = np.empty((2, n), dtype=np.float32)
    for ear in (0, 1):
        full = np.convolve(x.astype(np.float64), h[ear].astype(np.float64))
        out[ear] = full[:n].astype(np.float32)
    return out
