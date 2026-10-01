# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/ambisonic.py -- first-order ambisonic (B-format)
encode/decode for speaker arrays.

Hypothesis: first-order B-format is the right interchange for
  positioned stems -- encode once per source, decode per speaker
  layout, and the mix stays layout-agnostic.
Method:     AmbiX convention throughout: ACN channel ordering with
  SN3D normalization (the modern standard; see below). Decode by
  projecting the B-format vector onto each speaker direction
  (naive/projection decode -- cardioid per speaker).

CHANNEL ORDERING -- READ THIS, IT IS THE LOAD-BEARING PARAGRAPH.

  Two orderings exist in the wild and confusing them corrupts every
  render:

  * FuMa (Furse-Malham, the legacy): channel order W, X, Y, Z with
    maxN normalization (W scaled by 1/sqrt(2) relative to AmbiX).
  * AmbiX (the modern standard, used by YouTube, Facebook 360,
    and the ambiX/ICST plugins): ACN channel ordering with SN3D
    normalization.

  THIS MODULE USES AmbiX, i.e. ACN ordering + SN3D normalization:

    channel 0 (ACN 0): W = p / sqrt(2)
    channel 1 (ACN 1): Y = p * y_left
    channel 2 (ACN 2): Z = p * z_up
    channel 3 (ACN 3): X = p * x_fwd

  where (x_fwd, y_left, z_up) is the unit direction vector in the
  AmbiX coordinate frame: X forward, Y LEFT, Z up. Note the module's
  azimuth convention (0 = front, + = RIGHT, shared with hrtf.py) is
  mapped internally: y_left = -sin(az)*cos(el), x_fwd = cos(az)*cos(el),
  z_up = sin(el). SN3D means every channel has unit weight -- no
  per-channel scaling factors to remember, unlike FuMa/maxN.

  Files written by `encode()` are therefore directly consumable by
  any AmbiX-aware toolchain (ffmpeg's ambisonic support, the IEM
  plugin suite, YouTube spatial audio) -- label them AmbiX/ACN/SN3D
  on export.

DECODE: for speaker i at direction unit vector u_i, the feed is the
  dot product of the B-format vector with the speaker's SN3D harmonic
  vector: s_i = W/sqrt(2) + X*x_i + Y*y_i + Z*z_i. For a source
  exactly at speaker k this yields a cardioid lobe peaking at k --
  the classic naive decode. It is NOT max-rE or any optimized
  decode; for irregular arrays an optimized decoder (out of scope
  for v0.2.0) localizes better. The naive decode is exact and honest
  about what it is.

Hypothesis/Method/Observation/Result for the pair:
  Hypothesis: encode then decode over a symmetric speaker array
    reproduces each source's relative speaker balance.
  Method:     encode(mono, az, el) -> (4, N); decode(b, speaker_azs).
  Observation: a centered source (az 0) decodes to identical feeds on
    every speaker of a symmetric array; a source at a speaker's own
    azimuth peaks on that speaker.
  Result:     layout-agnostic positioning for stems and voices.

No medical or health claims of any kind -- this is speaker-array
spatialization for creative work.
"""
import numpy as np

SQRT2 = np.sqrt(2.0)


def _direction_vector(azimuth_deg, elevation_deg):
    """Unit vector in the AmbiX frame (X fwd, Y left, Z up).

    The module azimuth convention (+ = right) is mapped to the AmbiX
    Y-left axis: y_left = -sin(az) * cos(el).
    """
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    x_fwd = np.cos(az) * np.cos(el)
    y_left = -np.sin(az) * np.cos(el)
    z_up = np.sin(el)
    return np.array([x_fwd, y_left, z_up], dtype=np.float64)


def encode(mono, azimuth_deg, elevation_deg=0.0):
    """Encode mono into first-order B-format. Returns (4, N) float32.

    AmbiX (ACN/SN3D) channel order: [W, Y, Z, X] where
      W = p / sqrt(2), Y = p*y_left, Z = p*z_up, X = p*x_fwd.
    Raises ValueError on non-mono input.
    """
    p = np.asarray(mono, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"encode() needs mono (N,) input, got {p.shape}")
    x_fwd, y_left, z_up = _direction_vector(azimuth_deg, elevation_deg)
    b = np.stack([p / SQRT2,
                  p * y_left,
                  p * z_up,
                  p * x_fwd], axis=0)
    return b.astype(np.float32)


def decode(bformat, speaker_azimuths_deg, speaker_elevations_deg=None):
    """Decode B-format to speaker feeds. Returns (n_speakers, N).

    bformat: (4, N) AmbiX ACN/SN3D as produced by `encode()`.
    speaker_azimuths_deg: sequence of azimuths, same convention as
      encode (+ = right, 0 = front). Elevations default to 0.
    Method: naive/projection decode -- s_i = W/sqrt(2) + X*x_i +
      Y*y_i + Z*z_i (dot product with the speaker's SN3D vector).
    Raises ValueError on wrong channel count or empty speaker list.
    """
    b = np.asarray(bformat, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != 4:
        raise ValueError(f"decode() needs (4, N) B-format, got {b.shape}")
    azs = list(speaker_azimuths_deg)
    if not azs:
        raise ValueError("decode() needs at least one speaker azimuth")
    if speaker_elevations_deg is None:
        els = [0.0] * len(azs)
    else:
        els = list(speaker_elevations_deg)
        if len(els) != len(azs):
            raise ValueError("speaker azimuth/elevation counts differ: "
                             f"{len(azs)} vs {len(els)}")

    w, y, z, x = b[0], b[1], b[2], b[3]
    feeds = np.empty((len(azs), b.shape[1]), dtype=np.float64)
    for i, (az, el) in enumerate(zip(azs, els)):
        x_fwd, y_left, z_up = _direction_vector(az, el)
        feeds[i] = w / SQRT2 + x * x_fwd + y * y_left + z * z_up
    return feeds.astype(np.float32)
