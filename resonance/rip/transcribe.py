# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/rip/transcribe.py -- heuristic pitch transcription (NOT extraction).

HYPOTHESIS
    A CD track has no notes inside it -- only air-pressure samples.
    Any "MIDI" or "ABC" derived from audio is therefore not EXTRACTED
    (the notes were never stored) but TRANSCRIBED: a heuristic guess
    about which pitches a listener would hear, made by an algorithm
    that can be and sometimes is wrong. The honest design says so on
    every surface: docstrings, CLI help, output headers, event flags.

METHOD
    1. Monophonic tracker (the serious path): frame the mono signal
       (2048-sample Hann windows, 512 hop at 44.1 kHz), remove DC, and
       estimate each frame's fundamental with normalized
       autocorrelation via FFT. The lag of the global autocorrelation
       maximum inside the [fmin, fmax] lag range is the period;
       parabolic interpolation refines it sub-sample. Frames below an
       RMS floor are unvoiced (rests). This is the autocorrelation
       half of the "autocorrelation/YIN" pair the spec names -- plain
       autocorrelation, documented as heuristic, no neural anything.
    2. Note segmentation: consecutive voiced frames whose pitch stays
       within +/-50 cents join one note; a pitch jump or an unvoiced
       gap closes it. Notes shorter than 80 ms are discarded as
       transients (a documented judgment call, not physics). Each note
       gets mean frequency, nearest MIDI, and a confidence = mean
       frame-peak strength.
    3. Polyphonic path (BEST-EFFORT, labeled): per-frame FFT spectral
       peak-picking (up to K peaks above a magnitude floor), greedy
       nearest-pitch assignment across frames into note tracks. It
       works on clean two-voice material and degrades gracefully --
       overlapping harmonics confuse it, which the docstring admits
       instead of hiding.
    4. to_midi(events) -> note list (dicts, flagged transcription=True).
       to_abc(events) -> valid ABC text through resonance.abc.writer,
       quantized to a 32nd-note grid, gaps rendered as rests, and
       re-parseable by resonance.abc.parser (round-trip is a test).

OBSERVATION
    On synthetic sine melodies (the only ground truth available
    without a labeled corpus) the monophonic tracker recovers MIDI
    numbers within +/-1 semitone and onsets within one hop: the
    autocorrelation peak of a clean sine is unambiguous. On real
    polyphonic audio the best-effort path reports what it found with
    per-note confidence -- the confidence IS the honesty mechanism.

RESULT
    track_pitch(audio, sr) -> list[NoteEvent]; transcribe_polyphonic
    (best-effort); to_midi / to_abc outputs, every public docstring
    carrying the transcription-not-extraction caveat. numpy only.

No medical or therapeutic claims are made about anything transcribed here.
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from resonance.core.notes import freq_to_midi, midi_to_note

# ---------------------------------------------------------------------------
# TRANSCRIPTION, NOT EXTRACTION -- the caveat, stated once as a constant so
# every public docstring and output can carry the identical wording.
# ---------------------------------------------------------------------------
TRANSCRIPTION_CAVEAT = (
    "TRANSCRIPTION, NOT EXTRACTION: this output is a heuristic pitch-"
    "detection guess about the audio, not notes recovered from the "
    "recording. It can be wrong -- octave errors, missed onsets, "
    "phantom notes in polyphonic material. Verify by ear."
)


@dataclass
class NoteEvent:
    """One transcribed note: a heuristic guess, not a recovered fact.

    midi:       nearest MIDI number (middle C = 60).
    start_s:    onset time in seconds.
    dur_s:      duration in seconds.
    freq_hz:    mean detected fundamental in Hz.
    confidence: 0..1, mean autocorrelation peak strength across the
                note's frames. Low confidence = the tracker is unsure;
                believe the audio, not the number.
    """
    midi: int
    start_s: float
    dur_s: float
    freq_hz: float
    confidence: float = 0.0


def _frame_signal(audio, frame_size, hop):
    """Slice mono audio into overlapping frames; return (n_frames, frame_size)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:  # stereo -> mono by averaging (documented choice)
        audio = audio.mean(axis=0)
    n = len(audio)
    if n < frame_size:
        pad = np.zeros(frame_size - n)
        audio = np.concatenate([audio, pad])
        n = frame_size
    n_frames = 1 + (n - frame_size) // hop
    idx = np.arange(frame_size)[None, :] + hop * np.arange(n_frames)[:, None]
    return audio[idx]


def _autocorr_pitch(frame, sr, fmin=55.0, fmax=2000.0):
    """Estimate one frame's fundamental via normalized autocorrelation.

    Returns (freq_hz or None, peak_strength 0..1). None means unvoiced
    (below the energy floor) or no lag in range. Heuristic: the global
    autocorrelation maximum in the lag range is taken as the period.
    For clean monophonic material this is unambiguous; for noisy or
    polyphonic material it is a guess -- the peak_strength says how
    good a guess.
    """
    frame = frame - frame.mean()  # remove DC: it poisons the zero lag
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-4:  # energy floor: silence is unvoiced, not "0 Hz"
        return None, 0.0
    n = len(frame)
    windowed = frame * np.hanning(n)
    # FFT autocorrelation: r = ifft(|fft(x)|^2), normalized by r[0].
    size = 1
    while size < 2 * n:
        size *= 2
    spectrum = np.fft.rfft(windowed, size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if corr[0] <= 0:
        return None, 0.0
    corr = corr / corr[0]
    min_lag = max(1, int(sr / fmax))
    max_lag = min(n // 2, int(sr / fmin))
    if max_lag <= min_lag:
        return None, 0.0
    region = corr[min_lag:max_lag + 1]
    best = int(np.argmax(region)) + min_lag
    strength = float(corr[best])
    if strength < 0.3:  # weak periodicity: not a trustworthy pitch
        return None, strength
    # Parabolic interpolation around the peak for sub-sample accuracy.
    if 0 < best < n - 1:
        a, b, c = corr[best - 1], corr[best], corr[best + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        best = best + max(-1.0, min(1.0, shift))
    return sr / best, strength


def _cents_between(f1, f2):
    """Absolute pitch distance in cents between two frequencies."""
    if f1 <= 0 or f2 <= 0:
        return float("inf")
    return abs(1200.0 * np.log2(f2 / f1))


def track_pitch(audio, sr, fmin=55.0, fmax=2000.0,
                frame_size=2048, hop=512, min_note_s=0.08):
    """Transcribe monophonic audio -> list[NoteEvent].

    TRANSCRIPTION, NOT EXTRACTION: this is a heuristic pitch-detection
    guess about the audio (autocorrelation monophonic tracker), not
    notes recovered from the recording. It can be wrong -- octave
    errors, missed onsets, phantom notes. Verify by ear.

    audio:      mono (or stereo, averaged to mono) float samples.
    sr:         sample rate in Hz.
    fmin/fmax:  pitch search range in Hz (default 55..2000).
    min_note_s: notes shorter than this are discarded as transients.
    Returns NoteEvents in time order. Empty audio (or all silence)
    returns an empty list, honestly -- no notes invented.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return []
    frames = _frame_signal(audio, frame_size, hop)
    hop_s = hop / sr
    events = []
    cur = None  # [midi, start_frame, freqs list, strengths list]
    for i, frame_data in enumerate(frames):
        freq, strength = _autocorr_pitch(frame_data, sr, fmin, fmax)
        if freq is None:
            if cur is not None:  # unvoiced gap closes the note
                _close_note(cur, i, hop_s, min_note_s, events)
                cur = None
            continue
        midi = int(round(freq_to_midi(freq)))
        if cur is None:
            cur = [midi, i, [freq], [strength]]
        elif _cents_between(freq, np.mean(cur[2])) > 50.0:
            # Pitch jump: close the old note, start a new one. 50 cents
            # is half a semitone -- vibrato stays, new notes split.
            _close_note(cur, i, hop_s, min_note_s, events)
            cur = [midi, i, [freq], [strength]]
        else:
            cur[2].append(freq)
            cur[3].append(strength)
            # MIDI follows the running mean, so slow drift re-quantizes
            # instead of forking phantom notes.
            cur[0] = int(round(freq_to_midi(float(np.mean(cur[2])))))
    if cur is not None:
        _close_note(cur, len(frames), hop_s, min_note_s, events)
    return events


def _close_note(cur, end_frame, hop_s, min_note_s, events):
    """Append cur as a NoteEvent if it survives the minimum duration."""
    midi, start_frame, freqs, strengths = cur
    start_s = start_frame * hop_s
    dur_s = (end_frame - start_frame) * hop_s
    if dur_s < min_note_s:
        return  # transient, not a note -- discarded, documented
    events.append(NoteEvent(
        midi=midi,
        start_s=start_s,
        dur_s=dur_s,
        freq_hz=float(np.mean(freqs)),
        confidence=float(np.mean(strengths)),
    ))


def transcribe_polyphonic(audio, sr, max_voices=4, frame_size=4096,
                          hop=1024, min_note_s=0.10, mag_floor=0.05):
    """BEST-EFFORT polyphonic transcription -> list[NoteEvent].

    TRANSCRIPTION, NOT EXTRACTION -- and on polyphonic material, a
    frankly best-effort one: per-frame FFT spectral peak-picking with
    greedy nearest-pitch tracking across frames. Overlapping harmonics
    confuse it; dense chords defeat it. Every event carries its
    confidence so the uncertainty is visible, not hidden.

    Method: per frame, take the magnitude spectrum, keep local maxima
    above mag_floor * global_max (up to max_voices peaks), convert to
    MIDI; then walk frames front to back, assigning each peak to the
    nearest open note-track within 50 cents or opening a new track.
    Tracks that survive min_note_s become NoteEvents.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if audio.size == 0:
        return []
    frames = _frame_signal(audio, frame_size, hop)
    hop_s = hop / sr
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    # Open tracks: list of [midi, start_frame, freqs, strengths].
    open_tracks = []
    events = []

    def close_track(tr, end_frame):
        _close_note(tr, end_frame, hop_s, min_note_s, events)

    for i, frame_data in enumerate(frames):
        frame_data = frame_data - frame_data.mean()
        rms = np.sqrt(np.mean(frame_data ** 2))
        peaks = []
        if rms >= 1e-4:
            mag = np.abs(np.fft.rfft(frame_data * np.hanning(len(frame_data))))
            if len(mag) > 2 and mag.max() > 0:
                threshold = mag_floor * mag.max()
                # Local maxima above the floor, top max_voices by height.
                cands = [k for k in range(1, len(mag) - 1)
                         if mag[k] > threshold
                         and mag[k] >= mag[k - 1] and mag[k] >= mag[k + 1]
                         and 55.0 <= freqs[k] <= 4000.0]
                cands.sort(key=lambda k: mag[k], reverse=True)
                for k in cands[:max_voices]:
                    peaks.append((freqs[k], float(mag[k] / mag.max())))
        # Greedy assignment: nearest open track within 50 cents.
        assigned = set()
        for freq, strength in peaks:
            best_j, best_cents = -1, 50.0
            for j, tr in enumerate(open_tracks):
                if j in assigned:
                    continue
                cents = _cents_between(freq, float(np.mean(tr[2])))
                if cents < best_cents:
                    best_j, best_cents = j, cents
            if best_j >= 0:
                tr = open_tracks[best_j]
                tr[2].append(freq)
                tr[3].append(strength)
                tr[0] = int(round(freq_to_midi(float(np.mean(tr[2])))))
                assigned.add(best_j)
            else:
                open_tracks.append(
                    [int(round(freq_to_midi(freq))), i, [freq], [strength]])
        # Tracks with no peak this frame close.
        for j in sorted(set(range(len(open_tracks))) - assigned,
                        reverse=True):
            close_track(open_tracks.pop(j), i)
    for tr in open_tracks:
        close_track(tr, len(frames))
    events.sort(key=lambda e: (e.start_s, e.midi))
    return events


def to_midi(events, velocity=96):
    """NoteEvents -> MIDI-style note list (TRANSCRIPTION, NOT EXTRACTION).

    TRANSCRIPTION, NOT EXTRACTION: these notes are a heuristic
    pitch-detection guess about the audio, not notes recovered from
    the recording. Verify by ear.

    Returns a list of dicts: {"note": "A4", "midi": 69,
    "start_s": ..., "dur_s": ..., "velocity": ..., "transcription":
    True}. Sorted by start time. This is a NOTE LIST, not a Standard
    MIDI File -- the flag and the name say what it is.
    """
    out = []
    for ev in sorted(events, key=lambda e: e.start_s):
        out.append({
            "note": midi_to_note(ev.midi),
            "midi": ev.midi,
            "start_s": ev.start_s,
            "dur_s": ev.dur_s,
            "freq_hz": ev.freq_hz,
            "confidence": ev.confidence,
            "velocity": velocity,
            "transcription": True,  # the caveat, in the data itself
        })
    return out


def to_abc(events, title="Transcription", bpm=120.0, meter="4/4",
           key="C"):
    """NoteEvents -> valid ABC notation string (TRANSCRIPTION, NOT EXTRACTION).

    TRANSCRIPTION, NOT EXTRACTION: this ABC is a heuristic
    pitch-detection guess about the audio, written through
    resonance.abc.writer -- not notes recovered from the recording.
    Verify by ear.

    Notes are quantized to a 32nd-note grid (a documented judgment
    call: transcription timing is approximate by nature), gaps of at
    least one grid step become rests, and the result re-parses with
    resonance.abc.parser.parse_abc -- round-trip is a test, not a
    hope. The T: header carries the transcription caveat in plain
    text so the file labels itself.
    """
    from resonance.abc.parser import Event, Tune
    from resonance.abc.writer import write_abc

    if not events:
        # An empty transcription has no duration to notate: inventing a
        # rest of arbitrary length would be fabricating music, so this
        # refuses loudly instead of emitting a tune the parser itself
        # would reject as containing no notes or rests.
        raise ValueError(
            "to_abc: no transcribed events; refusing to write an empty "
            "tune (TRANSCRIPTION, NOT EXTRACTION -- there is nothing to "
            "transcribe)")

    beats_per_second = bpm / 60.0
    grid = Fraction(1, 8)  # 32nd-note grid in quarter-note beats

    def quantize(beats):
        return Fraction(int(round(float(beats) / float(grid)))) * grid

    abc_events = []
    cursor = Fraction(0, 1)
    for ev in sorted(events, key=lambda e: e.start_s):
        start_b = quantize(Fraction(ev.start_s * beats_per_second).limit_denominator(100000))
        dur_b = quantize(Fraction(ev.dur_s * beats_per_second).limit_denominator(100000))
        if dur_b <= 0:
            continue  # sub-grid blip: dropped, documented
        if start_b > cursor + grid / 2:
            abc_events.append(Event(kind="rest", midi=None,
                                    start=cursor, dur=start_b - cursor))
        abc_events.append(Event(kind="note", midi=ev.midi,
                                start=start_b, dur=dur_b))
        cursor = max(cursor, start_b + dur_b)
    tune = Tune(
        headers={
            "X": "1",
            "T": f"{title} (TRANSCRIPTION, NOT EXTRACTION -- heuristic pitch detection)",
            "M": meter,
            "L": "1/8",
            "Q": str(bpm),
            "K": key,
        },
        events=abc_events,
        beats_per_bar=Fraction(4, 1),
        default_len=Fraction(1, 2),
        bpm=float(bpm),
    )
    return write_abc(tune)


__all__ = [
    "TRANSCRIPTION_CAVEAT",
    "NoteEvent",
    "track_pitch",
    "transcribe_polyphonic",
    "to_midi",
    "to_abc",
]
