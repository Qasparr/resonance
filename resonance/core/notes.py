# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/notes.py -- note names, MIDI numbers, frequencies, Solfeggio data.

Hypothesis: a single source of truth for pitch (note name <-> MIDI <->
  frequency, A4 = 440 Hz) plus the nine traditional Solfeggio frequencies
  as plain data lets every generator root itself in a named pitch.
Method:     twelve-tone equal temperament: f = 440 * 2^((midi-69)/12);
  MIDI 69 = A4 by definition; note names parsed with sharps and flats.
Observation: A4 -> 440.0 exactly; C4 -> 261.6255... (middle C); the
  Solfeggio table carries the nine traditional frequencies with their
  traditional names, as data only -- no claims about effects are made.
Result:     round-trips (name -> midi -> name, midi -> freq -> midi) hold;
  solfeggio data is immutable and validated at import.

Honesty note: the Solfeggio frequencies belong to an experimental wellness
tradition. Listing their traditional names is cultural data, not a claim
of medical or therapeutic efficacy -- none is made anywhere in this
module or engine.
"""
import math
from types import MappingProxyType

# -- twelve-tone equal temperament -------------------------------------------
# Semitone names, sharps canonical. MIDI 60 = C4 (middle C), MIDI 69 = A4.
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Flat spellings map to the same semitone index: Db == C#, and so on.
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
A4_MIDI = 69
A4_FREQ = 440.0


def note_to_midi(name):
    """'A4' -> 69, 'C4' -> 60, 'F#3'/'Gb3' -> 54. Sharps and flats accepted.

    Raises ValueError on unparseable names. Octave follows scientific pitch
    notation: C4 is middle C.
    """
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"note_to_midi: bad note name {name!r}")
    # Split the pitch class from the octave: last char(s) are the digits.
    i = len(name)
    while i > 0 and (name[i - 1].isdigit() or name[i - 1] == "-"):
        i -= 1
    pitch, octave = name[:i], name[i:]
    pitch = _FLAT_TO_SHARP.get(pitch, pitch)  # flats resolve to sharps
    if pitch not in _SHARP_NAMES or octave in ("", "-"):
        raise ValueError(f"note_to_midi: bad note name {name!r}")
    try:
        octave = int(octave)
    except ValueError:
        raise ValueError(f"note_to_midi: bad octave in {name!r}") from None
    semitone = _SHARP_NAMES.index(pitch)
    # MIDI counts semitones from C-1 = 0, so C4 = 12*(4+1) + 0 = 60.
    return 12 * (octave + 1) + semitone


def midi_to_note(midi):
    """69 -> 'A4', 60 -> 'C4'. Inverse of note_to_midi (sharp spellings)."""
    midi = int(midi)
    if not 0 <= midi <= 127:
        raise ValueError(f"midi_to_note: MIDI range is 0-127, got {midi}")
    return f"{_SHARP_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi):
    """MIDI number -> frequency in Hz. A4 (69) = 440 Hz exactly.

    Twelve-tone equal temperament: each semitone multiplies frequency by
    2^(1/12), so f = 440 * 2^((midi - 69)/12).
    """
    return A4_FREQ * (2.0 ** ((int(midi) - A4_MIDI) / 12.0))


def freq_to_midi(freq):
    """Frequency in Hz -> nearest MIDI number (rounded). Inverse-ish."""
    freq = float(freq)
    if freq <= 0:
        raise ValueError(f"freq_to_midi: frequency must be positive, got {freq}")
    return int(round(A4_MIDI + 12.0 * math.log2(freq / A4_FREQ)))


def note_to_freq(name):
    """'A4' -> 440.0. Convenience: name -> midi -> frequency."""
    return midi_to_freq(note_to_midi(name))


# -- Solfeggio frequencies: traditional data ---------------------------------
# The nine frequencies of the Solfeggio tradition with their traditional
# names, as immutable data. This is cultural/wellness-tradition data only;
# no medical or therapeutic efficacy is claimed for any of them.
_SOLFEGGIO = {
    174: "Foundation",      # traditionally "UT" -- grounding tone
    285: "Quantum",         # traditionally associated with cellular renewal
    396: "Liberation",      # traditionally "UT/RE" -- guilt and fear
    417: "Transmutation",   # traditionally "RE" -- change and undoing
    528: "Transformation",  # traditionally "MI" -- the well-known "love" tone
    639: "Connection",      # traditionally "FA" -- relationships
    741: "Expression",      # traditionally "SOL" -- expression, solutions
    852: "Intuition",       # traditionally "LA" -- spiritual order
    963: "Oneness",         # traditionally "SI" -- unity, the crown tone
}
# MappingProxyType makes the table read-only: tradition data should not be
# mutated at runtime by accident.
SOLFEGGIO = MappingProxyType(_SOLFEGGIO)
SOLFEGGIO_FREQS = tuple(sorted(_SOLFEGGIO))  # (174, 285, ..., 963)


def solfeggio_name(freq):
    """Traditional name for a Solfeggio frequency, e.g. 528 -> 'Transformation'.

    Raises KeyError for frequencies not in the nine. Use `freq in SOLFEGGIO`
    to test membership.
    """
    return _SOLFEGGIO[int(freq)]


def nearest_solfeggio(freq):
    """Nearest of the nine Solfeggio frequencies to the given Hz value."""
    freq = float(freq)
    return min(SOLFEGGIO_FREQS, key=lambda f: abs(f - freq))
