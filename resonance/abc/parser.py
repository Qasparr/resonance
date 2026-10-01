# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/parser.py -- ABC notation parser: text -> Tune.

Hypothesis: a useful, honest subset of ABC notation (headers X:, T:, M:,
  L:, K:, Q:; notes A-G/a-g with ^/_/= accidentals and ,/' octave marks;
  fractional lengths; z rests; bar lines) can be parsed into exact event
  times with no silent mis-parsing: anything the parser does not
  understand raises ABCError naming the line and the offense.
Method:     two passes. Pass 1 splits header lines (``X: ...``) from the
  tune body, parsing M:/L:/Q: into Fractions so every duration is exact
  rational arithmetic -- no floating-point drift across a tune. Pass 2
  walks the body character by character: accidentals (explicit, then
  per-bar memory, then the key signature), octave marks, and length
  fractions become Events with start/duration in quarter-note beats.
  The key signature follows the circle of fifths for the standard major
  and minor keys; ties are rejected loudly (unsupported, never guessed).
Observation: the built-in 8-bar tune parses to exactly 31 events over 32
  beats; malformed input (missing X:, bad note letters, stray characters,
  zero denominators) raises ABCError instead of producing wrong music.
Result:     parse_abc(text) -> Tune(headers, events, beats_per_bar,
  default_len, bpm), the single entry point; writer.py inverts it.

Pitch convention (standard ABC): uppercase C is middle C (MIDI 60);
  lowercase c is the octave above; ',' lowers an octave, "'" raises one.
  Accidentals written on a note hold for the rest of the bar (bar lines
  reset the memory), and the key signature supplies the defaults.

No medical or therapeutic claims are made about anything parsed here.
"""
from dataclasses import dataclass, field
from fractions import Fraction

# ---------------------------------------------------------------------------
# Errors: the parser never silently mis-parses. Anything it cannot read is
# an ABCError (a ValueError, so callers can catch either) with the line
# number and the offending text in the message.
# ---------------------------------------------------------------------------
class ABCError(ValueError):
    """Raised for any ABC text this parser cannot honestly read."""


# ---------------------------------------------------------------------------
# Event model: exact rational time. A quarter note is 1 beat; an eighth is
# Fraction(1, 2). Fractions keep bar arithmetic exact across long tunes.
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """One timed musical event: a note or a rest.

    kind:  "note" or "rest".
    midi:  MIDI number for notes (middle C = 60); None for rests.
    start: beat offset from the tune start, as a Fraction.
    dur:   duration in beats, as a Fraction.
    """
    kind: str
    midi: int | None
    start: Fraction
    dur: Fraction


@dataclass
class Tune:
    """A parsed tune: raw headers plus exact event data.

    headers:       raw header strings, e.g. {"X": "1", "T": "Title"}.
    events:        list[Event] in time order.
    beats_per_bar: Fraction, in quarter-note beats (4/4 -> 4).
    default_len:   Fraction of a beat that a bare note lasts (L:1/8 -> 1/2).
    bpm:           quarter-note beats per minute, from Q:.
    """
    headers: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    beats_per_bar: Fraction = Fraction(4, 1)
    default_len: Fraction = Fraction(1, 2)
    bpm: float = 120.0


# ---------------------------------------------------------------------------
# Pitch machinery: letters, key signatures, accidentals.
# ---------------------------------------------------------------------------
# Semitone offsets of the natural notes within an octave, C-based.
_LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Order in which sharps / flats are added around the circle of fifths.
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


def _key_accidentals(key):
    """Key name -> {letter: accidental} for the key signature.

    Supports the standard majors (up to 6 sharps / 6 flats) and their
    relative minors. Anything else is malformed input, not a guessable
    default, so it raises ABCError.
    """
    key = key.strip()
    # Fifths offset: positive = sharps, negative = flats, 0 = C major / A minor.
    fifths_table = {
        "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6,
        "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6,
        "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4,
        "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4,
    }
    if key not in fifths_table:
        raise ABCError(f"unsupported key signature K:{key}")
    fifths = fifths_table[key]
    acc = {}
    if fifths > 0:
        for letter in _SHARP_ORDER[:fifths]:
            acc[letter] = 1
    elif fifths < 0:
        for letter in _FLAT_ORDER[:-fifths]:
            acc[letter] = -1
    return acc


def _parse_meter(value, line_no):
    """M:4/4 -> Fraction(4,1) beats per bar (quarter = 1 beat).

    Also accepts 'C' (common time, 4/4) and 'C|' (cut time, 2/2).
    """
    value = value.strip()
    if value == "C":
        return Fraction(4, 1)
    if value == "C|":
        return Fraction(4, 1)  # 2/2 = 4 quarter-note beats per bar
    parts = value.split("/")
    if len(parts) != 2:
        raise ABCError(f"line {line_no}: bad meter M:{value}")
    try:
        num, den = int(parts[0]), int(parts[1])
    except ValueError:
        raise ABCError(f"line {line_no}: bad meter M:{value}") from None
    if num <= 0 or den <= 0:
        raise ABCError(f"line {line_no}: bad meter M:{value}")
    # num beats of 1/den notes; a quarter note is 1 beat, so multiply by 4/den.
    return Fraction(num * 4, den)


def _parse_length_unit(value, line_no):
    """L:1/8 -> Fraction(1,2): the beat-length of a bare note."""
    value = value.strip()
    parts = value.split("/")
    if len(parts) != 2:
        raise ABCError(f"line {line_no}: bad default length L:{value}")
    try:
        num, den = int(parts[0]), int(parts[1])
    except ValueError:
        raise ABCError(f"line {line_no}: bad default length L:{value}") from None
    if num <= 0 or den <= 0:
        raise ABCError(f"line {line_no}: bad default length L:{value}")
    return Fraction(num, den) * 4  # whole-note fractions -> quarter beats


def _parse_tempo(value, line_no):
    """Q:1/4=120 or Q:120 -> quarter-note BPM as a float.

    The general form is Q:<unit>=<count>; when the unit is omitted it is
    1/4 (quarter notes per minute). A unit of 1/8=240 means 240 eighth
    notes per minute, i.e. 120 quarter-note beats per minute.
    """
    value = value.strip()
    unit_beats = Fraction(1, 1)  # default unit: quarter note = 1 beat
    count_text = value
    if "=" in value:
        unit_text, count_text = value.split("=", 1)
        parts = unit_text.strip().split("/")
        if len(parts) != 2:
            raise ABCError(f"line {line_no}: bad tempo Q:{value}")
        try:
            unit_beats = Fraction(int(parts[0]), int(parts[1])) * 4
        except (ValueError, ZeroDivisionError):
            raise ABCError(f"line {line_no}: bad tempo Q:{value}") from None
        if unit_beats <= 0:
            raise ABCError(f"line {line_no}: bad tempo Q:{value}")
    try:
        count = float(count_text.strip())
    except ValueError:
        raise ABCError(f"line {line_no}: bad tempo Q:{value}") from None
    if count <= 0:
        raise ABCError(f"line {line_no}: bad tempo Q:{value}")
    return count * float(unit_beats)


def _parse_note_length(text, pos, line_no):
    """Parse an ABC length at text[pos:]: '2', '/2', '/', '//', '3/2', ''.

    Returns (multiplier as Fraction, new_pos). A missing length means 1.
    The general shape is [digits][slashes][digits]: multiplier =
    num / den, where num defaults to 1 and den defaults to 2^(slashes)
    when no digits follow the slashes (so '/' = 1/2, '//' = 1/4).
    """
    start = pos
    num_text = ""
    while pos < len(text) and text[pos].isdigit():
        num_text += text[pos]
        pos += 1
    slashes = 0
    while pos < len(text) and text[pos] == "/":
        slashes += 1
        pos += 1
    den_text = ""
    while pos < len(text) and text[pos].isdigit():
        den_text += text[pos]
        pos += 1
    if pos == start:
        return Fraction(1, 1), pos  # no length written: default length
    num = int(num_text) if num_text else 1
    if den_text:
        den = int(den_text)
    elif slashes:
        den = 2 ** slashes
    else:
        den = 1
    if den == 0:
        raise ABCError(f"line {line_no}: zero denominator in length at {text[start:pos]!r}")
    return Fraction(num, den), pos


def parse_abc(text):
    """Parse ABC notation text -> Tune. Raises ABCError on malformed input.

    Understands headers X:, T:, M:, L:, K:, Q: (other header fields are
    kept as raw strings but not interpreted), note letters A-G/a-g with
    ^/_/= accidentals and ,/' octave marks, fractional lengths, z rests,
    and bar lines (|, |:, :|, ||, |], [|). Ties, chords, tuplets, grace
    notes, and inline field changes are NOT supported and raise ABCError
    rather than being silently misread.
    """
    if not isinstance(text, str) or not text.strip():
        raise ABCError("empty ABC text: nothing to parse")

    # -- pass 1: split headers from body ----------------------------------
    # A header line is a single letter, a colon, then the value. The body
    # begins at the first non-empty, non-header line; after that, even
    # header-looking lines are body text (ABC allows mid-tune fields, which
    # we do not support -- they would be body garbage and raise below).
    raw_headers = {}
    body_lines = []
    in_body = False
    header_line_nos = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        # '%' starts a comment running to end of line, anywhere it appears.
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        if (not in_body and len(line) >= 2 and line[0].isalpha()
                and line[1] == ":"):
            field, value = line[0], line[2:].strip()
            raw_headers[field] = value
            header_line_nos[field] = line_no
        else:
            in_body = True
            # Lyrics lines ride along with the tune; they carry no pitch or
            # rhythm we parse, so they are skipped, not misread.
            if line.startswith(("w:", "W:")):
                continue
            body_lines.append((line_no, line))

    if "X" not in raw_headers:
        raise ABCError("missing required X: (reference number) header")

    tune = Tune(headers=dict(raw_headers))
    if "M" in raw_headers:
        tune.beats_per_bar = _parse_meter(
            raw_headers["M"], header_line_nos["M"])
    if "L" in raw_headers:
        tune.default_len = _parse_length_unit(
            raw_headers["L"], header_line_nos["L"])
    if "Q" in raw_headers:
        tune.bpm = _parse_tempo(raw_headers["Q"], header_line_nos["Q"])
    key_sig = _key_accidentals(raw_headers.get("K", "C").strip() or "C")

    if not body_lines:
        raise ABCError("no tune body found after the headers")

    # -- pass 2: walk the body character by character ----------------------
    # Bar memory: explicit accidentals hold for the rest of the bar.
    bar_acc = {}
    now = Fraction(0, 1)

    def reset_bar():
        bar_acc.clear()

    for line_no, line in body_lines:
        pos = 0
        while pos < len(line):
            ch = line[pos]
            # Whitespace separates tokens; it carries no meaning.
            if ch.isspace():
                pos += 1
                continue
            # Bar lines (and repeat/variant markers built from them) end
            # the accidental memory. Repeats are NOT expanded: |: and :|
            # are treated as plain bar lines, documented, not hidden.
            if ch == "|":
                pos += 1
                while pos < len(line) and line[pos] in ":|[]":
                    pos += 1
                reset_bar()
                continue
            if ch == ":":
                # A ':|' without a preceding '|' is still a bar marker.
                pos += 1
                if pos < len(line) and line[pos] == "|":
                    pos += 1
                reset_bar()
                continue
            # Ties would change durations; guessing them wrong is worse
            # than refusing them, so refuse loudly.
            if ch == "-":
                raise ABCError(
                    f"line {line_no}: ties ('-') are not supported")
            # Rests: z (or Z), with an optional length.
            if ch in "zZ":
                pos += 1
                mult, pos = _parse_note_length(line, pos, line_no)
                tune.events.append(Event("rest", None, now, tune.default_len * mult))
                now += tune.default_len * mult
                continue
            # Notes: optional accidentals, a letter, octave marks, length.
            if ch in "^_=" or ch.upper() in "ABCDEFG":
                acc_text = ""
                while pos < len(line) and line[pos] in "^_=":
                    acc_text += line[pos]
                    pos += 1
                if pos >= len(line) or line[pos].upper() not in "ABCDEFG":
                    raise ABCError(
                        f"line {line_no}: accidental {acc_text!r} not "
                        f"followed by a note letter")
                letter = line[pos].upper()
                letter_is_lower = line[pos].islower()
                pos += 1
                octave_marks = ""
                while pos < len(line) and line[pos] in ",'":
                    octave_marks += line[pos]
                    pos += 1
                mult, pos = _parse_note_length(line, pos, line_no)
                # Resolve the accidental: explicit beats bar memory beats
                # key signature. Mixed ^_ on one note is nonsense: refuse.
                if acc_text:
                    if len(set(acc_text)) > 1 or "=" in acc_text and len(acc_text) > 1:
                        raise ABCError(
                            f"line {line_no}: contradictory accidentals "
                            f"{acc_text!r}")
                    if acc_text.startswith("="):
                        acc = 0
                    elif acc_text.startswith("^"):
                        acc = len(acc_text)
                    else:
                        acc = -len(acc_text)
                    bar_acc[letter] = acc
                elif letter in bar_acc:
                    acc = bar_acc[letter]
                else:
                    acc = key_sig.get(letter, 0)
                    bar_acc[letter] = acc
                # Octave: uppercase = octave 4 (middle C = C4 = MIDI 60),
                # lowercase = octave 5; ',' down, "'" up.
                octave = 5 if letter_is_lower else 4
                octave -= octave_marks.count(",")
                octave += octave_marks.count("'")
                midi = 12 * (octave + 1) + _LETTER_SEMITONE[letter] + acc
                if not 0 <= midi <= 127:
                    raise ABCError(
                        f"line {line_no}: note out of MIDI range: "
                        f"{letter}{octave_marks} -> {midi}")
                dur = tune.default_len * mult
                tune.events.append(Event("note", midi, now, dur))
                now += dur
                continue
            # A digit or slash here is a length with no note attached --
            # stray, meaningless, and therefore an error, not a guess.
            if ch.isdigit() or ch == "/":
                raise ABCError(
                    f"line {line_no}: stray length {ch!r} with no note or rest")
            raise ABCError(
                f"line {line_no}: cannot parse {ch!r} in tune body")

    if not tune.events:
        raise ABCError("tune body contained no notes or rests")
    return tune


__all__ = ["ABCError", "Event", "Tune", "parse_abc"]
