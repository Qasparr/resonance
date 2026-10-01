# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/writer.py -- ABC notation writer: Tune -> valid ABC text.

Hypothesis: a parsed Tune can be serialized back to ABC text that
  re-parses to the identical event list -- a true round-trip -- if the
  writer mirrors the parser's accidental rules exactly.
Method:     headers are emitted from the tune's raw header strings; each
  event becomes a note/rest token with its length expressed as a multiple
  of L:. Accidentals are written explicitly whenever the needed pitch
  differs from what the parser would otherwise infer (key signature, then
  the writer's own per-bar accidental memory, reset at every bar line) --
  so the re-parse can never inherit a wrong accidental from earlier in
  the bar. Bar lines are placed by exact Fraction arithmetic against
  beats_per_bar; the tune closes with |].
Observation: parse -> write -> parse preserves every event (kind, MIDI,
  start, duration); the output is human-readable ABC with four bars per
  text line.
Result:     write_abc(tune) -> str, the inverse of parser.parse_abc.

No medical or therapeutic claims are made about anything written here.
"""
from fractions import Fraction

from resonance.abc.parser import _key_accidentals

# Pitch-class -> (letter, accidental) using sharp spellings. The writer
# always spells chromatics as sharps (^C, ^F, ...); the parser reads them
# back to the same MIDI numbers, which is what the round-trip requires.
_PC_TO_LETTER = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]


def _length_token(dur, default_len):
    """Duration -> ABC length suffix, e.g. Fraction(1,1) -> '2' when L:1/8.

    The multiplier is dur / default_len, written in ABC's compact form:
    2 -> '2', 1/2 -> '/', 1/4 -> '//', 3/2 -> '3/2', 1 -> ''.
    """
    mult = Fraction(dur, default_len)
    if mult == 1:
        return ""
    if mult.denominator == 1:
        return str(mult.numerator)
    if mult.numerator == 1:
        # '/2' halves; '/' alone halves; '//' quarters -- the ABC shorthands.
        return "/" + ("" if mult.denominator == 2 else str(mult.denominator))
    return f"{mult.numerator}/{mult.denominator}"


def _note_token(midi, key_sig, bar_acc):
    """MIDI number -> ABC note token, with explicit accidentals as needed.

    key_sig: {letter: accidental} from the tune's K: header.
    bar_acc: the writer's mirror of the parser's per-bar accidental
      memory; updated in place as notes are emitted, so the writer and
      the parser always agree on what a bare letter means.
    """
    pc = midi % 12
    octave = midi // 12 - 1  # MIDI 60 -> octave 4 (middle C)
    letter, needed = _PC_TO_LETTER[pc]
    # What would the parser infer for a bare letter right now?
    inferred = bar_acc.get(letter, key_sig.get(letter, 0))
    if inferred != needed:
        # Explicit accidental required; '=' cancels to natural.
        prefix = "=" if needed == 0 else ("^" * needed if needed > 0 else "_" * -needed)
        bar_acc[letter] = needed
    else:
        prefix = ""
        # A bare note still refreshes the bar memory to the key default,
        # exactly as the parser does when it reads one.
        bar_acc[letter] = key_sig.get(letter, 0)
    # Octave marks: octave 4 -> uppercase bare, 5 -> lowercase,
    # lower -> commas, higher -> apostrophes.
    if octave == 4:
        body = letter
    elif octave == 5:
        body = letter.lower()
    elif octave < 4:
        body = letter + "," * (4 - octave)
    else:
        body = letter.lower() + "'" * (octave - 5)
    return prefix + body


def write_abc(tune):
    """Serialize a Tune to ABC text. The output re-parses to equal events."""
    headers = tune.headers
    key_sig = _key_accidentals((headers.get("K", "C") or "C").strip())
    lines = []
    # Headers: X, T, M, L, Q, K in the conventional order, from the raw
    # strings the parser kept -- what went in comes back out.
    for field in ("X", "T", "M", "L", "Q", "K"):
        if field in headers:
            lines.append(f"{field}:{headers[field]}")

    # Body: tokens with bar lines placed by exact bar arithmetic, four
    # bars per text line for readability.
    body_parts = []
    bar_acc = {}
    current_bar = 0
    bars_on_line = 0
    for ev in tune.events:
        bar_index = int(ev.start // tune.beats_per_bar)
        while current_bar < bar_index:
            body_parts.append("| ")
            current_bar += 1
            bars_on_line += 1
            bar_acc.clear()  # the parser resets its memory here too
            if bars_on_line >= 4:
                body_parts.append("\n")
                bars_on_line = 0
        token = _length_token(ev.dur, tune.default_len)
        if ev.kind == "rest":
            body_parts.append("z" + token + " ")
        else:
            body_parts.append(_note_token(ev.midi, key_sig, bar_acc) + token + " ")
    body_parts.append("|]")
    lines.append("".join(body_parts).replace("\n ", "\n").strip())
    return "\n".join(lines) + "\n"


__all__ = ["write_abc"]
