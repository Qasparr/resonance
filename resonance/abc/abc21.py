# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/abc21.py -- ABC 2.1 extensions: ties, chords, tuplets,
  grace notes, repeat expansion with first/second endings, multi-voice.

Hypothesis: the v0.1.0 parser (resonance/abc/parser.py) can be extended
  to the full ABC 2.1 construct set -- ties (`-`), chords (`[CEG]`),
  tuplets (`(3`), grace notes (`{g}`), repeats (`|: ... :|` with `[1`/`[2`
  endings), and multi-voice (`V:1`/`V:2`) -- while (a) keeping every
  duration as exact Fraction arithmetic, (b) wiring into the EXISTING
  event model so resonance/abc/render.py renders 2.1 tunes unchanged,
  and (c) refusing loudly (ABCError) anything still unsupported instead
  of rendering it wrong. The v0.1.0 parser is NOT modified: ties still
  raise there, so existing behavior and tests are untouched; this module
  is the opt-in 2.1 entry point.
Method:     parse_abc21(text) -> Tune21 runs four passes. Pass 1 splits
  headers from the body like the base parser, additionally collecting
  V: voice declarations (header section) and V: voice switches plus
  inline-field rejection (body). Pass 2 tokenizes the body into text /
  bar / voice tokens, classifying bar lines (`|:`, `:|`, `:|:`, `[1`,
  `[2`, plain). Pass 3 expands repeats at the token level: `|: head [1
  e1 :| [2 e2 |` becomes head, e1, head, e2 with plain bar lines in place
  of the markers, recording each repeat's beat boundaries (per voice)
  for the writer. Pass 4 scans the expanded tokens character by
  character: ties merge durations into the tied event (recording the
  split beats so the writer can re-emit `-`); chords emit simultaneous
  events sharing one start; tuplets scale the next r group durations by
  q/p as exact rationals; grace notes steal half the following note's
  duration, split equally; each voice keeps its own timeline from beat 0
  and tags its events. The writer write_abc21(tune) inverts all of it:
  chords from shared starts, tuplets from (p, q) tags (lengths written
  unscaled so re-parsing re-scales exactly once), grace groups from the
  grace flag (target length restored from the stolen time), repeats from
  the recorded beat map, voices as V: sections.
Observation: a tune exercising every construct parses to exact rational
  events (a triplet of eighth notes is three 1/3-beat events; `{g}a2`
  under L:1/8 is a 1/2-beat grace plus a 1/2-beat note), renders through
  the unchanged renderer, and survives parse -> write -> parse with
  identical events; every unsupported construct probed (nested repeats,
  slurs, inline K:, dangling ties/graces, ambiguous `(5`) raises ABCError
  naming the offense.
Result:     parse_abc21(text) -> Tune21, write_abc21(tune) -> str,
  tune_voice(tune, voice) -> Tune21, render_voice(tune, voice, ...) for
  per-voice audio; the base parser, writer, and renderer are untouched.

Pitch, accidental-memory, and key-signature rules mirror parser.py
exactly (explicit beats bar-memory beats key-signature; bar lines reset
the memory). Deliberate, documented semantics chosen where ABC 2.1
leaves room: a chord's clock advance is its FIRST note's duration while
each chord tone keeps its own written length (round-trips exactly);
grace notes steal half the following note (appoggiatura convention),
split equally, written lengths inside {...} accepted but not timing;
bare `(5`/`(7`/`(9` require an explicit `:q` (the standard's default
ratio for those depends on meter in a way this parser will not guess);
ties merge into the earlier event (one sustained note, the honest
musical meaning); a tie may land on a chord tone but never leave one
(`[CE]- X` raises -- ABC gives the tie no single pitch to mean).

No medical or therapeutic claims are made about anything parsed here.
"""
import re
from dataclasses import dataclass, field
from fractions import Fraction

from resonance.abc.parser import (
    _LETTER_SEMITONE,
    _key_accidentals,
    _parse_length_unit,
    _parse_meter,
    _parse_note_length,
    _parse_tempo,
    ABCError,
    Event,
    Tune,
)
from resonance.abc.render import render_tune
from resonance.abc.writer import _length_token, _note_token

# ---------------------------------------------------------------------------
# Extended event model. Subclassing keeps the v0.1.0 model untouched: a
# Tune21 IS a Tune, an Event21 IS an Event, so render.py, writer.py, and
# every existing test keep working on 2.1 output without modification.
# ---------------------------------------------------------------------------
@dataclass
class Event21(Event):
    """An Event plus ABC 2.1 metadata.

    voice:      the V: voice id ("1", "2", ...) or None for unvoiced music.
    grace:      True for grace-note events (timing stolen from the next).
    tuplet:     (p, q) for notes in a p-in-the-time-of-q tuplet, else None.
    tie_splits: beat offsets (Fractions, absolute) where this event was
      formed by merging tied notes -- the writer re-splits there to
      re-emit the '-' markup. Empty for untied notes.
    """
    voice: str | None = None
    grace: bool = False
    tuplet: tuple | None = None
    tie_splits: list = field(default_factory=list)


@dataclass
class Tune21(Tune):
    """A Tune plus voice order and the repeat-expansion beat map.

    voices:  voice ids in declaration/appearance order (header V: lines
      first, then inline ones).
    repeats: one dict per expanded repeat, in expansion order:
      {"voice": voice-or-None, "start": h0, ...} with Fraction beats.
      Variant repeats carry head_end, ending1_end, head2_end,
      ending2_end; simple repeats carry mid and end instead.
    """
    voices: list = field(default_factory=list)
    repeats: list = field(default_factory=list)


# Default tuplet ratios for bare `(p` markers, per the ABC 2.1 draft:
# (2 = 2 in the time of 3, (3 = 3 in the time of 2, (4 = 4 in the time
# of 3, (6 = 6 in the time of 2, (8 = 8 in the time of 3. For (5, (7,
# (9 the standard's default ratio depends on the meter in a way this
# parser refuses to guess: they require an explicit `(p:q[:r]`.
_TUPLET_DEFAULT_Q = {2: 3, 3: 2, 4: 3, 6: 2, 8: 3}


# ---------------------------------------------------------------------------
# Pass 2: body tokenizer. Splits body lines into a flat token stream so
# repeat expansion (pass 3) can work structurally instead of with
# fragile string surgery.
#
# Token shapes (tuples):
#   (line_no, "text", str)              -- note text for the char scanner
#   (line_no, "bar", (subtype, text))   -- subtype: "repeat_start" (|:),
#                                          "repeat_end" (:|), "variant"
#                                          ([1, [2, ...), "plain"
#   (line_no, "voice", voice_id)        -- V: voice switch
#   (line_no, "marker", mtype)          -- inserted by repeat expansion:
#                                          "|:", "[1", ":|", "[2", "|end"
# ---------------------------------------------------------------------------
def _tokenize_line(text, line_no):
    """One body line -> tokens; bar-ish syntax leaves the text stream.

    `[` followed by a note letter stays in the text (it opens a chord
    for the char scanner); `[|` (end bar) and `[<digits>` (variant
    endings) are bar tokens. `:`/`|` runs become repeat or plain bars.
    """
    tokens = []
    buf = []
    pos = 0

    def flush():
        if buf:
            tokens.append((line_no, "text", "".join(buf)))
            del buf[:]

    while pos < len(text):
        ch = text[pos]
        nxt = text[pos + 1:pos + 2]
        if ch == "|":
            if nxt == ":":
                flush()
                tokens.append((line_no, "bar", ("repeat_start", "|:")))
                pos += 2
            elif nxt in ("|", "]"):
                flush()
                tokens.append((line_no, "bar", ("plain", text[pos:pos + 2])))
                pos += 2
            else:
                flush()
                tokens.append((line_no, "bar", ("plain", "|")))
                pos += 1
        elif ch == ":":
            if text[pos:pos + 3] == ":|:":
                # End-then-start: a repeat end immediately followed by a
                # new repeat start.
                flush()
                tokens.append((line_no, "bar", ("repeat_end", ":|")))
                tokens.append((line_no, "bar", ("repeat_start", "|:")))
                pos += 3
            elif nxt == "|":
                flush()
                tokens.append((line_no, "bar", ("repeat_end", ":|")))
                pos += 2
            else:
                # A bare ':' is not a bar line (ABC bars are |, :|, |:,
                # ::) -- it belongs to a tuplet spec like (3:2:2, which
                # the char scanner consumes whole.
                buf.append(ch)
                pos += 1
        elif ch == "[":
            if nxt == "|":
                flush()
                tokens.append((line_no, "bar", ("plain", "[|")))
                pos += 2
            elif nxt.isdigit():
                m = re.match(r"\[(\d+)", text[pos:])
                flush()
                tokens.append((line_no, "bar", ("variant", m.group(1))))
                pos += 1 + len(m.group(1))
            else:
                buf.append(ch)  # a chord: the char scanner owns it
                pos += 1
        else:
            buf.append(ch)
            pos += 1
    flush()
    return tokens


# ---------------------------------------------------------------------------
# Pass 3: repeat expansion at the token level.
#
# `|: head [1 e1 :| [2 e2 |` expands to head, e1, head, e2 -- the classic
# first/second ending -- with plain "|" bars and "marker" tokens left in
# the stream so the char scanner (pass 4) can record each repeat's beat
# boundaries for the writer. A repeat without endings doubles its head.
# Markers never nest (ABC repeats do not nest); anything malformed is an
# ABCError, never a guess.
# ---------------------------------------------------------------------------
def _is_bar(tok, *subtypes):
    """True for ("bar", (subtype, ...)) tokens, optionally filtered."""
    return (tok[1] == "bar"
            and (not subtypes or tok[2][0] in subtypes))


def _expand_repeats(tokens):
    """Expand every |: ... :| repeat in the token stream.

    Returns the expanded token list. Raises ABCError on nested repeats,
    unmatched |: or :|, variant endings outside a repeat, [1 without [2,
    or a repeat end / variant appearing before any repeat start.
    """
    out = list(tokens)
    while True:
        starts = [i for i, t in enumerate(out) if _is_bar(t, "repeat_start")]
        if not starts:
            # No repeats left: any leftover end/variant markers are stray.
            for tok in out:
                if _is_bar(tok, "repeat_end"):
                    raise ABCError(
                        f"line {tok[0]}: ':|' without a matching '|:'")
                if _is_bar(tok, "variant"):
                    raise ABCError(
                        f"line {tok[0]}: '[{tok[2][1]}' variant ending "
                        f"outside a repeat")
            return out
        si = starts[0]
        # A repeat end or variant BEFORE the first start is stray, not
        # the start's problem: report it as such.
        for tok in out[:si]:
            if _is_bar(tok, "repeat_end"):
                raise ABCError(
                    f"line {tok[0]}: ':|' without a matching '|:'")
            if _is_bar(tok, "variant"):
                raise ABCError(
                    f"line {tok[0]}: '[{tok[2][1]}' variant ending "
                    f"outside a repeat")
        # Find the matching end; another start first means nesting.
        ei = None
        for i in range(si + 1, len(out)):
            if _is_bar(out[i], "repeat_start"):
                raise ABCError(
                    f"line {out[i][0]}: nested repeats are not supported")
            if _is_bar(out[i], "repeat_end"):
                ei = i
                break
        if ei is None:
            raise ABCError(f"line {out[si][0]}: '|:' without a matching ':|'")

        ln = out[si][0]
        bar = (ln, "bar", ("plain", "|"))

        def mark(mtype):
            return (ln, "marker", mtype)

        # A [1 variant may only appear between the start and the end.
        k1 = next((i for i in range(si + 1, ei) if _is_bar(out[i], "variant")),
                  None)
        if k1 is not None and out[k1][2][1] != "1":
            raise ABCError(
                f"line {out[k1][0]}: '[{out[k1][2][1]}' inside a repeat: "
                f"only '[1' can open the first ending")
        if k1 is None:
            head = out[si + 1:ei]
            out = (out[:si]
                   + [mark("|:"), bar] + head
                   + [mark(":|"), bar] + head
                   + [mark("|end"), bar] + out[ei + 1:])
            continue
        # First/second endings: [2 must follow the repeat end (allowing
        # only whitespace between), and the second ending runs to the
        # next bar line or the end of the tune.
        head = out[si + 1:k1]
        e1 = out[k1 + 1:ei]
        j = ei + 1
        while (j < len(out) and out[j][1] == "text"
               and not out[j][2].strip()):
            j += 1
        if not (j < len(out) and _is_bar(out[j], "variant")
                and out[j][2][1] == "2"):
            raise ABCError(
                f"line {out[ei][0]}: '[1' first ending without a "
                f"matching '[2' second ending")
        k = j + 1
        while k < len(out) and not _is_bar(out[k]):
            k += 1
        e2 = out[j + 1:k]
        rest = out[k:]
        out = (out[:si]
               + [mark("|:"), bar] + head
               + [mark("[1"), bar] + e1
               + [mark(":|"), bar] + head
               + [mark("[2"), bar] + e2
               + [mark("|end"), bar] + rest)
        # Loop: more repeats may follow (or precede) this one.


# ---------------------------------------------------------------------------
# Pass 4: the extended char scanner. Mirrors parser.py's accidental,
# octave, key-signature, and length rules exactly; adds ties, chords,
# tuplets, grace notes. One _BodyParser per parse_abc21 call; each voice
# keeps its own timeline (now) restarted at beat 0 on every V: switch.
# ---------------------------------------------------------------------------
class _BodyParser:
    """Token stream -> Tune21 events. See the module docstring for the
    semantics of each construct; anything unsupported raises ABCError."""

    def __init__(self, tune, key_sig, initial_voice=None):
        self.tune = tune
        self.key_sig = key_sig
        self.now = Fraction(0, 1)
        self.bar_acc = {}
        # None for a single-voice tune; otherwise the last header V:
        # (ABC semantics: the most recent V: field selects the voice for
        # the following music).
        self.voice = initial_voice
        self.pending_tie = None      # events awaiting a tie merge
        self.tuplet = None           # {"p","q","r","remaining"} or None
        self.grace_pending = []      # midi numbers awaiting their target
        self.last_note_events = None  # events of the last note group
        self.last_group_kind = None  # "note" | "rest" | None
        self.last_group_was_chord = False
        self.current_repeat = None   # repeat dict being recorded

    # -- token dispatch --------------------------------------------------
    def run(self, tokens):
        for tok in tokens:
            line_no, kind = tok[0], tok[1]
            if kind == "voice":
                self.on_voice(tok[2], line_no)
            elif kind == "bar":
                # Every bar line -- plain, repeat, or variant -- ends the
                # accidental memory, exactly like the base parser. Ties
                # deliberately survive bar lines: that is what ties are.
                self.bar_acc.clear()
            elif kind == "marker":
                self.on_marker(tok[2])
            else:  # text
                self.scan_text(tok[2], line_no)
        if self.grace_pending:
            raise ABCError(
                "grace notes at the end of the tune have no following note")
        if self.pending_tie is not None:
            raise ABCError("dangling tie '-' with no following note")
        if self.tuplet is not None:
            t = self.tuplet
            raise ABCError(
                f"tuplet '({t['p']}' expects {t['r']} notes but the tune "
                f"ran out with {t['remaining']} still missing")
        if not self.tune.events:
            raise ABCError("tune body contained no notes or rests")

    def on_voice(self, vid, line_no):
        """V: switch: a new, independent timeline from beat 0."""
        if self.pending_tie is not None:
            raise ABCError(
                f"line {line_no}: tie '-' crosses a voice change (V:{vid})")
        if self.tuplet is not None:
            raise ABCError(
                f"line {line_no}: tuplet crosses a voice change (V:{vid})")
        if self.grace_pending:
            raise ABCError(
                f"line {line_no}: grace notes have no following note "
                f"before the voice change (V:{vid})")
        if self.current_repeat is not None:
            raise ABCError(
                f"line {line_no}: repeat crosses a voice change (V:{vid}) "
                f"-- put the repeat marks inside each voice")
        self.voice = vid
        if vid not in self.tune.voices:
            self.tune.voices.append(vid)
        self.now = Fraction(0, 1)
        self.bar_acc.clear()
        self.last_note_events = None
        self.last_group_kind = None
        self.last_group_was_chord = False

    def on_marker(self, mtype):
        """Record repeat beat boundaries at the current (voice, now)."""
        if mtype == "|:":
            self.current_repeat = {"voice": self.voice, "start": self.now}
        elif mtype == "[1":
            self.current_repeat["head_end"] = self.now
        elif mtype == ":|":
            if "head_end" in self.current_repeat:
                self.current_repeat["ending1_end"] = self.now
            else:
                self.current_repeat["mid"] = self.now
        elif mtype == "[2":
            self.current_repeat["head2_end"] = self.now
        elif mtype == "|end":
            if "head2_end" in self.current_repeat:
                self.current_repeat["ending2_end"] = self.now
            else:
                self.current_repeat["end"] = self.now
            self.tune.repeats.append(self.current_repeat)
            self.current_repeat = None

    def on_tie(self, line_no):
        """A '-' after a note group: the next group merges into it.

        Ties may LAND on a chord tone (`C- [CE]` merges into the C) but
        may never LEAVE a chord (`[CE]- X` raises): a tie out of a chord
        has no single pitch for the '-' to mean, so it is refused loudly
        instead of guessed.
        """
        if self.grace_pending:
            raise ABCError(f"line {line_no}: grace notes cannot be tied")
        if self.last_group_kind == "rest":
            raise ABCError(f"line {line_no}: cannot tie a rest")
        if not self.last_note_events:
            raise ABCError(
                f"line {line_no}: stray tie '-' with no note before it")
        if self.last_group_was_chord:
            raise ABCError(
                f"line {line_no}: ties out of a chord are not supported")
        self.pending_tie = self.last_note_events

    # -- the char scanner -------------------------------------------------
    def scan_text(self, text, line_no):
        pos = 0
        n = len(text)
        while pos < n:
            ch = text[pos]
            if ch.isspace():
                pos += 1
                continue
            if ch == "[":
                recs, pos = self.parse_chord_at(text, pos, line_no)
                self.emit_group(recs, line_no)
                continue
            if ch == "{":
                midis, pos = self.parse_grace_at(text, pos, line_no)
                self.grace_pending.extend(midis)
                continue
            if ch == "(":
                pos = self.parse_tuplet_at(text, pos, line_no)
                continue
            if ch == ")":
                raise ABCError(
                    f"line {line_no}: ')' slur endings are not supported "
                    f"(only tuplet '(p' markers)")
            if ch == "-":
                self.on_tie(line_no)
                pos += 1
                continue
            if ch == "]":
                raise ABCError(
                    f"line {line_no}: ']' without a matching '['")
            if ch == "}":
                raise ABCError(
                    f"line {line_no}: '}}' without a matching '{{'")
            if ch in "zZ":
                pos += 1
                mult, pos = _parse_note_length(text, pos, line_no)
                self.emit_group(
                    [("rest", None, self.tune.default_len * mult)], line_no)
                continue
            if ch in "^_=" or ch.upper() in "ABCDEFG":
                rec, pos = self.parse_note_at(text, pos, line_no)
                self.emit_group([rec], line_no)
                continue
            if ch.isdigit() or ch == "/":
                raise ABCError(
                    f"line {line_no}: stray length {ch!r} with no note "
                    f"or rest")
            raise ABCError(
                f"line {line_no}: cannot parse {ch!r} in tune body")

    # -- note/chord/grace/tuplet parsers ----------------------------------
    def parse_note_at(self, text, pos, line_no):
        """One note: [accidentals] letter [octave marks] [length].

        Accidentals, bar memory, key signature, octave, and MIDI range
        follow parser.py exactly; returns (("note", midi, dur), pos).
        """
        acc_text = ""
        while pos < len(text) and text[pos] in "^_=":
            acc_text += text[pos]
            pos += 1
        if pos >= len(text) or text[pos].upper() not in "ABCDEFG":
            raise ABCError(
                f"line {line_no}: accidental {acc_text!r} not followed "
                f"by a note letter")
        letter = text[pos].upper()
        lower = text[pos].islower()
        pos += 1
        marks = ""
        while pos < len(text) and text[pos] in ",'":
            marks += text[pos]
            pos += 1
        mult, pos = _parse_note_length(text, pos, line_no)
        if acc_text:
            if (len(set(acc_text)) > 1
                    or ("=" in acc_text and len(acc_text) > 1)):
                raise ABCError(
                    f"line {line_no}: contradictory accidentals "
                    f"{acc_text!r}")
            if acc_text.startswith("="):
                acc = 0
            elif acc_text.startswith("^"):
                acc = len(acc_text)
            else:
                acc = -len(acc_text)
            self.bar_acc[letter] = acc
        elif letter in self.bar_acc:
            acc = self.bar_acc[letter]
        else:
            acc = self.key_sig.get(letter, 0)
            self.bar_acc[letter] = acc
        octave = 5 if lower else 4
        octave -= marks.count(",")
        octave += marks.count("'")
        midi = 12 * (octave + 1) + _LETTER_SEMITONE[letter] + acc
        if not 0 <= midi <= 127:
            raise ABCError(
                f"line {line_no}: note out of MIDI range: "
                f"{letter}{marks} -> {midi}")
        return ("note", midi, self.tune.default_len * mult), pos

    def parse_chord_at(self, text, pos, line_no):
        """`[CEG]`: simultaneous notes; returns (recs, pos).

        Each chord tone keeps its own written length; the clock advances
        by the FIRST tone's duration (the ABC 2.1 draft rule). Bar lines,
        rests, nested chords, and ties inside a chord are refused loudly.
        """
        pos += 1  # consume '['
        recs = []
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                raise ABCError(
                    f"line {line_no}: unterminated chord '['")
            ch = text[pos]
            if ch == "]":
                pos += 1
                break
            if ch == "[":
                raise ABCError(
                    f"line {line_no}: nested chords are not supported")
            if ch in "zZ":
                raise ABCError(
                    f"line {line_no}: rests cannot appear inside a chord")
            if ch == "-":
                raise ABCError(
                    f"line {line_no}: ties inside a chord are not "
                    f"supported (put the tie after the chord)")
            if ch == "(":
                raise ABCError(
                    f"line {line_no}: tuplets inside a chord are not "
                    f"supported (put the tuplet before the chord)")
            rec, pos = self.parse_note_at(text, pos, line_no)
            recs.append(rec)
        if not recs:
            raise ABCError(f"line {line_no}: empty chord '[]'")
        return recs, pos

    def parse_grace_at(self, text, pos, line_no):
        """`{gag}`: grace notes; returns (midi list, pos).

        Written lengths inside the braces are accepted but do not affect
        timing: the steal is split equally (see emit_group). Grace notes
        update the accidental memory like ordinary notes.
        """
        if self.tuplet is not None:
            raise ABCError(
                f"line {line_no}: grace notes inside a tuplet are not "
                f"supported")
        pos += 1  # consume '{'
        midis = []
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                raise ABCError(
                    f"line {line_no}: unterminated grace notes '{{'")
            ch = text[pos]
            if ch == "}":
                pos += 1
                break
            if ch == "{":
                raise ABCError(
                    f"line {line_no}: nested grace notes are not supported")
            if ch in "zZ":
                raise ABCError(
                    f"line {line_no}: rests cannot be grace notes")
            if ch == "-":
                raise ABCError(
                    f"line {line_no}: grace notes cannot be tied")
            if ch == "(":
                raise ABCError(
                    f"line {line_no}: tuplets inside grace notes are not "
                    f"supported")
            rec, pos = self.parse_note_at(text, pos, line_no)
            midis.append(rec[1])
        if not midis:
            raise ABCError(f"line {line_no}: empty grace notes '{{}}'")
        return midis, pos

    def parse_tuplet_at(self, text, pos, line_no):
        """`(p`, `(p:q`, `(p:q:r`: set the tuplet state for the next r
        groups. A `(` not starting a tuplet is a slur: refused loudly.
        Returns the new position."""
        pos += 1  # consume '('
        if pos >= len(text) or not text[pos].isdigit():
            raise ABCError(
                f"line {line_no}: '(' begins a tuplet '(p[:q[:r]]' here; "
                f"slurs are not supported")
        if self.tuplet is not None:
            raise ABCError(
                f"line {line_no}: nested tuplets are not supported")

        def digits():
            nonlocal pos
            out = ""
            while pos < len(text) and text[pos].isdigit():
                out += text[pos]
                pos += 1
            return out

        p = int(digits())
        q = None
        r = None
        if pos < len(text) and text[pos] == ":":
            pos += 1
            q_text = digits()
            if not q_text:
                raise ABCError(
                    f"line {line_no}: tuplet '({p}:' is missing its q")
            q = int(q_text)
            if pos < len(text) and text[pos] == ":":
                pos += 1
                r_text = digits()
                if not r_text:
                    raise ABCError(
                        f"line {line_no}: tuplet '({p}:{q}:' is missing "
                        f"its r")
                r = int(r_text)
        if p < 1:
            raise ABCError(f"line {line_no}: bad tuplet '({p}'")
        if q is None:
            q = _TUPLET_DEFAULT_Q.get(p)
            if q is None:
                raise ABCError(
                    f"line {line_no}: tuplet '({p}' needs an explicit "
                    f"time ratio -- write '({p}:q[:r]'")
        if q < 1:
            raise ABCError(f"line {line_no}: bad tuplet ratio q={q}")
        if r is None:
            r = p
        if r < 1:
            raise ABCError(f"line {line_no}: bad tuplet length r={r}")
        self.tuplet = {"p": p, "q": q, "r": r, "remaining": r}
        return pos

    # -- group emission ----------------------------------------------------
    def apply_grace(self, recs):
        """Steal half the target group's duration for pending grace notes.

        Mutates recs[0] (the target: first chord tone for chords) to its
        remaining duration; creates the grace events just before it. Each
        grace note gets an equal share of the steal. Returns the steal
        (the target's start offset), as a Fraction.
        """
        target_dur = recs[0][2]
        steal = target_dur / 2
        per = steal / len(self.grace_pending)
        for k, midi in enumerate(self.grace_pending):
            self.tune.events.append(Event21(
                "note", midi, self.now + k * per, per,
                voice=self.voice, grace=True))
        recs[0] = (recs[0][0], recs[0][1], target_dur - steal)
        return steal

    def emit_group(self, recs, line_no):
        """Emit one parsed group: a note, a rest, or a chord's tones.

        recs: [("note", midi, dur)] * n or [("rest", None, dur)]. Applies
        grace-note stealing, tuplet scaling (exact rational q/p), and
        tie merging, in that order. A written group always consumes one
        tuplet slot, even when a tie merges it into the previous group.
        """
        # 1. A written group is one tuplet slot, merged or not.
        finishing_tuplet = False
        if self.tuplet is not None:
            self.tuplet["remaining"] -= 1
            if self.tuplet["remaining"] == 0:
                finishing_tuplet = True
        # 2. Grace notes steal from this group (notes/chords only).
        start_offset = Fraction(0, 1)
        if self.grace_pending:
            if recs[0][0] == "rest":
                raise ABCError(
                    f"line {line_no}: grace notes must be followed by a "
                    f"note or chord, not a rest")
            if self.tuplet is not None:
                raise ABCError(
                    f"line {line_no}: grace notes before a tuplet are "
                    f"not supported")
            start_offset = self.apply_grace(recs)
            self.grace_pending = []
        # 3. Tuplet timing: p notes in the time of q (exact rational).
        tag = None
        if self.tuplet is not None:
            p, q = self.tuplet["p"], self.tuplet["q"]
            recs = [(kind, midi, dur * q / p) for kind, midi, dur in recs]
            tag = (p, q)
        if finishing_tuplet:
            self.tuplet = None
        # 4. Tie merge, or create fresh events.
        first_dur = recs[0][2]
        was_chord = len(recs) > 1
        if self.pending_tie is not None:
            if recs[0][0] == "rest":
                raise ABCError(
                    f"line {line_no}: cannot tie to a rest")
            # A tie landing on a chord merges the matching tone(s); any
            # unmatched chord tones are new notes starting here. A lone
            # note must match, else the tie is bogus -- at least one
            # rec must merge or the '-' connected different pitches.
            matched_any = False
            fresh = []
            for kind, midi, dur in recs:
                target = next(
                    (e for e in self.pending_tie if e.midi == midi), None)
                if target is None:
                    fresh.append((midi, dur))
                else:
                    matched_any = True
                    # The merge: durations add; the split beat is
                    # recorded so the writer can re-emit '-' exactly.
                    target.dur = target.dur + dur
                    target.tie_splits.append(self.now + start_offset)
            if not matched_any:
                raise ABCError(
                    f"line {line_no}: tie '-' connects different "
                    f"pitches (nothing in the tied group matches)")
            created = [Event21("note", midi, self.now + start_offset, dur,
                               voice=self.voice, tuplet=tag)
                       for midi, dur in fresh]
            self.tune.events.extend(created)
            self.last_note_events = self.pending_tie + created
            self.last_group_kind = "note"
            self.pending_tie = None
        else:
            created = []
            for kind, midi, dur in recs:
                if kind == "rest":
                    ev = Event21("rest", None, self.now + start_offset,
                                 dur, voice=self.voice)
                else:
                    ev = Event21("note", midi, self.now + start_offset,
                                 dur, voice=self.voice, tuplet=tag)
                created.append(ev)
            self.tune.events.extend(created)
            self.last_note_events = (
                None if recs[0][0] == "rest" else created)
            self.last_group_kind = recs[0][0]
        self.last_group_was_chord = was_chord
        # The clock advances by the group's full written duration: the
        # grace steal (start_offset) was already subtracted from
        # first_dur by apply_grace, so add it back -- the grace notes
        # occupied [now, now+steal) and the target [now+steal, ...).
        self.now += first_dur + start_offset


# ---------------------------------------------------------------------------
# Entry point: parse_abc21.
# ---------------------------------------------------------------------------
_VOICE_LINE = re.compile(r"^V:\s*(\S+)")
_FIELD_LINE = re.compile(r"^[A-Za-z]:")


def parse_abc21(text):
    """Parse ABC 2.1 notation text -> Tune21. Raises ABCError on malformed
    input -- including anything still unsupported, which is refused loudly
    rather than rendered wrong.

    Supported beyond the v0.1.0 core: ties (`-`, merged across bar lines),
    chords (`[CEG]`), tuplets (`(3`, `(p:q:r`), grace notes (`{g}` stealing
    half the following note), repeats (`|: ... :|`, `:|:`, `[1`/`[2`
    first/second endings -- expanded eagerly), and multi-voice (`V:1`,
    `V:2`, each voice an independent timeline from beat 0).

    Still unsupported (ABCError, never a silent misread): nested
    repeats, nested tuplets/chords/grace, slurs (`(`/`)`), decorations
    (`!`/`+`), inline field changes (`K:`/`M:`/`L:`/`Q:` mid-tune),
    ties out of chords, ties/rests/grace in illegal positions, bare
    `(5`/`(7`/`(9` tuplets without an explicit `:q`, and repeats or
    tuplets crossing a voice change.
    """
    if not isinstance(text, str) or not text.strip():
        raise ABCError("empty ABC text: nothing to parse")

    # -- pass 1: headers vs body ------------------------------------------
    # Like the base parser, plus: V: lines before the body declare voices
    # (kept in order); V: lines in the body switch the current voice; any
    # other mid-tune field line (K:, M:, ...) is an unsupported inline
    # field change and raises instead of being misread as notes.
    raw_headers = {}
    voice_decls = []
    last_header_voice = None  # the last V: before the body owns the
    # body's first music (ABC semantics: the most recent V: field
    # selects the voice; a V: line before any music line is a header
    # field, not a switch)
    body_items = []  # (line_no, "line", text) | (line_no, "voice", id)
    in_body = False
    seen_k = False  # the K: field ends the header in ABC: everything
    # after it is body, so a second K:/M:/... line is an inline field
    # change (unsupported -> raises), and V: lines there are voice
    # switches, not declarations
    header_line_nos = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("%", 1)[0].strip()  # '%' starts a comment
        if not line:
            continue
        if seen_k:
            in_body = True
        if (not in_body and len(line) >= 2 and line[0].isalpha()
                and line[1] == ":"):
            f, value = line[0], line[2:].strip()
            if f == "K":
                seen_k = True
            if f == "V":
                vid = value.split()[0] if value.split() else ""
                if not vid:
                    raise ABCError(f"line {line_no}: empty V: voice id")
                if vid not in voice_decls:
                    voice_decls.append(vid)
                last_header_voice = vid
            else:
                raw_headers[f] = value
                header_line_nos[f] = line_no
        else:
            in_body = True
            if line.startswith(("w:", "W:")):
                continue  # lyrics: skipped, not misread
            m = _VOICE_LINE.match(line)
            if m:
                body_items.append((line_no, "voice", m.group(1)))
                continue
            if _FIELD_LINE.match(line):
                raise ABCError(
                    f"line {line_no}: inline field changes are not "
                    f"supported (only V: voice switches)")
            body_items.append((line_no, "line", line))

    if "X" not in raw_headers:
        raise ABCError("missing required X: (reference number) header")

    tune = Tune21(headers=dict(raw_headers), voices=list(voice_decls))
    if "M" in raw_headers:
        tune.beats_per_bar = _parse_meter(
            raw_headers["M"], header_line_nos["M"])
    if "L" in raw_headers:
        tune.default_len = _parse_length_unit(
            raw_headers["L"], header_line_nos["L"])
    if "Q" in raw_headers:
        tune.bpm = _parse_tempo(raw_headers["Q"], header_line_nos["Q"])
    key_sig = _key_accidentals(raw_headers.get("K", "C").strip() or "C")

    if not body_items:
        raise ABCError("no tune body found after the headers")

    # -- passes 2+3: tokenize, then expand repeats -------------------------
    tokens = []
    for line_no, kind, payload in body_items:
        if kind == "voice":
            tokens.append((line_no, "voice", payload))
        else:
            tokens.extend(_tokenize_line(payload, line_no))
    tokens = _expand_repeats(tokens)

    # -- pass 4: scan to events --------------------------------------------
    _BodyParser(tune, key_sig,
                initial_voice=last_header_voice).run(tokens)
    return tune


# ---------------------------------------------------------------------------
# Voice helpers: split a multi-voice tune and render one voice alone.
# ---------------------------------------------------------------------------
def tune_voice(tune, voice):
    """Tune -> Tune21 holding only one voice's events (same headers and
    tempo; that voice's repeat records travel along). Renders through the
    unchanged render path, so each voice is an independently audible
    line."""
    sub = Tune21(
        headers=dict(tune.headers),
        beats_per_bar=tune.beats_per_bar,
        default_len=tune.default_len,
        bpm=tune.bpm,
        voices=[voice],
        repeats=[dict(r) for r in getattr(tune, "repeats", [])
                 if r.get("voice") == voice],
    )
    sub.events = [e for e in tune.events
                  if getattr(e, "voice", None) == voice]
    return sub


def render_voice(tune, voice, path=None, sr=44100, stereo=True):
    """Render one voice of a multi-voice tune to float32 audio."""
    return render_tune(tune_voice(tune, voice), path=path, sr=sr,
                       stereo=stereo)


# ---------------------------------------------------------------------------
# Writer: write_abc21 -- Tune -> ABC 2.1 text.
#
# Inverts parse_abc21: ties (from tie_splits), chords (shared starts),
# tuplets (from (p, q) tags, lengths written UNSCALED so re-parsing
# scales exactly once), grace notes (grace flag; the target's length is
# restored from the stolen time), repeats (from the recorded beat map),
# voices (V: sections). A plain v0.1.0 Tune degrades gracefully: no 2.1
# markup is invented -- in particular the writer never invents a tie
# between two separate same-pitch notes; only recorded tie_splits emit
# '-'. Ties that merged INTO a chord tone are written as sustained chord
# tones (ABC has no notation for them); the durations still round-trip.
# ---------------------------------------------------------------------------
def _section_voices(tune):
    """Voice ids in section order: unvoiced first, then declarations."""
    seen = []
    for e in tune.events:
        v = getattr(e, "voice", None)
        if v not in seen:
            seen.append(v)
    ordered = []
    if None in seen:
        ordered.append(None)
    for v in getattr(tune, "voices", []) or []:
        if v is not None and v in seen and v not in ordered:
            ordered.append(v)
    for v in seen:
        if v not in ordered:
            ordered.append(v)
    return ordered


def _units(evs):
    """Events -> sounding units: ("group", [events]) or
    ("grace", [grace events], [target events]). Chord tones (shared
    start) form one group; a grace run plus its whole target chord form
    one grace unit."""
    units = []
    i, n = 0, len(evs)
    while i < n:
        if getattr(evs[i], "grace", False):
            j = i
            while j < n and getattr(evs[j], "grace", False):
                j += 1
            if j < n:
                s = evs[j].start
                k = j
                while (k < n and evs[k].start == s
                       and not getattr(evs[k], "grace", False)):
                    k += 1
                units.append(("grace", evs[i:j], evs[j:k]))
                i = k
            else:
                # Grace notes with no target: unwritable as grace; emit
                # them as plain short notes rather than dropping music.
                units.append(("group", evs[i:j]))
                i = j
        else:
            s = evs[i].start
            k = i
            while (k < n and evs[k].start == s
                   and not getattr(evs[k], "grace", False)):
                k += 1
            units.append(("group", evs[i:k]))
            i = k
    return units


def _unit_tag(unit):
    """The tuplet tag shared by a unit's notes, or None if mixed/absent."""
    evs = unit[1] if unit[0] == "group" else unit[2]
    tags = {getattr(e, "tuplet", None) for e in evs if e.kind == "note"}
    if len(tags) == 1:
        return next(iter(tags))
    return None


def _unit_start(unit):
    evs = unit[1] if unit[0] == "group" else (unit[1] + unit[2])
    return evs[0].start if evs else Fraction(0, 1)


def _unscale(dur, tag, default_len):
    """Tuplet notes are written at their UNSCALED length so the parser's
    q/p scaling applies exactly once on re-parse."""
    if tag is not None:
        p, q = tag
        return dur * p / q
    return dur


def _emit_single(e, tag, key_sig, bar_acc, default_len, restore=Fraction(0, 1)):
    """One note/rest token; tie_splits re-emit the '-' markup.

    restore: extra duration added to the note (grace steal payback).
    """
    if e.kind == "rest":
        return "z" + _length_token(
            _unscale(e.dur + restore, tag, default_len), default_len)
    splits = sorted(getattr(e, "tie_splits", None) or [])
    bounds = [e.start] + [s for s in splits
                          if e.start < s < e.start + e.dur + restore]
    bounds.append(e.start + e.dur + restore)
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        d = _unscale(b - a, tag, default_len)
        segs.append(_note_token(e.midi, key_sig, bar_acc)
                    + _length_token(d, default_len))
    return "-".join(segs)


def _emit_unit(unit, key_sig, bar_acc, default_len):
    """One sounding unit -> ABC text (no bar lines or tuplet openers)."""
    tag = _unit_tag(unit)
    if unit[0] == "grace":
        grace_evs, target_evs = unit[1], unit[2]
        g = "{" + "".join(_note_token(e.midi, key_sig, bar_acc)
                          for e in grace_evs) + "}"
        if not target_evs:
            return g  # degenerate: grace emitted bare above
        steal = sum((e.dur for e in grace_evs), Fraction(0, 1))
        if len(target_evs) == 1:
            t = _emit_single(target_evs[0], None, key_sig, bar_acc,
                             default_len, restore=steal)
        else:
            tones = [_emit_single(target_evs[0], None, key_sig, bar_acc,
                                  default_len, restore=steal)]
            tones += [_emit_single(e, None, key_sig, bar_acc, default_len)
                      for e in target_evs[1:]]
            t = "[" + "".join(tones) + "]"
        return g + t
    evs = unit[1]
    if len(evs) == 1:
        return _emit_single(evs[0], tag, key_sig, bar_acc, default_len)
    # Chord: simultaneous tones, each with its own length.
    return ("[" + "".join(_emit_single(e, tag, key_sig, bar_acc,
                                       default_len) for e in evs) + "]")


_BARISH = ("|", "|:", ":|", "[1", "[2", "|]", "[|", "||")


def _write_section(tune, voice, key_sig):
    """One voice's events -> ABC body text with bars and repeat marks.

    Repeats are written COLLAPSED (`|: head [1 e1 :| [2 e2 |`): the
    parser expanded them eagerly, so the writer emits the head once and
    skips its second copy, using the repeat's recorded beat map. Bar
    lines are drawn on the collapsed (written) beat clock. Repeat
    markers clear the writer's accidental memory, matching the plain
    bars the parser inserts after each marker on re-parse.
    """
    evs = [e for e in tune.events if getattr(e, "voice", None) == voice]
    evs.sort(key=lambda e: (e.start, e.midi if e.midi is not None else -1))
    bpb = tune.beats_per_bar
    dl = tune.default_len
    units = _units(evs)

    # -- collapse repeats ---------------------------------------------
    # (collapsed_start, unit) in write order; (collapsed_start, marker)
    # in emission order. The skipped head copies are identical to the
    # written head (the parser built them by re-reading the same text).
    reps = sorted((r for r in getattr(tune, "repeats", [])
                   if r.get("voice") == voice),
                  key=lambda r: r["start"])
    collapsed = []  # (cstart, unit)
    markers = []    # (cstart, text)
    cnow = Fraction(0, 1)
    pos, n = 0, len(units)

    def span(u):
        sev = u[1] if u[0] == "group" else u[1] + u[2]
        s0 = sev[0].start
        return max(e.start + e.dur for e in sev) - s0

    def take_until(beat):
        nonlocal pos, cnow
        while pos < n and _unit_start(units[pos]) < beat:
            u = units[pos]
            collapsed.append((cnow, u))
            cnow += span(u)
            pos += 1

    def skip_until(beat):
        nonlocal pos
        while pos < n and _unit_start(units[pos]) < beat:
            pos += 1

    for rep in reps:
        take_until(rep["start"])
        markers.append((cnow, "|:"))
        if rep.get("ending2_end") is not None:  # first/second endings
            take_until(rep["head_end"])
            markers.append((cnow, "[1"))
            take_until(rep["ending1_end"])
            markers.append((cnow, ":|"))
            skip_until(rep["head2_end"])  # head's 2nd copy: written once
            markers.append((cnow, "[2"))
            take_until(rep["ending2_end"])
        else:  # simple repeat: head written once, copy skipped
            take_until(rep["mid"])
            markers.append((cnow, ":|"))
            skip_until(rep["end"])
    take_until(Fraction(10 ** 18, 1))

    # -- emit ------------------------------------------------------------
    marked_boundaries = {int(mb // bpb) for mb, mt in markers
                         if mb % bpb == 0 and mt in _BARISH}
    tags = [_unit_tag(u) for _, u in collapsed]
    parts = []
    bar_acc = {}
    current_bar = 0
    bars_on_line = 0
    mi = 0

    def flush_markers(upto):
        """Emit due repeat markers (they reset accidental memory, like
        the bar lines the parser inserts at the same spots)."""
        nonlocal mi
        while mi < len(markers) and markers[mi][0] <= upto:
            mb, mt = markers[mi]
            mi += 1
            parts.append(mt + " ")
            bar_acc.clear()

    for ui, (ustart, unit) in enumerate(collapsed):
        flush_markers(ustart)
        # Tuplet opener at the start of each maximal run of one tag.
        tag = tags[ui]
        if tag is not None and (ui == 0 or tags[ui - 1] != tag):
            run = 0
            j = ui
            while j < len(collapsed) and tags[j] == tag:
                run += 1
                j += 1
            parts.append(f"({tag[0]}:{tag[1]}:{run}")
        # Bar lines by exact bar arithmetic; a repeat marker already
        # sitting on the boundary draws it instead of a plain "|".
        bar_index = int(ustart // bpb)
        while current_bar < bar_index:
            if (current_bar + 1) not in marked_boundaries:
                parts.append("| ")
                bars_on_line += 1
                if bars_on_line >= 4:
                    parts.append("\n")
                    bars_on_line = 0
            current_bar += 1
            bar_acc.clear()
        parts.append(_emit_unit(unit, key_sig, bar_acc, dl) + " ")
    flush_markers(Fraction(10 ** 18, 1))  # trailing markers (e.g. final "|")
    if not parts or parts[-1].strip() not in _BARISH:
        parts.append("|]")
    return "".join(parts).replace("\n ", "\n").strip()


def write_abc21(tune):
    """Serialize a Tune/Tune21 to ABC 2.1 text.

    Round-trips through parse_abc21: ties, chords, tuplets, grace notes,
    repeats with endings, and multiple voices all survive. Plain v0.1.0
    tunes degrade gracefully (no 2.1 markup is invented).
    """
    headers = tune.headers
    key_sig = _key_accidentals((headers.get("K", "C") or "C").strip())
    lines = []
    for f in ("X", "T", "M", "L", "Q", "K"):
        if f in headers:
            lines.append(f"{f}:{headers[f]}")
    voices = _section_voices(tune)
    multi = any(v is not None for v in voices)
    if multi:
        for v in voices:
            if v is not None:
                lines.append(f"V:{v}")
    for v in voices:
        if multi and v is not None:
            lines.append(f"V:{v}")
        lines.append(_write_section(tune, v, key_sig))
    return "\n".join(lines) + "\n"


__all__ = [
    "Event21",
    "Tune21",
    "parse_abc21",
    "write_abc21",
    "tune_voice",
    "render_voice",
]
